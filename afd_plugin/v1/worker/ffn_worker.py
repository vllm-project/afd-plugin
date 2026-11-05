# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side worker for AFD GPU execution."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory, AFDRoutedExpertsConnector
from afd_plugin.connectors.base import AFDControlPlaneClosedError
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = logging.getLogger(__name__)
FFN_LOOP_JOIN_TIMEOUT_S = 5


class AFDFFNWorker(Worker):
    """FFN worker that owns the AFD daemon loop.

    The FFN side enters through native ``vllm serve`` after AFD configuration
    selects this worker. The native scheduler may still be present, but FFN
    work is driven by connector metadata.
    """

    afd_expected_role = "ffn"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )
        self._ffn_thread: threading.Thread | None = None
        self._ffn_shutdown_event: threading.Event | None = None
        self._ffn_loop_error: BaseException | None = None

    # Patch reason: native device initialization does not validate the AFD role
    # or select its model architecture before allocating GPU resources.
    # Patch functionality: validate and select the AFD model before delegating
    # native device, distributed, workspace and runner initialization.
    # Signature: matches vLLM v0.32.0 Worker.init_device exactly: (self).
    # Upstream source: vllm/v1/worker/gpu_worker.py, Worker.init_device;
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation exception: the large native device/distributed setup remains
    # in super().init_device(); this override contains only AFD preflight.
    # Removal plan: remove when native worker startup supports AFD preflight.
    def init_device(self):
        """Initialize the native GPU worker with the connector-driven runner."""

        # ### PATCH START: validate and select the AFD model.
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDFFNWorker.init_device",
            expected_role="ffn",
        )
        validate_gpu_model_runner_v2_config(
            self.vllm_config,
            expected_role="ffn",
            device_type=self.device_config.device_type,
        )
        afd_model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="cuda",
        )
        self.vllm_config.model_config = afd_model_config
        self.model_config = afd_model_config
        # ### PATCH END: validate and select the AFD model.

        super().init_device()
        torch.accelerator.empty_cache()

    # Upstream: vLLM 5df63a72, gpu_worker.py::Worker._make_model_runner.
    # Select the connector-driven runner before native model loading; routed
    # executors also require the Triton expert-weight layout. The native factory
    # annotates V1, so connector-driven V2 runners keep its return-type exception.
    def _make_model_runner(self) -> GPUModelRunner:
        """Construct the connector-driven runner via the native factory hook."""
        afd_config = parse_afd_config(self.vllm_config, expected_role="ffn")
        connector_cls = AFDConnectorFactory.get_connector_class(afd_config.connector)
        if issubclass(connector_cls, AFDRoutedExpertsConnector):
            from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
                GPUFFNRoutedExpertsModelRunner,
                select_routed_moe_backend,
            )

            select_routed_moe_backend(self.vllm_config)
            return GPUFFNRoutedExpertsModelRunner(self.vllm_config, self.device)  # type: ignore[return-value]
        return GPUFFNModelRunner(self.vllm_config, self.device)  # type: ignore[return-value]

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """FFN workers do not allocate KV cache."""

        return {}

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Skip KV cache allocation and start the FFN connector loop."""

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.model_runner.initialize_afd_connector()
        self.start_ffn_server_loop()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """FFN workers perform no warmup/capture; model execution is driven
        entirely by connector metadata.
        """

        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Fail fast if the default scheduler tries to execute FFN work."""

        raise RuntimeError(
            "AFD FFN workers are connector-driven; scheduler-driven "
            "execute_model() is not supported.",
        )

    def start_ffn_server_loop(self) -> None:
        if self._ffn_thread is not None and self._ffn_thread.is_alive():
            self.raise_ffn_loop_error_if_any()
            return

        self.raise_ffn_loop_error_if_any()
        connector = self.model_runner.connector
        if not connector.is_initialized:
            self.model_runner.initialize_afd_connector()

        self._ffn_shutdown_event = threading.Event()
        self._ffn_loop_error = None

        def ffn_worker_loop() -> None:
            try:
                self._run_ffn_server_loop()
            except Exception as exc:
                self._ffn_loop_error = exc
                logger.exception("AFD FFN worker loop failed")

        self._ffn_thread = threading.Thread(
            target=ffn_worker_loop,
            name="afd-ffn-worker-loop",
            daemon=True,
        )
        self._ffn_thread.start()

    def _run_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is None:
            return

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        while not event.is_set():
            control_plane = self.model_runner.connector.control_plane
            if control_plane is None:
                raise NotImplementedError(
                    "GPU FFN only supports control-plane-driven connectors; "
                    "connectors without a control plane (control_plane is None) "
                    "are not supported.",
                )

            try:
                payload = control_plane.recv_dp_metadata_list()
            except AFDControlPlaneClosedError:
                if not event.is_set():
                    raise
                break
            dp_metadata_list = payload.dp_metadata_list
            is_attn_graph_capturing = payload.is_graph_capturing
            is_warmup = payload.is_warmup
            is_graph_replaying = payload.is_graph_replaying

            if self.model_runner.use_cuda_graph and (
                is_warmup or is_attn_graph_capturing
            ):
                self.model_runner.capture_model(
                    dp_metadata_list=dp_metadata_list,
                    is_warmup=is_warmup,
                    is_attn_graph_capturing=is_attn_graph_capturing,
                )
            else:
                self.model_runner.execute_model(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    is_warmup=is_warmup,
                    is_graph_replaying=is_graph_replaying,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            # EngineCore and Worker can both stop the loop; retain its failure.
            raise RuntimeError("AFD FFN worker loop failed") from error

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is not None:
            event.set()
        control_plane = self.model_runner.connector.control_plane
        if control_plane is not None:
            control_plane.close()
        thread = self._ffn_thread
        if thread is not None:
            thread.join(timeout=FFN_LOOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError("AFD FFN worker loop did not stop before shutdown")
        self._ffn_thread = None
        self._ffn_shutdown_event = None
        self.raise_ffn_loop_error_if_any()

    # Upstream source: vLLM v0.32.0, vllm/v1/worker/gpu_worker.py,
    # Worker.shutdown; 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Patch reason: the native worker does not own AFD's metadata receiver.
    # Patch functionality: stop that receiver before releasing native resources,
    # including when the joined loop reports an earlier failure.
    # Signature: matches Worker.shutdown exactly: (self) -> None.
    # Delegation exception: native cleanup stays in super().shutdown().
    # Removal plan: remove when the native worker owns the AFD receiver lifecycle.
    def shutdown(self) -> None:
        # ### PATCH START: release native state only after the receiver exits.
        try:
            self.stop_ffn_server_loop()
        finally:
            if self._ffn_thread is None:
                super().shutdown()
        # ### PATCH END: release native state only after the receiver exits.


__all__ = ["AFDFFNWorker"]
