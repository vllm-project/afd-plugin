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

from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner import fail_if_unsupported_ubatching
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner
from afd_plugin.validation import assert_compatible_afd_stack

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

logger = logging.getLogger(__name__)


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

    def init_device(self):
        """Initialize the native GPU worker and swap in the FFN runner."""

        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDFFNWorker.init_device",
            expected_role="ffn",
        )
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD FFN runtime currently supports only the vLLM v1 "
                "GPUModelRunner interface; set VLLM_USE_V2_MODEL_RUNNER=0",
            )

        fail_if_unsupported_ubatching(self.vllm_config)

        super().init_device()
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
        )
        self.model_runner = GPUFFNModelRunner(self.vllm_config, self.device)

        torch.accelerator.empty_cache()

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
            if self.model_runner.connector.control_plane is None:
                raise NotImplementedError(
                    "GPU FFN only supports control-plane-driven connectors; "
                    "connectors without a control plane (control_plane is None) "
                    "are not supported.",
                )

            payload = self.model_runner.connector.control_plane.recv_dp_metadata_list()
            dp_metadata_list = payload.dp_metadata_list
            is_attn_graph_capturing = payload.is_graph_capturing
            is_warmup = payload.is_warmup

            if self.model_runner.use_cuda_graph and (
                is_warmup or is_attn_graph_capturing
            ):
                self.model_runner.capture_model(
                    dp_metadata_list=dp_metadata_list,
                    is_warmup=is_warmup,
                    is_attn_graph_capturing=is_attn_graph_capturing,
                    transport_spec=payload.transport_spec,
                )
            else:
                self.model_runner.execute_model(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    is_warmup=is_warmup,
                    transport_spec=payload.transport_spec,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            self._ffn_loop_error = None
            raise RuntimeError("AFD FFN worker loop failed") from error

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is not None:
            event.set()
        try:
            self.model_runner.shutdown()
        finally:
            thread = self._ffn_thread
            if thread is not None:
                thread.join(timeout=5)
            self._ffn_thread = None
            self._ffn_shutdown_event = None
        self.raise_ffn_loop_error_if_any()

    def shutdown(self) -> None:
        self.stop_ffn_server_loop()
        super().shutdown()


__all__ = ["AFDFFNWorker"]
