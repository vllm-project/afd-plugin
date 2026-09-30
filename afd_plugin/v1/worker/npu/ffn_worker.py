# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side worker for AFD execution."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.platforms import current_platform
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.validation import (
    NPU_FFN_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

logger = logging.getLogger(__name__)

FFN_SHUTDOWN_TIMEOUT_SECONDS = 5
FFN_STARTUP_THREAD_TIMEOUT_SECONDS = 30


class AFDNPUFFNWorker(NPUWorker):
    """FFN worker that owns a connector-driven NPU daemon loop."""

    afd_expected_role = "ffn"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Import after vLLM-Ascend completes platform initialization. Importing
        # its MoE modules from the general-plugin hook can race Ascend's own
        # ops package initialization and leave DeviceOperator partially loaded.
        import afd_plugin.compat.patches.npu.force_load_balance  # noqa: F401

        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)
        self._ffn_thread: threading.Thread | None = None
        self._ffn_shutdown_event: threading.Event | None = None
        self._ffn_loop_error: BaseException | None = None
        self._ffn_loop_started_event: threading.Event | None = None
        self._cpu_binding_attempted = False

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUFFNWorker.init_device",
            expected_role="ffn",
            expected_worker_qualname_override=NPU_FFN_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        if self.use_v2_model_runner:
            validate_npu_model_runner_v2_config(
                self.vllm_config,
                expected_role="ffn",
                device_type="npu",
            )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        self.model_runner = AFDNPUFFNModelRunner(self.vllm_config, self.device)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.start_ffn_server_loop()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        raise RuntimeError(
            "AFD NPU FFN workers are connector-driven; scheduler-driven "
            "execute_model() is not supported.",
        )

    def start_ffn_server_loop(self) -> None:
        if self._ffn_thread is not None and self._ffn_thread.is_alive():
            self.raise_ffn_loop_error_if_any()
            return

        self.raise_ffn_loop_error_if_any()
        connector = self.model_runner.connector
        is_async_cam = isinstance(connector, CAMAsyncAFDConnector)
        if is_async_cam:
            logger.info(
                "CAM FFN startup entered rank=%d graph=%s",
                connector.world_rank,
                self.model_runner._async_cam_ffn_graph_enabled,
            )
        try:
            if is_async_cam:
                logger.info("CAM FFN local warmup start rank=%d", connector.world_rank)
                self.model_runner.warmup_async_cam_ffn_graph()
                logger.info("CAM FFN local warmup done rank=%d", connector.world_rank)
            if not connector.is_initialized:
                self.model_runner.initialize_afd_connector()
            if is_async_cam:
                logger.info("CAM FFN startup store start rank=%d", connector.world_rank)
                connector.prepare_startup_coordination()
                logger.info("CAM FFN startup store done rank=%d", connector.world_rank)
                if self.model_runner._async_cam_ffn_graph_enabled:
                    executor = self.model_runner._layered_executor
                    if executor is None:
                        raise RuntimeError("CAM FFN graph has no layered executor")
                    connector.publish_ffn_mode(executor.layer_ids[0])
                    connector.wait_for_attention_warmup_prepared()
                    connector.publish_ffn_warmup_started()
                    self.model_runner.warmup_async_cam_ffn_communication()
                    connector.publish_ffn_warmup_done()
                    connector.wait_for_communication_warmup_done()
                    self.model_runner.capture_async_cam_ffn_graph()
                else:
                    connector.publish_ffn_mode(None)
        except Exception as exc:
            if is_async_cam and connector._startup_store is not None:
                connector.publish_ffn_startup(error=str(exc))
            raise

        self._bind_cpus_once()
        self._ffn_shutdown_event = threading.Event()
        self._ffn_loop_started_event = threading.Event()
        self._ffn_loop_error = None

        def ffn_worker_loop() -> None:
            try:
                self._run_ffn_server_loop()
            except Exception as exc:
                shutdown_event = self._ffn_shutdown_event
                if shutdown_event is not None and shutdown_event.is_set():
                    logger.debug(
                        "AFD NPU FFN receive loop stopped during shutdown",
                        exc_info=True,
                    )
                    return
                self._ffn_loop_error = exc
                if self._ffn_loop_started_event is not None:
                    self._ffn_loop_started_event.set()
                logger.exception("AFD NPU FFN worker loop failed")
                if is_async_cam and connector._startup_store is not None:
                    connector.publish_ffn_startup(error=str(exc))

        self._ffn_thread = threading.Thread(
            target=ffn_worker_loop,
            name="afd-npu-ffn-worker-loop",
            daemon=True,
        )
        try:
            self._ffn_thread.start()
            if is_async_cam:
                if not self._ffn_loop_started_event.wait(
                    timeout=FFN_STARTUP_THREAD_TIMEOUT_SECONDS
                ):
                    self.raise_ffn_loop_error_if_any()
                    raise TimeoutError("AFD NPU FFN service thread did not start")
                self.raise_ffn_loop_error_if_any()
                if not self._ffn_thread.is_alive():
                    raise RuntimeError(
                        "AFD NPU FFN service thread stopped during startup"
                    )
                connector.publish_ffn_startup()
        except Exception as exc:
            if is_async_cam and connector._startup_store is not None:
                connector.publish_ffn_startup(error=str(exc))
            raise

    def _bind_cpus_once(self) -> None:
        if self._cpu_binding_attempted:
            return
        self._cpu_binding_attempted = True

        if not get_ascend_config().enable_cpu_binding:
            return

        try:
            physical_npu_id = current_platform.device_id_to_physical_device_id(
                self.local_rank
            )
            bind_cpus(self.local_rank, npu_id=physical_npu_id)
        except Exception as exc:
            logger.warning(
                "Bind cpus failed in rank%s: %s Skip binding cpu.",
                self.local_rank,
                exc,
            )

    def _run_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is None:
            return

        torch.npu.set_device(self.device)
        if isinstance(self.model_runner.connector, CAMAsyncAFDConnector):
            self._ffn_loop_started_event.set()
        while not event.is_set():
            if self.model_runner.connector.control_plane is None:
                self.model_runner.execute_connector_driven_step()
                torch.npu.synchronize()
                continue

            payload = self.model_runner.connector.control_plane.recv_dp_metadata_list()
            dp_metadata_list = payload.dp_metadata_list
            is_attn_graph_capturing = payload.is_graph_capturing
            is_warmup = payload.is_warmup
            is_profile = payload.is_profile
            is_graph_replaying = payload.is_graph_replaying

            self.model_runner.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
                is_warmup=is_warmup,
                is_profile=is_profile,
                is_graph_replaying=is_graph_replaying,
            )
            torch.npu.synchronize()

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            self._ffn_loop_error = None
            raise RuntimeError("AFD NPU FFN worker loop failed") from error

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is not None:
            event.set()

        # A graph replay may be blocked in DR until Attention sends another
        # work item. Do not destroy its communicator or graph-owned tensors
        # while the worker thread is still using them.
        connector = self.model_runner.connector
        graph_cam = (
            isinstance(connector, CAMAsyncAFDConnector)
            and self.model_runner._async_cam_ffn_graph_enabled
        )
        thread = self._ffn_thread
        if graph_cam and thread is not None:
            thread.join(timeout=FFN_SHUTDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "AFD NPU FFN graph loop is still active; retain graph and "
                    "connector resources until the worker process exits"
                )
        # Release a completed graph before its communicator. The legacy path
        # still closes the connector before joining its receive thread.
        if graph_cam:
            self.model_runner.release_async_cam_ffn_graph()
            connector.close(release_buffers=False)
        else:
            connector.close()
        if thread is not None and not graph_cam:
            thread.join(timeout=FFN_SHUTDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "AFD NPU FFN worker loop did not stop after connector close",
                )
        if graph_cam:
            connector.release_closed_buffers()
        self._ffn_thread = None
        self._ffn_shutdown_event = None
        self._ffn_loop_started_event = None
        self.raise_ffn_loop_error_if_any()

    def shutdown(self) -> None:
        # Stop the connector-driven daemon before NPUWorker releases the model
        # runner and its tensors.
        self.stop_ffn_server_loop()
        super().shutdown()


__all__ = ["AFDNPUFFNWorker"]
