# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MRV2 Async CAM FFN runtime, composed from the model loader and connector."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

import torch
from vllm.logger import init_logger
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from afd_plugin.async_cam_startup import (
    GRAPH_STARTUP_TIMEOUT_SECONDS,
    POLL_SECONDS,
    FFNGraphPlan,
)
from afd_plugin.compat.npu import (
    ascend_forward_context,
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import (
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.npu.async_cam import (
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.envs import async_cam_layered_gmm_enabled
from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index
from afd_plugin.validation import (
    cam_receive_capacity,
    requested_ffn_mode,
    validate_npu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.tasks import SupportedTask
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


logger = init_logger(__name__)


class FFNModel(Protocol):
    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        group_list: torch.Tensor | None,
        dynamic_scales: torch.Tensor | None,
    ) -> torch.Tensor | AFDF2ATransferPayload: ...
    def get_experts_layer_indices(self) -> tuple[int, ...]: ...
    def get_async_cam_w4a8_layers(self) -> tuple[list[W4A8LayerWeights], str]: ...


class AsyncNPUFFNModelRunner:
    afd_expected_role = "ffn"

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        connector: CAMAsyncAFDConnector,
    ) -> None:
        validate_npu_model_runner_v2_config(
            vllm_config, expected_role="ffn", device_type=device.type
        )
        self.vllm_config, self.device, self.connector = vllm_config, device, connector
        self.model_config = vllm_config.model_config
        self.afd_config = parse_afd_config(vllm_config, expected_role="ffn")
        fail_if_unsupported_npu_afd_features(vllm_config, afd_config=self.afd_config)
        self.model: FFNModel | None = None
        self.model_memory_usage = 0
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.layer_ids: tuple[int, ...] = ()
        self._layered_executor: AsyncCAMW4A8Executor | None = None
        self._layered_gmm_requested = async_cam_layered_gmm_enabled()
        self.requested_mode = requested_ffn_mode(vllm_config)
        self.effective_mode = "eager"
        self._graph: torch.npu.NPUGraph | None = None
        self._graph_capture_count = self._graph_replay_count = 0
        self._graph_pending = self._graph_closed = False
        self._graph_receiver_exited = False
        self._graph_failure: Exception | None = None
        self._service_announced = self._is_shutdown = False
        self._failure: Exception | None = None
        self.prof = create_afd_npu_profiler("ffn")

    def load_model(self) -> None:
        # Import the loader only in the isolated, initialized NPU worker.
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.utils.mem_utils import DeviceMemoryProfiler
        from vllm_ascend.ascend_forward_context import (
            set_mc2_mask,
            set_mc2_tokens_capacity,
        )

        if self.model is not None:
            raise RuntimeError("Async FFN weights are already loaded")
        # Native MoE constructs its dispatchers even when CAM supplies routed
        # inputs. MRV2 initializes these shared helpers before model creation;
        # preserve that constructor contract without an Attention runner.
        speculative_config = self.vllm_config.speculative_config
        decode_query_len = 1 + (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        set_mc2_tokens_capacity(
            self.vllm_config,
            self.vllm_config.scheduler_config.max_num_seqs,
            decode_query_len,
        )
        set_mc2_mask(self.vllm_config, self.device)
        loader = get_model_loader(self.vllm_config.load_config)
        with DeviceMemoryProfiler() as profiler:
            self.model = cast(
                FFNModel,
                loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                ),
            )
        self.model_memory_usage = profiler.consumed_memory

    def get_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("Async FFN model has not been loaded")
        return cast(torch.nn.Module, self.model)

    def initialize_afd_connector(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "Async graph startup failed; retain resources"
            ) from self._failure
        if self.connector.is_initialized:
            return
        assert self.model is not None
        self.layer_ids = tuple(sorted(self.model.get_experts_layer_indices()))
        self._initialize_layered_executor()
        executor = self._layered_executor
        graph_error = (
            "requires layered W4A8 (AFD_ASYNC_CAM_LAYERED_GMM=1)"
            if executor is None
            else ""
        )
        self.connector.graph_startup.plan = FFNGraphPlan(
            requested=self.requested_mode,
            graph_error=graph_error,
            signature=executor.graph_signature(cam_receive_capacity())
            if executor
            else "",
            layer_ids=self.layer_ids,
        )
        try:
            self.connector.init_afd_connector()
            self.effective_mode = self.connector.graph_startup.mode
            if self.effective_mode == "full":
                assert executor is not None
                chunks = self.connector.graph_startup_chunks() * (
                    self.connector.attn_size // self.connector.tp_size
                )
                for _ in range(chunks):
                    self._ffn_forward_connector_driven(single=True)
                done = torch.npu.Event()
                done.record()
                self._wait_graph_device(done)
                self.connector.graph_startup.phase("warmup")
                stream = torch.npu.Stream(device=self.device)
                stream.wait_stream(torch.npu.current_stream(self.device))
                self._initialize_graph(stream)
                self._capture_graph()
                self.connector.graph_startup.phase("captured")
        except Exception as exc:
            self._failure = exc
            self.connector.graph_startup.fail(exc)
            raise
        logger.info(
            "Async FFN requested=%s mode=%s",
            self.requested_mode,
            self.effective_mode,
        )

    def _wait_graph_device(self, event: torch.npu.Event) -> None:
        deadline = time.monotonic() + GRAPH_STARTUP_TIMEOUT_SECONDS
        while not event.query():
            self.connector.graph_startup.check_failure()
            if time.monotonic() >= deadline:
                raise TimeoutError("Async graph warmup timed out; retain resources")
            time.sleep(POLL_SECONDS)

    def _initialize_graph(self, stream: torch.npu.Stream) -> None:
        self._graph_stream = stream
        self._graph_pool = torch.npu.graph_pool_handle()
        self._graph_tensors: list[torch.Tensor] = []

    def _check_graph_live(self) -> None:
        if self._graph_failure is not None:
            raise RuntimeError(
                "Async FFN graph failed; retain resources"
            ) from self._graph_failure
        if self._graph_closed:
            raise RuntimeError("Async FFN graph is closed")

    @torch.inference_mode()
    def _capture_graph(self) -> None:
        self._check_graph_live()
        if self._graph is not None:
            raise RuntimeError("Async FFN may only capture once")
        assert self._layered_executor is not None
        self._graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(
                self._graph, pool=self._graph_pool, stream=self._graph_stream
            ):
                self._graph_payload = self.connector.recv_attn_output()
                state = cast(AFDAsyncTransferState, self._graph_payload.context.states)
                assert state.token_nums_rankid_layeridx is not None
                assert state.group_list is not None
                assert state.dynamic_scales is not None
                self._graph_output = self._layered_executor(
                    self._graph_payload.hidden_states,
                    state.dynamic_scales,
                    state.group_list,
                    state.token_nums_rankid_layeridx,
                    retained_tensors=self._graph_tensors,
                )
                self.connector.send_ffn_output(
                    self._graph_output, self._graph_payload.context
                )
            self._graph_capture_count = 1
        except Exception as exc:
            self._graph_failure = exc
            raise

    @torch.inference_mode()
    def _submit_graph(self, count: int = 1) -> None:
        self._check_graph_live()
        if (
            self._graph is None
            or self._graph_capture_count != 1
            or self._graph_pending
            or self._graph_receiver_exited
            or count < 1
        ):
            raise RuntimeError("Async FFN graph is not ready for replay")
        self._graph_pending = True  # A failed launch may already have queued work.
        try:
            with torch.npu.stream(self._graph_stream):
                for _ in range(count):
                    self._graph.replay()
                    self._graph_replay_count += 1
        except Exception as exc:
            self._graph_failure = exc
            raise

    def _complete_graph(self) -> bool:
        self._check_graph_live()
        if not self._graph_pending:
            raise RuntimeError("Async FFN has no pending replay")
        try:
            # One fence per bounded batch. The original DR cannot be cancelled;
            # a blocked batch retains its graph, model and communication buffers.
            torch.npu.synchronize()
        except Exception as exc:
            self._graph_failure = exc
            raise
        self._graph_pending = False
        return True

    def _close_graph(self) -> None:
        if self._graph_closed:
            return
        self._check_graph_live()
        if self._graph_pending or not self._graph_receiver_exited:
            raise RuntimeError(
                "Async graph resources require receiver exit and device completion"
            )
        assert self._graph is not None
        self._graph.reset()
        self._graph = None
        self._graph_tensors.clear()
        del self._graph_payload, self._graph_output
        del self._graph_pool, self._graph_stream
        self._graph_closed = True

    def _initialize_layered_executor(self) -> None:
        assert self.model is not None
        reason = "switch disabled"
        if self._layered_gmm_requested:
            if not isinstance(self.connector, CAMAsyncAFDConnector):
                reason = "requires CAMAsyncAFDConnector"
            elif not self.afd_config.compute_gate_on_attention:
                reason = "requires Attention-side gate"
            else:
                layers, reason = self.model.get_async_cam_w4a8_layers()
                if layers:
                    if get_ascend_device_type() != AscendDeviceType.A3:
                        raise ValueError("layered W4A8 GMM requires Ascend 910C/A3")
                    if self.connector.dynamic_quant != 1:
                        raise ValueError("layered W4A8 GMM requires dynamicQuant=1")
                    if tuple(sorted(layer.layer_idx for layer in layers)) != tuple(
                        self.layer_ids
                    ):
                        raise ValueError(
                            "layered GMM weights do not cover the remote MoE layer IDs"
                        )
                    ensure_cam_async_ops_available()
                    self._layered_executor = AsyncCAMW4A8Executor(layers)
                    self.connector.layered_gmm_enabled = True
                    logger.info(
                        "AFD_ASYNC_CAM_LAYERED_GMM requested=1 actual=layered "
                        "quant=W4A8 layers=%d mode=%s layer_mapping=%s "
                        "swiglu_limit=%s ignored=%s",
                        len(layers),
                        "per-channel" if layers[0].per_channel else "per-group",
                        self._layered_executor.layer_id_to_slot is not None,
                        layers[0].swiglu_limit,
                        layers[0].swiglu_limit != 0.0,
                    )
                    return
        logger.info(
            "AFD_ASYNC_CAM_LAYERED_GMM requested=%s actual=legacy reason=%s",
            self._layered_gmm_requested,
            reason,
        )

    @torch.inference_mode()
    def execute_connector_driven_step(self) -> bool:
        step_afd_npu_profiler(self.prof)
        if self._graph is None:
            # Preserve main's eager layer loop and synchronization policy.
            self._ffn_forward_connector_driven()
            return True
        if not self._service_announced:
            self.connector.graph_startup.service_ready()
            self._service_announced = True
        # Match the reference's bounded submission count. Each DR still selects
        # ready work; this batch is not a request or model-forward boundary.
        self._submit_graph(len(self.layer_ids))
        return self._complete_graph()

    def _ffn_forward_connector_driven(
        self,
        *,
        single: bool = False,
    ) -> torch.Tensor | AFDF2ATransferPayload | None:
        assert self.model is not None
        stage_idx = 0
        rank_ffn_output = None
        connector = cast(CAMAsyncAFDConnector, self.connector)
        if self._layered_executor is not None:
            for _ in self.layer_ids[:1] if single else self.layer_ids:
                payload = connector.recv_attn_output()
                states = cast(AFDAsyncTransferState, payload.context.states)
                rank_ffn_output = self._layered_executor(
                    payload.hidden_states,
                    states.dynamic_scales,
                    states.group_list,
                    states.token_nums_rankid_layeridx,
                )
                connector.send_ffn_output(rank_ffn_output, payload.context)
            return rank_ffn_output

        for _ in self.layer_ids[:1] if single else self.layer_ids:
            work_item = connector.recv_ffn_work_item(
                stage_idx=stage_idx,
                max_num_tokens=self.max_num_tokens,
            )
            hidden_states = work_item.hidden_states
            metadata = work_item.context.metadata
            legacy_states = work_item.context.states
            if not isinstance(legacy_states, AFDAsyncTransferState):
                raise RuntimeError(
                    "CAM async FFN work item requires AFDAsyncTransferState",
                )
            layer_idx = work_item.layer_idx
            num_tokens = work_item.num_tokens
            afd_metadata = AFDForwardContextMetadata(
                tokens_start_loc=[0],
                requests_start_loc=[0],
                stage_idx=stage_idx,
                connector=self.connector,
                tokens_lens=[num_tokens],
                num_stages=1,
                tokens_unpadded_lens=[num_tokens],
            )

            with ascend_forward_context(
                vllm_config=self.vllm_config,
                afd_metadata=afd_metadata,
                model_instance=self.model,
                num_tokens=num_tokens,
                skip_mc2_mask=self.afd_config.compute_gate_on_attention,
            ) as forward_context:
                forward_context.dp_metadata = None
                forward_context.additional_kwargs["afd_metadata"] = metadata
                _set_moe_layer_index(forward_context, layer_idx)

                rank_ffn_output = self.model.compute_ffn_output(
                    hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    group_list=legacy_states.group_list,
                    dynamic_scales=legacy_states.dynamic_scales,
                )
                rank_ffn_output = connector.send_ffn_work_item_output(
                    work_item,
                    rank_ffn_output,
                )
        return rank_ffn_output

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def profile_run(self) -> None:
        pass

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ()

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        if self._failure is not None:
            raise RuntimeError(
                "Failed graph startup retains resources"
            ) from self._failure
        from vllm.v1.worker.gpu.shutdown import free_before_shutdown

        if self._graph is not None:
            self._close_graph()
        stop_afd_npu_profiler(self.prof)
        self.connector.close()
        self._layered_executor = None
        free_before_shutdown(self.vllm_config)
        self.model = None
        self._is_shutdown = True
