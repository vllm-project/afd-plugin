# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for AFD GPU execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config import update_config as update_vllm_config
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import DPMetadata, get_forward_context, set_forward_context
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.workspace import reset_workspace_manager

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.v1.worker.attention_model_runner import (
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
    validate_cuda_graph_mode,
)

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


class GPUFFNModelRunner(LoRAModelRunnerMixin):
    """FFN model runner for AFD GPU execution.

    FFN steps are driven by the connector control plane rather than the vLLM
    scheduler. GPU only supports control-plane-driven connectors, so the runner
    asserts ``connector.control_plane is not None`` at construction; connectors
    without a control plane (``control_plane is None``) are not supported.
    """

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: VllmConfig, device: object) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.load_config = vllm_config.load_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.afd_config = self.parse_config(vllm_config)
        fail_if_unsupported_ubatching(vllm_config)
        self.afd_cudagraph_policy = validate_cuda_graph_mode(
            vllm_config,
            role="ffn",
        )
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        # TODO: Async GPU connector will be supported in the future
        assert self.connector.control_plane is not None, (
            "GPU model runner only supports control-plane-driven connectors"
        )

        self.model: Any | None = None
        self.model_memory_usage = 0
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.use_cuda_graph = bool(
            self.afd_cudagraph_policy.enable_ffn_graph_cache,
        )
        self._cuda_graphs: dict[tuple, dict[str, Any]] = {}
        self._graph_memory_pool: Any | None = None
        self.prof = create_afd_gpu_profiler("ffn")

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def get_model(self) -> Any:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load the vLLM model."""

        model_loader = get_model_loader(self.load_config)
        with DeviceMemoryProfiler() as profiler:
            if self.model is None:
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config,
                )
            else:
                model_loader.load_weights(
                    self.model,
                    model_config=self.model_config,
                )
        self.connector.bind_model(self.model)
        self.model_memory_usage = profiler.consumed_memory

    def profile_run(self) -> None:
        pass

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None

    def execute_model(
        self,
        scheduler_output: SchedulerOutput | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        step_afd_gpu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner requires dp_metadata_list")
        graph_key = make_ffn_graph_key(dp_metadata_list)
        cuda_graph_info = self._cuda_graphs.get(graph_key)
        run_mode = graph_run_mode(
            is_warmup=is_warmup,
            is_graph_capturing=is_graph_capturing,
            graph_enabled=bool(self.use_cuda_graph),
            graph_exists=cuda_graph_info is not None,
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            cuda_graph_info["graph"].replay()
            return None

        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        return None

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        update_connector_state: bool = True,
    ) -> torch.Tensor | None:
        if update_connector_state:
            self.connector.control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                    is_warmup=is_warmup,
                ),
            )

        rank_ffn_output = None
        num_layers = max(int(self.num_layers or 0), 1)
        experts_layer_indices = frozenset(
            self.model.get_experts_layer_indices(),
        )
        layer_indices = (
            tuple(sorted(experts_layer_indices))
            if self.afd_config.compute_gate_on_attention
            else tuple(range(num_layers))
        )
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        with _ffn_forward_context(self.vllm_config) as forward_context:
            for layer_idx in layer_indices:
                uses_remote_experts = layer_idx in experts_layer_indices
                routing_spec = (
                    self.model.get_experts_routing_spec(layer_idx)
                    if uses_remote_experts and self.afd_config.compute_gate_on_attention
                    else None
                )
                for stage_idx in stage_ids:
                    payload = self.connector.recv_attn_output(
                        ubatch_idx=stage_idx,
                        routing_spec=routing_spec,
                    )
                    hidden_states = payload.hidden_states
                    context = payload.context
                    metadata = context.metadata
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = dp_metadata_list.get(
                            metadata.stage_idx,
                        )  # type: ignore
                        forward_context.additional_kwargs["afd_metadata"] = metadata
                        _set_moe_layer_index(forward_context, layer_idx)
                    if (
                        uses_remote_experts
                        and self.afd_config.compute_gate_on_attention
                    ):
                        router_logits = payload.router_logits
                        assert router_logits is not None
                        rank_ffn_output = self.model.compute_experts_output(
                            hidden_states,
                            layer_idx,
                            router_logits,
                        )
                    else:
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states,
                            layer_idx,
                        )
                    self.connector.send_ffn_output(rank_ffn_output, context)
        return rank_ffn_output

    def _execute_eager_mode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def update_config(self, overrides: dict[str, Any]) -> None:
        for config_name, config_overrides in overrides.items():
            config = getattr(self, config_name)
            updated_config = update_vllm_config(config, config_overrides)
            setattr(self, config_name, updated_config)

    def reload_weights(self) -> None:
        if self.model is None:
            raise RuntimeError("Cannot reload weights before model is loaded")
        self.load_model()

    def _dummy_run(
        self,
        cudagraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_attn_graph_capturing: bool,
    ) -> None:
        mode_name = getattr(cudagraph_runtime_mode, "name", str(cudagraph_runtime_mode))
        if mode_name.endswith(".FULL"):
            mode_name = "FULL"

        if mode_name == "FULL":
            if self._graph_memory_pool is None:
                self._graph_memory_pool = torch.cuda.graph_pool_handle()
            graph_key = make_ffn_graph_key(dp_metadata_list)
            cudagraph = torch.cuda.CUDAGraph()
            # DP metadata receive/update is a control-plane side effect and must
            # complete before CUDA graph capture starts.
            self.connector.control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                ),
            )
            with torch.cuda.graph(cudagraph, pool=self._graph_memory_pool):
                output = self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    update_connector_state=False,
                )
            self._cuda_graphs[graph_key] = {
                "graph": cudagraph,
                "input_hidden_states": output,
                "output": output,
            }
        else:
            self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_cuda_graph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner.capture_model requires metadata")

        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device=self.device):
                if is_warmup:
                    self.connector.control_plane.update_state_from_dp_metadata(
                        _make_dp_metadata_payload(
                            dp_metadata_list,
                            is_graph_capturing=False,
                            is_warmup=True,
                        ),
                    )
                    self._ffn_forward(
                        dp_metadata_list=dp_metadata_list,
                        is_graph_capturing=False,
                        is_warmup=True,
                        update_connector_state=False,
                    )
                else:
                    self._dummy_run(
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        return int(cuda_graph_size)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        raise RuntimeError("FFN runners do not sample tokens")

    def add_lora(self, lora_request: Any) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    @property
    def lora_config(self) -> None:
        return None

    @property
    def is_pooling_model(self) -> bool:
        return False

    def get_supported_tasks(self) -> tuple[Any, ...]:
        return ()

    # Patch reason: the FFN runner owns GPUModelRunner-equivalent CUDA state
    # without inheriting GPUModelRunner's shutdown implementation.
    # Patch functionality: mirror the pinned native GPU resource cleanup and
    # then close AFD-owned profiler and connector resources.
    # Signature: matches GPUModelRunner.shutdown; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def shutdown(self) -> None:
        # ### PATCH START: release native-equivalent and AFD-owned GPU state.
        stop_afd_gpu_profiler(self.prof)
        try:
            for graph_info in self._cuda_graphs.values():
                graph_info["graph"].reset()
            self._cuda_graphs.clear()
            self._graph_memory_pool = None
            self.vllm_config.compilation_config.static_forward_context.clear()
            self.model = None
            _ROPE_DICT.clear()
            reset_workspace_manager()
        finally:
            self.connector.close()
        # ### PATCH END: release native-equivalent and AFD-owned GPU state.


def _resolve_world_ranks() -> tuple[int, int]:
    group = get_world_group()
    return int(group.rank), int(group.local_rank)


@contextmanager
def _ffn_forward_context(vllm_config: VllmConfig):
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        yield get_forward_context()


def _set_moe_layer_index(forward_context: object, layer_idx: int) -> None:
    all_moe_layers = forward_context.all_moe_layers
    if not all_moe_layers:
        return

    target = f".layers.{int(layer_idx)}."
    for idx, layer_name in enumerate(all_moe_layers):
        if target in f".{layer_name}.":
            forward_context.moe_layer_index = idx
            return


def _make_dp_metadata_payload(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    *,
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
) -> AFDControlPayload:
    return AFDControlPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=is_graph_capturing,
        is_warmup=is_warmup,
    )


__all__ = ["GPUFFNModelRunner"]
