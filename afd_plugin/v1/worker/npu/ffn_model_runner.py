# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side model runner for AFD execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import DPMetadata
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, graph_capture

from afd_plugin.compat.npu import (
    ascend_forward_context,
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
    AFDTransferContext,
)
from afd_plugin.connectors.npu.async_cam import (
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
)
from afd_plugin.v1.worker.ffn_model_runner import _set_moe_layer_index

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    from afd_plugin.connectors import AFDConnectorBase

logger = init_logger(__name__)


class AFDNPUFFNModelRunner(NPUModelRunner):
    """Connector-driven NPU FFN runner for AFD execution."""

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        afd_config = self.parse_config(vllm_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        fail_if_unsupported_npu_afd_features(
            vllm_config,
            afd_config=afd_config,
        )
        rank, _ = _resolve_world_ranks()
        local_rank = int(device.index)
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.use_aclgraph = _use_npu_aclgraph(vllm_config, self)
        self._acl_graphs: dict[tuple, dict[str, Any]] = {}
        self.graph_pool = (
            current_platform.get_global_graph_pool() if self.use_aclgraph else None
        )
        self.prof = create_afd_npu_profiler("ffn")
        self._is_shutdown = False

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self) -> None:
        super().load_model()
        self.connector.bind_model(self.model)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None

    def profile_run(self) -> None:
        return None

    def execute_ffn_step(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN requires dp_metadata_list")
        if bool(self.use_aclgraph) and (is_graph_capturing or is_warmup):
            logger.debug(
                "AFD NPU FFN execute_ffn_step enters capture_model; "
                "key=%s is_graph_capturing=%s is_warmup=%s",
                self._make_graph_key(dp_metadata_list),
                is_graph_capturing,
                is_warmup,
            )
            self.capture_model(
                dp_metadata_list=dp_metadata_list,
                is_warmup=is_warmup,
                is_attn_graph_capturing=is_graph_capturing,
            )
            return None
        self.execute_model(dp_metadata_list=dp_metadata_list)
        return None

    def execute_connector_driven_step(self) -> None:
        if self.connector.control_plane is not None:
            raise RuntimeError(
                "execute_connector_driven_step requires a connector-driven "
                "AFD connector",
            )
        step_afd_npu_profiler(self.prof)
        self._ffn_forward_connector_driven()
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
        step_afd_npu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN is connector-driven")
        graph_key = self._make_graph_key(dp_metadata_list)
        acl_graphs = self._acl_graphs
        graph_info = acl_graphs.get(graph_key)
        graph_enabled = bool(self.use_aclgraph)
        run_mode = graph_run_mode(
            is_warmup=is_warmup and graph_enabled,
            is_graph_capturing=is_graph_capturing and graph_enabled,
            graph_enabled=graph_enabled,
            graph_exists=graph_info is not None,
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            logger.debug(
                "AFD NPU FFN replaying ACL graph; key=%s cached_graphs=%d",
                graph_key,
                len(acl_graphs),
            )
            graph_info["graph"].replay()
            return None
        if run_mode in (AFDGraphRunMode.WARMUP, AFDGraphRunMode.CAPTURE):
            return self.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )

        self._ffn_forward(dp_metadata_list=dp_metadata_list)
        return None

    def _make_graph_key(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> tuple:
        return make_ffn_graph_key(
            dp_metadata_list,
            attention_size=int(self.connector.attn_size),
            ffn_size=int(self.connector.ffn_size),
            fallback=int(self.max_num_tokens),
        )

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        aclgraph_runtime_mode: CUDAGraphMode | None = None,
        is_graph_capturing: bool = False,
        update_connector_state: bool = True,
    ) -> torch.Tensor | None:
        if update_connector_state:
            assert self.connector.control_plane, (
                "Only DP metadata control plane supports ",
                "update_connector_state == True.",
            )
            self.connector.control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                ),
            )
        num_stages = max(len(dp_metadata_list), 1)
        afd_metadata = AFDForwardContextMetadata(
            tokens_start_loc=[],
            requests_start_loc=[],
            stage_idx=0,
            connector=self.connector,
            tokens_lens=[],
            num_stages=num_stages,
        )
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        rank_ffn_output = None

        for layer_idx in _ffn_layer_indices(self):
            for stage_idx in stage_ids:
                num_tokens_across_dp = _ffn_token_counts_across_ranks(
                    self.connector,
                    dp_metadata_list,
                    stage_idx,
                    fallback=self.max_num_tokens,
                )
                num_tokens = _ffn_token_count_for_rank(
                    self.connector,
                    num_tokens_across_dp,
                )
                # DBO stages can have different token counts. Build a fresh
                # Ascend context for each stage so its MC2 padding mask matches
                # the hidden states received for that stage.
                dp_num_tokens_across_dp = _to_dp_level_token_counts(
                    num_tokens_across_dp,
                    dp_size=int(self.vllm_config.parallel_config.data_parallel_size),
                )
                with ascend_forward_context(
                    vllm_config=self.vllm_config,
                    afd_metadata=afd_metadata,
                    model_instance=self.model,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=dp_num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                ) as forward_context:
                    payload = self.connector.recv_attn_output(
                        ubatch_idx=stage_idx,
                        layer_idx=layer_idx,
                        max_num_tokens=self.max_num_tokens,
                    )
                    context = payload.context
                    metadata = context.metadata
                    states = context.states
                    hidden_states = payload.hidden_states
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    forward_context.dp_metadata = dp_metadata_list.get(stage_idx)
                    forward_context.additional_kwargs["afd_metadata"] = metadata
                    assert states, "Context.states must not be None"
                    _set_moe_layer_index(forward_context, layer_idx)

                    rank_ffn_output = self.model.compute_ffn_output(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                    )
                    _send_ffn_output(
                        self.connector,
                        rank_ffn_output,
                        context,
                        stage_idx=stage_idx,
                    )
        return rank_ffn_output

    def _ffn_forward_connector_driven(
        self,
    ) -> torch.Tensor | AFDF2ATransferPayload | None:
        stage_idx = 0
        rank_ffn_output = None
        connector = cast(CAMAsyncAFDConnector, self.connector)

        for _ in _ffn_layer_indices(self):
            work_item = connector.recv_ffn_work_item(
                stage_idx=stage_idx,
                max_num_tokens=self.max_num_tokens,
            )
            hidden_states = work_item.hidden_states
            metadata = work_item.context.metadata
            states = work_item.context.states
            if not isinstance(states, AFDAsyncTransferState):
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
            ) as forward_context:
                forward_context.dp_metadata = None
                forward_context.additional_kwargs["afd_metadata"] = metadata
                _set_moe_layer_index(forward_context, layer_idx)

                rank_ffn_output = self.model.compute_ffn_output(
                    hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    group_list=states.group_list,
                    dynamic_scales=states.dynamic_scales,
                    expand_x_shared=states.expand_x_shared,
                    dynamic_scales_shared=states.dynamic_scales_shared,
                )
                rank_ffn_output = connector.send_ffn_work_item_output(
                    work_item,
                    rank_ffn_output,
                )
        return rank_ffn_output

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_aclgraph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN capture requires dp_metadata_list")

        logger.debug(
            "AFD NPU FFN capture_model start; key=%s is_warmup=%s "
            "is_attn_graph_capturing=%s",
            self._make_graph_key(dp_metadata_list),
            is_warmup,
            is_attn_graph_capturing,
        )
        start_free_memory = int(torch.npu.mem_get_info()[0])
        set_cudagraph_capturing_enabled(True)
        try:
            if is_warmup:
                self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=False,
                )
            else:
                with graph_capture(device=self.device):
                    self._capture_graphs(
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_memory = int(torch.npu.mem_get_info()[0])
        graph_size = max(0, int(start_free_memory - end_free_memory))
        logger.debug(
            "AFD NPU FFN capture_model end; key=%s graph_size=%d",
            self._make_graph_key(dp_metadata_list),
            graph_size,
        )
        return graph_size

    def _capture_graphs(
        self,
        *,
        aclgraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_attn_graph_capturing: bool = True,
    ) -> None:
        assert self.connector.control_plane, (
            "Only DP metadata control plane supports graph capturing."
        )
        graph_key = self._make_graph_key(dp_metadata_list)
        if graph_key in self._acl_graphs:
            logger.debug(
                "AFD NPU FFN ACL graph capture skipped for existing key=%s",
                graph_key,
            )
            return

        logger.debug("AFD NPU FFN capturing ACL graph for key=%s", graph_key)
        graph = torch.npu.NPUGraph()
        logger.debug("AFD NPU FFN created NPUGraph for key=%s", graph_key)
        self.connector.control_plane.update_state_from_dp_metadata(
            _make_dp_metadata_payload(
                dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            ),
        )
        logger.debug("AFD NPU FFN updated connector state for key=%s", graph_key)
        with torch.npu.graph(graph, pool=self.graph_pool):
            logger.debug(
                "AFD NPU FFN entered NPU graph context for key=%s",
                graph_key,
            )
            output = self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                is_graph_capturing=is_attn_graph_capturing,
                update_connector_state=False,
            )
            logger.debug("AFD NPU FFN left _ffn_forward for key=%s", graph_key)
        self._acl_graphs[graph_key] = {
            "graph": graph,
            "output": output,
        }
        logger.debug("AFD NPU FFN captured ACL graph for key=%s", graph_key)

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        raise RuntimeError("AFD NPU FFN runners do not sample tokens")

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        stop_afd_npu_profiler(self.prof)
        if self.connector.is_initialized:
            self.connector.close()
        super().shutdown()
        self._is_shutdown = True


def _send_ffn_output(
    connector: AFDConnectorBase,
    ffn_output: torch.Tensor | AFDF2ATransferPayload,
    context: AFDTransferContext,
    *,
    stage_idx: int,
) -> None:
    if not isinstance(ffn_output, AFDF2ATransferPayload):
        connector.send_ffn_output(
            ffn_output,
            context,
            ubatch_idx=stage_idx,
        )
        return

    kwargs: dict[str, object] = {"ubatch_idx": stage_idx}
    if ffn_output.shared_output is not None:
        kwargs["expand_x_shared"] = ffn_output.shared_output
    connector.send_ffn_output(
        ffn_output.routed_output,
        context,
        **kwargs,
    )


def _ffn_layer_indices(runner: AFDNPUFFNModelRunner) -> range | list[int]:
    num_layers = max(int(runner.num_layers or 0), 1)
    if not runner.afd_config.compute_gate_on_attention:
        return range(num_layers)
    hf_config = runner.model_config.hf_config
    return [
        layer_idx
        for layer_idx in range(num_layers)
        if _is_moe_layer(hf_config, layer_idx)
    ]


def _is_moe_layer(hf_config: object, layer_idx: int) -> bool:
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)
    return (
        hf_config.n_routed_experts is not None
        and layer_idx >= hf_config.first_k_dense_replace
        and layer_idx % moe_layer_freq == 0
    )


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


def _ffn_token_counts_across_ranks(
    connector: AFDConnectorBase,
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    stage_idx: int,
    *,
    fallback: int,
) -> torch.Tensor:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    if dp_metadata is None:
        values = [max(1, int(fallback))] * int(connector.ffn_size)
    else:
        attention_counts = _to_int_list(dp_metadata.num_tokens_across_dp_cpu)
        # Expand DP-level counts to AFD-level counts when TP > 1.
        # With TP, attn_size = num_attention_ranks includes TP workers
        # but num_tokens_across_dp_cpu only has dp_size entries.
        # Each DP rank's token count is replicated tp_size times because
        # all TP workers within the same DP rank process the same tokens.
        if (
            len(attention_counts) < int(connector.attn_size)
            and int(connector.attn_size) % len(attention_counts) == 0
        ):
            tp_size = int(connector.attn_size) // len(attention_counts)
            attention_counts = [
                attention_counts[i // tp_size] for i in range(int(connector.attn_size))
            ]
        if (
            len(attention_counts) >= int(connector.attn_size)
            and int(connector.attn_size) >= int(connector.ffn_size)
            and int(connector.attn_size) % int(connector.ffn_size) == 0
        ):
            group_size = int(connector.attn_size) // int(connector.ffn_size)
            values = [
                max(1, sum(attention_counts[idx * group_size : (idx + 1) * group_size]))
                for idx in range(int(connector.ffn_size))
            ]
        else:
            values = [max(1, int(fallback))] * int(connector.ffn_size)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _ffn_token_count_for_rank(
    connector: AFDConnectorBase,
    num_tokens_across_dp: torch.Tensor,
) -> int:
    values = _to_int_list(num_tokens_across_dp)
    role_rank = int(connector.topology.role_rank)
    if role_rank >= len(values):
        return max(1, values[0] if values else 1)
    return max(1, int(values[role_rank]))


def _to_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in value.tolist()]


def _to_dp_level_token_counts(
    num_tokens_across_dp: torch.Tensor,
    *,
    dp_size: int,
) -> torch.Tensor:
    """Project AFD-level token counts back to DP-level for vLLM's forward context.

    ``num_tokens_across_dp`` has ``ffn_size`` entries (one per AFD role_rank).
    With TP > 1 each DP rank's count is replicated ``tp_size`` times, so the
    layout is ``[dp0, dp0, dp1, dp1, ...]``.  vLLM's ``DPMetadata.make()``
    expects exactly ``dp_size`` entries where ``[dp_rank] == batchsize``.
    """
    numel = len(num_tokens_across_dp)
    if numel == dp_size:
        return num_tokens_across_dp
    if dp_size <= 0 or numel % dp_size != 0:
        return num_tokens_across_dp
    tp_size = numel // dp_size
    # Take the first TP slot of each DP group (all TP slots are identical).
    indices = [dp_idx * tp_size for dp_idx in range(dp_size)]
    return num_tokens_across_dp[indices].contiguous()


def _use_npu_aclgraph(vllm_config: VllmConfig, runner: object) -> bool:
    inherited = bool(runner.use_aclgraph)
    if bool(vllm_config.model_config.enforce_eager):
        return False

    mode_name = vllm_config.compilation_config.cudagraph_mode.name
    return inherited or mode_name in {
        "FULL",
        "FULL_AND_PIECEWISE",
        "FULL_DECODE_ONLY",
        "PIECEWISE",
    }


__all__ = ["AFDNPUFFNModelRunner"]
