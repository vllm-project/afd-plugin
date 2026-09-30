# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Complete Attention-owned CAM MoE execution with native Ascend routing."""

import weakref

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.base_router import FusedMoERouter

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.envs import force_balanced_topk_ids_enabled
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    CAMDispatchLayout,
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)
from afd_plugin.model_executor.npu.async_cam_execution import (
    CAMAsyncPhase,
    require_cam_async_execution_context,
)
from afd_plugin.model_executor.remote_moe import AFDRemoteMoERunnerBase


def validate_remote_moe_config() -> None:
    # The Ascend wrapper can enable local EPLB despite explicit factory kwargs.
    from vllm_ascend.ascend_config import get_ascend_config

    eplb_config = get_ascend_config().eplb_config
    if eplb_config.dynamic_eplb:
        raise RuntimeError(
            "Remote MoE does not support Attention-local dynamic_eplb",
        )
    if eplb_config.expert_map_path is not None:
        raise RuntimeError(
            "Remote MoE does not support Attention-local expert_map_path",
        )
    if eplb_config.num_redundant_experts != 0:
        raise RuntimeError(
            "Remote MoE does not support Attention-local Ascend redundant experts",
        )


class AFDCAMAsyncMoERunner(AFDRemoteMoERunnerBase):
    """Complete routing, CAM transport, layout restoration, and shared experts."""

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
        *,
        num_shared_experts: int | None,
        attention_shared_experts: torch.nn.Module | None = None,
        shared_output_divisor_fp16: float = 1.0,
    ) -> None:
        super().__init__(
            layer_name=layer_name,
            moe_config=moe_config,
            router=router,
            routed_experts=routed_experts,
            enable_dbo=enable_dbo,
            gate=gate,
            shared_experts=shared_experts,
            shared_expert_gate=shared_expert_gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.num_shared_experts = num_shared_experts or 0
        # The outer MoE owns canonical shared weight names and their lifetime.
        self._attention_shared_experts = (
            None
            if attention_shared_experts is None
            else weakref.ref(attention_shared_experts)
        )
        self.shared_output_divisor_fp16 = shared_output_divisor_fp16

    def _route_native(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep Ascend imports lazy so GPU workers can load the same model shell.
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        assert self.gate is not None
        router_logits, _ = self.gate(hidden_states)
        experts = self.routed_experts
        # CAM transports routed-only IDs. Shared experts stay on Attention;
        # FFN owns routed scaling, so both native selector paths use unit scale.
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=experts.top_k,
            use_grouped_topk=experts.use_grouped_topk,
            renormalize=experts.renormalize,
            topk_group=experts.topk_group,
            num_expert_group=experts.num_expert_group,
            custom_routing_function=experts.custom_routing_function,
            scoring_func=experts.scoring_func,
            routed_scaling_factor=1.0,
            e_score_correction_bias=experts.e_score_correction_bias,
            mix_placement=False,
            num_logical_experts=router_logits.shape[1],
            num_shared_experts=self.num_shared_experts,
            num_experts=experts.global_num_experts,
        )
        if force_balanced_topk_ids_enabled():
            balanced_topk_ids = torch.arange(
                topk_ids.numel(), device=topk_ids.device, dtype=torch.int64
            ).reshape(topk_ids.shape)
            topk_ids.copy_(
                balanced_topk_ids.remainder(router_logits.shape[1]).to(topk_ids.dtype)
            )
        return topk_weights.to(torch.float32), topk_ids, router_logits

    def _dispatch_cam(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor | None,
        *,
        stage_idx: int,
        use_sequence_parallel: bool,
    ) -> tuple[torch.Tensor, CAMDispatchLayout]:
        """Send one routed shard while forward retains its completion state."""
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("Remote MoE execution requires AFD forward metadata")
        payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=use_sequence_parallel,
        )
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_id,
            stage_idx=stage_idx,
            seq_len=int(payload.hidden_states.shape[0]),
        )
        afd_metadata.connector.send_attn_output(
            payload.hidden_states,
            AFDTransferContext(metadata=metadata),
            topk_weights=payload.topk_weights,
            topk_ids=payload.topk_ids,
            router_logits=payload.router_logits,
        )
        return payload.hidden_states, payload.layout

    def _combine_cam(
        self,
        dispatch_ref: torch.Tensor,
        layout: CAMDispatchLayout,
        *,
        stage_idx: int,
    ) -> torch.Tensor:
        """Receive routed output and restore the dispatch's model token layout."""
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("Remote MoE execution requires AFD forward metadata")
        local_output = afd_metadata.connector.recv_ffn_output(
            ref_tensor=dispatch_ref,
            ubatch_idx=stage_idx,
        )
        return restore_cam_dispatch_output(local_output, layout)

    def _compute_attention_shared(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        if self._attention_shared_experts is None:
            return None
        shared_module = self._attention_shared_experts()
        assert shared_module is not None
        shared_output = shared_module(hidden_states)
        if hidden_states.dtype == torch.float16:
            shared_output = shared_output / self.shared_output_divisor_fp16
        return shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        execution = require_cam_async_execution_context()
        topk_weights, topk_ids, computed_logits = self._route_native(hidden_states)
        execution.checkpoint(self.layer_id, CAMAsyncPhase.ROUTED)
        dispatch_ref, layout = self._dispatch_cam(
            hidden_states,
            topk_weights,
            topk_ids,
            computed_logits,
            stage_idx=execution.stage_idx,
            use_sequence_parallel=execution.use_sequence_parallel,
        )
        shared_output = self._compute_attention_shared(hidden_states)
        execution.checkpoint(self.layer_id, CAMAsyncPhase.DISPATCHED)
        output = self._combine_cam(
            dispatch_ref,
            layout,
            stage_idx=execution.stage_idx,
        )
        if shared_output is not None:
            output = output + shared_output
        return output
