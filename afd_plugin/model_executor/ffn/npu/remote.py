# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM MoE runners and DeepSeek V4 routing on the Attention role."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.ffn.remote import (
    AFDRemoteMoERunner,
    AFDRemoteRoutedExperts,
    _current_afd_metadata,
    _send_attention_output,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import ReplicatedLinear


class AFDAscendRemoteRoutedExperts(AFDRemoteRoutedExperts):
    """The pinned Ascend factory mixes AscendRoutedExperts into this class."""

    def _replace_quant_method(self, quant_method: FusedMoEMethodBase) -> None:
        """Keep the weightless AFD method when Ascend installs its BF16 method."""


def cam_factory_kwargs() -> dict[str, Any]:
    # The model owns the checkpoint gate. CAM routes already include its scaling.
    return {
        "gate": None,
        "apply_routed_scale_to_output": False,
        "routed_experts_cls": AFDAscendRemoteRoutedExperts,
    }


def local_hash_input_ids(
    *,
    input_ids: torch.Tensor | None,
    router_tokens: int,
) -> torch.Tensor:
    """Validate IDs already sharded by the model alongside the routed tokens.

    The native model and AFD staged model both own the token split. A second
    split here would silently route a different set of tokens on Hash layers.
    """

    if input_ids is None:
        raise RuntimeError(
            "DSV4 Hash routing requires input_ids to send towards the FFN role, "
            "but the forward context carries none. This path routes by token "
            "identity and has no fallback, so the runner must install the "
            "request's input_ids before the model forward.",
        )
    ids = input_ids.reshape(-1).to(torch.int64)
    if ids.numel() != router_tokens:
        raise RuntimeError(
            "DSV4 Hash routing cannot align the ids sent to FFN with the local "
            f"tokens: ids={ids.numel()} router_tokens={router_tokens}",
        )
    return ids


def _cam_gate_topk(
    experts: RoutedExperts,
    gate: ReplicatedLinear,
    gate_input: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the local V4 gate in FP32 and select routes on the same token rows."""
    logits = torch.nn.functional.linear(gate_input.float(), gate.weight_fp32)
    bias = gate.e_score_correction_bias
    if bias is not None and bias.dtype != logits.dtype:
        bias = bias.to(logits.dtype)

    if experts.scoring_func == "sqrtsoftplus":
        tid2eid = gate.tid2eid
        if tid2eid is not None:
            token_ids = local_hash_input_ids(
                input_ids=get_forward_context().input_ids
                if input_ids is None
                else input_ids,
                router_tokens=logits.shape[0],
            )
            input_ids = torch.where(token_ids == -1, 0, token_ids)
            tid2eid = tid2eid.to(torch.int32)
        else:
            input_ids = None
        weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=logits,
            k=experts.top_k,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=experts.topk_group,
            group_count=experts.num_expert_group,
            routed_scaling_factor=experts.routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
    elif experts.scoring_func in ("softmax", "sigmoid"):
        import torch_npu

        weights, ids, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            k=experts.top_k,
            k_group=experts.topk_group,
            group_count=experts.num_expert_group,
            group_select_mode=1,
            renorm=int(experts.renormalize),
            norm_type=0 if experts.scoring_func == "softmax" else 1,
            out_flag=False,
            routed_scaling_factor=experts.routed_scaling_factor,
            eps=1e-20,
            bias=bias,
        )
    else:
        raise RuntimeError(
            "Unsupported non-Hash DSV4 routing scoring function: "
            f"{experts.scoring_func!r}"
        )

    # Read the profile flag inside the op so graph replay uses the live routes.
    torch.ops.vllm.afd_balance_routing_ids_(ids, logits.shape[1], False)
    return weights.to(torch.float32), ids


class AFDCAMP2PMoERunner(AFDRemoteMoERunner):
    """Send the V2 activation or the V4 activation with its IDs or routes."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        attention_gate: ReplicatedLinear | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        if attention_gate is not None:
            weights, ids = _cam_gate_topk(
                self.routed_experts, attention_gate, router_logits, input_ids
            )
            return self._exchange(
                hidden_states, routing=(ids, weights), shared_experts=shared_experts
            )
        if input_ids is not None:
            return self._exchange(hidden_states, input_ids=input_ids)
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _exchange(
        self,
        hidden_states: torch.Tensor,
        *,
        shared_experts: nn.Module | None = None,
        **send_kwargs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        metadata = _current_afd_metadata()
        stage_idx = _send_attention_output(
            metadata, hidden_states, layer_idx=self.layer_idx, **send_kwargs
        )
        shared = shared_experts(hidden_states) if shared_experts is not None else None
        hidden_states = maybe_apply_dbo_yield(hidden_states, role="attention")
        output = metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states, ubatch_idx=stage_idx
        )
        return output if shared is None else output + shared


class AFDAsyncCAMMoERunner(AFDRemoteMoERunner):
    """Dispatch local routes and combine on the microbatch that owns the call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_sequence_parallel_moe = (
            get_current_vllm_config().parallel_config.use_sequence_parallel_moe
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        attention_gate: ReplicatedLinear | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        from afd_plugin.connectors.npu.async_cam_runtime.async_cam_dbo import (
            send_async_cam_remote_call,
        )
        from afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout import (
            prepare_cam_dispatch_payload,
            restore_cam_dispatch_output,
        )

        assert attention_gate is not None, "Async CAM requires an Attention gate"
        topk_weights, topk_ids = _cam_gate_topk(
            self.routed_experts, attention_gate, router_logits, input_ids
        )
        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            use_sequence_parallel=self.use_sequence_parallel_moe,
        )
        afd_metadata = _current_afd_metadata()
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=afd_metadata.stage_idx,
                seq_len=int(dispatch_payload.hidden_states.shape[0]),
            )
        )
        receive = send_async_cam_remote_call(
            afd_metadata.connector,
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
        )
        shared_output = (
            shared_experts(hidden_states) if shared_experts is not None else None
        )
        maybe_apply_dbo_yield(hidden_states, role="attention")
        output = receive()
        output = restore_cam_dispatch_output(output, dispatch_payload.layout)
        if shared_output is not None:
            output = output + shared_output
        return output
