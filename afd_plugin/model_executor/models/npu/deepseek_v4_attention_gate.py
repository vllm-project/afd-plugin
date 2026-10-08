# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 Attention-side routing helpers for Async CAM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4AttentionGateRemoteMoE,
    )


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


def compute_attention_gate_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    hidden_states: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
    hidden_states_fp32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4 routing without entering vLLM's native MoE communicator.

    AFD Async CAM owns cross-role dispatch and receives tokens and IDs already
    sharded by the model. Preserve the target decoder's exact FP32 RMSNorm
    result and the precast gate weights, then call the CANN selectors on those
    local tokens without entering the native MoE runner's EP communication.
    """

    router_input = (
        hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
    )
    router_logits = torch.nn.functional.linear(router_input, moe.gate.weight_fp32)
    if moe.scoring_func == "sqrtsoftplus":
        topk_weights, topk_ids = _compute_sqrtsoftplus_topk(
            moe, router_logits, input_ids=input_ids
        )
    else:
        topk_weights, topk_ids = _compute_standard_topk(moe, router_logits)
    return topk_weights.to(torch.float32), topk_ids


def _compute_sqrtsoftplus_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4's sqrtsoftplus CANN router without native MoE communication."""

    if moe.scoring_func != "sqrtsoftplus":
        raise RuntimeError(
            "DSV4 Hash routing requires scoring_func='sqrtsoftplus', got "
            f"{moe.scoring_func!r}",
        )

    tid2eid = moe.gate.tid2eid
    if tid2eid is not None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        input_ids = local_hash_input_ids(
            input_ids=(forward_context.input_ids if input_ids is None else input_ids),
            router_tokens=router_logits.shape[0],
        )
        input_ids = torch.where(input_ids == -1, 0, input_ids)
        tid2eid = tid2eid.to(torch.int32)
    else:
        input_ids = None
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits,
        k=moe.top_k,
        bias=correction_bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return topk_weights, topk_ids


def _compute_standard_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the non-Hash CANN selector without native MoE communication."""

    from vllm_ascend.device.device_op import DeviceOperator

    norm_type_by_scoring_func = {"softmax": 0, "sigmoid": 1}
    try:
        norm_type = norm_type_by_scoring_func[moe.scoring_func]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported non-Hash DSV4 routing scoring function: {moe.scoring_func!r}",
        ) from exc
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=moe.top_k,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        group_select_mode=1,
        renorm=int(moe.renormalize),
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        bias_opt=correction_bias,
    )
    return topk_weights, topk_ids
