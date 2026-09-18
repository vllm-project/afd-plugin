# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 Attention-side routing helpers for Async CAM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4AttentionGateRemoteMoE,
    )


def local_hash_input_ids(
    *,
    input_ids: torch.Tensor | None,
    router_tokens: int,
    flash_comm_v1_enabled: bool,
    pad_size: int,
) -> torch.Tensor:
    """Return the rank-local token ids that a Hash layer routes on.

    A DSV4 Hash layer routes by token identity rather than by router logits, so
    the FFN rank executing that layer needs the ids of exactly the tokens it
    computes on. On Attention the forward context carries the *global* ids while
    FlashComm v1 shards router logits across TP ranks, so the global vector must
    receive the same padding and contiguous TP split as the logits before it can
    be sent.

    Both the local routing path and the AFD send path call this, so the ids that
    cross the boundary describe the same tokens the Attention-side routing used.

    Args:
        input_ids: Global ids from the forward context, or ``None``.
        router_tokens: Token count of this rank's router logits.
        flash_comm_v1_enabled: Whether FlashComm v1 is active for this forward.
        pad_size: FlashComm v1 padding applied to the activation.

    Returns:
        A one-dimensional ``int64`` tensor of ``router_tokens`` local ids.

    Raises:
        RuntimeError: If ids are unavailable, or if the ids do not describe
            exactly ``router_tokens`` tokens. Both would otherwise let a
            token-keyed router select experts for the wrong tokens.
    """

    if input_ids is None:
        raise RuntimeError(
            "DSV4 Hash routing requires input_ids to send towards the FFN role, "
            "but the forward context carries none. This path routes by token "
            "identity and has no fallback, so the runner must install the "
            "request's input_ids before the model forward.",
        )
    ids = input_ids.reshape(-1).to(torch.int64)
    if flash_comm_v1_enabled and ids.numel() != router_tokens:
        from vllm.distributed import get_tp_group
        from vllm_ascend.distributed.utils import split_tensor_along_first_dim

        if pad_size > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_size))
        group = get_tp_group()
        ids = split_tensor_along_first_dim(
            ids,
            num_partitions=group.world_size,
            contiguous_split_chunks=True,
        )[group.rank_in_group]
    if ids.numel() != router_tokens:
        raise RuntimeError(
            "DSV4 Hash routing cannot align the ids sent to FFN with the local "
            f"tokens: ids={ids.numel()} router_tokens={router_tokens}",
        )
    return ids


def hash_input_ids_from_context(
    *,
    forward_context: ForwardContext,
    router_tokens: int,
) -> torch.Tensor:
    """Return the ids to send for a Hash layer, raising if the context has none.

    The ids channel belongs to the transfer rather than to one layer: the FFN
    role cannot tell a Hash layer from a non-Hash one, so it asks for ids on
    every layer. Answering with activations alone would leave it reading an ids
    slot the operator never wrote.
    """

    return local_hash_input_ids(
        input_ids=forward_context.input_ids,
        router_tokens=router_tokens,
        flash_comm_v1_enabled=forward_context.flash_comm_v1_enabled,
        pad_size=forward_context.pad_size,
    )


def compute_attention_gate_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4 routing without entering vLLM's native MoE communicator.

    AFD Async CAM owns the cross-role dispatch.  The vLLM-Ascend fused
    selector's hash path instead assumes native EP/SP communication and calls
    ``forward_context.moe_comm_method.pad_and_split_input_ids``.  That object
    is intentionally absent on Attention ranks, including the KV-cache profile
    forward.  Use the same CANN routing operators directly on local Attention
    tokens, then hand their IDs and weights to CAM dispatch.
    """

    router_logits, _ = moe.gate(hidden_states)
    if moe.scoring_func == "sqrtsoftplus":
        topk_weights, topk_ids = _compute_sqrtsoftplus_topk(moe, router_logits)
    else:
        topk_weights, topk_ids = _compute_standard_topk(moe, router_logits)
    return topk_weights.to(torch.float32), topk_ids


def _compute_sqrtsoftplus_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4's sqrtsoftplus CANN router without native MoE communication."""

    if moe.scoring_func != "sqrtsoftplus":
        raise RuntimeError(
            "DSV4 Hash routing requires scoring_func='sqrtsoftplus', got "
            f"{moe.scoring_func!r}",
        )

    tid2eid = moe.gate.tid2eid
    input_ids = None
    if tid2eid is not None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        input_ids = local_hash_input_ids(
            input_ids=getattr(forward_context, "input_ids", None),
            router_tokens=router_logits.shape[0],
            flash_comm_v1_enabled=forward_context.flash_comm_v1_enabled,
            pad_size=forward_context.pad_size,
        )
        input_ids = torch.where(input_ids == -1, 0, input_ids)
        tid2eid = tid2eid.to(torch.int32)
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
