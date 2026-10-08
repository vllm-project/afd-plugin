# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep AFD hash-routing ids aligned with the rows the router sees.

``moe_gating_top_k_hash`` reads one id per row of ``router_logits`` and indexes its
token-to-expert table with that id without validating it, so the ids tensor has to
cover every row the kernel iterates. vLLM-Ascend's fused selector aligns the ids to
the MoE's sequence-parallel layout first: it pads them to ``padded_num_tokens``,
splits them across the tensor-parallel group, and splits them again for FlashComm
v1. That is right when the MoE chunks the activations it routes the same way, which
is what the native sequence-parallel path does.

The AFD FFN role never chunks them: it routes the complete A2E tile it received,
and the plugin rejects sequence-parallel MoE for DeepSeek-V4. The ids the connector
installs therefore already describe exactly the rows of ``router_logits``, and the
re-alignment can only shrink the ids buffer below what the kernel reads. The kernel
then turns unrelated device memory into token ids, and the out-of-range table index
faults the AIV core with an MTE DDR error that names only the core
(``MoeGatingTopKHash_..._10004``, errcode 95).

This module patches the selector so that ids which already describe the router's
rows are used unchanged. Ids that do not match keep the upstream alignment, which
the native sequence-parallel path relies on.
"""

from __future__ import annotations

import torch
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import split_tensor_along_first_dim

_PATCHED = False


def ids_describe_router_rows(
    input_ids: torch.Tensor,
    router_logits: torch.Tensor,
) -> bool:
    """Report whether the ids already cover every row the router routes.

    One id per router row is what the hash operator reads, so ids that already
    match need no alignment. Aligning them anyway -- padding, then splitting
    across the tensor-parallel group -- is what makes the ids buffer shorter than
    the rows the kernel iterates on the AFD FFN role.
    """

    return int(input_ids.numel()) == int(router_logits.shape[0])


# Upstream source: vllm-ascend commit 80d8c194f,
# vllm_ascend/ops/fused_moe/experts_selector.py,
# _select_experts_with_fusion_ops.
# Patch reason: the upstream hash path always re-aligns the ids to the MoE's
# sequence-parallel layout (DP all-gather or pad-and-split, then a FlashComm v1
# split), while the AFD FFN role routes a complete A2E tile whose ids already cover
# every router row. Re-aligning them shortens the ids buffer, and the hash kernel
# then reads past it and faults the AIV core.
# Patch functionality: skip the alignment when the ids already describe the router
# rows and use them unchanged; ids that do not match keep the upstream path.
# Signature: matches upstream; no added parameters.
def _select_experts_with_fusion_ops(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    tid2eid=None,
    input_ids=None,
):
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            # ### PATCH START: AFD hash ids already describe the router rows
            if not ids_describe_router_rows(input_ids, router_logits):
                if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                    prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                    input_ids = prepare_finalize.all_gather_input_id_with_dp_group(
                        input_ids
                    )
                else:
                    input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(
                        input_ids
                    )

                if (
                    forward_context.flash_comm_v1_enabled
                    and forward_context.moe_comm_type != MoECommType.ALLGATHER
                ):
                    # Process for Flash Comm V1
                    tp_size = get_tp_group().world_size
                    tp_rank = get_tp_group().rank_in_group
                    splitted_input = split_tensor_along_first_dim(
                        input_ids, num_partitions=tp_size
                    )
                    input_ids = splitted_input[tp_rank].contiguous()
            # ### PATCH END: AFD hash ids already describe the router rows
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=e_score_correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid_ones,
            k_group=topk_group,
            group_count=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            # The hash custom op currently rejects renorm != 0. Apply
            # norm_topk_prob in Python below before returning to MoE compute.
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        return topk_weights, topk_ids
    norm_type = 0 if scoring_func == "softmax" else 1
    if (
        e_score_correction_bias is not None
        and e_score_correction_bias.dtype != router_logits.dtype
    ):
        e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=e_score_correction_bias,
    )

    return topk_weights, topk_ids


def apply_afd_hash_ids_alignment_patch() -> None:
    """Patch the fused selector of the installed vLLM-Ascend.

    The rebind reads the attribute directly rather than probing for it: AFD
    patches the pinned vLLM-Ascend revision, so a revision that renamed or dropped
    the selector has to fail here and now instead of silently leaving the upstream
    re-alignment in place. The FFN worker imports
    ``vllm_ascend.ops.fused_moe.experts_selector`` for the force-load-balance
    patch before calling this, so the module is importable either way.
    """

    global _PATCHED
    if _PATCHED:
        return

    from vllm_ascend.ops.fused_moe import experts_selector

    experts_selector._select_experts_with_fusion_ops = _select_experts_with_fusion_ops
    _PATCHED = True


__all__ = [
    "apply_afd_hash_ids_alignment_patch",
    "ids_describe_router_rows",
]
