# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side gate helpers for DeepSeek-V2 on CUDA.

The async GPU connector dispatches tokens that are already routed: one row per
``(token, topk_slot)`` partial, grouped by local expert, with a ``group_list``
of per-expert counts. The FFN side therefore must not run routing again -- it
only needs the grouped GEMM over its local experts.

That shape is a ``topk == 1`` problem: give every arriving row its own expert id
and its own weight, and vLLM's ``fused_experts`` computes exactly the local
expert output, already scaled, in the epilogue it runs anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from afd_plugin.connectors.metadata import AFDF2ATransferPayload

if TYPE_CHECKING:
    from torch import nn


def compute_attention_gate_moe_ffn(
    layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    expand_x_shared: torch.Tensor | None = None,
) -> AFDF2ATransferPayload:
    """Run this rank's local experts over pre-routed tokens.

    Args:
        layer: The AFD DeepSeek decoder layer owning the MoE module.
        hidden_states: ``[num_partials, hidden]`` rows sorted by local expert.
        group_list: ``[expert_per_rank]`` per-expert row counts. Must sum to
            ``hidden_states.shape[0]``.
        expand_x_shared: Optional ``[num_shared, hidden]`` shared-expert rows.

    Returns:
        Routed output in the same row order as ``hidden_states``, plus the
        shared-expert output when the model has shared experts.
    """
    # ``mlp.experts`` is a MoERunner; the weight parameters live on its
    # RoutedExperts, and the shared experts hang off the runner rather than
    # off the MoE module.
    runner = layer.mlp.experts
    routed_experts = runner.routed_experts
    counts = group_list.to(torch.int64)
    num_rows = int(hidden_states.shape[0])
    num_local_experts = counts.numel()
    if num_rows == 0:
        # Routing can leave a peer with nothing -- common in decode, where a
        # single token's topk may land entirely on the other FFN rank.
        routed_output = hidden_states.new_empty((0, hidden_states.shape[-1]))
        shared = runner._shared_experts
        return AFDF2ATransferPayload(
            routed_output=routed_output,
            shared_output=(
                shared._layer(expand_x_shared)
                if shared is not None
                and expand_x_shared is not None
                and expand_x_shared.shape[0] > 0
                else None
            ),
        )
    # ``output_size`` keeps this off the device: without it repeat_interleave
    # reads the counts back to the host to size its output, which is a
    # synchronize on every work item. It also enforces what the removed
    # ``counts.sum() == num_rows`` check used to, raising if they disagree.
    expert_ids = torch.repeat_interleave(
        torch.arange(
            num_local_experts,
            device=hidden_states.device,
            dtype=torch.int32,
        ),
        counts,
        output_size=num_rows,
    ).unsqueeze(1)
    # Mirrors the NPU gate path: scale the routed branch unless fp16, where the
    # native model instead scales the shared branch down. ``fused_experts``
    # multiplies every row by its topk weight in an epilogue it runs anyway, so
    # feeding the factor in there costs nothing, where scaling its output cost a
    # full pass over ``[num_partials, hidden]``. The topk weighting itself is
    # not applied here -- the connector owns it, and applies it when it reduces
    # the partials back to one row per token.
    routed_scaling_factor = runner.routed_scaling_factor
    scale_shared_instead = hidden_states.dtype == torch.float16
    row_weights = torch.full(
        (num_rows, 1),
        1.0 if scale_shared_instead else routed_scaling_factor,
        dtype=torch.float32,
        device=hidden_states.device,
    )

    routed_output = fused_experts(
        hidden_states,
        routed_experts.w13_weight,
        routed_experts.w2_weight,
        row_weights,
        expert_ids,
        global_num_experts=num_local_experts,
        expert_map=None,
    )

    shared_output = None
    shared_experts = runner._shared_experts
    # A dispatch's round-robin shared slice is empty for a rank whenever the
    # ubatch holds fewer tokens than FFN ranks, so zero rows are a normal item
    # shape, not a missing payload.
    if (
        shared_experts is not None
        and expand_x_shared is not None
        and expand_x_shared.shape[0] > 0
    ):
        # Call the wrapped MLP rather than SharedExperts.forward: the wrapper is
        # a stateful scheduler for the runner's own multi-stream pipeline and
        # returns None unless its expected ordering matches. AFD feeds shared
        # tokens as their own batch, so that machinery does not apply.
        shared_output = shared_experts._layer(expand_x_shared)

    if scale_shared_instead and shared_output is not None:
        shared_output = shared_output * (1.0 / routed_scaling_factor)

    return AFDF2ATransferPayload(
        routed_output=routed_output,
        shared_output=shared_output,
    )


__all__ = ["compute_attention_gate_moe_ffn"]
