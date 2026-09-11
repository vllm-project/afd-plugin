# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Numerical equivalence of the async GPU dispatch/compute/combine chain.

Run with one GPU::

    python tests/e2e/async_gpu_moe_equivalence.py

The connector routes tokens itself and hands the FFN side pre-grouped rows, so
the expert compute runs as a topk==1 problem with unit weights and the real
topk weighting is applied during combine. This checks that the whole chain
reproduces a naive per-token MoE reference.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from afd_plugin.model_executor.models.gpu.deepseek_v2_attention_gate import (
    compute_attention_gate_moe_ffn,
)

from afd_plugin.connectors.gpu.async_gpu import plan_dispatch

NUM_TOKENS = 32
HIDDEN = 128
INTERMEDIATE = 256
TOPK = 4
NUM_EXPERTS = 8
FFN_SIZE = 2
EXPERT_PER_RANK = NUM_EXPERTS // FFN_SIZE


def reference_moe(x, w13, w2, topk_ids, topk_weights):
    """Naive per-token MoE: sum over each token's topk experts."""
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            hidden = x[token].to(torch.float32) @ w13[expert].to(torch.float32).T
            gate, up = hidden.chunk(2, dim=-1)
            y = (F.silu(gate) * up) @ w2[expert].to(torch.float32).T
            out[token] += float(topk_weights[token, slot]) * y
    return out


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16

    x = torch.randn(NUM_TOKENS, HIDDEN, device=device, dtype=dtype)
    w13 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            device=device,
            dtype=dtype,
        )
        / HIDDEN**0.5
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=dtype,
        )
        / INTERMEDIATE**0.5
    )
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS)[:TOPK] for _ in range(NUM_TOKENS)],
    ).to(device, torch.int32)
    topk_weights = torch.rand(NUM_TOKENS, TOPK, device=device, dtype=torch.float32)

    expected = reference_moe(x, w13, w2, topk_ids, topk_weights)

    # --- what the connector + FFN runner actually do -----------------------
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=FFN_SIZE,
        expert_per_rank=EXPERT_PER_RANK,
    )
    # A destination locates its own partials from the two header words the
    # sender fills on the device; nothing about the routing is read back here,
    # which is what the send path does now too.
    starts = plan.segment_start.cpu().tolist()
    routed = plan.routed_per_rank.cpu().tolist()

    accumulator = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float32, device=device)
    for ffn_rank in range(FFN_SIZE):
        base = ffn_rank * EXPERT_PER_RANK
        group_list = plan.counts[base : base + EXPERT_PER_RANK]
        partials = slice(starts[ffn_rank], starts[ffn_rank] + routed[ffn_rank])
        expand = plan.expand_idx[partials].to(torch.int64)

        # This FFN rank owns experts [base, base + EXPERT_PER_RANK).
        layer = SimpleNamespace(
            mlp=SimpleNamespace(
                experts=SimpleNamespace(
                    routed_experts=SimpleNamespace(
                        w13_weight=w13[base : base + EXPERT_PER_RANK].contiguous(),
                        w2_weight=w2[base : base + EXPERT_PER_RANK].contiguous(),
                    ),
                    _shared_experts=None,
                    routed_scaling_factor=1.0,
                ),
            ),
        )
        # The whole batch crosses the wire; the FFN side gathers one row per
        # partial from it before the grouped GEMM, which applies the partial
        # weights in its own epilogue.
        payload = compute_attention_gate_moe_ffn(
            layer,
            hidden_states=x.index_select(0, expand),
            group_list=group_list,
            expand_x_shared=None,
        )
        # ...then weights and reduces back to one row per token, in the payload
        # dtype, leaving a zero row for tokens this rank held no expert for.
        reduced = torch.zeros(
            NUM_TOKENS, HIDDEN, dtype=payload.routed_output.dtype, device=device
        )
        weighted = payload.routed_output * plan.weights[partials].unsqueeze(1).to(
            payload.routed_output.dtype
        )
        reduced.index_add_(0, expand, weighted)
        accumulator += reduced.to(torch.float32)

    diff = (accumulator - expected).abs()
    rel = diff.max() / expected.abs().max()
    print(f"max abs diff={diff.max():.4e}  max rel={rel:.4e}")
    torch.testing.assert_close(accumulator, expected, rtol=6e-2, atol=6e-2)
    print("PASS: dispatch -> local experts -> combine matches naive MoE")


if __name__ == "__main__":
    main()
