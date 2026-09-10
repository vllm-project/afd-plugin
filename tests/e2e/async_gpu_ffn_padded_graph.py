# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Replay the FFN grouped GEMM from a graph captured at a padded shape.

Run with one GPU::

    python tests/e2e/async_gpu_ffn_padded_graph.py

The FFN side of the async connector never learns the next work item's shape
ahead of time -- there is no control plane -- which is why it ran eagerly. The
claim this checks is that it does not need to: a grouped GEMM takes its
grouping from a device-side count vector rather than from its row count, so one
row count can be captured and every smaller item padded up to it.

What makes that non-obvious is that the padding changes the *grouping*, not
just the row count: the pad rows are charged to the last expert, so the count
vector differs on every replay. A graph that had baked its expert assignment in
would return the previous item's answer. Each replay here uses a different
routing and is checked against an eager run of the same routing.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

from afd_plugin.model_executor.models.gpu.deepseek_v2_attention_gate import (
    compute_attention_gate_moe_ffn,
)
from afd_plugin.v1.worker.cuda_graph import pad_counts_to_shape

HIDDEN = 128
INTERMEDIATE = 256
EXPERT_PER_RANK = 4
MAX_ROWS = 24
SCALING = 1.7
NUM_REPLAYS = 4


def build_layer(device):
    gen = torch.Generator(device="cpu").manual_seed(5)
    w13 = (
        torch.randn(EXPERT_PER_RANK, 2 * INTERMEDIATE, HIDDEN, generator=gen)
        / HIDDEN**0.5
    ).to(device, torch.bfloat16)
    w2 = (
        torch.randn(EXPERT_PER_RANK, HIDDEN, INTERMEDIATE, generator=gen)
        / INTERMEDIATE**0.5
    ).to(device, torch.bfloat16)
    return SimpleNamespace(
        mlp=SimpleNamespace(
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(w13_weight=w13, w2_weight=w2),
                _shared_experts=None,
                routed_scaling_factor=SCALING,
            ),
        ),
    )


def routing_for(iteration: int, device):
    """A different per-expert split, and a different row count, each time."""
    gen = torch.Generator(device="cpu").manual_seed(300 + iteration)
    counts = torch.randint(0, 5, (EXPERT_PER_RANK,), generator=gen)
    rows = int(counts.sum())
    if rows == 0:
        counts[0] = 3
        rows = 3
    hidden = torch.randn(rows, HIDDEN, generator=gen).to(device, torch.bfloat16)
    return hidden, counts.to(device, torch.int32), rows


def main() -> None:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    layer = build_layer(device)

    static_hidden = torch.zeros(MAX_ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    static_counts = torch.zeros(EXPERT_PER_RANK, dtype=torch.int32, device=device)

    def padded_compute():
        return compute_attention_gate_moe_ffn(
            layer,
            hidden_states=static_hidden,
            group_list=static_counts,
            expand_x_shared=None,
        )

    # Warm first: the fused MoE picks a kernel on its first call, and that
    # choice has to be settled before capture -- autotuning synchronizes.
    static_counts[-1] = MAX_ROWS
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        padded_compute()
    torch.cuda.current_stream(device).wait_stream(warmup)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        payload = padded_compute()
    print(f"captured the grouped GEMM at {MAX_ROWS} rows", flush=True)

    for replay in range(NUM_REPLAYS):
        hidden, counts, rows = routing_for(replay, device)

        expected = compute_attention_gate_moe_ffn(
            layer,
            hidden_states=hidden,
            group_list=counts,
            expand_x_shared=None,
        ).routed_output

        static_hidden[:rows].copy_(hidden)
        static_counts.copy_(counts)
        pad_counts_to_shape(static_counts, padded_rows=MAX_ROWS, actual_rows=rows)
        assert int(static_counts.sum()) == MAX_ROWS
        graph.replay()
        torch.cuda.synchronize(device)

        got = payload.routed_output[:rows]
        torch.testing.assert_close(
            got.to(torch.float32),
            expected.to(torch.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        print(
            f"replay {replay}: rows={rows} split={counts.tolist()} matches eager",
            flush=True,
        )

    print("PASS: padded FFN grouped GEMM replays correctly", flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("this test needs a GPU")
    main()
    sys.exit(0)
