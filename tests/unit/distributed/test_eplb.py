# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for the shared EPLB observability helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.distributed.eplb import (  # noqa: E402
    describe_eplb_placement,
    describe_eplb_rank_load,
)

# Two layers of a 12-slot world over 2 FFN ranks: layer 0 keeps the identity
# placement with logical 8 replicated in a redundant slot, layer 1 swaps
# logical 5 and 6 across the ranks and drops the replica.
LAYER_IDENTITY = [0, 1, 2, 3, 4, 8, 6, 7, 8, 9, 10, 11]
LAYER_SWAPPED = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11]


def test_describe_eplb_placement_lists_experts_per_rank():
    """The placement description names each rank's logical experts and the
    replicated ones, per MoE layer."""
    placement = torch.tensor([LAYER_IDENTITY, LAYER_SWAPPED])

    lines = describe_eplb_placement(placement, ffn_size=2)

    assert len(lines) == 2
    assert lines[0] == (
        "layer 0: rank 0=[0, 1, 2, 3, 4, 8] | rank 1=[6, 7, 8, 9, 10, 11] "
        "| replicas: 8x2"
    )
    assert lines[1] == (
        "layer 1: rank 0=[0, 1, 2, 3, 4, 6] | rank 1=[5, 7, 8, 9, 10, 11] "
        "| replicas: none"
    )


def test_describe_eplb_rank_load_sums_each_ranks_slots():
    """A rank's recorded load is the sum over its contiguous slot block,
    across the window steps and layers."""
    # Two window steps, one layer: rank 0's slots (0-5) carry 1 each, rank
    # 1's slots (6-11) carry 2 each.
    layer_load = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    window = torch.tensor([layer_load, layer_load]).unsqueeze(1)

    load = describe_eplb_rank_load(window, ffn_size=2)

    assert load == "rank 0=12 tokens, rank 1=24 tokens"
