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
    eplb_map_loss,
    logical_load_from_window,
    summarize_rank_load,
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


def test_logical_load_from_window_sums_the_window_through_the_map():
    """The window is summed over its steps and gathered through the placement
    that recorded it, giving tokens per logical expert; replicas beyond an
    expert's count are ignored rather than double counted."""
    step_load = [4, 2, 1, 1]
    window = torch.tensor([step_load, step_load, step_load]).unsqueeze(1)
    # Logical 0 is replicated on slots 0 and 1, logical 1 on 1 and 0, and the
    # remaining slots are all logicals 2 and 3 hold.
    logical_to_physical = torch.tensor([[[0, 1], [1, 0], [2, -1], [3, -1]]])
    replica_count = torch.tensor([[2, 2, 1, 1]])

    logical_load = logical_load_from_window(window, logical_to_physical, replica_count)

    assert logical_load.tolist() == [[3 * (4 + 2), 3 * (2 + 4), 3 * 1, 3 * 1]]


def test_summarize_rank_load_matches_vllms_balancedness():
    """Balancedness is the ratio of the per-layer mean and max rank load,
    each summed over layers."""
    per_rank = torch.tensor([[18, 6], [12, 24]])

    summary = summarize_rank_load(per_rank)

    assert summary.avg_tokens == pytest.approx(30.0)
    assert summary.max_tokens == pytest.approx(42.0)
    assert summary.balancedness == pytest.approx(30.0 / 42.0)


def test_summarize_rank_load_reports_zero_without_tokens():
    """A step that recorded no load has no ratio to report."""
    summary = summarize_rank_load(torch.zeros((2, 2)))

    assert (summary.avg_tokens, summary.max_tokens, summary.balancedness) == (0, 0, 0)


def test_summarize_rank_load_reaches_one_when_perfectly_balanced():
    """Equal rank load is a balancedness of 1."""
    per_rank = torch.tensor([[5, 5], [7, 7]])

    assert summarize_rank_load(per_rank).balancedness == pytest.approx(1.0)


def test_eplb_map_loss_splits_a_replicated_expert_over_its_replicas():
    """EPLB estimates a logical expert's load as divided evenly over its
    replicas, so an idle replica still counts as half of the load."""
    # Layer 0 of LAYER_IDENTITY: logical 8 sits in slots 5 (rank 0) and 8
    # (rank 1), and the whole of its load was recorded on one replica.
    logical_load = torch.zeros((1, 12))
    logical_load[0, 8] = 20

    placement = torch.tensor([LAYER_IDENTITY])
    loss = eplb_map_loss(logical_load, placement, ffn_size=2)

    # 20 tokens over 2 replicas -> 10 per rank, which is perfectly balanced,
    # even though the recorded per-slot load sits on rank 0 alone.
    assert loss.avg_tokens == pytest.approx(10.0)
    assert loss.max_tokens == pytest.approx(10.0)
    assert loss.balancedness == pytest.approx(1.0)


def test_eplb_map_loss_rises_when_a_hot_expert_moves_to_the_light_rank():
    """The metric the policy minimises improves when the hot expert stops
    sharing a rank with the other hot expert."""
    identity = [0, 1, 2, 3]
    # Logical 0 (50 tokens) and logical 1 (30) start on rank 0, which also
    # gives logical 2 (10) and logical 3 (10) to rank 1.
    logical_load = torch.tensor([[50, 30, 10, 10]], dtype=torch.float32)
    balanced = [0, 3, 2, 1]

    before = eplb_map_loss(logical_load, torch.tensor([identity]), ffn_size=2)
    after = eplb_map_loss(logical_load, torch.tensor([balanced]), ffn_size=2)

    assert before.balancedness == pytest.approx(50.0 / 80.0)
    assert after.balancedness == pytest.approx(50.0 / 60.0)


def test_eplb_map_loss_ignores_slots_no_logical_expert_uses():
    """Slots mapped to -1 carry no load and no replica, so they neither add
    to a rank's total nor dilute an expert's load."""
    logical_load = torch.tensor([[12.0, 6.0, 3.0]])
    placement = torch.tensor([[0, 1, 2, -1]])

    loss = eplb_map_loss(logical_load, placement, ffn_size=2)

    # Rank 0 holds the first two slots (18 tokens), rank 1 the third (3).
    # Read without the guard the -1 slot would wrap to the last logical
    # expert and hand rank 1 its 3 tokens twice, a 0.6667 loss.
    assert loss.avg_tokens == pytest.approx(10.5)
    assert loss.max_tokens == pytest.approx(18.0)
    assert loss.balancedness == pytest.approx(10.5 / 18.0)
