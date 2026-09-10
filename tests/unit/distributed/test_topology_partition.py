# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Subgroup partition tests for the P2P rank mapping.

``build_rank_mapping`` spreads the Attention ranks over the FFN ranks in
contiguous blocks that differ in size by at most one (attention ``a`` joins
the subgroup of FFN rank ``a * F // A``). When ``F`` divides ``A`` this is the
grouping the connector has always built, which
``test_divisible_layouts_keep_the_historical_grouping`` pins; the other tests
cover every ``A >= F`` pair, including the layouts that grouping could not
express.
"""

from __future__ import annotations

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed.topology import build_rank_mapping

# Every A >= F pair with A, F in 1..8, divisible and not.
_GRID = [(a, f) for a in range(1, 9) for f in range(1, a + 1)]


def _config(role: str, attention: int, ffn: int) -> AFDConfig:
    return AFDConfig(
        role=role,
        connector="P2pNcclAFDConnector",
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _mapping(role: str, role_rank: int, attention: int, ffn: int):
    return build_rank_mapping(_config(role, attention, ffn), role_rank)


@pytest.mark.parametrize(("attention", "ffn"), _GRID)
def test_every_attention_rank_belongs_to_exactly_one_subgroup(attention, ffn):
    subgroups = {
        ffn_rank: _mapping("ffn", ffn_rank, attention, ffn).subgroup_ranks
        for ffn_rank in range(ffn)
    }

    # The FFN rank leads its own subgroup, and the Attention members partition
    # the Attention world in world order.
    assert [ranks[0] for ranks in subgroups.values()] == list(range(ffn))
    peers = [rank for ranks in subgroups.values() for rank in ranks[1:]]
    assert peers == list(range(ffn, ffn + attention))

    # Block sizes differ by at most one, and none is empty.
    sizes = [len(ranks) - 1 for ranks in subgroups.values()]
    assert min(sizes) >= 1
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(("attention", "ffn"), _GRID)
def test_both_roles_agree_on_the_subgroup_they_share(attention, ffn):
    for attention_rank in range(attention):
        mapping = _mapping("attention", attention_rank, attention, ffn)
        owner = _mapping("ffn", mapping.subgroup_index, attention, ffn)
        assert mapping.subgroup_ranks == owner.subgroup_ranks
        assert mapping.subgroup_ranks[mapping.rank_in_subgroup] == mapping.world_rank


@pytest.mark.parametrize(("attention", "ffn"), _GRID)
def test_world_and_p2p_ranks_follow_the_role_layout(attention, ffn):
    min_size = min(attention, ffn)
    dp_destinations: list[int] = []
    for role, size in (("ffn", ffn), ("attention", attention)):
        for role_rank in range(size):
            mapping = _mapping(role, role_rank, attention, ffn)
            assert mapping.min_size == min_size
            if role == "ffn":
                assert mapping.world_rank == role_rank
                assert mapping.p2p_rank == role_rank
            else:
                assert mapping.world_rank == ffn + role_rank
                assert mapping.p2p_rank == role_rank + min_size
            dp_destinations.extend(mapping.dp_metadata_destinations)

    # Only Attention ranks send DP metadata, and each FFN rank receives it
    # from exactly one of them.
    assert sorted(dp_destinations) == list(range(ffn))


@pytest.mark.parametrize(
    ("attention", "ffn", "expected"),
    [
        (3, 2, [(0, 2, 3), (1, 4)]),
        (5, 2, [(0, 2, 3, 4), (1, 5, 6)]),
        (5, 3, [(0, 3, 4), (1, 5, 6), (2, 7)]),
        (6, 4, [(0, 4, 5), (1, 6), (2, 7, 8), (3, 9)]),
    ],
)
def test_partition_literal_examples(attention, ffn, expected):
    assert [
        _mapping("ffn", ffn_rank, attention, ffn).subgroup_ranks
        for ffn_rank in range(ffn)
    ] == expected


@pytest.mark.parametrize(("attention", "ffn"), [(a, f) for a, f in _GRID if a % f == 0])
def test_divisible_layouts_keep_the_historical_grouping(attention, ffn):
    ratio = attention // ffn
    for ffn_rank in range(ffn):
        mapping = _mapping("ffn", ffn_rank, attention, ffn)
        assert len(mapping.subgroup_ranks) - 1 == ratio
        assert mapping.subgroup_ranks == (
            ffn_rank,
            *(ffn + ffn_rank * ratio + offset for offset in range(ratio)),
        )
