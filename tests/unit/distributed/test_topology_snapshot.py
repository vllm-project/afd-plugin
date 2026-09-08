# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Characterization snapshots of the P2P rank mapping.

These tests freeze ``build_rank_mapping`` for the topologies the connector
already supported, so that relaxing the divisibility rule can be shown to leave
their mappings byte-for-byte unchanged.

Two layers of protection:

1. Literal golden snapshots for representative topologies (1A1F, 2A2F, 4A2F).
2. Structural invariants over a wider grid of legal ``(A, F)`` pairs.
"""

from __future__ import annotations

import dataclasses

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed.topology import (
    build_rank_mapping,
)


def _config(role: str, attention: int, ffn: int) -> AFDConfig:
    return AFDConfig(
        role=role,
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _mapping_dict(role: str, attention: int, ffn: int, role_rank: int) -> dict:
    mapping = build_rank_mapping(_config(role, attention, ffn), role_rank)
    return dataclasses.asdict(mapping)


# Captured from main @ 603c111 (pre-M2N). Do not regenerate to make a failing
# test pass: a diff here means gather-mode behavior changed.
_GOLDEN: dict[tuple[int, int], dict[tuple[str, int], dict]] = {
    (1, 1): {
        ("ffn", 0): {
            "role": "ffn",
            "role_rank": 0,
            "world_rank": 0,
            "p2p_rank": 0,
            "attention_size": 1,
            "ffn_size": 1,
            "min_size": 1,
            "ratio": 1,
            "subgroup_index": 0,
            "rank_in_subgroup": 0,
            "subgroup_ranks": (0, 1),
            "dp_metadata_destinations": (),
        },
        ("attention", 0): {
            "role": "attention",
            "role_rank": 0,
            "world_rank": 1,
            "p2p_rank": 1,
            "attention_size": 1,
            "ffn_size": 1,
            "min_size": 1,
            "ratio": 1,
            "subgroup_index": 0,
            "rank_in_subgroup": 1,
            "subgroup_ranks": (0, 1),
            "dp_metadata_destinations": (0,),
        },
    },
    (2, 2): {
        ("ffn", 0): {
            "role": "ffn",
            "role_rank": 0,
            "world_rank": 0,
            "p2p_rank": 0,
            "attention_size": 2,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 1,
            "subgroup_index": 0,
            "rank_in_subgroup": 0,
            "subgroup_ranks": (0, 2),
            "dp_metadata_destinations": (),
        },
        ("ffn", 1): {
            "role": "ffn",
            "role_rank": 1,
            "world_rank": 1,
            "p2p_rank": 1,
            "attention_size": 2,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 1,
            "subgroup_index": 1,
            "rank_in_subgroup": 0,
            "subgroup_ranks": (1, 3),
            "dp_metadata_destinations": (),
        },
        ("attention", 0): {
            "role": "attention",
            "role_rank": 0,
            "world_rank": 2,
            "p2p_rank": 2,
            "attention_size": 2,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 1,
            "subgroup_index": 0,
            "rank_in_subgroup": 1,
            "subgroup_ranks": (0, 2),
            "dp_metadata_destinations": (0,),
        },
        ("attention", 1): {
            "role": "attention",
            "role_rank": 1,
            "world_rank": 3,
            "p2p_rank": 3,
            "attention_size": 2,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 1,
            "subgroup_index": 1,
            "rank_in_subgroup": 1,
            "subgroup_ranks": (1, 3),
            "dp_metadata_destinations": (1,),
        },
    },
    (4, 2): {
        ("ffn", 0): {
            "role": "ffn",
            "role_rank": 0,
            "world_rank": 0,
            "p2p_rank": 0,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 0,
            "rank_in_subgroup": 0,
            "subgroup_ranks": (0, 2, 3),
            "dp_metadata_destinations": (),
        },
        ("ffn", 1): {
            "role": "ffn",
            "role_rank": 1,
            "world_rank": 1,
            "p2p_rank": 1,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 1,
            "rank_in_subgroup": 0,
            "subgroup_ranks": (1, 4, 5),
            "dp_metadata_destinations": (),
        },
        ("attention", 0): {
            "role": "attention",
            "role_rank": 0,
            "world_rank": 2,
            "p2p_rank": 2,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 0,
            "rank_in_subgroup": 1,
            "subgroup_ranks": (0, 2, 3),
            "dp_metadata_destinations": (0,),
        },
        ("attention", 1): {
            "role": "attention",
            "role_rank": 1,
            "world_rank": 3,
            "p2p_rank": 3,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 0,
            "rank_in_subgroup": 2,
            "subgroup_ranks": (0, 2, 3),
            "dp_metadata_destinations": (1,),
        },
        ("attention", 2): {
            "role": "attention",
            "role_rank": 2,
            "world_rank": 4,
            "p2p_rank": 4,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 1,
            "rank_in_subgroup": 1,
            "subgroup_ranks": (1, 4, 5),
            "dp_metadata_destinations": (),
        },
        ("attention", 3): {
            "role": "attention",
            "role_rank": 3,
            "world_rank": 5,
            "p2p_rank": 5,
            "attention_size": 4,
            "ffn_size": 2,
            "min_size": 2,
            "ratio": 2,
            "subgroup_index": 1,
            "rank_in_subgroup": 2,
            "subgroup_ranks": (1, 4, 5),
            "dp_metadata_destinations": (),
        },
    },
}

_GOLDEN_CASES = [
    (attention, ffn, role, role_rank)
    for (attention, ffn), mappings in _GOLDEN.items()
    for (role, role_rank) in mappings
]

_LEGAL_GRID = [
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 1),
    (4, 1),
    (4, 2),
    (4, 4),
    (6, 2),
    (6, 3),
    (8, 2),
    (8, 4),
]


@pytest.mark.parametrize(
    ("attention", "ffn", "role", "role_rank"),
    _GOLDEN_CASES,
)
def test_rank_mapping_matches_golden_snapshot(attention, ffn, role, role_rank):
    assert (
        _mapping_dict(role, attention, ffn, role_rank)
        == _GOLDEN[(attention, ffn)][(role, role_rank)]
    )


@pytest.mark.parametrize(("attention", "ffn"), _LEGAL_GRID)
def test_rank_mapping_invariants(attention, ffn):
    ratio = attention // ffn
    min_size = min(attention, ffn)

    seen_subgroups: dict[int, tuple[int, ...]] = {}
    dp_destination_union: list[int] = []

    for role, size in (("ffn", ffn), ("attention", attention)):
        for role_rank in range(size):
            mapping = build_rank_mapping(_config(role, attention, ffn), role_rank)

            if role == "ffn":
                assert mapping.world_rank == role_rank
                assert mapping.p2p_rank == role_rank
                assert mapping.subgroup_index == role_rank
            else:
                assert mapping.world_rank == ffn + role_rank
                assert mapping.p2p_rank == role_rank + min_size
                assert mapping.subgroup_index == role_rank // ratio

            assert mapping.ratio == ratio
            assert mapping.min_size == min_size
            assert len(mapping.subgroup_ranks) == 1 + ratio
            assert mapping.subgroup_ranks[0] == mapping.subgroup_index
            assert (
                mapping.subgroup_ranks[mapping.rank_in_subgroup] == mapping.world_rank
            )

            prior = seen_subgroups.setdefault(
                mapping.subgroup_index,
                mapping.subgroup_ranks,
            )
            assert prior == mapping.subgroup_ranks

            dp_destination_union.extend(mapping.dp_metadata_destinations)

    # The DP metadata senders must cover every FFN rank exactly once.
    assert sorted(dp_destination_union) == list(range(ffn))
    # Subgroups must partition the attention ranks.
    attention_members = sorted(
        rank for ranks in seen_subgroups.values() for rank in ranks[1:]
    )
    assert attention_members == list(range(ffn, ffn + attention))


# The ``A >= F`` constraint is unchanged and still raises; only the divisibility
# rule is relaxed. ``test_p2p_connector.py`` pins the surviving error path and
# ``test_topology_partition.py`` pins the layouts this grid excludes.
