# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Global role ranks and contiguous, balanced P2P Attention/FFN subgroups.

Each Attention rank a belongs to FFN rank a * F // A. Divisible layouts keep
historical grouping; the same contract also covers uneven A >= F layouts.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed.topology import build_rank_mapping, resolve_role_rank


def _vllm_config(
    *,
    dp_size: int = 1,
    dp_rank: int = 0,
    pcp_size: int = 1,
    tp_size: int = 1,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
        ),
    )


def _afd_config(
    *,
    role: str = "attention",
    num_attention_ranks: int = 1,
    num_ffn_ranks: int = 1,
):
    return AFDConfig(
        role=role,
        num_attention_ranks=num_attention_ranks,
        num_ffn_ranks=num_ffn_ranks,
    )


def _install_parallel_state(
    monkeypatch,
    *,
    pcp_rank: int = 0,
    tp_rank: int = 0,
) -> None:
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    distributed_module = ModuleType("vllm.distributed")
    distributed_module.__path__ = []
    parallel_state_module = ModuleType("vllm.distributed.parallel_state")
    vars(parallel_state_module).update(
        get_pcp_group=lambda: SimpleNamespace(rank_in_group=pcp_rank),
        get_tensor_model_parallel_rank=lambda: tp_rank,
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state_module,
    )


@pytest.mark.parametrize(
    "role,dp_size,dp_rank,pcp_size,pcp_rank,tp_size,tp_rank,role_size,expected",
    [
        pytest.param("attention", 1, 0, 1, 0, 1, 0, 1, 0, id="single-worker"),
        pytest.param("attention", 2, 1, 1, 0, 1, 0, 2, 1, id="dp2"),
        pytest.param("attention", 1, 0, 1, 0, 2, 1, 4, 1, id="tp2"),
        pytest.param("attention", 1, 0, 2, 1, 1, 0, 2, 1, id="pcp2"),
        pytest.param("attention", 3, 2, 8, 0, 1, 0, 24, 16, id="global-dp-pcp-first"),
        pytest.param("attention", 3, 2, 8, 7, 1, 0, 24, 23, id="global-dp-pcp-last"),
        pytest.param("attention", 2, 1, 1, 0, 2, 1, 4, 3, id="dp2-tp2"),
        pytest.param("attention", 2, 1, 2, 1, 2, 1, 8, 7, id="dp2-pcp2-tp2"),
        pytest.param("ffn", 2, 1, 1, 0, 1, 0, 2, 1, id="ffn-role-size"),
    ],
)
def test_resolve_role_rank_linearizes_global_dp_and_local_pcp_tp(
    monkeypatch,
    role,
    dp_size,
    dp_rank,
    pcp_size,
    pcp_rank,
    tp_size,
    tp_rank,
    role_size,
    expected,
):
    _install_parallel_state(monkeypatch, pcp_rank=pcp_rank, tp_rank=tp_rank)
    config = _afd_config(
        role=role,
        num_attention_ranks=role_size if role == "attention" else 1,
        num_ffn_ranks=role_size if role == "ffn" else 1,
    )
    role_rank = resolve_role_rank(
        _vllm_config(
            dp_size=dp_size, dp_rank=dp_rank, pcp_size=pcp_size, tp_size=tp_size
        ),
        config,
    )
    assert role_rank == expected


def test_resolve_role_rank_rejects_derived_rank_outside_role_size(monkeypatch):
    _install_parallel_state(monkeypatch, tp_rank=0)
    with pytest.raises(ValueError, match="out of range"):
        resolve_role_rank(
            _vllm_config(dp_size=2, dp_rank=1, tp_size=2),
            _afd_config(num_attention_ranks=2),
        )


# Every A >= F pair with A, F in 1..8, divisible and not.
PARTITION_LAYOUTS = tuple((a, f) for a in range(1, 9) for f in range(1, a + 1))


def _mapping(role: str, role_rank: int, attention: int, ffn: int):
    return build_rank_mapping(
        _afd_config(role=role, num_attention_ranks=attention, num_ffn_ranks=ffn),
        role_rank,
    )


@pytest.mark.parametrize(("attention", "ffn"), PARTITION_LAYOUTS)
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


@pytest.mark.parametrize(("attention", "ffn"), PARTITION_LAYOUTS)
def test_both_roles_agree_on_the_subgroup_they_share(attention, ffn):
    for attention_rank in range(attention):
        mapping = _mapping("attention", attention_rank, attention, ffn)
        owner = _mapping("ffn", mapping.subgroup_index, attention, ffn)
        assert mapping.subgroup_ranks == owner.subgroup_ranks
        assert mapping.subgroup_ranks[mapping.rank_in_subgroup] == mapping.world_rank


@pytest.mark.parametrize(("attention", "ffn"), PARTITION_LAYOUTS)
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


@pytest.mark.parametrize(
    ("attention", "ffn"), [(a, f) for a, f in PARTITION_LAYOUTS if a % f == 0]
)
def test_divisible_layouts_keep_the_historical_grouping(attention, ffn):
    ratio = attention // ffn
    for ffn_rank in range(ffn):
        mapping = _mapping("ffn", ffn_rank, attention, ffn)
        assert len(mapping.subgroup_ranks) - 1 == ratio
        assert mapping.subgroup_ranks == (
            ffn_rank,
            *(ffn + ffn_rank * ratio + offset for offset in range(ratio)),
        )
