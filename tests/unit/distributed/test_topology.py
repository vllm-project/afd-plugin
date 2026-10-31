from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed import (
    build_rank_mapping,
    build_window_expert_layout,
    build_window_rank_mapping,
    resolve_role_rank,
)


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
    connector: str = "P2pNcclAFDConnector",
):
    return AFDConfig(
        connector=connector,
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
    parallel_state_module.get_pcp_group = lambda: SimpleNamespace(
        rank_in_group=pcp_rank,
    )
    parallel_state_module.get_tensor_model_parallel_rank = lambda: tp_rank
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state_module,
    )


@pytest.mark.parametrize(("pcp_rank", "expected_role_rank"), [(0, 16), (7, 23)])
def test_resolve_role_rank_uses_global_dp_rank_for_dp3_pcp8_node1(
    monkeypatch,
    pcp_rank,
    expected_role_rank,
):
    _install_parallel_state(monkeypatch, pcp_rank=pcp_rank)

    role_rank = resolve_role_rank(
        _vllm_config(dp_size=3, dp_rank=2, pcp_size=8),
        _afd_config(num_attention_ranks=24, num_ffn_ranks=8),
    )

    assert role_rank == expected_role_rank


def test_resolve_role_rank_linearizes_dp_pcp_and_tp(monkeypatch):
    _install_parallel_state(monkeypatch, pcp_rank=1, tp_rank=1)

    role_rank = resolve_role_rank(
        _vllm_config(dp_size=2, dp_rank=1, pcp_size=2, tp_size=2),
        _afd_config(num_attention_ranks=8),
    )

    assert role_rank == 7


def test_resolve_role_rank_uses_selected_role_size():
    role_rank = resolve_role_rank(
        _vllm_config(dp_size=2, dp_rank=1),
        _afd_config(role="ffn", num_attention_ranks=1, num_ffn_ranks=2),
    )

    assert role_rank == 1


def test_resolve_role_rank_rejects_derived_rank_outside_role_size(monkeypatch):
    _install_parallel_state(monkeypatch, tp_rank=0)

    with pytest.raises(ValueError, match="out of range"):
        resolve_role_rank(
            _vllm_config(dp_size=2, dp_rank=1, tp_size=2),
            _afd_config(num_attention_ranks=2),
        )


def test_build_window_rank_mapping_connects_all_opposite_role_ranks():
    attention_config = _afd_config(
        role="attention",
        num_attention_ranks=8,
        num_ffn_ranks=16,
        connector="WindowAFDConnector",
    )
    ffn_config = _afd_config(
        role="ffn",
        num_attention_ranks=8,
        num_ffn_ranks=16,
        connector="WindowAFDConnector",
    )

    attention = build_window_rank_mapping(attention_config, 7)
    ffn = build_window_rank_mapping(ffn_config, 15)

    assert attention.world_rank == 23
    assert attention.peer_ranks == tuple(range(16))
    assert ffn.world_rank == 15
    assert ffn.peer_ranks == tuple(range(16, 24))


@pytest.mark.parametrize(
    ("ffn_rank", "kind", "start", "count"),
    [
        (0, "shared", 256, 1),
        (1, "routed", 0, 18),
        (2, "routed", 18, 17),
        (15, "routed", 239, 17),
    ],
)
def test_build_window_expert_layout_is_shared_first_and_balanced(
    ffn_rank,
    kind,
    start,
    count,
):
    layout = build_window_expert_layout(
        routed_expert_num=256,
        ffn_size=16,
        ffn_rank=ffn_rank,
    )

    assert layout.kind == kind
    assert layout.local_expert_start == start
    assert layout.local_expert_count == count


@pytest.mark.parametrize(
    ("role", "role_rank", "subgroup_index", "ffn_peers", "attention_peers"),
    [
        ("attention", 0, 0, (0, 1), (4,)),
        ("attention", 1, 1, (2, 3), (5,)),
        ("ffn", 0, 0, (0, 1), (4,)),
        ("ffn", 1, 0, (0, 1), (4,)),
        ("ffn", 2, 1, (2, 3), (5,)),
        ("ffn", 3, 1, (2, 3), (5,)),
    ],
)
def test_build_rank_mapping_supports_integer_ffn_fanout(
    role,
    role_rank,
    subgroup_index,
    ffn_peers,
    attention_peers,
):
    config = _afd_config(
        role=role,
        num_attention_ranks=2,
        num_ffn_ranks=4,
        connector="P2pHcclAFDConnector",
    )

    mapping = build_rank_mapping(config, role_rank)

    assert mapping.ratio == 2
    assert mapping.attention_fans_out is True
    assert mapping.subgroup_index == subgroup_index
    assert mapping.ffn_peer_ranks == ffn_peers
    assert mapping.attention_peer_ranks == attention_peers
    assert mapping.subgroup_ranks == (*ffn_peers, *attention_peers)
    assert mapping.p2p_rank == mapping.world_rank


def test_build_rank_mapping_rejects_ffn_fanout_for_nccl():
    with pytest.raises(ValueError, match="other than P2pHcclAFDConnector"):
        build_rank_mapping(
            _afd_config(
                num_attention_ranks=2,
                num_ffn_ranks=4,
                connector="P2pNcclAFDConnector",
            ),
            0,
        )


def test_build_rank_mapping_rejects_non_integer_ffn_fanout():
    with pytest.raises(ValueError, match="num_ffn_ranks to be a multiple"):
        build_rank_mapping(
            _afd_config(
                num_attention_ranks=2,
                num_ffn_ranks=3,
                connector="P2pHcclAFDConnector",
            ),
            0,
        )
