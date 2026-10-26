# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed import resolve_role_rank, tp_rank_before_distributed_init


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


def test_resolve_role_rank_uses_given_tp_rank_without_vllm_groups(monkeypatch):
    """A TP rank passed by the caller replaces vLLM's TP group lookup.

    Why: routed-experts connectors resolve their role rank before vLLM's
    distributed init, when the TP group does not exist yet.
    """
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", None)

    role_rank = resolve_role_rank(
        _vllm_config(dp_size=2, dp_rank=1, tp_size=2),
        _afd_config(num_attention_ranks=4),
        tp_rank=1,
    )

    assert role_rank == 3


def test_resolve_role_rank_rejects_given_tp_rank_outside_tp_size():
    with pytest.raises(ValueError, match="tp_rank must be in"):
        resolve_role_rank(
            _vllm_config(tp_size=2),
            _afd_config(num_attention_ranks=2),
            tp_rank=2,
        )


def _pre_init_config(*, tp_size: int, pp_size: int = 1, pcp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


@pytest.mark.parametrize(
    ("worker_rank", "tp_size", "expected"),
    [(0, 1, 0), (3, 1, 0), (0, 2, 0), (1, 2, 1), (5, 4, 1)],
)
def test_tp_rank_before_distributed_init_is_worker_rank_mod_tp(
    worker_rank, tp_size, expected
):
    """TP is vLLM's innermost rank dimension, so the worker rank gives it.

    Why: this holds for DP-local and global worker ranks alike;
    ``check_worker_connector`` re-checks it after vLLM's init.
    """
    assert (
        tp_rank_before_distributed_init(_pre_init_config(tp_size=tp_size), worker_rank)
        == expected
    )


@pytest.mark.parametrize(("pp_size", "pcp_size"), [(2, 1), (1, 2)])
def test_tp_rank_before_distributed_init_rejects_pp_and_pcp(pp_size, pcp_size):
    with pytest.raises(ValueError, match="PP = PCP = 1"):
        tp_rank_before_distributed_init(
            _pre_init_config(tp_size=2, pp_size=pp_size, pcp_size=pcp_size), 1
        )
