# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone launch configuration preserves the existing CAM contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from afd_plugin.config import connector_extra_config_from_source, parse_afd_config
from afd_plugin.ffn_faas.platforms.npu.settings import (
    FFNConfig,
    validate_native_expert_map,
    validate_static_experts,
)


@pytest.fixture
def settings():
    return FFNConfig(Path("/model"), attention_ranks=12, attention_tp=4, ffn_ranks=4)


def test_existing_cam_configuration_roundtrip(settings):
    config = settings.additional_config()
    afd = parse_afd_config(config, expected_role="ffn")
    assert afd.connector == "CAMAsyncAFDConnector"
    assert afd.async_dp and afd.compute_gate_on_attention
    assert (afd.num_attention_ranks, afd.num_ffn_ranks) == (12, 4)
    assert connector_extra_config_from_source(config) == {
        "dynamicQuant": 1,
        "attn_ranks_per_dp": 4,
        "ffn_two_stream": False,
    }


def test_explicit_native_stream_budgets(settings):
    settings = replace(
        settings,
        full_graph=True,
        ffn_two_stream=True,
        ffn_compute_aic_num=20,
        ffn_compute_aiv_num=40,
        ffn_combine_aiv_num=8,
        hccl_buffer_size=256,
    )
    assert connector_extra_config_from_source(settings.additional_config()) == {
        "dynamicQuant": 1,
        "attn_ranks_per_dp": 4,
        "ffn_two_stream": True,
        "ffn_compute_aic_num": 20,
        "ffn_compute_aiv_num": 40,
        "ffn_combine_aiv_num": 8,
        "hccl_buffer_size": 256,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"attention_ranks": 0},
        {"attention_ranks": 11},
        {"ffn_ranks": 0},
        {"attention_tp": 0},
        {"max_num_batched_tokens": 2047},
        {"rank": 4},
        {"local_rank": -1},
        {"device": -1},
        {"port": 0},
        {"port": 65536},
        {"shutdown_timeout": 0},
        {"shutdown_timeout": float("nan")},
        {"shutdown_timeout": float("inf")},
        {"ffn_compute_aic_num": 0},
    ],
)
def test_invalid_launch_rejected_before_native_runtime(settings, change):
    with pytest.raises(ValueError):
        replace(settings, **change)


@pytest.mark.parametrize(
    "rank,entries,owned",
    [
        (0, [0, 1, -1, -1], {0, 1}),
        (1, [-1, -1, 0, 1], {2, 3}),
    ],
)
def test_native_expert_map_matches_cam(rank, entries, owned):
    assert (
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=rank, expert_map=entries
        )
        == owned
    )


@pytest.mark.parametrize(
    "entries",
    [
        [0, 0, -1, -1],
        [-1, -1, 0, 1],
        [1, 0, -1, -1],
        [0, 1],
    ],
)
def test_native_expert_map_mismatch_rejected(entries):
    with pytest.raises(ValueError):
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=entries
        )


@pytest.mark.parametrize("ffn_ranks", [1, 2])
def test_implicit_native_map_only_when_all_experts_are_local(ffn_ranks):
    if ffn_ranks == 1:
        assert validate_native_expert_map(
            num_experts=4, ffn_ranks=1, ffn_rank=0, expert_map=None
        ) == frozenset(range(4))
    else:
        with pytest.raises(ValueError, match="explicit map"):
            validate_native_expert_map(
                num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=None
            )


@pytest.mark.parametrize(
    "eplb,elastic,redundant,dynamic",
    [
        (True, False, 0, False),
        (False, True, 0, False),
        (False, False, 1, False),
        (False, False, 0, True),
    ],
)
def test_static_cam_adapter_rejects_expert_relocation(
    eplb, elastic, redundant, dynamic
):
    with pytest.raises(ValueError, match="static expert placement"):
        validate_static_experts(
            enable_eplb=eplb,
            enable_elastic_ep=elastic,
            redundant_experts=redundant,
            dynamic_eplb=dynamic,
        )
