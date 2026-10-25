# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone launch configuration preserves the existing CAM contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from afd_plugin.config import connector_extra_config_from_source, parse_afd_config
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig


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
