# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN settings keep only launcher and service-fixed facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from afd_plugin.ffn_faas.platforms.gpu.settings import (
    ASYNC_FFN_DRAIN_TIMEOUT_S,
    GPUAsyncFFNConfig,
    GPUFFNConfig,
)


def _settings(**overrides):
    return GPUFFNConfig(
        **{
            "model": Path("/model"),
            "ffn_ranks": 2,
            "rank": 1,
            **overrides,
        }
    )


def _async_settings(**overrides):
    return GPUAsyncFFNConfig(
        **{
            "model": Path("/model"),
            "ffn_ranks": 2,
            "rank": 1,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "graph, dbo, graph_mode, num_lanes",
    [
        (False, False, "eager", 1),
        (True, False, "FULL_DECODE_ONLY", 1),
        (True, True, "FULL_DECODE_ONLY", 2),
    ],
    ids=["eager", "graph", "graph-dbo"],
)
def test_execution_mode_and_lanes(graph, dbo, graph_mode, num_lanes):
    settings = _settings(graph=graph, dbo=dbo)

    assert settings.graph_mode == graph_mode
    assert settings.num_lanes == num_lanes


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": 2}, "invalid F-only rank"),
        ({"ffn_ranks": 0}, "invalid F-only rank"),
        ({"local_rank": -1}, "local rank must be nonnegative"),
        ({"shutdown_timeout": float("inf")}, "finite and positive"),
        (
            {"connector_extra_config": {"payload_format": "fp8_block128_fp32"}},
            "payload_format='bf16'",
        ),
        (
            {"connector_extra_config": {"num_lanes": 2}},
            "disagrees with no --enable-dbo",
        ),
        (
            {"dbo": True, "connector_extra_config": {"num_lanes": 1}},
            "disagrees with --enable-dbo",
        ),
    ],
    ids=[
        "rank",
        "ffn-ranks",
        "local-rank",
        "shutdown-timeout",
        "fp8-payload",
        "lanes-without-dbo",
        "lanes-with-dbo",
    ],
)
def test_invalid_launch_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _settings(**overrides)


def test_matching_explicit_lane_count_is_accepted():
    assert _settings(dbo=True, connector_extra_config={"num_lanes": 2}).num_lanes == 2


@pytest.mark.parametrize(
    "graph, dbo, graph_mode, num_stages",
    [
        (False, False, "eager", 1),
        (True, False, "FULL_DECODE_ONLY", 1),
        (True, True, "FULL_DECODE_ONLY", 2),
    ],
    ids=["eager", "graph", "graph-dbo"],
)
def test_async_execution_mode_and_stages(graph, dbo, graph_mode, num_stages):
    settings = _async_settings(graph=graph, dbo=dbo)

    assert settings.graph_mode == graph_mode
    assert settings.num_stages == num_stages
    assert settings.shutdown_timeout == ASYNC_FFN_DRAIN_TIMEOUT_S


def test_async_valid_extra_config_is_accepted():
    settings = _async_settings(
        dbo=True,
        connector_extra_config={
            "attn_ranks_per_dp": 2,
            "ring_depth": 2,
            "recv_poll_timeout_ms": 25,
        },
    )
    assert settings.num_stages == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": 2}, "invalid F-only rank"),
        ({"ffn_ranks": 0}, "invalid F-only rank"),
        ({"local_rank": -1}, "local rank must be nonnegative"),
        ({"shutdown_timeout": float("inf")}, "finite and positive"),
        (
            {"connector_extra_config": {"unknown_option": 1}},
            "Unknown async GPU connector field",
        ),
        (
            {"connector_extra_config": {"recv_poll_timeout_ms": 0}},
            "recv_poll_timeout_ms must be positive",
        ),
        (
            {"dbo": True, "connector_extra_config": {"ring_depth": 1}},
            "ring_depth must cover every live stage",
        ),
    ],
    ids=[
        "rank",
        "ffn-ranks",
        "local-rank",
        "shutdown-timeout",
        "unknown-extra-field",
        "non-positive-extra-field",
        "ring-depth-below-dbo-stages",
    ],
)
def test_async_invalid_launch_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _async_settings(**overrides)
