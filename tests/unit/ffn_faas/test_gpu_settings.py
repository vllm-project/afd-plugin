# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN settings produce the Worker-path DeepEP M2N FFN config."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from afd_plugin.config import connector_extra_config_from_source, parse_afd_config
from afd_plugin.ffn_faas.platforms.gpu.settings import (
    GPUFFNConfig,
    parse_connector_extra_config,
)


def _settings(**overrides):
    return GPUFFNConfig(
        **{
            "model": Path("/model"),
            "attention_ranks": 2,
            "ffn_ranks": 2,
            "rank": 1,
            **overrides,
        }
    )


def test_afd_config_matches_a_worker_path_m2n_ffn():
    settings = _settings(
        port=6239,
        afd_timeout_s=600,
        connector_extra_config=MappingProxyType({"expert_num_sms": 96}),
    )
    config = settings.afd_additional_config()

    afd = parse_afd_config(config, expected_role="ffn")
    assert afd.connector == "DeepEPM2NAFDConnector"
    assert afd.compute_gate_on_attention
    assert (afd.num_attention_ranks, afd.num_ffn_ranks) == (2, 2)
    assert (afd.port, afd.afd_process_group_timeout_s) == (6239, 600)
    assert connector_extra_config_from_source(config) == {"expert_num_sms": 96}


@pytest.mark.parametrize(
    "graph, dbo, eager, compilation_config",
    [
        (False, False, True, {}),
        (True, False, False, {"cudagraph_mode": "FULL_DECODE_ONLY"}),
        (True, True, False, {"cudagraph_mode": "FULL_DECODE_ONLY"}),
    ],
    ids=["eager", "graph", "graph-dbo"],
)
def test_engine_args_describe_an_f_only_ep_rank(graph, dbo, eager, compilation_config):
    kwargs = _settings(graph=graph, dbo=dbo).engine_kwargs()

    assert kwargs["distributed_executor_backend"] == "external_launcher"
    assert (kwargs["data_parallel_size"], kwargs["data_parallel_rank"]) == (2, 1)
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["enable_expert_parallel"] and kwargs["enable_ep_weight_filter"]
    assert kwargs["enforce_eager"] is eager
    assert kwargs["compilation_config"] == compilation_config
    assert kwargs["enable_dbo"] is dbo
    # vLLM's idle default all2all backend is what the M2N handshake requires.
    assert "all2all_backend" not in kwargs


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": 2}, "invalid F rank"),
        ({"attention_ranks": 0}, "rank counts must be positive"),
        ({"port": 0}, "valid port"),
        ({"afd_timeout_s": 0}, "--afd-timeout must be positive"),
        ({"shutdown_timeout": float("inf")}, "finite and positive"),
        (
            {"connector_extra_config": {"payload_format": "fp8_block128_fp32"}},
            "payload_format='bf16'",
        ),
        ({"connector_extra_config": {"num_lanes": 2}}, "disagrees with no --dbo"),
        (
            {"dbo": True, "connector_extra_config": {"num_lanes": 1}},
            "disagrees with --dbo",
        ),
    ],
    ids=[
        "rank",
        "attention-ranks",
        "port",
        "afd-timeout",
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
    "text, message",
    [("{", "not valid JSON"), ("[1]", "must be a JSON object")],
    ids=["malformed", "not-object"],
)
def test_connector_extra_config_must_be_a_json_object(text, message):
    with pytest.raises(ValueError, match=message):
        parse_connector_extra_config(text)
