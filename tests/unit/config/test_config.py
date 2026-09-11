# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import (
    AFDConfig,
    afd_config_from_mapping,
    connector_extra_config_from_mapping,
    has_afd_config,
    is_afd_active,
    is_afd_async_dp,
    parse_afd_config,
    parse_optional_afd_config,
)


def test_parse_empty_additional_config_is_inactive():
    assert parse_optional_afd_config({}) is None

    with pytest.raises(ValueError, match="requires additional_config"):
        parse_afd_config({})


def test_parse_canonical_additional_config_namespace():
    config = parse_afd_config(
        {
            "afd": {
                "role": "ffn",
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
            },
        },
        expected_role="ffn",
    )

    assert config.role == "ffn"
    assert config.afd_role == "ffn"
    assert config.is_ffn_server


def test_parse_vllm_like_config_object():
    vllm_config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "attention",
                "connector": "P2pNcclAFDConnector",
            },
        },
    )

    config = parse_afd_config(vllm_config, expected_role="attention")

    assert config.is_attention_server


def test_config_object_requires_additional_config_attribute():
    with pytest.raises(AttributeError, match="additional_config"):
        parse_optional_afd_config(object())


def test_compute_gate_on_attention_is_common_config():
    config = parse_afd_config(
        {
            "afd": {
                "role": "ffn",
                "compute_gate_on_attention": "true",
            },
        },
        expected_role="ffn",
    )

    assert config.compute_gate_on_attention is True


def test_common_config_coerces_integer_bool_values():
    assert (
        afd_config_from_mapping(
            {"compute_gate_on_attention": 1}
        ).compute_gate_on_attention
        is True
    )
    assert (
        afd_config_from_mapping(
            {"compute_gate_on_attention": 0}
        ).compute_gate_on_attention
        is False
    )


def test_connector_extra_config_is_extracted_for_connector_parser():
    raw = {
        "connector": "CAMAsyncAFDConnector",
        "role": "attention",
        "connector_extra_config": {
            "async_moe_ubatching": "true",
        },
    }

    assert connector_extra_config_from_mapping(raw) == {
        "async_moe_ubatching": "true",
    }


def test_parse_async_dp_config_from_async_alias():
    config = parse_afd_config(
        {
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async": "true",
            },
        },
    )

    assert config.async_dp is True


def test_async_dp_requires_async_connector():
    with pytest.raises(ValueError, match="AFD async mode requires one of"):
        parse_afd_config(
            {
                "afd": {
                    "connector": "CAMP2pAFDConnector",
                    "role": "attention",
                    "async": True,
                },
            },
        )


@pytest.mark.parametrize(
    "connector",
    ["CAMAsyncAFDConnector", "GpuAsyncAFDConnector"],
)
def test_async_dp_accepts_every_async_connector(connector):
    config = parse_afd_config(
        {
            "afd": {
                "connector": connector,
                "role": "attention",
                "async": True,
            },
        },
    )
    assert config.connector == connector
    assert config.async_dp


def test_original_common_afd_field_aliases_are_supported():
    raw = {
        "afd_role": "ffn",
        "afd_connector": "P2pNcclAFDConnector",
        "afd_host": "localhost",
        "afd_port": 2345,
    }
    config = afd_config_from_mapping(raw)

    assert config.role == "ffn"
    assert config.connector == "P2pNcclAFDConnector"
    assert config.afd_host == "localhost"
    assert config.afd_port == 2345


def test_has_afd_config_only_checks_presence():
    source = {"afd": {"role": "decode"}}

    assert has_afd_config(source) is True
    with pytest.raises(ValueError, match="AFD role must be one of"):
        is_afd_active(source)


def test_is_afd_active_requires_common_config_validity():
    assert is_afd_active({"afd": {"role": "attention"}}) is True
    assert is_afd_active({}) is False


def test_is_afd_async_dp_is_selector_not_activation_validator():
    source = {
        "afd": {
            "connector": "CAMAsyncAFDConnector",
            "async": True,
            "role": "decode",
        },
    }

    assert is_afd_async_dp(SimpleNamespace(additional_config=source)) is True
    with pytest.raises(ValueError, match="AFD role must be one of"):
        is_afd_active(source)


def test_integer_like_config_values_are_coerced():
    class IntLike:
        def __int__(self) -> int:
            return 2

    config = afd_config_from_mapping(
        {
            "num_attention_ranks": IntLike(),
            "num_ffn_ranks": IntLike(),
        },
    )

    assert config.num_attention_ranks == 2
    assert config.num_ffn_ranks == 2


def test_common_config_rejects_float_int_values():
    with pytest.raises(TypeError, match="num_attention_ranks must be an integer"):
        afd_config_from_mapping({"num_attention_ranks": 2.5})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"enabled": True}, "unknown AFD config field"),
        ({"role": "decode"}, "AFD role must be one of"),
        ({"connector": "tcp"}, "AFD connector must be one of"),
        ({"afd_role_rank": 0}, "unknown AFD config field"),
        ({"num_attention_servers": 2}, "unknown AFD config field"),
        ({"num_ffn_servers": 2}, "unknown AFD config field"),
        ({"afd_server_rank": 0}, "unknown AFD config field"),
        ({"unknown": True}, "unknown AFD config field"),
    ],
)
def test_validation_errors_are_clear(raw, message):
    with pytest.raises((TypeError, ValueError), match=message):
        afd_config_from_mapping(raw)


def test_role_mismatch_fails_fast():
    with pytest.raises(ValueError, match="AFD role mismatch"):
        afd_config_from_mapping(
            {"role": "ffn"},
            expected_role="attention",
        )


def test_compute_hash_changes_for_graph_affecting_fields():
    attention = AFDConfig(role="attention")
    ffn = AFDConfig(role="ffn")

    assert attention.compute_hash() != ffn.compute_hash()
