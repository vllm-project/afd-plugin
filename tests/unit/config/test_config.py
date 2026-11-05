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


def test_connector_extra_fields_participate_in_the_graph_hash():
    extra = {"ffn_two_stream": True, "ffn_compute_aic_num": 12}
    raw = {
        "connector": "CAMAsyncAFDConnector",
        "role": "ffn",
        "connector_extra_config": extra,
    }
    first = afd_config_from_mapping(raw)
    extra["ffn_compute_aic_num"] = 10
    second = afd_config_from_mapping(raw)
    assert first.compute_hash() != second.compute_hash()
    raw["connector_extra_config"] = {"ffn_compute_aic_num": 10, "ffn_two_stream": True}
    assert second.compute_hash() == afd_config_from_mapping(raw).compute_hash()


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
            "dynamicQuant": "1",
        },
    }

    assert connector_extra_config_from_mapping(raw) == {
        "dynamicQuant": "1",
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
    with pytest.raises(ValueError, match="requires connector='CAMAsyncAFDConnector'"):
        parse_afd_config(
            {
                "afd": {
                    "connector": "CAMP2pAFDConnector",
                    "role": "attention",
                    "async": True,
                },
            },
        )


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


def test_afd_process_group_timeout_s_defaults_and_is_configurable():
    assert afd_config_from_mapping({}).afd_process_group_timeout_s == 120

    config = afd_config_from_mapping({"afd_process_group_timeout_s": "300"})

    assert config.afd_process_group_timeout_s == 300


def test_afd_process_group_timeout_s_must_be_positive():
    with pytest.raises(
        ValueError, match="afd_process_group_timeout_s must be positive"
    ):
        afd_config_from_mapping({"afd_process_group_timeout_s": 0})


def test_has_afd_config_only_checks_presence():
    source = {"afd": {"role": "decode"}}

    assert has_afd_config(source) is True
    with pytest.raises(ValueError, match="AFD role must be one of"):
        is_afd_active(source)


def test_is_afd_active_requires_common_config_validity():
    assert is_afd_active({"afd": {"role": "attention"}}) is True
    assert is_afd_active({}) is False


@pytest.mark.parametrize(
    "connector,enabled,expected",
    [
        ("CAMAsyncAFDConnector", True, True),
        ("CAMAsyncAFDConnector", False, False),
        ("CAMP2pAFDConnector", False, False),
    ],
)
def test_is_afd_async_dp_is_selector_not_activation_validator(
    connector, enabled, expected
):
    source = {
        "afd": {
            "connector": connector,
            "async": enabled,
            "role": "decode",
        },
    }

    assert is_afd_async_dp(SimpleNamespace(additional_config=source)) is expected
    with pytest.raises(ValueError, match="AFD role must be one of"):
        is_afd_active(source)


def test_common_config_rejects_float_int_values():
    with pytest.raises(TypeError, match="num_attention_ranks must be an integer"):
        afd_config_from_mapping({"num_attention_ranks": 2.5})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"role": "decode"}, "AFD role must be one of"),
        ({"connector": "tcp"}, "AFD connector must be one of"),
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
