# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the shared EPLB capability check (CPU-only, no vLLM)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    CAMP2P_CONNECTOR,
    DEEPEP_M2N_CONNECTOR,
    AFDConfig,
)
from afd_plugin.eplb import capability
from afd_plugin.eplb.capability import (
    EplbCapabilityRequest,
    EplbPath,
    EplbRouterLocation,
    EplbUnsupportedError,
    check_eplb_support,
    eplb_request_from_vllm_config,
)

P2P = "P2pNcclAFDConnector"
MESSAGE_PREFIX = "AFD EPLB unsupported: "


def _request(**overrides) -> EplbCapabilityRequest:
    fields = {
        "role": "ffn",
        "connector": P2P,
        "router_location": EplbRouterLocation.FFN,
        "enable_eplb": False,
        "num_redundant_experts": 0,
        "use_async_eplb": False,
        "uses_dbo_or_ubatching": False,
        "is_standalone_ffn_service": False,
        "supports_expert_weight_views": True,
        "handshake_reports_eplb": None,
    }
    fields.update(overrides)
    return EplbCapabilityRequest(**fields)


@pytest.fixture
def all_rows_enabled(monkeypatch):
    """Simulate every backend sub-task having landed."""
    monkeypatch.setattr(
        capability,
        "_ENABLED_EPLB_ROWS",
        frozenset(capability._EPLB_PATH_TABLE),
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("connector", [P2P, DEEPEP_M2N_CONNECTOR, CAMP2P_CONNECTOR])
@pytest.mark.parametrize("router", list(EplbRouterLocation))
def test_eplb_off_returns_none_for_every_combination(role, connector, router):
    request = _request(role=role, connector=connector, router_location=router)
    assert check_eplb_support(request) is EplbPath.NONE


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_redundant_experts_without_eplb_raise(role):
    # Closes the silent drop of redundant experts (e.g. Async CAM).
    request = _request(
        role=role,
        connector=AFD_ASYNC_CONNECTOR,
        router_location=EplbRouterLocation.ATTENTION,
        num_redundant_experts=2,
    )
    with pytest.raises(EplbUnsupportedError, match="num_redundant_experts"):
        check_eplb_support(request)


def test_error_is_runtime_error_with_fixed_message_shape():
    request = _request(
        role="ffn",
        connector=CAMP2P_CONNECTOR,
        enable_eplb=True,
    )
    with pytest.raises(RuntimeError) as exc_info:
        check_eplb_support(request)
    assert isinstance(exc_info.value, EplbUnsupportedError)
    message = str(exc_info.value)
    assert message.startswith(MESSAGE_PREFIX)
    assert message.endswith(f"(connector={CAMP2P_CONNECTOR}, role=ffn, router=ffn)")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"use_async_eplb": True}, "asynchronous EPLB"),
        ({"uses_dbo_or_ubatching": True}, "DBO or ubatching"),
        ({"is_standalone_ffn_service": True}, "standalone FFN service"),
        ({"supports_expert_weight_views": False}, "expert weight views"),
        ({"role": "attention"}, "FFN role only"),
    ],
)
def test_always_on_rules_raise_even_for_a_supported_row(
    all_rows_enabled, overrides, reason
):
    request = _request(enable_eplb=True, **overrides)
    with pytest.raises(EplbUnsupportedError, match=reason):
        check_eplb_support(request)


def test_npu_row_is_rejected_until_the_ascend_backend_lands():
    request = _request(
        connector=CAMP2P_CONNECTOR,
        router_location=EplbRouterLocation.FFN,
        enable_eplb=True,
    )
    with pytest.raises(EplbUnsupportedError, match="no EPLB mechanism"):
        check_eplb_support(request)


@pytest.mark.parametrize(
    ("connector", "router", "role", "handshake", "expected"),
    [
        (P2P, EplbRouterLocation.FFN, "ffn", None, EplbPath.FFN_LOCAL),
        (
            DEEPEP_M2N_CONNECTOR,
            EplbRouterLocation.ATTENTION,
            "ffn",
            None,
            EplbPath.FFN_LOCAL,
        ),
        (
            DEEPEP_M2N_CONNECTOR,
            EplbRouterLocation.ATTENTION,
            "attention",
            True,
            EplbPath.ATTENTION_MAPPED,
        ),
    ],
)
def test_decision_table_rows_return_their_path(
    all_rows_enabled, connector, router, role, handshake, expected
):
    request = _request(
        role=role,
        connector=connector,
        router_location=router,
        enable_eplb=role == "ffn",
        handshake_reports_eplb=handshake,
    )
    assert check_eplb_support(request) is expected


def test_m2n_attention_without_local_flag_and_no_handshake_report_is_none():
    request = _request(
        role="attention",
        connector=DEEPEP_M2N_CONNECTOR,
        router_location=EplbRouterLocation.ATTENTION,
        handshake_reports_eplb=False,
    )
    assert check_eplb_support(request) is EplbPath.NONE


@pytest.mark.parametrize(
    ("connector", "router", "role"),
    [
        (CAMP2P_CONNECTOR, EplbRouterLocation.FFN, "ffn"),
        (CAMP2P_CONNECTOR, EplbRouterLocation.ATTENTION, "ffn"),
        (AFD_ASYNC_CONNECTOR, EplbRouterLocation.ATTENTION, "ffn"),
        (AFD_ASYNC_CONNECTOR, EplbRouterLocation.ATTENTION, "attention"),
        (P2P, EplbRouterLocation.ATTENTION, "ffn"),
        (DEEPEP_M2N_CONNECTOR, EplbRouterLocation.FFN, "ffn"),
    ],
)
def test_rows_absent_from_the_table_raise(all_rows_enabled, connector, router, role):
    request = _request(
        role=role,
        connector=connector,
        router_location=router,
        enable_eplb=role == "ffn",
        handshake_reports_eplb=True if role == "attention" else None,
    )
    with pytest.raises(EplbUnsupportedError, match="no EPLB mechanism"):
        check_eplb_support(request)


def test_only_gpu_rows_are_enabled_by_default():
    p2p_ffn = (EplbRouterLocation.FFN, P2P, "ffn")
    m2n_ffn = (EplbRouterLocation.ATTENTION, DEEPEP_M2N_CONNECTOR, "ffn")
    m2n_attention = (
        EplbRouterLocation.ATTENTION,
        DEEPEP_M2N_CONNECTOR,
        "attention",
    )
    assert frozenset({p2p_ffn, m2n_ffn, m2n_attention}) == capability._ENABLED_EPLB_ROWS


def test_request_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _request().enable_eplb = True  # type: ignore[misc]


def _vllm_config(**parallel_overrides) -> SimpleNamespace:
    parallel = {
        "enable_eplb": False,
        "eplb_config": SimpleNamespace(num_redundant_experts=0, use_async=False),
        "enable_dbo": False,
        "use_ubatching": False,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(parallel_config=SimpleNamespace(**parallel))


@pytest.mark.parametrize("compute_gate_on_attention", [False, True])
def test_request_from_vllm_config_maps_fields(compute_gate_on_attention):
    afd_config = AFDConfig(
        role="ffn",
        connector=DEEPEP_M2N_CONNECTOR,
        compute_gate_on_attention=compute_gate_on_attention,
    )
    vllm_config = _vllm_config(
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=4, use_async=True),
        use_ubatching=True,
    )

    request = eplb_request_from_vllm_config(
        vllm_config,
        afd_config,
        supports_expert_weight_views=False,
        handshake_reports_eplb=True,
    )

    assert request == EplbCapabilityRequest(
        role="ffn",
        connector=DEEPEP_M2N_CONNECTOR,
        router_location=(
            EplbRouterLocation.ATTENTION
            if compute_gate_on_attention
            else EplbRouterLocation.FFN
        ),
        enable_eplb=True,
        num_redundant_experts=4,
        use_async_eplb=True,
        uses_dbo_or_ubatching=True,
        is_standalone_ffn_service=False,
        supports_expert_weight_views=False,
        handshake_reports_eplb=True,
    )


def test_request_from_vllm_config_treats_dbo_as_ubatching():
    request = eplb_request_from_vllm_config(
        _vllm_config(enable_dbo=True),
        AFDConfig(),
        supports_expert_weight_views=True,
    )
    assert request.uses_dbo_or_ubatching is True
