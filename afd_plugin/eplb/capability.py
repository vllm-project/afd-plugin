# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared, CPU-safe EPLB capability check for AFD.

The check answers "who runs the router, and does this connector have a
mechanism to keep that role's expert map and load counters consistent". It is
keyed by router location, connector and role rather than by platform. Every
input is a plain value, so no vLLM import is needed to evaluate or unit-test it.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from afd_plugin.config import DEEPEP_M2N_CONNECTOR

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig

P2P_CONNECTOR: Final[str] = "P2pNcclAFDConnector"


class EplbRouterLocation(enum.Enum):
    FFN = "ffn"
    ATTENTION = "attention"


class EplbPath(enum.Enum):
    NONE = "none"  # EPLB off, no redundant experts
    FFN_LOCAL = "ffn_local"  # Path 1: FFN owns map and counters
    ATTENTION_MAPPED = "attention_mapped"  # Path 2 / M2N: Attention maps ids


class EplbUnsupportedError(RuntimeError):
    """Raised when an EPLB request is not supported by this AFD deployment."""


@dataclass(frozen=True)
class EplbCapabilityRequest:
    """Plain-value description of one EPLB request for one role."""

    role: Literal["attention", "ffn"]
    # One of ``config.SUPPORTED_AFD_CONNECTORS``.
    connector: str
    router_location: EplbRouterLocation
    # Local flag; False on the M2N Attention role.
    enable_eplb: bool
    num_redundant_experts: int
    # ``eplb_config.use_async``.
    use_async_eplb: bool
    uses_dbo_or_ubatching: bool
    # ffn_faas path, no vLLM engine.
    is_standalone_ffn_service: bool
    # Model and quant method can serve ``expert_weights``.
    supports_expert_weight_views: bool
    # M2N Attention only; None before the handshake.
    handshake_reports_eplb: bool | None = None


_REASON_REDUNDANT_WITHOUT_EPLB: Final[str] = (
    "num_redundant_experts > 0 requires EPLB to be enabled"
)
_REASON_ATTENTION_ENABLE_EPLB: Final[str] = "EPLB must be enabled on the FFN role only"
_REASON_STANDALONE_FFN: Final[str] = "the standalone FFN service does not support EPLB"
_REASON_ASYNC_EPLB: Final[str] = "asynchronous EPLB is not supported"
_REASON_DBO_OR_UBATCHING: Final[str] = "EPLB does not support DBO or ubatching"
_REASON_NO_WEIGHT_VIEWS: Final[str] = (
    "model or quantization method cannot serve EPLB expert weight views"
)
_REASON_UNSUPPORTED_COMBINATION: Final[str] = (
    "no EPLB mechanism exists for this router location, connector and role"
)

# Rows are the only (router location, connector, role) combinations that can
# return a path.
_EPLB_PATH_TABLE: Final[Mapping[tuple[EplbRouterLocation, str, str], EplbPath]] = (
    MappingProxyType(
        {
            (EplbRouterLocation.FFN, P2P_CONNECTOR, "ffn"): EplbPath.FFN_LOCAL,
            (
                EplbRouterLocation.ATTENTION,
                DEEPEP_M2N_CONNECTOR,
                "ffn",
            ): EplbPath.FFN_LOCAL,
            (
                EplbRouterLocation.ATTENTION,
                DEEPEP_M2N_CONNECTOR,
                "attention",
            ): EplbPath.ATTENTION_MAPPED,
        },
    )
)


# Rows of ``_EPLB_PATH_TABLE`` whose backend has landed. GPU support was
# carried over from ``enable-eplb``; NPU rows stay disabled until the Ascend
# backend lands.
_ENABLED_EPLB_ROWS: Final[frozenset[tuple[EplbRouterLocation, str, str]]] = frozenset(
    {
        (EplbRouterLocation.FFN, P2P_CONNECTOR, "ffn"),
        (EplbRouterLocation.ATTENTION, DEEPEP_M2N_CONNECTOR, "ffn"),
        (EplbRouterLocation.ATTENTION, DEEPEP_M2N_CONNECTOR, "attention"),
    },
)


def _unsupported(request: EplbCapabilityRequest, reason: str) -> EplbUnsupportedError:
    return EplbUnsupportedError(
        f"AFD EPLB unsupported: {reason} "
        f"(connector={request.connector}, role={request.role}, "
        f"router={request.router_location.value})",
    )


def check_eplb_support(request: EplbCapabilityRequest) -> EplbPath:
    """Return the EPLB path for ``request`` or raise ``EplbUnsupportedError``."""

    if request.num_redundant_experts > 0 and not request.enable_eplb:
        raise _unsupported(request, _REASON_REDUNDANT_WITHOUT_EPLB)

    # The M2N Attention role runs without the local flag and learns about EPLB
    # from the connector handshake.
    eplb_requested = request.enable_eplb or request.handshake_reports_eplb is True
    if not eplb_requested:
        return EplbPath.NONE

    if request.role == "attention" and request.enable_eplb:
        raise _unsupported(request, _REASON_ATTENTION_ENABLE_EPLB)
    if request.is_standalone_ffn_service:
        raise _unsupported(request, _REASON_STANDALONE_FFN)
    if request.use_async_eplb:
        raise _unsupported(request, _REASON_ASYNC_EPLB)
    if request.uses_dbo_or_ubatching:
        raise _unsupported(request, _REASON_DBO_OR_UBATCHING)
    if request.role == "ffn" and not request.supports_expert_weight_views:
        raise _unsupported(request, _REASON_NO_WEIGHT_VIEWS)

    row = (request.router_location, request.connector, request.role)
    path = _EPLB_PATH_TABLE.get(row)
    if path is None or row not in _ENABLED_EPLB_ROWS:
        raise _unsupported(request, _REASON_UNSUPPORTED_COMBINATION)
    return path


def eplb_request_from_vllm_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    supports_expert_weight_views: bool,
    is_standalone_ffn_service: bool = False,
    handshake_reports_eplb: bool | None = None,
) -> EplbCapabilityRequest:
    """Build a request from vLLM and AFD configs for the current role.

    Config-level validators cannot see the model, so they pass
    ``supports_expert_weight_views=True`` and leave that rule to the model
    constructors, which know whether the model serves expert weight views.
    """

    parallel_config = vllm_config.parallel_config
    return EplbCapabilityRequest(
        role=afd_config.role,
        connector=afd_config.connector,
        router_location=(
            EplbRouterLocation.ATTENTION
            if afd_config.compute_gate_on_attention
            else EplbRouterLocation.FFN
        ),
        enable_eplb=parallel_config.enable_eplb,
        num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
        use_async_eplb=parallel_config.eplb_config.use_async,
        uses_dbo_or_ubatching=(
            parallel_config.enable_dbo or parallel_config.use_ubatching
        ),
        is_standalone_ffn_service=is_standalone_ffn_service,
        supports_expert_weight_views=supports_expert_weight_views,
        handshake_reports_eplb=handshake_reports_eplb,
    )


__all__ = [
    "EplbCapabilityRequest",
    "EplbPath",
    "EplbRouterLocation",
    "EplbUnsupportedError",
    "check_eplb_support",
    "eplb_request_from_vllm_config",
]
