# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD configuration parsed from vLLM ``additional_config``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from afd_plugin.config_utils import coerce_extra_bool as _coerce_bool
from afd_plugin.config_utils import coerce_extra_int as _coerce_int

if TYPE_CHECKING:
    from vllm.config import VllmConfig

AFD_ADDITIONAL_CONFIG_KEY: Final[str] = "afd"
AFD_ASYNC_NPU_CONNECTOR: Final[str] = "CAMAsyncAFDConnector"
AFD_ASYNC_GPU_CONNECTOR: Final[str] = "GpuAsyncAFDConnector"
# Connectors that drive FFN work from the connector receive loop instead of a DP
# metadata control plane, and therefore need the async-DP engine patches. The
# patches are platform-neutral; only the connector below them differs.
AFD_ASYNC_CONNECTORS: Final[frozenset[str]] = frozenset(
    {AFD_ASYNC_NPU_CONNECTOR, AFD_ASYNC_GPU_CONNECTOR},
)
AFDRole = Literal["attention", "ffn"]

SUPPORTED_AFD_ROLES: Final[tuple[str, ...]] = ("attention", "ffn")
SUPPORTED_AFD_CONNECTORS: Final[tuple[str, ...]] = (
    "P2pNcclAFDConnector",
    "CAMP2pAFDConnector",
    AFD_ASYNC_NPU_CONNECTOR,
    AFD_ASYNC_GPU_CONNECTOR,
)

_ALIASES: Final[dict[str, str]] = {
    "afd_connector": "connector",
    "afd_role": "role",
    "afd_port": "port",
    "afd_host": "host",
    "async": "async_dp",
}


@dataclass(frozen=True)
class AFDConfig:
    """Plugin-owned AFD configuration.

    The plugin uses shorter keys inside ``additional_config["afd"]`` while
    preserving read-only compatibility aliases for legacy ``afd_*`` field
    names. Per-process role ranks are runtime state resolved from vLLM's
    parallel placement and are intentionally not stored in this configuration.
    """

    # Connector implementation name used to create the backend data path.
    connector: str = "P2pNcclAFDConnector"
    # Whether AFD owns async data-parallel runtime patches for this process.
    async_dp: bool = False
    # Role owned by this process: Attention sends hidden states; FFN receives.
    role: AFDRole = "attention"
    # Port used by the AFD connector rendezvous/control path.
    port: int = 1239
    # Host used by the AFD connector rendezvous/control path.
    host: str = "127.0.0.1"
    # Number of AFD Attention ranks participating in this topology.
    num_attention_ranks: int = 1
    # Number of AFD FFN ranks participating in this topology.
    num_ffn_ranks: int = 1
    # Whether Attention computes MoE gate outputs before sending to FFN.
    compute_gate_on_attention: bool = False

    @property
    def afd_connector(self) -> str:
        return self.connector

    @property
    def afd_role(self) -> AFDRole:
        return self.role

    @property
    def afd_port(self) -> int:
        return self.port

    @property
    def afd_host(self) -> str:
        return self.host

    @property
    def is_attention_server(self) -> bool:
        return self.role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        return self.role == "ffn"

    def compute_hash(self) -> str:
        """Return a stable hash for graph-affecting AFD settings."""

        factors: list[str | bool | int] = [
            self.connector,
            self.async_dp,
            self.role,
            self.num_attention_ranks,
            self.num_ffn_ranks,
        ]
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def validate(self, *, expected_role: str | None = None) -> None:
        validate_afd_config(self, expected_role=expected_role)


def _normalize_mapping(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized: dict[str, Any] = {}
    connector_extra_config: dict[str, Any] | None = None
    valid_fields = set(AFDConfig.__dataclass_fields__)  # type: ignore[attr-defined]

    for key, value in raw.items():
        normalized_key = _ALIASES.get(key, key)
        if normalized_key == "connector_extra_config":
            if not isinstance(value, Mapping):
                raise TypeError(f"{key} must be a mapping")
            connector_extra_config = dict(value)
            continue
        if normalized_key not in valid_fields:
            raise ValueError(
                "unknown AFD config field "
                f"{key!r}; put connector-specific values under "
                "'connector_extra_config'",
            )
        if normalized_key in normalized:
            raise ValueError(
                f"duplicate AFD config field for {normalized_key!r}: "
                f"both alias and canonical key were provided",
            )
        normalized[normalized_key] = value

    if "async_dp" in normalized:
        normalized["async_dp"] = _coerce_bool(
            normalized["async_dp"],
            field_name="async",
        )

    for field_name in (
        "port",
        "num_attention_ranks",
        "num_ffn_ranks",
    ):
        if field_name in normalized:
            normalized[field_name] = _coerce_int(
                normalized[field_name],
                field_name=field_name,
            )

    if "compute_gate_on_attention" in normalized:
        normalized["compute_gate_on_attention"] = _coerce_bool(
            normalized["compute_gate_on_attention"],
            field_name="compute_gate_on_attention",
        )

    return normalized, connector_extra_config or {}


def afd_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    validate: bool = True,
    expected_role: str | None = None,
) -> AFDConfig:
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"AFD config must be a mapping, got {type(raw).__name__}",
        )
    normalized, _connector_extra_config = _normalize_mapping(raw)
    config = AFDConfig(**normalized)

    if validate:
        config.validate(expected_role=expected_role)
    return config


def connector_extra_config_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"AFD config must be a mapping, got {type(raw).__name__}",
        )
    _normalized, connector_extra_config = _normalize_mapping(raw)
    return connector_extra_config


def _additional_config_from_source(source: Any) -> Mapping[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        additional_config = source
    else:
        additional_config = source.additional_config
    if additional_config is None:
        return None
    if not isinstance(additional_config, Mapping):
        raise TypeError(
            "additional_config must be a mapping when parsing AFD config, "
            f"got {type(additional_config).__name__}",
        )
    return additional_config


def _afd_raw_from_source(source: Any) -> Mapping[str, Any] | None:
    additional_config = _additional_config_from_source(source)
    if additional_config is None:
        return None
    if AFD_ADDITIONAL_CONFIG_KEY not in additional_config:
        return None
    afd_raw = additional_config[AFD_ADDITIONAL_CONFIG_KEY]
    if not isinstance(afd_raw, Mapping):
        raise TypeError(
            f"AFD config must be a mapping, got {type(afd_raw).__name__}",
        )
    return afd_raw


def has_afd_config(source: Any) -> bool:
    additional_config = _additional_config_from_source(source)
    return bool(
        additional_config is not None and AFD_ADDITIONAL_CONFIG_KEY in additional_config
    )


def parse_optional_afd_config(
    source: Any,
    *,
    validate: bool = True,
    expected_role: str | None = None,
) -> AFDConfig | None:
    """Parse ``additional_config["afd"]`` when present."""

    afd_raw = _afd_raw_from_source(source)
    if afd_raw is None:
        return None
    return afd_config_from_mapping(
        afd_raw,
        validate=validate,
        expected_role=expected_role,
    )


def parse_afd_config(
    source: Any,
    *,
    validate: bool = True,
    expected_role: str | None = None,
) -> AFDConfig:
    """Parse required active ``additional_config["afd"]``."""

    config = parse_optional_afd_config(
        source,
        validate=validate,
        expected_role=expected_role,
    )
    if config is None:
        raise ValueError(
            'AFD config requires additional_config["afd"]; omit it to disable AFD',
        )
    return config


def connector_extra_config_from_source(source: Any) -> dict[str, Any]:
    afd_raw = _afd_raw_from_source(source)
    if afd_raw is None:
        return {}
    return connector_extra_config_from_mapping(afd_raw)


def is_afd_active(source: Any) -> bool:
    """Return true only when a present AFD config passes common validation."""

    return parse_optional_afd_config(source, validate=True) is not None


def is_afd_async_dp(vllm_config: VllmConfig) -> bool:
    """Return whether ``vllm_config`` selects AFD's async connector mode.

    This is a lightweight selector for import-time async-DP patches, not a full
    activation validator. Use ``is_afd_active`` or ``parse_afd_config`` when
    common config validity is required.
    """

    config = parse_optional_afd_config(vllm_config, validate=False)
    return (
        config is not None
        and config.async_dp
        and config.connector in AFD_ASYNC_CONNECTORS
    )


def validate_afd_config(
    config: AFDConfig,
    *,
    expected_role: str | None = None,
) -> None:
    """Validate AFD config values without importing vLLM or CUDA modules."""

    if config.role not in SUPPORTED_AFD_ROLES:
        raise ValueError(
            f"AFD role must be one of {SUPPORTED_AFD_ROLES!r}, got {config.role!r}",
        )
    if expected_role is not None and config.role != expected_role:
        raise ValueError(
            f"AFD role mismatch: expected {expected_role!r}, got {config.role!r}",
        )
    if config.connector not in SUPPORTED_AFD_CONNECTORS:
        raise ValueError(
            "AFD connector must be one of "
            f"{SUPPORTED_AFD_CONNECTORS!r}, got {config.connector!r}",
        )
    if config.async_dp and config.connector not in AFD_ASYNC_CONNECTORS:
        raise ValueError(
            "AFD async mode requires one of "
            f"{sorted(AFD_ASYNC_CONNECTORS)!r}, got {config.connector!r}",
        )
    if config.connector == "P2pNcclAFDConnector":
        from afd_plugin.distributed import validate_p2p_topology

        validate_p2p_topology(config)
    if not config.host:
        raise ValueError("AFD host must be non-empty")
    if not 0 < config.port < 65536:
        raise ValueError(f"AFD port must be in 1..65535, got {config.port}")
    if config.num_attention_ranks <= 0:
        raise ValueError(
            f"num_attention_ranks must be positive, got {config.num_attention_ranks}",
        )
    if config.num_ffn_ranks <= 0:
        raise ValueError(
            f"num_ffn_ranks must be positive, got {config.num_ffn_ranks}",
        )


__all__ = [
    "AFDConfig",
    "AFD_ASYNC_NPU_CONNECTOR",
    "AFD_ASYNC_CONNECTORS",
    "AFD_ASYNC_GPU_CONNECTOR",
    "afd_config_from_mapping",
    "AFD_ADDITIONAL_CONFIG_KEY",
    "AFDRole",
    "SUPPORTED_AFD_CONNECTORS",
    "SUPPORTED_AFD_ROLES",
    "connector_extra_config_from_mapping",
    "connector_extra_config_from_source",
    "has_afd_config",
    "is_afd_async_dp",
    "is_afd_active",
    "parse_afd_config",
    "parse_optional_afd_config",
    "validate_afd_config",
]
