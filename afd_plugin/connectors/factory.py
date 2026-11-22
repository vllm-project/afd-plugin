# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Factory for plugin-owned AFD connectors."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from afd_plugin.config import (
    AFDConfig,
    connector_extra_config_from_source,
    parse_afd_config,
)
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.distributed import resolve_role_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class _ConnectorRegistration(NamedTuple):
    load_connector: Callable[[], type[AFDConnectorBase]]
    # (module path, class name) of the Attention-role MoE runner that
    # ``build_expert`` builds for this connector; None when the connector has
    # no factory-built Attention MoE path.
    attention_moe_runner: tuple[str, str] | None


class AFDConnectorFactory:
    _registry: dict[str, _ConnectorRegistration] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        *,
        attention_moe_runner: tuple[str, str] | None = None,
        replace: bool = False,
    ) -> None:
        if name in cls._registry and not replace:
            raise ValueError(f"connector {name!r} is already registered")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            # ``getattr`` rather than ``vars(module)``: packages in this tree resolve
            # their names through a module-level ``__getattr__``, so the class is not
            # necessarily present in the module dictionary.
            connector_cls = getattr(module, class_name)
            if not issubclass(connector_cls, AFDConnectorBase):
                raise TypeError(
                    f"{module_path}.{class_name} is not an AFDConnectorBase",
                )
            return connector_cls

        cls._registry[name] = _ConnectorRegistration(loader, attention_moe_runner)

    @classmethod
    def create_connector(
        cls,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig | None = None,
    ) -> AFDConnectorBase:
        if afd_config is None:
            config = parse_afd_config(vllm_config)
        else:
            afd_config.validate()
            config = afd_config
        if config.connector not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {config.connector}")
        connector_cls = cls._registry[config.connector].load_connector()
        role_rank = resolve_role_rank(vllm_config, config)
        return connector_cls(
            rank,
            local_rank,
            vllm_config,
            config,
            role_rank,
        )

    @classmethod
    def get_connector_class(cls, connector_name: str) -> type[AFDConnectorBase]:
        if connector_name not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {connector_name}")
        return cls._registry[connector_name].load_connector()

    @classmethod
    def get_attention_moe_runner(cls, connector_name: str) -> tuple[str, str] | None:
        """Return the (module path, class name) of the connector's
        Attention-role MoE runner, or None when it registers none."""
        if connector_name not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {connector_name}")
        return cls._registry[connector_name].attention_moe_runner

    @classmethod
    def parse_connector_extra_info(
        cls,
        connector_name: str,
        source: VllmConfig,
    ) -> ConnectorExtraInfo:
        connector_cls = cls.get_connector_class(connector_name)
        return connector_cls.parse_extra_config(
            connector_extra_config_from_source(source),
        )


AFDConnectorFactory.register_connector(
    "P2pNcclAFDConnector",
    "afd_plugin.connectors.gpu.p2p",
    "P2pNcclAFDConnector",
    attention_moe_runner=(
        "afd_plugin.model_executor.ffn.remote",
        "AFDRemoteMoERunner",
    ),
)
AFDConnectorFactory.register_connector(
    "DeepEPM2NAFDConnector",
    "afd_plugin.connectors.gpu.deepep_m2n.connector",
    "DeepEPM2NAFDConnector",
    # vLLM's runner routes, with a shared-experts slot per microbatch; AFD's
    # routed experts send the routes.
    attention_moe_runner=(
        "afd_plugin.model_executor.ffn.remote",
        "AFDRoutedExpertsMoERunner",
    ),
)
AFDConnectorFactory.register_connector(
    "CAMP2pAFDConnector",
    "afd_plugin.connectors.npu.camp2p",
    "CAMP2pAFDConnector",
)
AFDConnectorFactory.register_connector(
    "CAMAsyncAFDConnector",
    "afd_plugin.connectors.npu.async_cam",
    "CAMAsyncAFDConnector",
)


__all__ = ["AFDConnectorFactory"]
