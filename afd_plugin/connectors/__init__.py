# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.connectors.base import (
        AFDConnectorBase,
        AFDControlPlane,
        AFDRoutedExpertsConnector,
        ConnectorExtraInfo,
    )
    from afd_plugin.connectors.factory import AFDConnectorFactory
    from afd_plugin.connectors.metadata import (
        AFDA2FTransferPayload,
        AFDControlPayload,
        AFDDPMetadata,
        AFDExpertRoutingSpec,
        AFDF2ATransferPayload,
        AFDForwardContextMetadata,
        AFDSingleDPMetadata,
        AFDTransferContext,
        AFDTransferMetadata,
        AFDTransferState,
        FFNLocalExperts,
        LocalExpertWeights,
    )


def __getattr__(name: str):
    if name in {
        "AFDConnectorBase",
        "AFDControlPlane",
        "AFDRoutedExpertsConnector",
        "ConnectorExtraInfo",
    }:
        module = "base"
    elif name == "AFDConnectorFactory":
        module = "factory"
    elif name in __all__:
        module = "metadata"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


__all__ = [
    "AFDRoutedExpertsConnector",
    "AFDConnectorBase",
    "AFDControlPlane",
    "ConnectorExtraInfo",
    "AFDExpertRoutingSpec",
    "AFDTransferState",
    "AFDTransferContext",
    "AFDConnectorFactory",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDF2ATransferPayload",
    "AFDForwardContextMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
    "FFNLocalExperts",
    "LocalExpertWeights",
]
