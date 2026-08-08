# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDConnectorExtension,
    AFDControlPlane,
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
)

__all__ = [
    "AFDConnectorBase",
    "AFDConnectorExtension",
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
]
