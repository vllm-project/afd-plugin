# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDControlPlaneClosedError,
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
    "AFDControlPlane",
    "ConnectorExtraInfo",
    "AFDExpertRoutingSpec",
    "AFDTransferState",
    "AFDTransferContext",
    "AFDConnectorFactory",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDControlPlaneClosedError",
    "AFDF2ATransferPayload",
    "AFDForwardContextMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
]
