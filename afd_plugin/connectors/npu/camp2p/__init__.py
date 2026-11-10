# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous CAMP2P connector package.

The connector implementation lives in :mod:`~.connector` and the Attention-side routing
payload in :mod:`~.external_routing`. This module only re-exports the public surface so
callers keep importing from ``afd_plugin.connectors.npu.camp2p``.
"""

from __future__ import annotations

from afd_plugin.connectors.npu.camp2p.connector import (
    CAMP2pAFDConnector,
    CAMP2PAFDConnectorData,
    CAMP2pAFDControlPlane,
    CAMP2PExtraInfo,
    CAMP2PTransferState,
    build_camp2p_topology,
    prepare_token_id_transfer,
    received_token_ids,
)

__all__ = [
    "CAMP2pAFDConnector",
    "CAMP2pAFDControlPlane",
    "CAMP2PAFDConnectorData",
    "CAMP2PExtraInfo",
    "CAMP2PTransferState",
    "build_camp2p_topology",
    "prepare_token_id_transfer",
    "received_token_ids",
]
