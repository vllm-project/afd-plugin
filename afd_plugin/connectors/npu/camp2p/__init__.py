# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous CAMP2P connector package.

The connector implementation lives in :mod:`~.connector` and the Attention-side routing
payload in :mod:`~.external_routing`. Names are resolved lazily so importing the payload
does not pull in the connector's vLLM and Ascend dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def __getattr__(name: str) -> Any:
    if name in __all__:
        from afd_plugin.connectors.npu.camp2p import connector

        return getattr(connector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
