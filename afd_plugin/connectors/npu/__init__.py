# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU-specific AFD connector implementations."""

from afd_plugin.connectors.npu.camp2p import (
    CAMP2pAFDConnector,
    CAMP2PExtraInfo,
    CAMP2PTransferState,
    build_camp2p_topology,
)
from afd_plugin.connectors.npu.p2p_hccl import (
    HCCLMTPHeader,
    HCCLP2PTransferState,
    P2pHcclAFDConnector,
    P2pHcclAFDControlPlane,
)

__all__ = [
    "CAMP2pAFDConnector",
    "CAMP2PExtraInfo",
    "CAMP2PTransferState",
    "build_camp2p_topology",
    "HCCLMTPHeader",
    "HCCLP2PTransferState",
    "P2pHcclAFDConnector",
    "P2pHcclAFDControlPlane",
]
