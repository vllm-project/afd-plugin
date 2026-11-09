# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU-specific AFD connector implementations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.camp2p import (
        CAMP2pAFDConnector,
        CAMP2PExtraInfo,
        CAMP2PTransferState,
        build_camp2p_topology,
    )


def __getattr__(name: str):
    if name in __all__:
        from afd_plugin.connectors.npu import camp2p

        return getattr(camp2p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAMP2pAFDConnector",
    "CAMP2PExtraInfo",
    "CAMP2PTransferState",
    "build_camp2p_topology",
]
