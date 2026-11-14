# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU-specific AFD connector implementations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.connectors.gpu.async_gpu import GpuAsyncAFDConnector
    from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector


def __getattr__(name: str):
    if name == "P2pNcclAFDConnector":
        module = "p2p"
    elif name == "GpuAsyncAFDConnector":
        module = "async_gpu"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


__all__ = ["P2pNcclAFDConnector", "GpuAsyncAFDConnector"]
