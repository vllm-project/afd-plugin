# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async Mega communication and fused expert backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .connector import (
        CAMAsyncMegaAFDConnector,
        MegaExtraInfo,
        MegaLane,
        validate_mega_config,
    )

__all__ = [
    "CAMAsyncMegaAFDConnector",
    "MegaExtraInfo",
    "MegaLane",
    "validate_mega_config",
]


def __getattr__(name: str) -> Any:
    # Layout/weight helpers must remain usable without the Ascend runtime.
    if name in __all__:
        from . import connector

        return getattr(connector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
