# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared utilities for standalone FFN model owners."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import nn


class ParameterStorageOwner(Protocol):
    """A module-like owner whose distinct parameter storages are counted."""

    def parameters(self) -> Iterator[nn.Parameter]: ...


def parameter_storage_bytes(model: ParameterStorageOwner) -> int:
    """Bytes of the distinct storages behind ``model``'s parameters.

    Views that share storage (for example the connector's expert views) are
    counted once.
    """
    storage_sizes: dict[tuple[str, int], int] = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storage_sizes[(str(parameter.device), storage.data_ptr())] = storage.nbytes()
    return sum(storage_sizes.values())
