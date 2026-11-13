# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe checkpoint path parsing shared by AFD model wrappers."""

from __future__ import annotations


def weight_layer_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
    """Return ``(layer index, stage, remainder)`` for a decoder path."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2], tuple(parts[marker_idx + 3 :])
    return None
