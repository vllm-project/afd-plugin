# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared validation for models attached outside a Worker model runner."""

from __future__ import annotations


def validate_external_model_attachment(
    *,
    model_attached: bool,
    connector_initialized: bool,
    model_size_bytes: int,
) -> None:
    """Require an external model attachment before connector startup."""
    if model_attached or connector_initialized:
        raise RuntimeError("FFN model must be attached once, before connector startup")
    if model_size_bytes < 0:
        raise ValueError("model memory usage must be non-negative")
