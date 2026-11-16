# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared helpers for parsing AFD configuration values."""

from __future__ import annotations

from typing import SupportsInt, cast


def coerce_extra_bool(value: object, *, field_name: str) -> bool:
    """Parse a boolean option from bool, integer, or string input."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean, got {value!r}")


def coerce_extra_int(value: object, *, field_name: str) -> int:
    """Parse an integer option without truncating floats."""

    if isinstance(value, bool | float):
        raise TypeError(f"{field_name} must be an integer, got {value!r}")
    try:
        # int() validates the raw value; normalize conversion failures below.
        return int(cast(str | SupportsInt, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{field_name} must be an integer, got {value!r}",
        ) from exc


def coerce_extra_positive_int(value: object, *, field_name: str) -> int:
    """Parse a strictly positive integer connector option."""

    coerced = coerce_extra_int(value, field_name=field_name)
    if coerced <= 0:
        raise ValueError(f"{field_name} must be positive, got {coerced}")
    return coerced


def coerce_optional_extra_positive_int(
    value: object,
    *,
    field_name: str,
) -> int | None:
    """Parse an optional strictly positive integer connector option."""

    if value in (None, ""):
        return None
    return coerce_extra_positive_int(value, field_name=field_name)


__all__ = [
    "coerce_extra_bool",
    "coerce_extra_int",
    "coerce_extra_positive_int",
    "coerce_optional_extra_positive_int",
]
