# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Version-aware vLLM compatibility helpers."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Final

TARGET_VLLM_VERSION: Final[str] = "0.32.0"


def get_target_vllm_version() -> str:
    """Return the common vLLM pin for both backends."""
    return TARGET_VLLM_VERSION


def _parse_release(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"cannot parse vLLM version {value!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def get_installed_vllm_version() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def is_vllm_version_supported(installed_version: str | None = None) -> bool:
    if installed_version is None:
        installed_version = get_installed_vllm_version()
    if installed_version is None:
        return False

    return _parse_release(installed_version) == _parse_release(
        get_target_vllm_version()
    )


def assert_vllm_version_supported(*, strict: bool = True) -> None:
    installed_version = get_installed_vllm_version()
    if is_vllm_version_supported(installed_version):
        return

    message = (
        "AFD plugin currently supports exactly vLLM "
        f"{get_target_vllm_version()}; installed vLLM version is "
        f"{installed_version or 'not installed'}"
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def is_target_vllm_compatible() -> bool:
    """Return whether compatibility patches may install.

    Tolerates development builds of the target release and missing vLLM so
    CPU-safe imports keep working.
    """
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    try:
        return is_vllm_version_supported(str(version_value))
    except ValueError:
        return False


__all__ = [
    "TARGET_VLLM_VERSION",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "get_target_vllm_version",
    "is_target_vllm_compatible",
    "is_vllm_version_supported",
]
