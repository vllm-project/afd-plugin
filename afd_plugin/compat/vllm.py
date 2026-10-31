# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Version-aware vLLM compatibility helpers."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Final

TARGET_VLLM_VERSION: Final[str] = "0.26.0"
SUPPORTED_VLLM_VERSIONS: Final[tuple[str, ...]] = ("0.23.0", TARGET_VLLM_VERSION)


def _parse_release(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"cannot parse vLLM version {value!r}")
    return tuple(int(part) for part in match.groups())


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

    installed_release = _parse_release(installed_version)
    return any(
        installed_release == _parse_release(supported_version)
        for supported_version in SUPPORTED_VLLM_VERSIONS
    )


def assert_vllm_version_supported(*, strict: bool = True) -> None:
    installed_version = get_installed_vllm_version()
    if is_vllm_version_supported(installed_version):
        return

    message = (
        "AFD plugin currently supports exactly vLLM versions "
        f"{', '.join(SUPPORTED_VLLM_VERSIONS)}; installed vLLM version is "
        f"{installed_version or 'not installed'}"
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


__all__ = [
    "TARGET_VLLM_VERSION",
    "SUPPORTED_VLLM_VERSIONS",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "is_vllm_version_supported",
]
