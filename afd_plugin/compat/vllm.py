# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Version-aware vLLM compatibility helpers."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Final

TARGET_VLLM_VERSION: Final[str] = "0.34.0"
TARGET_NPU_VLLM_VERSION: Final[str] = TARGET_VLLM_VERSION
SUPPORTED_VLLM_VERSIONS: Final[tuple[str, ...]] = (TARGET_VLLM_VERSION,)


def get_target_vllm_version(device_type: str | None = None) -> str:
    """Return the vLLM pin shared by CUDA and Ascend."""
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


def is_vllm_version_supported(
    installed_version: str | None = None,
    *,
    device_type: str | None = None,
) -> bool:
    if installed_version is None:
        installed_version = get_installed_vllm_version()
    if installed_version is None:
        return False

    return _parse_release(installed_version) == _parse_release(
        get_target_vllm_version(device_type)
    )


def assert_vllm_version_supported(
    *,
    strict: bool = True,
    device_type: str | None = None,
) -> None:
    installed_version = get_installed_vllm_version()
    if is_vllm_version_supported(installed_version, device_type=device_type):
        return

    message = (
        "AFD plugin currently supports exactly vLLM "
        f"{get_target_vllm_version(device_type)}; installed vLLM version is "
        f"{installed_version or 'not installed'}"
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def is_target_vllm_compatible(device_type: str | None = None) -> bool:
    """Return whether compatibility patches may install.

    Both backends accept development builds of the shared pin. A missing
    vLLM version is permitted for CPU-safe imports of shared patches, when
    ``device_type`` is omitted.
    """
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return device_type is None
    try:
        installed = _parse_release(str(version_value))
    except ValueError:
        return False
    return installed == _parse_release(get_target_vllm_version(device_type))


__all__ = [
    "SUPPORTED_VLLM_VERSIONS",
    "TARGET_NPU_VLLM_VERSION",
    "TARGET_VLLM_VERSION",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "get_target_vllm_version",
    "is_target_vllm_compatible",
    "is_vllm_version_supported",
]
