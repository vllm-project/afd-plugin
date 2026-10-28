# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Version-aware vLLM compatibility helpers."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Final

TARGET_VLLM_VERSION: Final[str] = "0.32.0"
NPU_TARGET_VLLM_VERSION: Final[str] = "0.30.0"


def get_target_vllm_version(device_type: str | None = None) -> str:
    """Select the pinned release without loading vLLM on CPU-only hosts."""
    if device_type is None:
        try:
            # Runtime platform discovery stays lazy for CPU-safe imports.
            from vllm.platforms import current_platform
        except ImportError:
            return TARGET_VLLM_VERSION
        device_type = current_platform.device_type
    return NPU_TARGET_VLLM_VERSION if device_type == "npu" else TARGET_VLLM_VERSION


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
    installed_version: str | None = None, *, device_type: str | None = None
) -> bool:
    if installed_version is None:
        installed_version = get_installed_vllm_version()
    if installed_version is None:
        return False

    return _parse_release(installed_version) == _parse_release(
        get_target_vllm_version(device_type)
    )


def assert_vllm_version_supported(
    *, strict: bool = True, device_type: str | None = None
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


def is_target_vllm_compatible(*, device_type: str | None = None) -> bool:
    """Return whether compatibility patches may install.

    Tolerates development builds of the selected backend's target release
    and missing vLLM so CPU-safe imports keep working.
    """
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    try:
        return is_vllm_version_supported(str(version_value), device_type=device_type)
    except ValueError:
        return False


__all__ = [
    "TARGET_VLLM_VERSION",
    "NPU_TARGET_VLLM_VERSION",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "get_target_vllm_version",
    "is_target_vllm_compatible",
    "is_vllm_version_supported",
]
