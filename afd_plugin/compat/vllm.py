# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Version-aware vLLM compatibility helpers."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Final

# The GPU backend moves to 0.34 while the Ascend backend keeps its tested
# 0.32 source pair (upgrade tracked separately in RFC #613).
TARGET_VLLM_VERSION: Final[str] = "0.34.0"
TARGET_NPU_VLLM_VERSION: Final[str] = "0.32.0"


def runtime_device_type() -> str:
    """Return the active vLLM accelerator without importing torch_npu."""
    try:
        from vllm.platforms import current_platform

        device_type = getattr(current_platform, "device_type", "") or ""
    except Exception:
        device_type = ""
    if device_type:
        return str(device_type)
    return "npu" if find_spec("torch_npu") is not None else "cuda"


def get_target_vllm_version(device_type: str | None = None) -> str:
    """Return the pinned vLLM version for the active backend."""
    if device_type is None:
        device_type = runtime_device_type()
    return (
        TARGET_NPU_VLLM_VERSION
        if device_type == "npu"
        else TARGET_VLLM_VERSION
    )


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
    "TARGET_NPU_VLLM_VERSION",
    "TARGET_VLLM_VERSION",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "get_target_vllm_version",
    "is_target_vllm_compatible",
    "is_vllm_version_supported",
    "runtime_device_type",
]
