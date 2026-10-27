# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compatibility helpers for the target vLLM runtime."""

from afd_plugin.compat.vllm import (
    NPU_TARGET_VLLM_VERSION,
    TARGET_VLLM_VERSION,
    assert_vllm_version_supported,
    get_installed_vllm_version,
    get_target_vllm_version,
    is_target_vllm_compatible,
    is_vllm_version_supported,
)

__all__ = [
    "NPU_TARGET_VLLM_VERSION",
    "TARGET_VLLM_VERSION",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "get_target_vllm_version",
    "is_target_vllm_compatible",
    "is_vllm_version_supported",
]
