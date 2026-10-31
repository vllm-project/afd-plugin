# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compatibility helpers for the target vLLM runtime."""

from afd_plugin.compat.vllm import (
    SUPPORTED_VLLM_VERSIONS,
    TARGET_VLLM_VERSION,
    assert_vllm_version_supported,
    get_installed_vllm_version,
    is_vllm_version_supported,
)

__all__ = [
    "TARGET_VLLM_VERSION",
    "SUPPORTED_VLLM_VERSIONS",
    "assert_vllm_version_supported",
    "get_installed_vllm_version",
    "is_vllm_version_supported",
]
