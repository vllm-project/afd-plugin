# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public Ascend runtime compatibility facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.forward_context import (
    ascend_forward_context,
)
from afd_plugin.compat.npu.runtime_config import (
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_PATCHES_APPLIED = False


def apply_afd_ascend_config_patch_if_needed() -> None:
    """Apply patches required while vLLM builds an AFD NPU config."""

    from afd_plugin.compat.patches.npu.ascend_platform import (
        apply_afd_ascend_dbo_config_patch,
    )

    if not apply_afd_ascend_dbo_config_patch():
        raise RuntimeError(
            "AFD NPU DBO config patch requires vLLM-Ascend NPUPlatform",
        )
    from afd_plugin.compat.patches.npu.ascend_config import (
        apply_afd_ascend_config_patch,
    )

    apply_afd_ascend_config_patch()


def apply_afd_async_dp_engine_patch_if_needed(vllm_config: VllmConfig) -> bool:
    """Finalize the async Attention engine entry point after Ascend patches."""

    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    if (
        afd_config is None
        or afd_config.role != "attention"
        or not is_afd_async_dp(vllm_config)
    ):
        return False

    from afd_plugin.compat.patches.async_dp_engine import (
        apply_async_dp_engine_patch,
    )

    if not apply_async_dp_engine_patch():
        raise RuntimeError(
            "AFD async-DP engine patch requires the pinned vLLM version",
        )
    return True


def apply_afd_ascend_patches_if_needed() -> None:
    """Apply plugin-owned runtime patches after Ascend initialization."""

    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from afd_plugin.compat.patches.npu.mla_graph import (
        apply_afd_mla_graph_patch,
    )

    apply_afd_ascend_config_patch_if_needed()
    if not apply_afd_mla_graph_patch():
        raise RuntimeError(
            "AFD NPU MLA graph patch requires the vLLM-Ascend MLA resolver",
        )
    _PATCHES_APPLIED = True


__all__ = [
    "apply_afd_async_dp_engine_patch_if_needed",
    "apply_afd_ascend_config_patch_if_needed",
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "npu_afd_num_ubatches",
]
