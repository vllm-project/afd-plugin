# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch vLLM-Ascend platform config normalization for AFD-owned DBO.

Upstream source: ``vllm_ascend/platform.py`` at target commit
``bd69bad88fc19e1aeeea585416d408df8bda8fef``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from afd_plugin.config import parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ASCEND_PLATFORM_PATCH_ATTR = "_afd_plugin_ascend_platform_patch_state"


@dataclass(frozen=True)
class _AFDDBOConfigSnapshot:
    enable_dbo: bool
    ubatch_size: int
    all2all_backend: str


def apply_afd_ascend_dbo_config_patch() -> bool:
    """Preserve AFD-owned DBO settings during vLLM-Ascend config normalization.

    vLLM-Ascend's platform compatibility pass disables DBO/ubatching fields and
    can rewrite ``all2all_backend`` for ordinary NPU runs. AFD owns its NPU
    ubatching path, so this patch snapshots those fields for AFD-enabled configs,
    lets upstream normalization run, then restores the AFD DBO values and
    backend. The patch is a no-op when vLLM-Ascend is not importable. Returns
    whether this process has installed the wrapper (or had already installed
    it), so callers do not cache a failed early import during plugin
    initialization.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except ImportError:
        return False

    if hasattr(NPUPlatform, _ASCEND_PLATFORM_PATCH_ATTR):
        return True

    original_check_and_update_config = NPUPlatform.check_and_update_config

    # Patch reason: vLLM-Ascend resets DBO fields in _fix_incompatible_config and
    # later rewrites all2all_backend in check_and_update_config, while AFD owns
    # the Ascend DBO/ubatching path and temporarily supplies a validation-safe
    # backend.
    # Patch functionality: preserves upstream normalization for non-AFD configs and
    # restores AFD DBO fields plus the temporary ubatching backend after upstream
    # normalization for AFD-enabled configs.
    # Expansion exception: upstream check_and_update_config is platform-owned
    # normalization; keep narrow original-function delegation so this patch only
    # owns the AFD DBO preservation.
    # Signature: matches upstream; no added parameters.
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        del cls
        # ### PATCH START: AFD DBO config preservation
        saved = _snapshot_afd_dbo_config(vllm_config)
        try:
            original_check_and_update_config(vllm_config)
        finally:
            if saved is not None:
                _restore_afd_dbo_config(vllm_config, saved)
        # ### PATCH END: AFD DBO config preservation

    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
    setattr(
        NPUPlatform,
        _ASCEND_PLATFORM_PATCH_ATTR,
        original_check_and_update_config,
    )
    return True


def _snapshot_afd_dbo_config(
    vllm_config: VllmConfig,
) -> _AFDDBOConfigSnapshot | None:
    if not _has_valid_afd_config(vllm_config):
        return None
    parallel_config = vllm_config.parallel_config
    return _AFDDBOConfigSnapshot(
        enable_dbo=parallel_config.enable_dbo,
        ubatch_size=parallel_config.ubatch_size,
        all2all_backend=parallel_config.all2all_backend,
    )


def _restore_afd_dbo_config(
    vllm_config: VllmConfig,
    saved: _AFDDBOConfigSnapshot,
) -> None:
    parallel_config = vllm_config.parallel_config
    if not saved.enable_dbo and saved.ubatch_size == 0:
        return
    parallel_config.enable_dbo = saved.enable_dbo
    parallel_config.ubatch_size = saved.ubatch_size
    parallel_config.all2all_backend = saved.all2all_backend


def _has_valid_afd_config(vllm_config: VllmConfig) -> bool:
    try:
        return parse_optional_afd_config(vllm_config, validate=True) is not None
    except (TypeError, ValueError):
        return False


__all__ = ["apply_afd_ascend_dbo_config_patch"]
