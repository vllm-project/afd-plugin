# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config normalization shim for AFD-owned runtime behavior.

vLLM 0.30.0 validates native microbatching by requiring a supported all2all
backend. AFD ubatching uses plugin connectors instead, so this patch only
relaxes that assertion for configs with active ``additional_config["afd"]``.
It also replaces the platform's default worker with the role-specific AFD
worker when ``worker_cls`` was left as ``"auto"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import vllm.config.vllm as config_module
import vllm.engine.arg_utils as arg_utils_module

from afd_plugin.compat.vllm import (
    is_target_vllm_compatible as _is_target_vllm_compatible,
)
from afd_plugin.config import parse_optional_afd_config
from afd_plugin.validation import afd_worker_qualname_for_platform_default

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

_ORIGINAL_CREATE_ENGINE_CONFIG_ATTR = "_afd_plugin_original_create_engine_config"
_ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR = "_afd_plugin_original_vllm_config_post_init"
_AFD_TEMP_BACKEND = "deepep_low_latency"
_original_create_engine_config: Callable[..., Any] | None = None
_original_vllm_config_post_init: Callable[..., Any] | None = None


# Patch reason: vLLM validates native ubatching against DeepEP backends and
# selects a generic worker before AFD can bind its NPU child lifecycle.
# Patch functionality: install the Ascend config shim before validation, use a
# temporary backend only for AFD GPU validation, select the role worker when
# worker_cls is auto, finalize the NPU backend, and bind Ascend async-DP and
# EngineCore hooks after the full config exists.
# Removal plan: remove when vLLM supports connector-owned ubatching and
# role-aware worker and child-process setup.
# Expansion exception: upstream create_engine_config is a large config builder;
# keep original-function delegation scoped to AFD config lifecycle changes.
# Signature: matches upstream; no added parameters.
def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """Create the VllmConfig."""

    assert _original_create_engine_config is not None
    # ### PATCH START: AFD config preflight
    worker_cls_was_auto = _uses_auto_worker_value(self.worker_cls)
    is_afd_npu = _apply_afd_npu_config_patches(self)
    needs_engine_args_bypass = not is_afd_npu and _should_relax_engine_args_backend(
        self
    )
    # ### PATCH END: AFD config preflight
    if not needs_engine_args_bypass:
        config = _original_create_engine_config(
            self,
            usage_context,
            headless,
        )
    else:
        # ### PATCH START: AFD ubatching all2all backend validation
        # vLLM validates native ubatching against DeepEP backends. AFD ubatching
        # uses plugin connectors, so temporarily present a supported backend while
        # upstream builds and validates VllmConfig on GPU. Ascend scopes this
        # bypass inside __post_init__ to retain its native SP derivation.
        original_backend = self.all2all_backend
        self.all2all_backend = _AFD_TEMP_BACKEND
        try:
            config = _original_create_engine_config(
                self,
                usage_context,
                headless,
            )
        finally:
            self.all2all_backend = original_backend
        config.parallel_config.all2all_backend = original_backend
        # ### PATCH END: AFD ubatching all2all backend validation

    # ### PATCH START: AFD post-config setup
    if worker_cls_was_auto:
        _select_afd_worker_for_auto(config)
    # Preserve the actual backend before serializing the child config.
    if is_afd_npu:
        from afd_plugin.compat.npu import fix_all2all_backend_for_afd

        fix_all2all_backend_for_afd(config)
    # Ascend platform initialization wraps EngineCoreProc.run_engine_core after
    # general plugins load. Finalize AFD Attention scheduling and the early
    # child config binding before vLLM captures the subprocess target.
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        from afd_plugin.compat.npu import (
            apply_afd_ascend_engine_core_config_patch_if_needed,
            apply_afd_async_dp_engine_patch_if_needed,
        )

        apply_afd_async_dp_engine_patch_if_needed(config)
        apply_afd_ascend_engine_core_config_patch_if_needed(config)
    # ### PATCH END: AFD post-config setup
    return config


# Patch reason: EngineCore handshakes explicitly rerun VllmConfig.__post_init__
# after the config's actual AFD all2all backend has been restored.
# Patch functionality: temporarily presents a validation-safe backend during
# explicit AFD ubatching revalidation, then restores the actual backend.
# NPU platform normalization sees the real backend, including cached Ascend
# config revalidation, so SP graph filtering cannot observe temporary DeepEP.
# Install Ascend patches before fresh child-process validation, even in eager mode.
# Expansion exception: upstream VllmConfig.__post_init__ is a large validation
# pipeline; keep narrow original-function delegation so this patch only owns
# the AFD backend validation bypass.
# Removal plan: remove when vLLM accepts connector-owned ubatching.
# Signature: matches upstream; no added parameters.
def __post_init__(self):
    """Verify configs are valid & consistent with each other."""

    assert _original_vllm_config_post_init is not None
    # ### PATCH START: AFD child-process Ascend config ordering
    is_afd_npu = _apply_afd_npu_config_patches(self)
    # ### PATCH END: AFD child-process Ascend config ordering
    if not _should_relax_vllm_config_backend(self):
        return _original_vllm_config_post_init(self)

    # ### PATCH START: AFD repeated ubatching backend validation
    parallel_config = self.parallel_config
    original_backend = parallel_config.all2all_backend
    if is_afd_npu:
        from afd_plugin.compat.patches.npu.ascend_platform import (
            AFDAll2AllValidation,
        )

        self._afd_all2all_validation = AFDAll2AllValidation(original_backend)
    parallel_config.all2all_backend = _AFD_TEMP_BACKEND
    try:
        result = _original_vllm_config_post_init(self)
        if is_afd_npu:
            original_backend = self._afd_all2all_validation.backend
    finally:
        parallel_config.all2all_backend = original_backend
        if is_afd_npu:
            del self._afd_all2all_validation
    # ### PATCH END: AFD repeated ubatching backend validation
    return result


def _apply_afd_npu_config_patches(config: EngineArgs | VllmConfig) -> bool:
    if parse_optional_afd_config(config.additional_config) is None:
        return False

    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        return False

    from afd_plugin.compat.npu import apply_afd_ascend_config_patch_if_needed

    apply_afd_ascend_config_patch_if_needed()
    return True


def _uses_auto_worker_value(worker_cls: str | type[Any]) -> bool:
    return isinstance(worker_cls, str) and worker_cls.strip() == "auto"


def _select_afd_worker_for_auto(vllm_config: VllmConfig) -> None:
    afd_config = parse_optional_afd_config(vllm_config)
    if afd_config is None:
        return

    from vllm.platforms import current_platform

    platform_worker_qualname = vllm_config.parallel_config.worker_cls
    if not isinstance(platform_worker_qualname, str):
        raise ValueError(
            "platform worker_cls must be a qualname string before AFD automatic "
            f"selection, got {type(platform_worker_qualname).__name__}",
        )
    vllm_config.parallel_config.worker_cls = afd_worker_qualname_for_platform_default(
        afd_config.role,
        platform_worker_qualname,
        is_cuda=current_platform.is_cuda(),
        device_type=current_platform.device_type,
    )


def _should_relax_engine_args_backend(engine_args: EngineArgs) -> bool:
    if not _is_target_vllm_compatible():
        return False
    afd_config = parse_optional_afd_config(engine_args.additional_config)
    if afd_config is None:
        return False
    if not engine_args.enable_dbo and engine_args.ubatch_size <= 1:
        return False

    backend = engine_args.all2all_backend
    return backend not in {
        "deepep_low_latency",
        "deepep_high_throughput",
        "nixl_ep",
    }


def _should_relax_vllm_config_backend(vllm_config: VllmConfig) -> bool:
    if not _is_target_vllm_compatible():
        return False
    if parse_optional_afd_config(vllm_config) is None:
        return False

    parallel_config = vllm_config.parallel_config
    if not parallel_config.use_ubatching:
        return False

    backend = parallel_config.all2all_backend
    return backend not in {
        "deepep_low_latency",
        "deepep_high_throughput",
        "nixl_ep",
    }


if _is_target_vllm_compatible():
    if not hasattr(arg_utils_module, _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR):
        setattr(
            arg_utils_module,
            _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR,
            arg_utils_module.EngineArgs.create_engine_config,
        )

    if not hasattr(config_module, _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR):
        setattr(
            config_module,
            _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR,
            config_module.VllmConfig.__post_init__,
        )

    _original_create_engine_config = getattr(
        arg_utils_module,
        _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR,
    )
    _original_vllm_config_post_init = getattr(
        config_module,
        _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR,
    )

    arg_utils_module.EngineArgs.create_engine_config = create_engine_config
    config_module.VllmConfig.__post_init__ = __post_init__
    arg_utils_module.logger.debug("AFD config validation patch applied")


__all__: list[str] = []
