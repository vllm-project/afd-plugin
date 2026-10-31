# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config normalization shim for AFD-owned runtime behavior.

vLLM 0.26.0 validates native microbatching by requiring a supported all2all
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

from afd_plugin.compat.vllm import is_vllm_version_supported
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


# Patch reason: vLLM validates native ubatching by requiring a DeepEP all2all
# backend, while AFD ubatching is implemented by plugin connectors.
# Patch functionality: temporarily uses a supported backend only during
# upstream EngineArgs-to-VllmConfig validation for AFD configs.
# Expansion exception: upstream create_engine_config is a large config builder;
# keep a narrow original-function delegation so this patch only owns the AFD
# validation bypass.
# Signature: matches upstream; no added parameters.
def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """Create the VllmConfig."""

    assert _original_create_engine_config is not None
    # ### PATCH START: AFD automatic worker selection
    worker_cls_was_auto = _uses_auto_worker_value(self.worker_cls)
    # ### PATCH END: AFD automatic worker selection
    # ### PATCH START: AFD Ascend config patch ordering
    if parse_optional_afd_config(self.additional_config) is not None:
        from vllm.platforms import current_platform

        if current_platform.device_type == "npu":
            from afd_plugin.compat.npu import (
                apply_afd_ascend_config_patch_if_needed,
            )

            apply_afd_ascend_config_patch_if_needed()
    # ### PATCH END: AFD Ascend config patch ordering
    if not _should_relax_engine_args_backend(self):
        config = _original_create_engine_config(
            self,
            usage_context,
            headless,
        )
    else:
        # ### PATCH START: AFD ubatching all2all backend validation
        # vLLM validates native ubatching against DeepEP backends. AFD ubatching
        # uses plugin connectors, so temporarily present a supported backend while
        # upstream builds and validates VllmConfig. The Ascend platform wrapper
        # preserves this temporary value across its default-worker normalization.
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

    # ### PATCH START: AFD automatic worker selection
    if worker_cls_was_auto:
        _select_afd_worker_for_auto(config)
    # ### PATCH END: AFD automatic worker selection
    return config


# Patch reason: EngineCore handshakes explicitly rerun VllmConfig.__post_init__
# after the config's actual AFD all2all backend has been restored.
# Patch functionality: temporarily presents a validation-safe backend during
# explicit AFD ubatching revalidation, then restores the actual backend.
# Expansion exception: upstream VllmConfig.__post_init__ is a large validation
# pipeline; keep narrow original-function delegation so this patch only owns
# the AFD backend validation bypass.
# Signature: matches upstream; no added parameters.
def __post_init__(self):
    """Verify configs are valid & consistent with each other."""

    assert _original_vllm_config_post_init is not None
    if not _should_relax_vllm_config_backend(self):
        return _original_vllm_config_post_init(self)

    # ### PATCH START: AFD repeated ubatching backend validation
    parallel_config = self.parallel_config
    original_backend = parallel_config.all2all_backend
    parallel_config.all2all_backend = _AFD_TEMP_BACKEND
    try:
        result = _original_vllm_config_post_init(self)
    finally:
        parallel_config.all2all_backend = original_backend
    # ### PATCH END: AFD repeated ubatching backend validation
    return result


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


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return is_vllm_version_supported(version_text)


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
