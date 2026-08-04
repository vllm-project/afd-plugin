# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vLLM AFD plugin: Attention-FFN Disaggregation support."""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from afd_plugin.config import AFDConfig, parse_afd_config, parse_optional_afd_config


def __getattr__(name: str):
    if name in {
        "AFDAttentionModelRunner",
        "AFDAttentionWorker",
        "AFDFFNWorker",
        "AFDUBatchWrapper",
        "GPUFFNModelRunner",
    }:
        from afd_plugin.v1 import worker

        return getattr(worker, name)
    if name == "assert_compatible_afd_stack":
        from afd_plugin.validation import assert_compatible_afd_stack

        return assert_compatible_afd_stack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = version("vllm-afd-plugin")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root=str(Path(__file__).resolve().parents[1]))
    except (ImportError, LookupError):
        __version__ = "0.0.0+unknown"


_logger = logging.getLogger(__name__)
_registered = False

_DEEPSEEK_MODEL_REGISTRATIONS = {
    "DeepseekForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekForCausalLM"
    ),
    "DeepseekV2ForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV2ForCausalLM"
    ),
    "DeepseekV3ForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    ),
    "DeepseekV32ForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    ),
    "DeepseekV4ForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v4:AFDDeepseekV4ForCausalLM"
    ),
    "GlmMoeDsaForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDGlmMoeDsaForCausalLM"
    ),
}


def register_afd() -> None:
    """Entry point for ``vllm.general_plugins``.

    Perform plugin runtime registration when vLLM is available. Importing this
    package or calling this function remains safe without vLLM installed, which
    keeps local CPU smoke tests useful on non-CUDA machines.
    """

    global _registered
    if _registered:
        _logger.debug("AFD plugin: register_afd() already completed")
        return

    _logger.debug("AFD plugin: register_afd() called")
    if importlib.util.find_spec("vllm") is None:
        _logger.debug("AFD plugin: vLLM not found, skipping runtime registration")
        _registered = True
        return

    try:
        from afd_plugin.compat.vllm import assert_vllm_version_supported

        assert_vllm_version_supported(strict=False)
    except Exception:
        _logger.debug(
            "AFD plugin: vLLM version check could not be completed",
            exc_info=True,
        )

    try:
        import afd_plugin.compat.patches.async_dp_engine  # noqa: F401
        import afd_plugin.compat.patches.async_dp_forward_context  # noqa: F401
        import afd_plugin.compat.patches.config_validation  # noqa: F401
        import afd_plugin.compat.patches.engine_core  # noqa: F401
    except Exception:
        _logger.debug(
            "AFD plugin: compatibility patches could not be applied",
            exc_info=True,
        )

    try:
        from afd_plugin.v1.worker.dbo import register_dbo_yield_custom_op

        register_dbo_yield_custom_op()
    except Exception:
        _logger.debug(
            "AFD plugin: DBO yield custom op could not be registered",
            exc_info=True,
        )

    # NPU compatibility patches are applied during AFD config construction and
    # worker startup, after vLLM-Ascend completes its platform initialization.

    from vllm.model_executor.models import ModelRegistry

    for model_arch, model_cls in _DEEPSEEK_MODEL_REGISTRATIONS.items():
        ModelRegistry.register_model(f"AFD{model_arch}", model_cls)

    _registered = True


__all__ = [
    "AFDConfig",
    "AFDAttentionModelRunner",
    "AFDAttentionWorker",
    "AFDFFNWorker",
    "GPUFFNModelRunner",
    "assert_compatible_afd_stack",
    "parse_afd_config",
    "parse_optional_afd_config",
    "__version__",
    "_DEEPSEEK_MODEL_REGISTRATIONS",
    "register_afd",
]
