# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vLLM AFD plugin: Attention-FFN Disaggregation support."""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

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


def _force_spawn_multiprocessing_if_requested() -> None:
    """Pin Python's default context for the A3 Python 3.12 runtime.

    ``VLLM_WORKER_MULTIPROC_METHOD`` controls vLLM's explicit context only.
    Some NPU runtime helpers use Python's default context and otherwise retain
    the task's inherited ``forkserver`` setting, which cannot restore the
    signal-handler state in this environment.  Keep the override opt-in so it
    remains isolated to the affected CAM async recipe.
    """
    if os.environ.get("AFD_FORCE_SPAWN_MULTIPROCESSING") != "1":
        return
    multiprocessing.set_start_method("spawn", force=True)

    # A few optional runtime helpers request ``forkserver`` explicitly instead
    # of consulting the default start method.  On the A3 Python 3.12 image the
    # forkserver cannot restore its inherited signal-handler state, so every
    # child it starts exits before running user code.  Keep this narrowly
    # opt-in with the recipe environment variable above: callers requesting a
    # forkserver receive the equivalent spawn context instead.
    contexts = cast(Any, multiprocessing.context)._concrete_contexts
    contexts["forkserver"] = contexts["spawn"]

    # TE Fusion keeps a module reference to ``multiprocessing`` and calls its
    # public ``get_context("forkserver")`` API directly inside each model
    # worker.  Replacing the registry alone is not sufficient for all Python
    # 3.12 context instances, so redirect that explicit request as well.
    multiprocessing_module = cast(Any, multiprocessing)
    if not getattr(multiprocessing_module, "_afd_spawn_context_redirect", False):
        original_get_context = multiprocessing.get_context

        def get_spawn_context(method: str | None = None):
            if method == "forkserver":
                method = "spawn"
            return original_get_context(method)

        multiprocessing.get_context = get_spawn_context
        multiprocessing_module._afd_spawn_context_redirect = True


# vLLM model workers import the configured worker class directly in spawned
# child interpreters; they do not invoke the general-plugin entry point below.
# Apply the opt-in setting at package import time so it is also in effect before
# Ascend TE Fusion initializes its compilation workers.
_force_spawn_multiprocessing_if_requested()

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
        "afd_plugin.model_executor.models.npu.deepseek_v4:AFDDeepseekV4ForCausalLM"
        if importlib.util.find_spec("torch_npu") is not None
        else "afd_plugin.model_executor.models.deepseek_v4:AFDDeepseekV4ForCausalLM"
    ),
    "GlmMoeDsaForCausalLM": (
        "afd_plugin.model_executor.models.deepseek_v2:AFDGlmMoeDsaForCausalLM"
    ),
}

_QWEN_MODEL_REGISTRATIONS = {
    "Qwen3MoeForCausalLM": (
        "afd_plugin.model_executor.models.qwen3_moe:AFDQwen3MoeForCausalLM"
    ),
}

_QWEN3_5_MODEL_REGISTRATIONS = MappingProxyType(
    {
        "Qwen3_5MoeForConditionalGeneration": (
            "afd_plugin.model_executor.models.qwen3_5:AFDQwen3_5MoeForConditionalGeneration"
        ),
    }
)

_MODEL_REGISTRATIONS = MappingProxyType(
    {
        **_DEEPSEEK_MODEL_REGISTRATIONS,
        **_QWEN_MODEL_REGISTRATIONS,
        **_QWEN3_5_MODEL_REGISTRATIONS,
    }
)


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

    _force_spawn_multiprocessing_if_requested()
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

    # One import per patch, each isolated. A single try block around the whole
    # list means the first failure silently skips every patch after it: a stale
    # module name here once disabled the ubatch positions and split patches,
    # which surfaced two layers away as "positions is required for C128A
    # metadata build" inside DeepSeek-V4's kernel warmup. Warn rather than
    # debug for the same reason -- a patch that did not load is not a detail.
    for _patch in (
        "async_dp_engine",
        "async_dp_forward_context",
        "config_validation",
        "dp_coordinator_timeout",
        "engine_core",
        "ffn_local_moe_prepare",
        "ubatch_positions",
        "ubatch_split",
    ):
        try:
            import_module(f"afd_plugin.compat.patches.{_patch}")
        except Exception:
            _logger.warning(
                "AFD plugin: compatibility patch %r could not be applied",
                _patch,
                exc_info=True,
            )

    from afd_plugin.model_executor.routing_simulator import (
        register_afd_balanced_routing_strategy,
    )

    register_afd_balanced_routing_strategy()

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

    for model_arch, model_cls in _MODEL_REGISTRATIONS.items():
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
    "_MODEL_REGISTRATIONS",
    "_QWEN_MODEL_REGISTRATIONS",
    "_QWEN3_5_MODEL_REGISTRATIONS",
    "register_afd",
]
