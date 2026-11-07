# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The plugin entry point registers lazy AFD aliases without replacing vLLM."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

import afd_plugin


@pytest.fixture(params=["cuda", "npu"])
def registration(monkeypatch, request):
    native = {
        name: object()
        for name in (
            "DeepseekV2ForCausalLM",
            "DeepseekV4ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        )
    }
    models = dict(native)
    sent = []

    def register_model(name, path):
        sent.append((name, path))
        models[name] = path

    modules: dict[
        str, dict[str, ModuleSpec | list[str] | SimpleNamespace | Callable[..., None]]
    ] = {
        "vllm": {"__spec__": ModuleSpec("vllm", loader=None), "__path__": []},
        "vllm.platforms": {
            "current_platform": SimpleNamespace(device_type=request.param)
        },
        "vllm.model_executor": {"__path__": []},
        "vllm.model_executor.models": {
            "ModelRegistry": SimpleNamespace(register_model=register_model)
        },
        "afd_plugin.compat.vllm": {"assert_vllm_version_supported": lambda **kw: None},
        "afd_plugin.model_executor.routing_simulator": {
            "register_afd_balanced_routing_strategy": lambda: None
        },
        "afd_plugin.v1.worker.dbo": {"register_dbo_yield_custom_op": lambda: None},
        "afd_plugin.compat.patches.npu.ascend_config": {
            "apply_afd_ascend_config_patch": lambda: None
        },
    }
    for name in (
        "async_dp_engine",
        "async_dp_forward_context",
        "config_validation",
        "engine_core",
    ):
        modules[f"afd_plugin.compat.patches.{name}"] = {}
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kw: (
            (ModuleSpec(name, loader=None) if request.param == "npu" else None)
            if name == "torch_npu"
            else find_spec(name, *args, **kw)
        ),
    )
    monkeypatch.delenv("AFD_FORCE_SPAWN_MULTIPROCESSING", raising=False)
    # A fresh module isolates registration state from test order without
    # asserting or changing the real package's private registration flag.
    spec = importlib.util.spec_from_file_location(
        "afd_registration_test", afd_plugin.__file__
    )
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return SimpleNamespace(
        plugin=plugin, models=models, native=native, sent=sent, device=request.param
    )


def test_register_afd_preserves_native_models_and_is_idempotent(registration):
    deepseek = "afd_plugin.model_executor.models.deepseek_v2:"
    expected = {
        "AFDDeepseekForCausalLM": deepseek + "AFDDeepseekForCausalLM",
        "AFDDeepseekV2ForCausalLM": deepseek + "AFDDeepseekV2ForCausalLM",
        "AFDDeepseekV3ForCausalLM": deepseek + "AFDDeepseekV3ForCausalLM",
        "AFDDeepseekV32ForCausalLM": deepseek + "AFDDeepseekV3ForCausalLM",
        "AFDGlmMoeDsaForCausalLM": deepseek + "AFDGlmMoeDsaForCausalLM",
        "AFDDeepseekV4ForCausalLM": (
            "afd_plugin.model_executor.models.npu.deepseek_v4:AFDDeepseekV4ForCausalLM"
            if registration.device == "npu"
            else "afd_plugin.model_executor.models.deepseek_v4:AFDDeepseekV4ForCausalLM"
        ),
        "AFDDSparkDraftModel": (
            "afd_plugin.model_executor.models.npu.deepseek_v4_dspark:"
            "AFDDSparkDeepseekV4ForCausalLM"
        ),
        "AFDQwen3MoeForCausalLM": (
            "afd_plugin.model_executor.models.qwen3_moe:AFDQwen3MoeForCausalLM"
        ),
        "AFDQwen3MoeRoutedExperts": (
            "afd_plugin.model_executor.models.qwen3_moe:AFDQwen3MoeRoutedExperts"
        ),
        "AFDQwen3_5MoeForConditionalGeneration": (
            "afd_plugin.model_executor.models.qwen3_5:"
            "AFDQwen3_5MoeForConditionalGeneration"
        ),
    }
    registration.plugin.register_afd()
    registration.plugin.register_afd()

    assert registration.models == {**registration.native, **expected}
    # ModelRegistry is the external registration seam; compare requests without
    # requiring the implementation's iteration order.
    assert sorted(registration.sent) == sorted(expected.items())


def test_entry_point_is_registered():
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [ep for ep in entry_points if ep.name == "afd"]
    assert matches
    assert matches[0].value == "afd_plugin:register_afd"
