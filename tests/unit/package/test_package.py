# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The plugin entry point registers lazy AFD aliases without replacing vLLM."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

import afd_plugin
from afd_plugin.compat.npu import ops


def test_models_metadata_compat_export(monkeypatch):
    from afd_plugin.model_executor import models

    context = ModuleType("afd_plugin.forward_context")
    metadata_reader = object()
    context.__dict__["get_afd_metadata_from_forward_context"] = metadata_reader
    monkeypatch.setitem(sys.modules, context.__name__, context)

    assert models.get_afd_metadata_from_forward_context is metadata_reader


@pytest.fixture(
    params=[
        pytest.param(("cuda", False), id="cuda-without-torch-npu"),
        pytest.param(("cuda", True), id="cuda-with-torch-npu"),
        pytest.param(("npu", False), id="npu-without-torch-npu"),
        pytest.param(("npu", True), id="npu-with-torch-npu"),
    ],
)
def registration(monkeypatch, request):
    device, has_torch_npu = request.param
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
        "vllm.platforms": {"current_platform": SimpleNamespace(device_type=device)},
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
    torch_npu = None
    if has_torch_npu:
        torch_npu = ModuleType("torch_npu")
        torch_npu.__spec__ = ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
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
        plugin=plugin, models=models, native=native, sent=sent, device=device
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


@pytest.mark.parametrize("registration", [("npu", True)], indirect=True)
def test_register_publishes_a5_vendor_before_runtime_setup(
    registration, monkeypatch, tmp_path
):
    library = tmp_path / "op_api/lib/libafd_cust_opapi.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(ops, "get_afd_cann_vendor_path", lambda: tmp_path)
    monkeypatch.delenv("AFD_CUST_OPAPI_LIB_PATH", raising=False)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/native/vendor")
    observed = []

    def check_version(*, strict):
        observed.append(
            (
                os.environ.get("AFD_CUST_OPAPI_LIB_PATH"),
                os.environ["ASCEND_CUSTOM_OPP_PATH"],
            )
        )

    monkeypatch.setattr(
        sys.modules["afd_plugin.compat.vllm"],
        "assert_vllm_version_supported",
        check_version,
    )
    registration.plugin.register_afd()
    assert observed == [
        (str(library), os.pathsep.join(("/native/vendor", str(tmp_path))))
    ]


def test_entry_point_is_registered():
    """The installed distribution has to expose the vLLM plugin entry point.

    Importing the package from a source tree does not register installation
    metadata, so the entry point can only be checked on an installed build.
    """
    try:
        distribution = importlib.metadata.distribution("vllm-afd-plugin")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("the plugin is not installed in this environment")
    matches = [
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "vllm.general_plugins" and entry_point.name == "afd"
    ]
    assert matches
    assert matches[0].value == "afd_plugin:register_afd"
