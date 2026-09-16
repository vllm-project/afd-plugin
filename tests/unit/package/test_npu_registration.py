# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import builtins
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

import afd_plugin


@pytest.fixture
def registration_env(monkeypatch):
    events = []
    monkeypatch.setattr(afd_plugin, "_registered", False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace())
    for name in (
        "afd_plugin.compat.patches.async_dp_engine",
        "afd_plugin.compat.patches.async_dp_forward_context",
        "afd_plugin.compat.patches.config_validation",
        "afd_plugin.compat.patches.engine_core",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    routing = ModuleType("afd_plugin.model_executor.routing_simulator")
    routing.register_afd_balanced_routing_strategy = lambda: events.append("routing")
    monkeypatch.setitem(sys.modules, routing.__name__, routing)
    dbo = ModuleType("afd_plugin.v1.worker.dbo")
    dbo.register_dbo_yield_custom_op = lambda: events.append("dbo")
    monkeypatch.setitem(sys.modules, dbo.__name__, dbo)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="npu")
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    models = ModuleType("vllm.model_executor.models")
    models.ModelRegistry = SimpleNamespace(
        register_model=lambda name, value: events.append(("model", name)),
    )
    monkeypatch.setitem(sys.modules, models.__name__, models)
    return platforms, events


def test_npu_plugin_registration_precedes_direct_scheduler_config(
    monkeypatch, registration_env
):
    _, events = registration_env
    scheduler = ModuleType("vllm_ascend.patch.platform.patch_balance_schedule")

    def unpatched_factory(config):
        raise TypeError("unexpected keyword afd")

    scheduler.init_ascend_config = unpatched_factory
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    namespace = ModuleType("afd_plugin.compat.patches.npu.ascend_config")

    def install_namespace():
        events.append("namespace")
        # The namespace patch's own tests exercise actual native alias rebinding
        # and strict kwargs validation. This spy tests the registration boundary.
        scheduler.init_ascend_config = lambda config: config.additional_config["afd"]

    namespace.apply_afd_ascend_config_patch = install_namespace
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    afd_plugin.register_afd()
    config = SimpleNamespace(additional_config={"afd": {"role": "attention"}})
    assert scheduler.init_ascend_config(config) == {"role": "attention"}
    first_model = next(i for i, event in enumerate(events) if isinstance(event, tuple))
    assert events.index("namespace") < first_model
    afd_plugin.register_afd()
    assert events.count("namespace") == 1


def test_gpu_registration_does_not_import_npu_namespace(monkeypatch, registration_env):
    platforms, events = registration_env
    platforms.current_platform.device_type = "cuda"
    original_import = builtins.__import__

    def reject_npu_import(name, *args, **kwargs):
        if name == "afd_plugin.compat.patches.npu.ascend_config":
            raise AssertionError("GPU registration imported the NPU namespace patch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_npu_import)
    afd_plugin.register_afd()
    assert any(isinstance(event, tuple) for event in events)


def test_no_vllm_registration_does_not_import_platform(monkeypatch):
    monkeypatch.setattr(afd_plugin, "_registered", False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    original_import = builtins.__import__

    def reject_platform_import(name, *args, **kwargs):
        if name == "vllm.platforms":
            raise AssertionError("CPU-only registration imported vLLM platform")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_platform_import)
    afd_plugin.register_afd()
    afd_plugin.register_afd()
    assert afd_plugin._registered


def test_namespace_install_failure_is_not_swallowed(monkeypatch, registration_env):
    _, events = registration_env
    namespace = ModuleType("afd_plugin.compat.patches.npu.ascend_config")

    def fail_install():
        raise RuntimeError("namespace install failed")

    namespace.apply_afd_ascend_config_patch = fail_install
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    with pytest.raises(RuntimeError, match="namespace install failed"):
        afd_plugin.register_afd()
    assert not afd_plugin._registered
    assert not any(isinstance(event, tuple) for event in events)
