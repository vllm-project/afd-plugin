# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler import (
    afd_npu_profiler_config,
    create_afd_npu_profiler,
    create_runtime_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from tests.unit.fakes.profiler import StepProfiler as _StepProfiler

_ENV_NAMES = (
    "AFD_NPU_ATTENTION_PROFILER_ENABLE",
    "AFD_NPU_ATTENTION_PROFILER_WAIT",
    "AFD_NPU_ATTENTION_PROFILER_WARMUP",
    "AFD_NPU_ATTENTION_PROFILER_ACTIVE",
    "AFD_NPU_ATTENTION_PROFILER_REPEAT",
    "AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST",
    "AFD_NPU_ATTENTION_PROFILER_DIR",
    "AFD_NPU_ATTENTION_PROFILER_WITH_STACK",
    "AFD_NPU_FFN_PROFILER_ENABLE",
    "AFD_NPU_FFN_PROFILER_WAIT",
    "AFD_NPU_FFN_PROFILER_WARMUP",
    "AFD_NPU_FFN_PROFILER_ACTIVE",
    "AFD_NPU_FFN_PROFILER_REPEAT",
    "AFD_NPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_NPU_FFN_PROFILER_DIR",
    "AFD_NPU_FFN_PROFILER_WITH_STACK",
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE",
    "VLLM_ASCEND_FFN_PROFILER_ENABLE",
    "VLLM_TORCH_PROFILER_DIR",
)


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_npu_profiler_defaults_are_disabled():
    attention = afd_npu_profiler_config("attention")
    ffn = afd_npu_profiler_config("ffn")

    assert attention.enabled is False
    assert attention.wait == 2
    assert attention.warmup == 1
    assert attention.active == 10
    assert attention.repeat == 1
    assert attention.skip_first == 1500
    assert attention.trace_dir == "/tmp/profile/attn"
    assert attention.with_stack is False
    assert ffn.enabled is False
    assert ffn.active == 20
    assert ffn.trace_dir == "/tmp/profile/ffn"
    assert ffn.with_stack is False


def test_npu_profiler_uses_only_plugin_owned_enable_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FFN_PROFILER_ENABLE", "1")

    assert afd_npu_profiler_config("ffn").enabled is False

    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "1")

    assert afd_npu_profiler_config("ffn").enabled is True


def test_npu_profiler_dir_falls_back_to_vllm_torch_profiler_dir(monkeypatch):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/vllm-profile")

    assert afd_npu_profiler_config("attention").trace_dir == "/tmp/vllm-profile"

    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_DIR", "/tmp/afd-attn")

    assert afd_npu_profiler_config("attention").trace_dir == "/tmp/afd-attn"


def test_create_npu_profiler_uses_configured_schedule(monkeypatch):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WAIT", "3")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WARMUP", "4")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ACTIVE", "5")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_REPEAT", "6")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_SKIP_FIRST", "7")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_DIR", "/tmp/afd-ffn")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WITH_STACK", "true")

    profiler = create_afd_npu_profiler("ffn")

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True
    assert profiler_module.schedule_kwargs == {
        "wait": 3,
        "warmup": 4,
        "active": 5,
        "repeat": 6,
        "skip_first": 7,
    }
    assert profiler_module.profile_kwargs is not None
    assert profiler_module.profile_kwargs["record_shapes"] is True
    assert profiler_module.profile_kwargs["with_stack"] is True
    assert profiler_module.profile_kwargs["with_modules"] is True
    assert profiler_module.trace_dir == "/tmp/afd-ffn"


def test_step_npu_profiler_ignores_disabled_profiler():
    step_afd_npu_profiler(None)

    profiler = _StepProfiler()
    step_afd_npu_profiler(profiler)

    assert profiler.steps == 1


def test_runtime_profiler_reuses_backend_options_and_handler_without_step_schedule(
    monkeypatch,
):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(profiler=profiler_module)
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "1")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WITH_STACK", "1")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_DIR", "/static-output")
    create_afd_npu_profiler("ffn")
    assert profiler_module.profile_kwargs is not None
    expected = dict(profiler_module.profile_kwargs)
    expected["schedule"] = None
    expected["on_trace_ready"] = ("handler", "/session/worker/full")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "0")
    controllers = []
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler._profiler_action_controller",
        SimpleNamespace(ProfActionController=lambda *args: controllers.append(args)),
    )

    profiler = create_runtime_npu_profiler("ffn", "/session/worker/full")
    from afd_plugin.compat.npu.profiler import _stop_ffn_trace

    assert profiler.prof_if.stop_trace.__func__ is _stop_ffn_trace
    assert controllers == [(profiler, profiler.prof_if, profiler.on_trace_ready)]

    assert profiler is profiler_module.created_profiler
    assert profiler_module.created_profiler.started is True
    assert profiler_module.profile_kwargs == expected
    assert profiler_module.trace_dir == "/session/worker/full"
    assert afd_npu_profiler_config("ffn").enabled is False
    assert afd_npu_profiler_config("ffn").trace_dir == "/static-output"


def test_stop_npu_profiler_ignores_disabled_profiler():
    stop_afd_npu_profiler(None)

    profiler = _StepProfiler()
    stop_afd_npu_profiler(profiler)

    assert profiler.stopped is True


class _FakeTorchNPUProfiler:
    class ExportType:
        Text = "text"

    class ProfilerLevel:
        Level0 = "level0"
        Level2 = "level2"
        Level_none = "level_none"

    class AiCMetrics:
        AiCoreNone = "aicore_none"

    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    def __init__(self):
        self.created_profiler = _StepProfiler()
        self.schedule_kwargs: dict[str, object] = {}
        self.profile_kwargs: dict[str, object] = {}
        self.trace_dir = None
        self._ExperimentalConfig = self._experimental_config

    def _experimental_config(self, **kwargs):
        return kwargs

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir, **kwargs):
        self.handler_kwargs = kwargs
        self.trace_dir = trace_dir
        return ("handler", trace_dir)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        self.created_profiler.prof_if = SimpleNamespace()
        self.created_profiler.on_trace_ready = kwargs["on_trace_ready"]
        return self.created_profiler


@pytest.mark.parametrize("valid,flops", [(False, False), (True, False), (True, True)])
def test_ffn_stop_preserves_native_cleanup_without_synchronizing(
    monkeypatch, valid, flops
):
    from afd_plugin.compat.npu.profiler import _stop_ffn_trace

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler.profiler_interface",
        SimpleNamespace(
            FlopsHookManager=SimpleNamespace(uninstall=lambda: calls.append("flops")),
            _stop_profiler=lambda: calls.append("native"),
            _disable_event_record=lambda: calls.append("events"),
        ),
    )
    interface = SimpleNamespace(
        _is_env_valid=valid,
        with_flops=flops,
        experimental_config=SimpleNamespace(tx_enabled=True),
        stop_gc_detect=lambda: calls.append("gc"),
    )
    _stop_ffn_trace(interface)
    expected = (["flops"] if flops else []) + ["native", "gc", "events"]
    assert calls == (expected if valid else [])


def test_lightweight_collects_only_layer_ranges_without_inline_analysis(monkeypatch):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(profiler=profiler_module)
    )
    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_WITH_STACK", "1")
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler._profiler_action_controller",
        SimpleNamespace(ProfActionController=lambda *args: None),
    )
    disabled = []
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler.profiler_interface",
        SimpleNamespace(
            _disable_event_record=lambda: disabled.append(
                profiler_module.created_profiler.started
            )
        ),
    )
    create_runtime_npu_profiler(
        "attention", "/session/worker/lightweight", lightweight=True
    )
    assert disabled == [True]
    config = profiler_module.profile_kwargs
    assert config is not None
    assert config["activities"] == ["npu"]
    for field in (
        "record_shapes",
        "profile_memory",
        "with_flops",
        "with_stack",
        "with_modules",
    ):
        assert config[field] is False
    assert config["schedule"] is None
    experimental = config["experimental_config"]
    assert isinstance(experimental, dict)
    assert experimental["profiler_level"] == "level_none"
    assert experimental["aic_metrics"] == "aicore_none"
    assert profiler_module.handler_kwargs == {"analyse_flag": False}
    assert experimental["mstx"] is True
    assert experimental["mstx_domain_include"] == ["afd.cam"]


@pytest.mark.parametrize("lightweight", [False, True])
def test_graph_profiling_uses_native_kernel_collection(monkeypatch, lightweight):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(profiler=profiler_module)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler._profiler_action_controller",
        SimpleNamespace(ProfActionController=lambda *args: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.profiler.profiler_interface",
        SimpleNamespace(_disable_event_record=lambda: None),
    )
    create_runtime_npu_profiler(
        "ffn", "/session/worker/graph", lightweight=lightweight, graph=True
    )
    config = profiler_module.profile_kwargs
    assert config is not None
    assert config["schedule"] is None
    assert config["activities"] == (["npu"] if lightweight else ["cpu", "npu"])
    assert config["record_shapes"] is not lightweight
    experimental = config["experimental_config"]
    assert isinstance(experimental, dict)
    assert experimental["profiler_level"] == ("level0" if lightweight else "level2")
    assert experimental["mstx"] is False
    assert experimental["mstx_domain_include"] == []
