from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler import (
    afd_npu_profiler_config,
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)

_ENV_NAMES = (
    "AFD_NPU_ATTENTION_PROFILER_ENABLE",
    "AFD_NPU_ATTENTION_PROFILER_WAIT",
    "AFD_NPU_ATTENTION_PROFILER_WARMUP",
    "AFD_NPU_ATTENTION_PROFILER_ACTIVE",
    "AFD_NPU_ATTENTION_PROFILER_REPEAT",
    "AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST",
    "AFD_NPU_ATTENTION_PROFILER_DIR",
    "AFD_NPU_ATTENTION_PROFILER_WITH_STACK",
    "AFD_NPU_ATTENTION_PROFILER_RANKS",
    "AFD_NPU_FFN_PROFILER_ENABLE",
    "AFD_NPU_FFN_PROFILER_WAIT",
    "AFD_NPU_FFN_PROFILER_WARMUP",
    "AFD_NPU_FFN_PROFILER_ACTIVE",
    "AFD_NPU_FFN_PROFILER_REPEAT",
    "AFD_NPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_NPU_FFN_PROFILER_DIR",
    "AFD_NPU_FFN_PROFILER_WITH_STACK",
    "AFD_NPU_FFN_PROFILER_RANKS",
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
    assert attention.ranks == frozenset({0})
    assert ffn.enabled is False
    assert ffn.active == 20
    assert ffn.trace_dir == "/tmp/profile/ffn"
    assert ffn.with_stack is False
    assert ffn.ranks == frozenset({0})


def test_npu_profiler_parses_rank_selection(monkeypatch):
    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_RANKS", "0,2,2")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", "all")

    assert afd_npu_profiler_config("attention").ranks == frozenset({0, 2})
    assert afd_npu_profiler_config("ffn").ranks is None


def test_npu_profiler_rejects_invalid_rank_selection(monkeypatch):
    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_RANKS", "0,-1")

    with pytest.raises(ValueError, match="non-negative ranks"):
        afd_npu_profiler_config("attention")


def test_create_npu_profiler_skips_unselected_rank(monkeypatch):
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")

    assert create_afd_npu_profiler("ffn", role_rank=1) is None


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
    assert profiler_module.profile_kwargs["record_shapes"] is True
    assert profiler_module.profile_kwargs["with_stack"] is True
    assert profiler_module.profile_kwargs["with_modules"] is True
    assert profiler_module.trace_dir == "/tmp/afd-ffn"


def test_step_npu_profiler_ignores_disabled_profiler():
    step_afd_npu_profiler(None)

    profiler = _StepProfiler()
    step_afd_npu_profiler(profiler)

    assert profiler.steps == 1


def test_stop_npu_profiler_ignores_disabled_profiler():
    stop_afd_npu_profiler(None)

    profiler = _StepProfiler()
    stop_afd_npu_profiler(profiler)

    assert profiler.stopped is True


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self):
        self.steps += 1


class _FakeTorchNPUProfiler:
    class ExportType:
        Text = "text"

    class ProfilerLevel:
        Level2 = "level2"

    class AiCMetrics:
        AiCoreNone = "aicore_none"

    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    def __init__(self):
        self.created_profiler = _StepProfiler()
        self.schedule_kwargs = None
        self.profile_kwargs = None
        self.trace_dir = None
        self._ExperimentalConfig = self._experimental_config

    def _experimental_config(self, **kwargs):
        return kwargs

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir):
        self.trace_dir = trace_dir
        return ("handler", trace_dir)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        return self.created_profiler
