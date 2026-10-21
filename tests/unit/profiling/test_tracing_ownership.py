# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pipeline transitions reuse stream tracks and never allocate per-call records."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from afd_plugin.profiling.model import begin_profile_call, install_pipeline_hook
from afd_plugin.profiling.session import Capture


@pytest.fixture
def tracer(monkeypatch):
    native = SimpleNamespace(_mark=Mock(), _mark_on_host=Mock())
    runtime = SimpleNamespace(
        _C=SimpleNamespace(
            _mstx=native, _npu_getCurrentStream=Mock(return_value=(11, 0, 20))
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    tracing = importlib.import_module("afd_plugin.profiling.tracing")
    monkeypatch.setattr(tracing, "torch_npu", runtime)
    tracer = tracing.DeviceTracer(SimpleNamespace(index=0), "f0")
    capture = Capture(
        "test",
        1,
        SimpleNamespace(submit=Mock(), snapshot=lambda: {"dropped_records": 0}),
        accepting=True,
        deadline_ns=time.monotonic_ns() + int(60e9),
    )
    tracer.session = SimpleNamespace(capture=capture, _wake=threading.Event())
    return tracer, capture, native


def test_three_pipeline_states_without_stage_names_or_identity(tracer):
    tracer, capture, native = tracer
    execution = tracer.new_work()
    assert tracer.retain_execution(execution)
    for kind in (
        "COMPUTE",
        "COMPUTE",
        "COMMUNICATION",
        "COMPUTE",
        "WAIT",
        "WAIT",
        "COMPUTE",
    ):
        tracer.transition(execution, kind)
    tracer.release_execution(execution)
    tracer.idle(capture)
    assert [c.args[0] for c in native._mark.call_args_list] == [
        "afd.cam|COMPUTE",
        "afd.cam|COMMUNICATION",
        "afd.cam|COMPUTE",
        "afd.cam|WAIT",
        "afd.cam|COMPUTE",
        "afd.cam|WAIT",
    ]
    tracer.end_capture(capture)
    native._mark_on_host.assert_called_once_with("afd.cam|END", "afd.cam")
    tracer.finish_capture(capture)
    assert capture.writer.submit.call_args.args[0]["native_marks"] == 7
    assert tracer.pending_count() == 0


def test_streams_and_generations_keep_independent_states(tracer):
    tracer, capture, native = tracer
    first = tracer.execution()
    assert tracer.retain_execution(first)
    tracing = importlib.import_module("afd_plugin.profiling.tracing")
    tracing.torch_npu._C._npu_getCurrentStream.return_value = (12, 0, 20)
    second = tracer.execution()
    tracer.transition(first, "COMPUTE")
    tracer.transition(second, "COMPUTE")
    assert native._mark.call_count == 2
    tracer.close_admission(capture)
    assert tracer.new_work() is None
    tracer.transition(first, "WAIT")
    tracer.release_execution(first)
    tracer.end_capture(capture)
    tracer.finish_capture(capture)
    tracer.begin_capture()
    tracer.transition(first, "COMPUTE")
    assert native._mark.call_count == 3
    assert not tracer._streams


def test_unresolved_work_is_reported_at_drain_deadline(tracer):
    tracer, capture, _ = tracer
    tracer.retain_execution(tracer.new_work())
    tracer.finish_capture(capture)
    assert "unresolved" in capture.error
    assert tracer.snapshot()["unresolved_spans"] == 1


def test_native_failure_never_interrupts_model_execution(tracer):
    tracer, capture, native = tracer
    native._mark.side_effect = RuntimeError("failed")
    tracer.transition(tracer.new_work(), "COMPUTE")
    assert "native pipeline mark failed" in capture.error
    assert tracer.snapshot()["dropped_spans"] == 1


def test_layer_release_does_not_break_adjacent_compute(tracer):
    tracer, capture, native = tracer
    for _ in range(100):
        execution = tracer.execution()
        assert tracer.retain_execution(execution)
        tracer.transition(execution, "COMPUTE")
        tracer.release_execution(execution)
    assert native._mark.call_count == 1
    tracer.idle(capture)
    assert native._mark.call_count == 2
    assert native._mark.call_args.args[0] == "afd.cam|WAIT"


@pytest.mark.parametrize("start_layer", [0, 1])
def test_compiled_model_entry_retains_and_releases_every_call(tracer, start_layer):
    pytest.importorskip("torch")
    from torch import nn

    tracer, capture, _ = tracer
    metadata = SimpleNamespace(
        connector=SimpleNamespace(profile_session=SimpleNamespace(tracer=tracer))
    )

    class Layer(nn.Module):
        def forward(self):
            execution = begin_profile_call(metadata)
            assert execution is not None
            assert tracer.pending_count() == 1
            tracer.release_execution(execution)

    class CompiledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.start_layer = start_layer
            self.layers = nn.ModuleList([nn.Identity()] * start_layer + [Layer()])

        def forward(self):
            return self.layers[self.start_layer]()

        def __call__(self):
            # Match support_torch_compile's eager path: no Module._call_impl.
            metadata.profile_pipeline = None
            return self.forward()

    model = CompiledModel()
    install_pipeline_hook(model, lambda: metadata)
    tracing = importlib.import_module("afd_plugin.profiling.tracing")
    for _ in range(tracing.MAX_PENDING_EXECUTIONS + 1):
        model()
        assert tracer.pending_count() == 0
    tracer.finish_capture(capture)
    assert tracer.snapshot()["unresolved_spans"] == 0
    assert tracer.snapshot()["dropped_spans"] == 0
