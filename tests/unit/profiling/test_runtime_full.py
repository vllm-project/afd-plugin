# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from afd_plugin.profiling import native as full_module
from afd_plugin.profiling import npu as npu_module
from afd_plugin.profiling import session as session_module
from afd_plugin.profiling.session import WorkerSession
from afd_plugin.profiling.worker import CAMProfileWorkerMixin
from tests.unit.profiling.test_session import (
    FakeClock,
    FakeTracer,
    commit_command,
    finish_writer,
    prepare_command,
    stop_command,
)


@pytest.fixture
def runtime_session(tmp_path, monkeypatch):
    clock = FakeClock()
    backend = SimpleNamespace(step=Mock(), stop=Mock())
    create = Mock(return_value=backend)
    monkeypatch.setattr(session_module, "time", clock)
    monkeypatch.setattr(full_module, "time", clock)
    monkeypatch.setattr(session_module, "PROFILER_OPERATION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(full_module, "create_runtime_npu_profiler", create)
    monkeypatch.setattr(full_module, "archive_native_trace", lambda path: None)
    session = WorkerSession(
        "attention-dp0",
        "attention",
        tmp_path,
        None,
        {},
        tracer_factory=FakeTracer,
        profiler_factory=partial(full_module.RuntimeProfiler, "attention"),
    )
    yield session, clock, create, backend
    if session.capture is not None:
        assert session.capture.writer.close(1)


@pytest.mark.parametrize("mode", ["FULL", "LIGHTWEIGHT"])
def test_modes_start_on_owner_thread_stop_at_safe_point_and_reuse(
    runtime_session, mode
):
    session, clock, create, backend = runtime_session
    paths = []
    for generation in range(2):
        session.prepare(prepare_command(mode=mode, duration_s=1), backend_busy=False)
        capture = session.capture
        paths.append(capture.profiler.trace_dir)
        assert (session.tracer is None) == (mode == "FULL")
        session.commit(commit_command(session, clock))
        session.tick()
        assert create.call_count == generation
        clock.advance(0.1)
        session.tick()
        assert session.state == "STARTING"
        session.advance_profiler(allow_stop=False)
        session.tick()
        assert session.state == "RUNNING"
        assert capture.started_at_ns == clock.time_ns()
        assert capture.admits() == (mode == "LIGHTWEIGHT")
        session.stop(stop_command(session))
        session.advance_profiler(allow_stop=False)
        assert backend.stop.call_count == generation
        finish_writer(session)
        assert session.state == "COMPLETED"
        assert not capture.admits() and session.profiler_reusable
    assert paths[0] != paths[1]
    assert create.call_count == backend.stop.call_count == 2


@pytest.mark.parametrize("phase", ["start", "stop"])
@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_native_failure_or_deadline_blocks_backend_reuse(
    runtime_session, phase, failure
):
    session, clock, create, backend = runtime_session
    session.prepare(prepare_command(mode="FULL", duration_s=1), backend_busy=False)
    session.commit(commit_command(session, clock))
    clock.advance(0.1)
    session.tick()
    if phase == "stop":
        session.advance_profiler()
        session.tick()
        session.stop(stop_command(session))
    operation = create if phase == "start" else backend.stop
    if failure == "error":
        operation.side_effect = RuntimeError(f"native {phase} failed")
        session.advance_profiler()
    else:
        clock.advance(0.051)
    finish_writer(session)
    assert session.state == "FAILED"
    assert phase in session.capture.error
    assert session.status()["details"]["profiler_poisoned"]
    assert not session.profiler_reusable
    for mode in ("FULL", "LIGHTWEIGHT"):
        with pytest.raises(ValueError, match="restart worker"):
            session.prepare(prepare_command(mode=mode), backend_busy=False)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_enabled_worker_creates_tracer_only_for_lightweight(
    tmp_path, monkeypatch, role, initialized, graph
):
    class Connector(SimpleNamespace):
        pass

    connector = Connector(
        profile_worker_id=f"{role}-dp0",
        topology=SimpleNamespace(role=role),
        graph_startup=SimpleNamespace(mode="eager"),
        world_rank=0,
        attn_size=1,
        ffn_size=1,
        tp_size=1,
        comm_args=object() if initialized else None,
        is_initialized=initialized,
    )
    module = ModuleType("afd_plugin.connectors.npu.async_cam")
    monkeypatch.setattr(module, "CAMAsyncAFDConnector", Connector, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    monkeypatch.setenv("AFD_CAM_PROFILE_DIR", str(tmp_path))
    starts = []
    waiters = []
    tracer_calls = []
    waiter = SimpleNamespace()
    tracer = FakeTracer()
    monkeypatch.setattr(
        WorkerSession, "start", lambda session, device: starts.append((session, device))
    )

    def create_waiter(device, session):
        waiters.append((device, session))
        return waiter

    monkeypatch.setattr(npu_module, "FFNDeviceWaiter", create_waiter)
    worker = CAMProfileWorkerMixin()
    worker.device = "npu:0"
    worker.model_runner = SimpleNamespace(connector=connector, prof=None)

    def create_tracer(received_connector):
        tracer_calls.append(received_connector)
        return tracer

    monkeypatch.setattr(worker, "_create_cam_tracer", create_tracer)
    worker.initialize_cam_profiling()
    session = worker.cam_profile_session
    assert connector.profile_session is session
    assert starts == [(session, worker.device)]
    assert waiters == ([(worker.device, session)] if role == "ffn" else [])
    assert session.device_waiter is (waiter if role == "ffn" else None)
    assert session.tracer is None
    assert not tracer_calls
    connector.graph_startup.mode = "full" if graph else "eager"

    try:
        assert worker.cam_profile_prepare(prepare_command(mode="FULL"))["error"] is None
        assert session.capture.profiler.graph is (graph and role == "ffn")
        assert session.tracer is None
        assert not tracer_calls
        worker.cam_profile_stop(stop_command(session))
        finish_writer(session)
        command = prepare_command()
        assert worker.cam_profile_prepare(command)["error"] is None
        assert worker.cam_profile_prepare(dict(command))["error"] is None
        assert tracer_calls == [connector]
        assert tracer.session is session
        assert tracer.beginnings == 1
        assert session.tracer is tracer
    finally:
        if session.capture is not None:
            assert session.capture.writer.close(1.0)
