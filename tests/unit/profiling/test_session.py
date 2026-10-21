# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from afd_plugin.profiling import session as session_module
from afd_plugin.profiling import storage
from afd_plugin.profiling.session import WorkerSession
from afd_plugin.profiling.worker import CAMProfileWorkerMixin


class FakeClock:
    def __init__(self):
        self.wall = 1_700_000_000_000_000_000
        self.monotonic = 1_000_000_000

    def time_ns(self):
        return self.wall

    def monotonic_ns(self):
        return self.monotonic

    def advance(self, seconds):
        elapsed = int(seconds * 1_000_000_000)
        self.wall += elapsed
        self.monotonic += elapsed


class ImmediateProfiler:
    def __init__(self, path, mode, clock):
        self.trace_dir = path
        self.mode = mode
        self.clock = clock
        self.state = "PREPARED"
        self.started_at_ns = None
        self.error = None
        self.busy = False

    @property
    def finished(self):
        return self.state in {"COMPLETED", "FAILED"}

    def start(self):
        self.state = "STARTING"
        self.busy = True

    def step(self, **kwargs):
        if self.state == "STARTING":
            self.state = "RUNNING"
            self.started_at_ns = self.clock.time_ns()
        elif self.state == "STOPPING":
            self.state = "COMPLETED"
            self.busy = False

    def stop(self):
        self.state = "STOPPING" if self.busy else "COMPLETED"

    def snapshot(self):
        return {"state": self.state, "trace_dir": str(self.trace_dir)}


class FakeTracer:
    def __init__(self):
        self.session: WorkerSession | None = None
        self.pending = 0
        self.collections = 0
        self.finalized_generations = set()
        self.beginnings = 0

    def begin_capture(self):
        self.beginnings += 1

    def close_admission(self, capture):
        capture.accepting = False

    def pending_count(self):
        return self.pending

    def snapshot(self):
        return {
            "pending_spans": self.pending,
            "dropped_spans": 0,
            "unresolved_spans": 0,
        }

    def collect(self, capture):
        self.collections += 1

    def finalized(self, capture):
        return capture.generation in self.finalized_generations

    def end_capture(self, capture):
        pass

    def finish_capture(self, capture):
        self.finalized_generations.add(capture.generation)
        capture.writer.submit(
            {"record_type": "FINISHED", "generation": capture.generation}
        )
        if self.pending:
            capture.error = "device records unresolved at drain deadline"


@pytest.fixture
def worker_session(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(session_module, "time", clock)
    tracer = FakeTracer()
    session = WorkerSession(
        "attention-dp0",
        "attention",
        tmp_path,
        tracer,
        {"pid": 7},
        profiler_factory=lambda path, mode: ImmediateProfiler(path, mode, clock),
    )
    yield session, tracer, clock
    if session.capture:
        assert session.capture.writer.close(1.0)


def prepare_command(**overrides):
    command = {
        "session_id": str(uuid.uuid4()),
        "command_seq": 1,
        "mode": "LIGHTWEIGHT",
        "duration_s": 0.2,
        "lease_s": 10,
    }
    command.update(overrides)
    return command


def commit_command(session, clock, **overrides):
    command = {
        "session_id": session.capture.session_id,
        "command_seq": 2,
        "start_at_ns": clock.time_ns() + 100_000_000,
        "clock_by_worker": {session.worker_id: {"offset_ns": 0, "uncertainty_ns": 0}},
        "lease_s": 10,
    }
    command.update(overrides)
    return command


def stop_command(session):
    return {"session_id": session.capture.session_id, "command_seq": 3}


def finish_writer(session):
    session.tick()
    session.advance_profiler()
    session.tick()
    assert session.capture.writer.close(1.0)
    session.tick()


def test_lifecycle_uses_monotonic_admission_and_per_worker_clock(worker_session):
    session, tracer, clock = worker_session
    prepared = session.prepare(prepare_command(), backend_busy=False)
    capture = session.capture
    assert prepared["state"] == "PREPARED"
    assert prepared["host_clock_id"] == session_module.host_clock_id()
    assert not capture.admits()
    offset = 5_000_000_000
    local_start = clock.time_ns() + 100_000_000
    command = commit_command(
        session,
        clock,
        start_at_ns=local_start - offset,
        clock_by_worker={
            session.worker_id: {"offset_ns": offset, "uncertainty_ns": 17}
        },
    )
    assert session.commit(command)["state"] == "STARTING"
    assert capture.start_host_ns == local_start - offset
    assert capture.clock_offset_ns == offset
    assert capture.clock_uncertainty_ns == 17
    assert capture.start_ns == clock.monotonic_ns() + 100_000_000
    assert not capture.admits()
    clock.advance(0.1)
    assert not capture.admits()
    session.tick()
    session.advance_profiler()
    session.tick()
    assert capture.admits()
    assert session.status()["state"] == "RUNNING"
    assert session.status()["details"]["started_at_ns"] == clock.time_ns()
    clock.advance(0.2)
    assert not capture.admits()
    finish_writer(session)
    status = session.status()
    assert status["state"] == "COMPLETED"
    assert status["error"] is None
    assert status["worker_id"] == "attention-dp0"
    assert status["pid"] == 7
    assert status["capabilities"] == ["LIGHTWEIGHT", "FULL"]
    assert tracer.finalized_generations == {capture.generation}
    path = session.output_dir / capture.session_id / "attention-dp0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["SESSION", "FINISHED"]
    assert records[0]["host_clock_id"] == prepared["host_clock_id"]


def test_command_retries_are_idempotent_and_stale_commands_are_rejected(worker_session):
    session, _, clock = worker_session
    with pytest.raises(ValueError):
        session.prepare(prepare_command(session_id="../escape"), backend_busy=False)
    assert session.status()["session_id"] is None
    prepare = prepare_command()
    session.prepare(prepare, backend_busy=False)
    capture = session.capture
    assert session.prepare(dict(prepare), backend_busy=False)["state"] == "PREPARED"
    assert session.capture is capture
    commit = commit_command(session, clock)
    session.commit(commit)
    assert session.commit(dict(commit))["state"] == "STARTING"
    with pytest.raises(ValueError, match="stale|conflicting"):
        session.prepare(prepare, backend_busy=False)
    with pytest.raises(ValueError, match="stale|conflicting"):
        session.commit({**commit, "start_at_ns": commit["start_at_ns"] + 1})
    stop = stop_command(session)
    session.stop(stop)
    deadline = capture.drain_deadline_ns
    clock.advance(0.1)
    assert session.stop(dict(stop))["state"] == "STOPPING"
    assert capture.drain_deadline_ns == deadline
    assert not capture.admits()
    finish_writer(session)
    assert session.stop(stop)["state"] == "COMPLETED"


@pytest.mark.parametrize(
    "clocks",
    [
        None,
        [],
        {},
        {"other-worker": {"offset_ns": 0, "uncertainty_ns": 0}},
        {"attention-dp0": None},
        {"attention-dp0": {}},
        {"attention-dp0": {"offset_ns": 0}},
        {"attention-dp0": {"offset_ns": True, "uncertainty_ns": 0}},
        {"attention-dp0": {"offset_ns": 0.0, "uncertainty_ns": 0}},
        {"attention-dp0": {"offset_ns": 0, "uncertainty_ns": -1}},
        {"attention-dp0": {"offset_ns": 0, "uncertainty_ns": False}},
    ],
)
def test_commit_requires_explicit_valid_worker_clock(worker_session, clocks):
    session, _, clock = worker_session
    session.prepare(prepare_command(), backend_busy=False)
    command = commit_command(session, clock, clock_by_worker=clocks)
    if clocks is None:
        command.pop("clock_by_worker")
    with pytest.raises(ValueError, match="worker clock"):
        session.commit(command)
    assert session.state == "PREPARED"
    assert session.command_seq == 1
    assert not session.capture.accepting
    assert session.capture.start_ns == 0
    assert session.capture.error is None


def test_prepared_lease_expires_without_commit_and_flushes_idle_output(worker_session):
    session, _, clock = worker_session
    session.prepare(prepare_command(lease_s=0.05), backend_busy=False)
    clock.advance(0.05)
    finish_writer(session)
    assert session.state == "COMPLETED"
    assert session.capture.stop_requested
    assert session.capture.started_at_ns is None


def test_writer_flush_timeout_fails_and_blocks_reuse_until_closed(
    worker_session, monkeypatch
):
    session, _, clock = worker_session
    entered = threading.Event()
    release = threading.Event()
    original_dumps = storage.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(storage.json, "dumps", blocked_dumps)
    session.prepare(prepare_command(), backend_busy=False)
    session.commit(commit_command(session, clock))
    assert entered.wait(1.0)
    try:
        session.stop(stop_command(session))
        session.tick()
        clock.advance(session_module.DRAIN_TIMEOUT_S * 2 + 0.1)
        session.tick()
        assert session.state == "FAILED"
        assert "flush" in session.capture.error.lower()
        with pytest.raises(ValueError, match="writer|flush"):
            session.prepare(prepare_command(), backend_busy=False)
    finally:
        release.set()
        assert session.capture.writer.close(1.0)


@pytest.mark.parametrize("delayed_start", [False, True])
@pytest.mark.parametrize("automatic_stop", [False, True])
def test_worker_rpc_mixin_preserves_native_profiler_exclusivity(
    worker_session, delayed_start, automatic_stop
):
    session, _, _ = worker_session

    class NativeProfiler:
        # Model the pinned WorkerProfiler's delayed start and max-iteration stop.
        def start(self):
            self._active = True
            self.is_running = not delayed_start

        def step(self):
            if self.is_running:
                self.stop()
            elif self._active:
                self.is_running = True

        def stop(self):
            self._active = False
            self.is_running = False

    class NativeWorker:
        profiler: NativeProfiler | None = None

        def profile(self, is_start=True, profile_prefix=None):
            if self.profiler is None:
                self.profiler = NativeProfiler()
            if is_start:
                self.profiler.start()
            else:
                self.profiler.stop()
            return is_start, profile_prefix

    class Worker(CAMProfileWorkerMixin, NativeWorker):
        pass

    worker = Worker()
    worker.cam_profile_session = session
    worker.model_runner = SimpleNamespace(prof=None)
    assert worker.cam_profile_status({})["state"] == "IDLE"
    assert worker.profile(True, "native") == (True, "native")
    assert worker.profiler is not None
    if delayed_start:
        assert not worker.profiler.is_running
        assert "profiler" in worker.cam_profile_prepare(prepare_command())["error"]
        worker.profiler.step()
    assert worker.profiler.is_running
    rejected = worker.cam_profile_prepare(prepare_command())
    assert "profiler" in rejected["error"]
    assert session.status()["error"] is None
    if automatic_stop:
        worker.profiler.step()
    else:
        assert worker.profile(False, "native") == (False, "native")
    assert not worker.profiler._active
    worker.model_runner.prof = object()
    assert "profiler" in worker.cam_profile_prepare(prepare_command())["error"]
    worker.model_runner.prof = None
    assert worker.cam_profile_prepare(prepare_command())["state"] == "PREPARED"
    with pytest.raises(ValueError, match="runtime CAM"):
        worker.profile()
    assert worker.profile(False, "native") == (False, "native")
    worker.cam_profile_stop(stop_command(session))
    finish_writer(session)
    assert worker.profile() == (True, None)


def test_disabled_worker_does_not_initialize_device_profiling(monkeypatch):
    monkeypatch.delenv("AFD_CAM_PROFILE_ENABLE", raising=False)
    worker = CAMProfileWorkerMixin()
    worker.initialize_cam_profiling()
    assert worker.cam_profile_session is None
    with pytest.raises(ValueError, match="not enabled"):
        worker.cam_profile_status({})


def test_ffn_stop_fence_is_queried_only_for_pending_native_stop(worker_session):
    session, _, clock = worker_session
    session.prepare(prepare_command(), backend_busy=False)
    session.commit(commit_command(session, clock))
    clock.advance(0.1)
    session.tick()

    def forbidden():
        raise AssertionError("unnecessary native query")

    session.advance_profiler(stop_ready=forbidden)
    session.tick()
    assert session.state == "RUNNING"
    session.advance_profiler(stop_ready=forbidden)
    session.stop(stop_command(session))
    session.advance_profiler(stop_ready=lambda: False)
    assert session.capture.profiler.state == "STOPPING"
    session.advance_profiler(stop_ready=lambda: True)
    assert session.capture.profiler.finished
