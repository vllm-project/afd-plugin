# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import threading

import pytest

from afd_plugin.profiling import native as full
from afd_plugin.profiling.native import RuntimeProfiler


@pytest.fixture(
    params=[
        ("FULL", False),
        ("LIGHTWEIGHT", False),
        ("FULL", True),
        ("LIGHTWEIGHT", True),
    ]
)
def backend(tmp_path, monkeypatch, request):
    mode, graph = request.param
    calls = []
    stopping = threading.Event()
    release = threading.Event()
    release.set()

    def record(operation):
        calls.append((operation, threading.get_ident()))

    class NativeProfiler:
        failure = None
        block_start = False

        def step(self):
            record("step")
            if self.failure == "step":
                raise RuntimeError("step failed")

        def stop(self):
            record("stop")
            stopping.set()
            assert release.wait(2.0)
            if self.failure == "stop":
                raise RuntimeError("stop failed")

    native = NativeProfiler()
    received = []

    def factory(role, trace_dir, **kwargs):
        assert kwargs == {"lightweight": mode == "LIGHTWEIGHT", "graph": graph}
        received.append((role, trace_dir))
        record("factory_start")
        if native.failure == "start":
            raise RuntimeError("start failed")
        if native.block_start:
            stopping.set()
            assert release.wait(2.0)
        return native

    monkeypatch.setattr(full, "create_runtime_npu_profiler", factory)
    monkeypatch.setattr(full, "archive_native_trace", lambda path: None)
    owner = RuntimeProfiler("ffn", tmp_path, mode, graph=graph)
    yield owner, native, calls, received, stopping, release
    release.set()
    owner.stop()
    owner.step()


def test_native_lifecycle_runs_only_on_original_runner_thread(backend):
    owner, _, calls, _, _, _ = backend
    owner.step()
    owner.start()
    assert owner.state == "STARTING"
    assert owner.busy
    assert owner.started_at_ns is None
    assert not calls
    runner_tid = threading.get_ident()
    owner.step()
    assert owner.state == "RUNNING"
    owner.step()
    owner.stop()
    assert owner.state == "STOPPING"
    assert not owner.finished
    owner.step()
    assert owner.finished
    assert not owner.busy
    assert [name for name, _ in calls] == ["factory_start", "step", "step", "stop"]
    assert {tid for _, tid in calls} == {runner_tid}
    status = owner.snapshot()
    if owner.mode == "LIGHTWEIGHT":
        for key in ("archive_state", "archive_error", "archive_path"):
            status.pop(key)
    assert status == {
        "state": "COMPLETED",
        "trace_dir": str(owner.trace_dir),
        "started_at_ns": owner.started_at_ns,
        "error": None,
        "busy": False,
    }
    assert status["started_at_ns"] > 0


@pytest.mark.parametrize("started", [False, True])
def test_stop_before_any_runner_step_never_touches_native_backend(backend, started):
    owner, _, calls, _, _, _ = backend
    if started:
        owner.start()
    owner.stop()
    owner.step()
    assert owner.finished
    assert owner.state == "COMPLETED"
    assert not owner.busy
    assert not calls


@pytest.mark.parametrize("failure", ["start", "step", "stop"])
def test_native_failure_is_reported_without_breaking_model_execution(backend, failure):
    owner, native, _, _, _, _ = backend
    owner.start()
    if failure == "stop":
        owner.step()
        owner.stop()
    native.failure = failure
    owner.step()
    assert owner.finished
    assert owner.state == "FAILED"
    assert owner.error == f"RuntimeError: {failure} failed"
    assert not owner.busy
    owner.stop()
    assert owner.state == "FAILED"


def test_blocked_native_stop_keeps_control_requests_reachable(backend):
    owner, _, calls, _, stopping, release = backend
    release.clear()
    initialized = threading.Event()
    stop_requested = threading.Event()

    def run_model():
        owner.step()
        initialized.set()
        assert stop_requested.wait(2.0)
        owner.step()

    owner.start()
    runner = threading.Thread(target=run_model)
    runner.start()
    try:
        assert initialized.wait(1.0)
        owner.stop()
        stop_requested.set()
        assert stopping.wait(1.0)
        owner.stop()
        status = owner.snapshot()
        assert status["state"] == "STOPPING"
        assert status["busy"]
        assert not owner.finished
    finally:
        release.set()
        runner.join(2.0)
    assert not runner.is_alive()
    assert owner.finished
    assert len({tid for _, tid in calls}) == 1
    assert calls[0][1] != threading.get_ident()


def test_stop_during_native_start_is_serviced_by_the_same_runner_step(backend):
    owner, native, calls, _, entering, release = backend
    native.block_start = True
    release.clear()
    owner.start()
    runner = threading.Thread(target=owner.step)
    runner.start()
    try:
        assert entering.wait(1.0)
        owner.stop()
        assert owner.snapshot()["state"] == "STOPPING"
        assert owner.busy
        assert not owner.finished
    finally:
        release.set()
        runner.join(2.0)
    assert not runner.is_alive()
    assert owner.finished
    assert owner.state == "COMPLETED"
    assert [name for name, _ in calls] == ["factory_start", "stop"]
    assert len({tid for _, tid in calls}) == 1


def test_archive_thread_failure_does_not_poison_native_backend(backend, monkeypatch):
    owner, _, _, _, _, _ = backend
    owner.start()
    owner.step()

    def fail_start(self):
        raise RuntimeError("cannot start archive thread")

    monkeypatch.setattr(full.threading.Thread, "start", fail_start)
    owner.stop()
    owner.step()
    assert owner.state == "COMPLETED" and owner.error is None
    if owner.mode == "LIGHTWEIGHT":
        assert owner.archive_state == "FAILED"
        assert "cannot start archive thread" in owner.archive_error
    else:
        assert owner.archive_state is None
