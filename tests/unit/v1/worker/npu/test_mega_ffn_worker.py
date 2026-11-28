# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAMAsyncMegaAFDConnector worker loop and shutdown lifecycle."""

import threading
from types import SimpleNamespace

import pytest

from tests.unit.fakes.npu_runtime import new_ffn_worker as _new_ffn_worker


def test_npu_ffn_worker_drives_mega_runner_without_device_sync(monkeypatch):
    worker = _new_ffn_worker()
    from afd_plugin.v1.worker.npu import ffn_worker

    event = threading.Event()
    calls: list[str | tuple[str, SimpleNamespace]] = []

    def execute_connector_driven_step():
        calls.append("step")
        event.set()

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = object.__new__(ffn_worker.MegaNPUFFNModelRunner)
    worker.model_runner.execute_connector_driven_step = execute_connector_driven_step

    monkeypatch.setattr(
        ffn_worker.torch.npu,
        "set_device",
        lambda device: calls.append(("bind", device)),
    )
    monkeypatch.setattr(
        ffn_worker.torch.npu, "synchronize", lambda: calls.append("sync")
    )
    worker._run_ffn_server_loop()

    # Mega polls its single lane completion event; the worker must not add
    # a device-wide fence to each submission.
    assert calls == [("bind", worker.device), "step"]


def test_npu_ffn_worker_joins_mega_loop_before_connector_release(monkeypatch):
    worker = _new_ffn_worker()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker._ffn_shutdown_event = threading.Event()
    calls: list[str | tuple[str, float]] = []

    class _StoppingThread:
        alive = False

        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return self.alive

    runner = object.__new__(ffn_worker.MegaNPUFFNModelRunner)
    runner.shutdown = lambda: calls.append("release")
    worker._ffn_thread = _StoppingThread()
    worker.model_runner = runner

    worker.stop_ffn_server_loop()

    assert calls == [
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        "release",
    ]


def test_npu_ffn_worker_retains_mega_resources_while_step_is_pending(monkeypatch):
    worker = _new_ffn_worker()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker._ffn_shutdown_event = threading.Event()
    calls: list[str | tuple[str, float]] = []

    class _StoppingThread:
        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return True

    runner = object.__new__(ffn_worker.MegaNPUFFNModelRunner)
    runner.shutdown = lambda: calls.append("release")
    worker._ffn_thread = _StoppingThread()
    worker.model_runner = runner

    with pytest.raises(RuntimeError, match="retain resources"):
        worker.stop_ffn_server_loop()

    assert calls == [("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS)]
