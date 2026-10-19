# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The FFN graph E2E must prove real execution, not just HTTP success."""

import json
import os
import select
import signal
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.e2e.models.deepseek_v4_flash import ffn_graph


@pytest.mark.parametrize(
    "phase,signum",
    [
        ("running", signal.SIGTERM),
        ("running", signal.SIGINT),
        ("cleanup", signal.SIGTERM),
        ("cleanup", signal.SIGINT),
        ("success", None),
        ("failure", None),
    ],
)
def test_signals_do_not_interrupt_service_cleanup(monkeypatch, tmp_path, signum, phase):
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ffn_graph",
            "--mode",
            "eager",
            "--model",
            "/unused",
            "--afd-host",
            "127.0.0.1",
            "--output-dir",
            str(output),
        ],
    )
    # Readiness is a pipe handshake, so SIGTERM cannot precede SIG_IGN setup.
    popen = subprocess.Popen
    children = []
    logs = []

    def launch(_command, **kwargs):
        logs.append(kwargs["stdout"])
        child = popen(
            [
                sys.executable,
                "-c",
                "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); signal.pause()",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        children.append(child)
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 5)[0], "Child not ready"
        assert child.stdout.readline() == "ready\n"
        child.stdout.close()
        return child

    monkeypatch.setattr(ffn_graph.subprocess, "Popen", launch)
    client = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(is_success=True))
    monkeypatch.setattr(ffn_graph.httpx, "Client", lambda **k: nullcontext(client))
    monkeypatch.setattr(ffn_graph, "snapshots", lambda *a: {})
    monkeypatch.setattr(ffn_graph, "make_requests", lambda *a: [])
    monkeypatch.setattr(ffn_graph, "audit_execution", lambda *a: None)

    def requests(*args):
        if phase == "running":
            signal.raise_signal(signum)
        if phase == "failure":
            raise RuntimeError("request failed")
        return []

    monkeypatch.setattr(ffn_graph, "run_requests", requests)
    terminate = ffn_graph.terminate_process_groups

    def cleanup(processes, **kwargs):
        # First signal during cleanup, or a repeated signal after interruption.
        if signum is not None:
            signal.raise_signal(signum)
            signal.raise_signal(
                signal.SIGINT if signum == signal.SIGTERM else signal.SIGTERM
            )
        return terminate(processes, **{**kwargs, "termination_timeout_s": 0})

    monkeypatch.setattr(ffn_graph, "terminate_process_groups", cleanup)
    save = ffn_graph.save

    def save_with_signal(path, value):
        if path.name == "cleanup.json":
            assert all(log.closed for log in logs)
            if signum is not None:
                signal.raise_signal(signum)
        save(path, value)

    monkeypatch.setattr(ffn_graph, "save", save_with_signal)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if signum is not None:
            with pytest.raises(SystemExit) as caught:
                ffn_graph.main()
            assert caught.value.code == 128 + signum
        elif phase == "failure":
            with pytest.raises(RuntimeError, match="request failed"):
                ffn_graph.main()
        else:
            ffn_graph.main()
        assert len(children) == 2
        assert all(child.poll() == -signal.SIGKILL for child in children)
        for child in children:
            with pytest.raises(ProcessLookupError):
                os.killpg(child.pid, 0)
        report = json.loads((output / "cleanup.json").read_text())
        assert report["failures"] == []
        assert len(report["forced_cleanup"]) == 2
        assert set(report["exit_codes"].values()) == {-signal.SIGKILL}
        assert (output / "passed.json").exists() == (phase == "success")
        assert (output / "inference-passed.json").exists() == (
            phase in ("cleanup", "success")
        )
        assert all(
            signal.getsignal(sig) == handler for sig, handler in previous.items()
        )
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
