# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import signal

import pytest

from tests.e2e import process_utils

LEADER_PID = 101
FORCED = "forced SIGKILL for process group 101 after 0s timeout"
SURVIVED = "process group 101 still alive after SIGKILL"
NON_UTF8_COMM = b"bad\xff\xfe"
RUN_FFN_ENVIRONMENT = {"AFD_E2E_RUN_ID": "run-1", "AFD_E2E_PROCESS_ROLE": "ffn"}


class FakeLeader:
    def __init__(self, pid=LEADER_PID):
        self.pid = pid
        self.wait_timeouts = []

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)


class FakeSignals:
    """Records os.kill/os.killpg and models which signal ends each group."""

    def __init__(self):
        self.calls = []
        self.fatal_signals = {}
        self.exited_leaders = set()
        self.dead_groups = set()

    def kill(self, pid, sig):
        self.calls.append(("kill", pid, sig))
        if pid in self.exited_leaders:
            raise ProcessLookupError
        self._deliver(pid, sig)

    def killpg(self, pgid, sig):
        self.calls.append(("killpg", pgid, sig))
        if sig != 0:
            self._deliver(pgid, sig)
        elif pgid in self.dead_groups:
            raise ProcessLookupError

    def _deliver(self, pgid, sig):
        if self.fatal_signals.get(pgid) == sig:
            self.dead_groups.add(pgid)


@pytest.fixture
def fake_signals(monkeypatch):
    signals = FakeSignals()
    monkeypatch.setattr(process_utils.os, "kill", signals.kill)
    monkeypatch.setattr(process_utils.os, "killpg", signals.killpg)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)
    return signals


def _terminate(processes, proc_root, **timeouts):
    timeouts = {
        "termination_timeout_s": 0,
        "poll_interval_s": 0,
        "reap_timeout_s": 0,
    } | timeouts
    return process_utils.terminate_process_groups(
        processes,  # type: ignore[arg-type]
        proc_root=proc_root,
        **timeouts,
    )


def _stat(pid, state, pgrp, comm=b"python3"):
    return b"%d (%s) %s 1 %d %d 0\n" % (pid, comm, state, pgrp, pgrp)


@pytest.mark.parametrize(
    ("exited_leaders", "expected_calls"),
    [
        pytest.param(
            set(),
            [("kill", 101, signal.SIGTERM), ("killpg", 101, 0)],
            id="leader-signaled-first",
        ),
        pytest.param(
            {LEADER_PID},
            [
                ("kill", 101, signal.SIGTERM),
                ("killpg", 101, signal.SIGTERM),
                ("killpg", 101, 0),
            ],
            id="exited-leader-falls-back-to-group",
        ),
    ],
)
def test_terminate_signals_leader_before_group(
    fake_signals, tmp_path, exited_leaders, expected_calls
):
    fake_signals.fatal_signals[LEADER_PID] = signal.SIGTERM
    fake_signals.exited_leaders = exited_leaders

    assert _terminate([FakeLeader()], tmp_path) == []
    assert fake_signals.calls == expected_calls


@pytest.mark.parametrize(
    ("fatal_signal", "stats", "expected"),
    [
        pytest.param(signal.SIGKILL, {}, [FORCED], id="sigkill-ends-group"),
        pytest.param(None, {}, [FORCED, SURVIVED], id="group-survives-sigkill"),
        pytest.param(
            None,
            {
                4481: _stat(4481, b"Z", LEADER_PID, b"VLLM::Worker) x"),
                4482: _stat(4482, b"S", 202, NON_UTF8_COMM),
            },
            [],
            id="only-zombies-left",
        ),
        pytest.param(
            None,
            {
                4481: _stat(4481, b"Z", LEADER_PID),
                4485: _stat(4485, b"R", LEADER_PID, NON_UTF8_COMM),
            },
            [FORCED, SURVIVED],
            id="live-member-beside-zombie",
        ),
        pytest.param(
            None,
            {4486: b"4486 (python3) S 1 pgrp pgrp 0\n"},
            [
                "liveness check failed for process group 101",
                FORCED,
                "post-SIGKILL liveness check failed for process group 101",
                SURVIVED,
            ],
            id="unparsable-pgrp",
        ),
    ],
)
def test_terminate_escalates_and_reports_surviving_groups(
    fake_signals, tmp_path, fatal_signal, stats, expected
):
    fake_signals.fatal_signals[LEADER_PID] = fatal_signal
    (tmp_path / "self").mkdir()
    for pid, stat in stats.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_bytes(stat)

    failures = _terminate([FakeLeader()], tmp_path)

    # Drop exception details after ':' so the table pins every message.
    assert [failure.split(":")[0] for failure in failures] == expected


def test_terminate_uses_one_deadline_and_reaps_every_process(
    fake_signals, monkeypatch, tmp_path
):
    first_process, second_process = FakeLeader(101), FakeLeader(102)
    fake_signals.fatal_signals = {101: signal.SIGTERM, 102: signal.SIGKILL}
    monotonic_values = iter([100.0, 121.0, 200.0, 201.0, 204.0])
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: next(monotonic_values))

    failures = _terminate(
        [first_process, second_process],
        tmp_path,
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == ["forced SIGKILL for process group 102 after 20s timeout"]
    assert fake_signals.calls == [
        ("kill", 101, signal.SIGTERM),
        ("kill", 102, signal.SIGTERM),
        ("killpg", 101, 0),
        ("killpg", 102, 0),
        ("killpg", 102, signal.SIGKILL),
        ("killpg", 102, 0),
    ]
    assert first_process.wait_timeouts == [4]
    assert second_process.wait_timeouts == [1]


def test_find_processes_matching_environment_uses_exact_entries(monkeypatch, tmp_path):
    environments = {
        101: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0",
        102: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=attention\0",
        103: b"AFD_E2E_RUN_ID=run-10\0AFD_E2E_PROCESS_ROLE=ffn\0",
    }
    for pid, environment in environments.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "environ").write_bytes(environment)
    (tmp_path / "self").mkdir()
    closed_pidfds: list[int] = []
    monkeypatch.setattr(
        process_utils.os, "pidfd_open", lambda pid: pid + 1000, raising=False
    )
    monkeypatch.setattr(process_utils.os, "close", closed_pidfds.append)

    matching_processes = process_utils.find_processes_matching_environment(
        RUN_FFN_ENVIRONMENT, proc_root=tmp_path
    )

    assert [(process.pid, process.pidfd) for process in matching_processes] == [
        (101, 1101)
    ]
    process_utils.close_process_identities(matching_processes)
    assert set(closed_pidfds[:2]) == {1102, 1103}
    assert closed_pidfds[-1] == 1101


def test_kill_matching_process_does_not_signal_recycled_pid(monkeypatch, tmp_path):
    (tmp_path / "101").mkdir()
    environment_path = tmp_path / "101" / "environ"
    environment_path.write_bytes(b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0")
    opened_pidfds = iter((501, 502))
    closed_pidfds: list[int] = []
    signal_calls: list[tuple[int, int]] = []

    def fake_pidfd_open(pid):
        assert pid == 101
        return next(opened_pidfds)

    def fake_pidfd_send_signal(pidfd, sig):
        signal_calls.append((pidfd, sig))
        # The matched process exits and PID 101 is immediately recycled for an
        # unrelated process. Signaling the old pidfd must not target the reuse.
        environment_path.write_bytes(b"UNRELATED_PROCESS=1\0")
        raise ProcessLookupError

    def fail_numeric_pid_signal(*_args):
        raise AssertionError("numeric PID signaled")

    monkeypatch.setattr(process_utils.os, "pidfd_open", fake_pidfd_open, raising=False)
    monkeypatch.setattr(process_utils.os, "close", closed_pidfds.append)
    monkeypatch.setattr(process_utils.os, "kill", fail_numeric_pid_signal)
    monkeypatch.setattr(
        process_utils.signal,
        "pidfd_send_signal",
        fake_pidfd_send_signal,
        raising=False,
    )
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.kill_processes_matching_environment(
        RUN_FFN_ENVIRONMENT,
        timeout_s=0,
        poll_interval_s=0,
        process_name="FFN",
        proc_root=tmp_path,
    )

    assert failures == []
    assert signal_calls == [(501, signal.SIGKILL)]
    assert closed_pidfds == [501, 502]
