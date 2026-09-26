# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import signal

from tests.e2e import process_utils


def test_terminate_process_groups_signals_the_leader_before_the_group(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        nonlocal group_alive
        leader_signal_calls.append((pid, sig))
        group_alive = False

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == signal.SIGTERM:
            group_alive = False
        if sig == 0 and not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert leader_signal_calls == [(101, signal.SIGTERM)]
    assert group_signal_calls == [(101, 0)]


def test_terminate_process_groups_cleans_children_after_leader_exits(monkeypatch):
    class ExitedLeader:
        pid = 101

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def leader_is_gone(*_args):
        raise ProcessLookupError

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == signal.SIGTERM:
            group_alive = False
        elif not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", leader_is_gone, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [ExitedLeader()],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert group_alive is False


def test_terminate_process_groups_reports_force_kill_escalation(monkeypatch):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == 9:
            group_alive = False

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert not any("still alive after SIGKILL" in failure for failure in failures)


def test_terminate_process_groups_reports_a_group_that_survives_sigkill(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_liveness_checks = 0

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_liveness_checks
        if sig == 0:
            group_liveness_checks += 1

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)
    assert group_liveness_checks >= 2


def _write_proc_stat(proc_root, pid, *, state, pgrp, comm: str | bytes = "python3"):
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    if isinstance(comm, str):
        comm = comm.encode()
    (process_dir / "stat").write_bytes(
        b"%d (%s) %s 1 %d %d 0\n" % (pid, comm, state.encode(), pgrp, pgrp),
    )


def _patch_group_that_never_disappears(monkeypatch):
    monkeypatch.setattr(process_utils.os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(process_utils.os, "killpg", lambda _pid, _sig: None)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)


class _UnreapedLeader:
    pid = 101

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_terminate_process_groups_ignores_a_group_of_only_zombies(
    monkeypatch,
    tmp_path,
):
    _patch_group_that_never_disappears(monkeypatch)
    _write_proc_stat(tmp_path, 4481, state="Z", pgrp=101, comm="VLLM::Worker) x")
    _write_proc_stat(tmp_path, 4482, state="S", pgrp=202)
    (tmp_path / "self").mkdir()

    failures = process_utils.terminate_process_groups(
        [_UnreapedLeader()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert failures == []


def test_terminate_process_groups_still_reports_a_live_member_beside_zombies(
    monkeypatch,
    tmp_path,
):
    _patch_group_that_never_disappears(monkeypatch)
    _write_proc_stat(tmp_path, 4481, state="Z", pgrp=101)
    _write_proc_stat(tmp_path, 4483, state="D", pgrp=101)

    failures = process_utils.terminate_process_groups(
        [_UnreapedLeader()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)


def test_terminate_process_groups_tolerates_a_non_utf8_comm_outside_the_group(
    monkeypatch,
    tmp_path,
):
    # Any process may set a comm that is not valid UTF-8 via prctl(PR_SET_NAME).
    _patch_group_that_never_disappears(monkeypatch)
    _write_proc_stat(tmp_path, 4481, state="Z", pgrp=101)
    _write_proc_stat(tmp_path, 4484, state="S", pgrp=303, comm=b"bad\xff\xfe")

    failures = process_utils.terminate_process_groups(
        [_UnreapedLeader()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert failures == []


def test_terminate_process_groups_counts_a_non_utf8_comm_member_as_alive(
    monkeypatch,
    tmp_path,
):
    _patch_group_that_never_disappears(monkeypatch)
    _write_proc_stat(tmp_path, 4481, state="Z", pgrp=101)
    _write_proc_stat(tmp_path, 4485, state="R", pgrp=101, comm=b"bad\xff\xfe")

    failures = process_utils.terminate_process_groups(
        [_UnreapedLeader()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)


def test_terminate_process_groups_reports_an_unparsable_process_group(
    monkeypatch,
    tmp_path,
):
    _patch_group_that_never_disappears(monkeypatch)
    process_dir = tmp_path / "4486"
    process_dir.mkdir()
    (process_dir / "stat").write_bytes(b"4486 (python3) S 1 pgrp pgrp 0\n")

    failures = process_utils.terminate_process_groups(
        [_UnreapedLeader()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    assert any(failure.startswith("liveness check failed") for failure in failures)
    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("post-SIGKILL liveness check failed" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)


def test_find_processes_matching_environment_uses_exact_entries(
    monkeypatch,
    tmp_path,
):
    environments = {
        101: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0",
        102: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=attention\0",
        103: b"AFD_E2E_RUN_ID=run-10\0AFD_E2E_PROCESS_ROLE=ffn\0",
    }
    for pid, environment in environments.items():
        process_dir = tmp_path / str(pid)
        process_dir.mkdir()
        (process_dir / "environ").write_bytes(environment)
    (tmp_path / "self").mkdir()
    closed_pidfds: list[int] = []
    monkeypatch.setattr(
        process_utils.os,
        "pidfd_open",
        lambda pid: pid + 1000,
    )
    monkeypatch.setattr(
        process_utils.os,
        "close",
        closed_pidfds.append,
    )

    matching_processes = process_utils.find_processes_matching_environment(
        {
            "AFD_E2E_RUN_ID": "run-1",
            "AFD_E2E_PROCESS_ROLE": "ffn",
        },
        proc_root=tmp_path,
    )

    assert [process.pid for process in matching_processes] == [101]
    assert [process.pidfd for process in matching_processes] == [1101]
    process_utils.close_process_identities(matching_processes)
    assert set(closed_pidfds[:2]) == {1102, 1103}
    assert closed_pidfds[-1] == 1101


def test_kill_matching_process_does_not_signal_recycled_pid(monkeypatch, tmp_path):
    process_dir = tmp_path / "101"
    process_dir.mkdir()
    environment_path = process_dir / "environ"
    environment_path.write_bytes(
        b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0",
    )
    opened_pidfds = iter((501, 502))
    closed_pidfds: list[int] = []
    signal_calls: list[tuple[int, int]] = []

    def fake_pidfd_open(pid):
        assert pid == 101
        return next(opened_pidfds)

    def fake_close(pidfd):
        closed_pidfds.append(pidfd)

    def fake_pidfd_send_signal(pidfd, sig):
        signal_calls.append((pidfd, sig))
        # The matched process exits and PID 101 is immediately recycled for an
        # unrelated process. Signaling the old pidfd must not target the reuse.
        environment_path.write_bytes(b"UNRELATED_PROCESS=1\0")
        raise ProcessLookupError

    def fail_numeric_pid_signal(*_args):
        raise AssertionError("numeric PID signaled")

    monkeypatch.setattr(process_utils.os, "pidfd_open", fake_pidfd_open)
    monkeypatch.setattr(process_utils.os, "close", fake_close)
    monkeypatch.setattr(process_utils.os, "kill", fail_numeric_pid_signal)
    monkeypatch.setattr(
        process_utils.signal,
        "pidfd_send_signal",
        fake_pidfd_send_signal,
    )
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.kill_processes_matching_environment(
        {
            "AFD_E2E_RUN_ID": "run-1",
            "AFD_E2E_PROCESS_ROLE": "ffn",
        },
        timeout_s=0,
        poll_interval_s=0,
        process_name="FFN",
        proc_root=tmp_path,
    )

    assert failures == []
    assert signal_calls == [(501, signal.SIGKILL)]
    assert closed_pidfds == [501, 502]


def test_terminate_process_groups_uses_one_deadline_and_reaps_every_process(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = 0

    first_process = FakeProcess(101)
    second_process = FakeProcess(102)
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        leader_signal_calls.append((pid, sig))
        if pid == first_process.pid and sig == signal.SIGTERM:
            first_process.returncode = 0

    def fake_killpg(pid, sig):
        group_signal_calls.append((pid, sig))
        process = first_process if pid == first_process.pid else second_process
        if sig == 0 and process.returncode is not None:
            raise ProcessLookupError
        if pid == second_process.pid and sig == signal.SIGKILL:
            second_process.returncode = 0

    monotonic_values = iter([100.0, 121.0, 200.0, 201.0, 204.0])
    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        process_utils.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    failures = process_utils.terminate_process_groups(
        [first_process, second_process],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert len(failures) == 1
    assert "forced SIGKILL" in failures[0]
    assert leader_signal_calls == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
    ]
    assert group_signal_calls == [
        (101, 0),
        (102, 0),
        (102, 9),
        (102, 0),
    ]
    assert first_process.wait_calls == [4]
    assert second_process.wait_calls == [1]
