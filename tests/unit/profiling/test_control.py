# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
import errno
import json
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from afd_plugin.profiling import control


class FakeFleet:
    def __init__(self):
        self.offsets = {"attention": 5_000_000_000, "ffn": -7_000_000_000}
        self.workers: dict[str, list[dict]] = {
            role: [
                {
                    "worker_id": f"{role}-dp{rank}",
                    "role": role,
                    "state": "IDLE",
                    "session_id": None,
                    "command_seq": 0,
                    "wall_time_ns": 0,
                    "monotonic_ns": 0,
                    "host_clock_id": f"{role}-boot-id",
                    "host": "same-hostname",
                    "capabilities": ["LIGHTWEIGHT"],
                    "error": None,
                    "details": {},
                }
                for rank in range(2)
            ]
            for role in ("attention", "ffn")
        }
        self.calls = []
        self.start_at_ns = None
        self.prepare_error = None
        self.drop_after_commit = False
        self.stuck_stop = False
        self.delay_s = 0.0
        self.partial = None

    async def handle(self, request):
        assert request.headers["authorization"] == "Bearer test-token"
        role = request.url.host
        action = request.url.path.rsplit("/", 1)[1]
        command = json.loads(request.content)
        self.calls.append((role, action, command))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        workers = self.workers[role]
        if action == "prepare":
            for worker in workers:
                worker.update(
                    session_id=command["session_id"], command_seq=1, state="PREPARED"
                )
                if role == self.prepare_error:
                    worker.update(state="FAILED", error="Resource unavailable")
        elif action == "commit":
            self.start_at_ns = command["start_at_ns"]
            for worker in workers:
                worker.update(command_seq=2, state="STARTING")
        elif action == "stop":
            for worker in workers:
                worker.update(
                    session_id=command["session_id"],
                    command_seq=3,
                    state="STOPPING" if self.stuck_stop else "COMPLETED",
                    error=None,
                )
                if self.partial:
                    if self.partial == "writer":
                        worker["details"]["writer"] = {"dropped_records": 1}
                    else:
                        worker["details"][self.partial] = 1
        elif self.start_at_ns and time.time_ns() >= self.start_at_ns:
            for worker in workers:
                if worker["state"] == "STARTING":
                    worker["state"] = "RUNNING"
                    worker["details"]["started_at_ns"] = time.time_ns()
        for worker in workers:
            worker["wall_time_ns"] = time.time_ns() + self.offsets[role]
            worker["monotonic_ns"] = time.monotonic_ns()
        if self.drop_after_commit and self.start_at_ns and role == "ffn":
            workers = workers[:1]
        return httpx.Response(
            200, json={"workers": workers, "wall_time_ns": time.time_ns()}
        )


@pytest.fixture(autouse=True)
def short_deadlines(monkeypatch):
    monkeypatch.setattr(control, "POLL_INTERVAL_S", 0.002)
    monkeypatch.setattr(control, "START_TIMEOUT_S", 0.05)
    monkeypatch.setattr(control, "STOP_TIMEOUT_S", 0.025)


def make_coordinator(tmp_path, fleet):
    client = httpx.AsyncClient(transport=httpx.MockTransport(fleet.handle))
    coordinator = control.ProfileCoordinator(
        ["http://attention", "http://ffn"], "test-token", tmp_path, client
    )
    return coordinator, client


@pytest.mark.parametrize("mode", ["LIGHTWEIGHT", "FULL"])
def test_session_covers_all_dp_workers_and_adjusts_each_host_clock(tmp_path, mode):
    async def run():
        fleet = FakeFleet()
        for workers in fleet.workers.values():
            for worker in workers:
                worker["capabilities"] = ["LIGHTWEIGHT", "FULL"]
                worker["details"]["native_profiler"] = {
                    "trace_dir": f"cam-profiles/{worker['worker_id']}/{mode.lower()}",
                    "state": "COMPLETED",
                }
        fleet.delay_s = 0.002
        coordinator, client = make_coordinator(tmp_path, fleet)
        async with client:
            session = coordinator.start(control.StartRequest(mode=mode, duration=0.01))
            await asyncio.wait_for(session.task, 2.0)
        assert session.state == "COMPLETED"
        assert session.snapshot()["expected_workers"] == 4
        assert session.data_quality == "COMPLETE"
        assert coordinator.active is None
        for _role, action, command in fleet.calls:
            if action == "prepare":
                assert command["mode"] == mode
            if action != "commit":
                continue
            assert set(command) == {
                "session_id",
                "command_seq",
                "start_at_ns",
                "clock_by_worker",
                "lease_s",
            }
            assert command["start_at_ns"] == session.start_at_ns
            assert len(command["clock_by_worker"]) == 4
            for worker_id, clock in command["clock_by_worker"].items():
                worker_role = worker_id.split("-")[0]
                assert (
                    abs(clock["offset_ns"] - fleet.offsets[worker_role])
                    <= clock["uncertainty_ns"]
                )
        manifest = json.loads((session.directory / "manifest.json").read_text())
        assert manifest["state"] == "COMPLETED"
        assert manifest["mode"] == mode
        assert all(
            w["details"]["native_profiler"]["trace_dir"] for w in manifest["workers"]
        )
        assert manifest["clock_method"] == "host_clock_interval_intersection"
        assert not (session.directory / "manifest.json.tmp").exists()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["LIGHTWEIGHT", "FULL"])
def test_local_workers_keep_zero_offset_despite_rpc_delays(tmp_path, mode):
    async def run():
        fleet = FakeFleet()
        fleet.offsets = {"attention": 0, "ffn": 0}
        fleet.delay_s = 0.003
        coordinator, client = make_coordinator(tmp_path, fleet)
        coordinator.host_clock_id = "local-boot-id"
        for workers in fleet.workers.values():
            for worker in workers:
                worker["host_clock_id"] = coordinator.host_clock_id
                worker["capabilities"] = [mode]
        async with client:
            session = coordinator.start(control.StartRequest(mode=mode, duration=0.005))
            await asyncio.wait_for(session.task, 2.0)
        assert session.state == "COMPLETED", session.errors
        assert len(session.clock_by_worker) == 4
        assert all(
            clock == {"offset_ns": 0, "uncertainty_ns": 0}
            for clock in session.clock_by_worker.values()
        )

    asyncio.run(run())


def test_request_bounds_include_asymmetric_worker_queueing(tmp_path, monkeypatch):
    async def run():
        fleet = FakeFleet()
        coordinator, client = make_coordinator(tmp_path, fleet)
        origin = 1_700_000_000_000_000_000
        walls = iter([origin, origin + 101])
        monotonic = iter([1_000, 1_101])
        monkeypatch.setattr(
            control,
            "time",
            SimpleNamespace(
                time_ns=lambda: next(walls), monotonic_ns=lambda: next(monotonic)
            ),
        )

        async def response(request):
            # Two workers sample near opposite ends of the same RPC round trip.
            workers = fleet.workers["attention"]
            for worker, delay in zip(workers, [7, 96], strict=True):
                worker["wall_time_ns"] = origin - 1_000 + delay
            return httpx.Response(200, json={"workers": workers})

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as probe:
            coordinator.client = probe
            _, bounds, elapsed = await coordinator._request(
                "http://attention", "status", {}
            )
        await client.aclose()
        assert elapsed == 101
        assert bounds == {
            "attention-dp0": (-1_094, -993),
            "attention-dp1": (-1_005, -904),
        }
        assert all(lower <= -1_000 <= upper for lower, upper in bounds.values())

    asyncio.run(run())


def test_host_bounds_intersect_all_workers_and_probes_with_integer_rounding(
    tmp_path, monkeypatch
):
    async def run():
        fleet = FakeFleet()
        coordinator, client = make_coordinator(tmp_path, fleet)
        workers = fleet.workers["attention"]
        session = control.Session(
            str(uuid.uuid4()),
            control.StartRequest(mode="LIGHTWEIGHT", duration=1),
            tmp_path,
            workers=workers,
        )
        probes = iter(
            [
                {"attention-dp0": (-100, 0), "attention-dp1": (-90, -10)},
                {"attention-dp0": (-75, -30), "attention-dp1": (-60, -35)},
                {"attention-dp0": (-45, -34), "attention-dp1": (-43, -36)},
            ]
        )

        async def exchange(current, action, command):
            assert current is session and action == "status"
            return next(probes), 100

        monkeypatch.setattr(coordinator, "_exchange", exchange)
        async with client:
            await coordinator._calibrate_clocks(session)
        # Feasible offsets are [-43, -36]; the integer bound must include both ends.
        assert list(session.clock_by_worker.values()) == [
            {"offset_ns": -40, "uncertainty_ns": 4},
            {"offset_ns": -40, "uncertainty_ns": 4},
        ]

    asyncio.run(run())


@pytest.mark.parametrize("local_host", [False, True])
def test_inconsistent_host_samples_roll_back_without_committing(tmp_path, local_host):
    async def run():
        fleet = FakeFleet()
        coordinator, client = make_coordinator(tmp_path, fleet)
        coordinator.host_clock_id = "local-boot-id"
        for workers in fleet.workers.values():
            for worker in workers:
                worker["host_clock_id"] = (
                    coordinator.host_clock_id if local_host else "same-boot-id"
                )
        async with client:
            session = coordinator.start(
                control.StartRequest(mode="LIGHTWEIGHT", duration=1)
            )
            await asyncio.wait_for(session.task, 2.0)
        assert session.state == "FAILED"
        assert session.data_quality == "INVALID"
        assert any("Inconsistent host clock" in error for error in session.errors)
        assert all(action != "commit" for _, action, _ in fleet.calls)
        assert {role for role, action, _ in fleet.calls if action == "stop"} == {
            "attention",
            "ffn",
        }

    asyncio.run(run())


def test_prepare_failure_rolls_back_every_target_and_never_commits(tmp_path):
    async def run():
        fleet = FakeFleet()
        fleet.prepare_error = "ffn"
        coordinator, client = make_coordinator(tmp_path, fleet)
        async with client:
            session = coordinator.start(
                control.StartRequest(mode="LIGHTWEIGHT", duration=1)
            )
            await session.task
        assert session.state == "FAILED"
        assert session.data_quality == "INVALID"
        assert any("Resource unavailable" in error for error in session.errors)
        assert {role for role, action, _ in fleet.calls if action == "stop"} == {
            "attention",
            "ffn",
        }
        assert all(action != "commit" for _, action, _ in fleet.calls)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["duplicate", "missing", "stop_timeout", "capability"]
)
def test_partial_or_invalid_roster_cannot_complete(tmp_path, failure):
    async def run():
        fleet = FakeFleet()
        if failure == "duplicate":
            fleet.workers["ffn"][0]["worker_id"] = "attention-dp0"
        elif failure == "missing":
            fleet.drop_after_commit = True
        elif failure == "stop_timeout":
            fleet.stuck_stop = True
        else:
            fleet.workers["ffn"][0]["capabilities"] = []
        coordinator, client = make_coordinator(tmp_path, fleet)
        async with client:
            session = coordinator.start(
                control.StartRequest(mode="LIGHTWEIGHT", duration=0.005)
            )
            await asyncio.wait_for(session.task, 2.0)
        assert session.state == "FAILED"
        assert session.errors
        assert coordinator.active is None

    asyncio.run(run())


@pytest.mark.parametrize("dropped", ["unresolved_spans", "dropped_metadata", "writer"])
def test_stop_is_idempotent_and_closed_spans_can_be_partial(tmp_path, dropped):
    async def run():
        fleet = FakeFleet()
        fleet.partial = dropped
        coordinator, client = make_coordinator(tmp_path, fleet)
        async with client:
            request = control.StartRequest(mode="LIGHTWEIGHT", duration=100)
            session = coordinator.start(request, "retry-key")
            assert coordinator.start(request, "retry-key") is session
            with pytest.raises(HTTPException) as conflict:
                coordinator.start(request)
            assert conflict.value.status_code == 409
            with pytest.raises(HTTPException) as conflict:
                coordinator.start(
                    control.StartRequest(mode="LIGHTWEIGHT", duration=1), "retry-key"
                )
            assert conflict.value.status_code == 409
            while session.state == "STARTING":
                await asyncio.sleep(0.001)
            assert coordinator.stop(session.session_id).state == "STOPPING"
            assert coordinator.stop(session.session_id).state == "STOPPING"
            await asyncio.wait_for(session.task, 2.0)
            assert coordinator.stop(session.session_id).state == "COMPLETED"
        assert session.data_quality == "PARTIAL"
        assert len([call for call in fleet.calls if call[1] == "stop"]) == 2

    asyncio.run(run())


def test_public_routes_require_auth_and_reject_invalid_requests(tmp_path):
    fleet = FakeFleet()
    coordinator, outbound = make_coordinator(tmp_path, fleet)
    app = control.create_app(coordinator)
    auth = {"Authorization": "Bearer test-token"}
    with TestClient(app) as client:
        assert (
            client.post(
                "/cam/profile/start", json={"mode": "LIGHTWEIGHT", "duration": 1}
            ).status_code
            == 401
        )
        assert client.get("/cam/profile/missing").status_code == 401
        assert client.post("/cam/profile/missing/stop").status_code == 401
        assert client.get("/cam/profile/missing", headers=auth).status_code == 404
        for body in (
            {"mode": "BAD", "duration": 1},
            {"mode": "LIGHTWEIGHT", "duration": 0},
            {"mode": "LIGHTWEIGHT", "duration": 604801},
            {"mode": "LIGHTWEIGHT", "duration": 1, "path": "/tmp/untrusted"},
        ):
            assert (
                client.post("/cam/profile/start", json=body, headers=auth).status_code
                == 400
            )
        response = client.post(
            "/cam/profile/start",
            json={"mode": "LIGHTWEIGHT", "duration": 1},
            headers=auth,
        )
        assert response.status_code == 202
        session_id = response.json()["session_id"]
        assert client.get(f"/cam/profile/{session_id}", headers=auth).status_code == 200
        assert (
            client.post(f"/cam/profile/{session_id}/stop", headers=auth).status_code
            == 202
        )
    asyncio.run(outbound.aclose())


def save_previous_session(coordinator, fleet, *, state="RUNNING"):
    session_id = str(uuid.uuid4())
    directory = coordinator.output_directory / session_id
    directory.mkdir()
    for workers in fleet.workers.values():
        for worker in workers:
            worker.update(session_id=session_id, state=state, command_seq=2)
    session = control.Session(
        session_id,
        control.StartRequest(mode="LIGHTWEIGHT", duration=10),
        directory,
        state=state,
        workers=[worker for workers in fleet.workers.values() for worker in workers],
        roster={
            f"http://{role}": tuple(worker["worker_id"] for worker in workers)
            for role, workers in fleet.workers.items()
        },
        start_at_ns=time.time_ns(),
    )
    coordinator._persist(session)
    return session


@pytest.mark.parametrize("stalled_action", ["stop", "status"])
def test_cleanup_deadline_bounds_stalled_worker_rpc(
    tmp_path, monkeypatch, stalled_action
):
    async def run():
        coordinator, client = make_coordinator(tmp_path, FakeFleet())
        original = coordinator._request

        async def request(target, action, command):
            if action == stalled_action:
                await asyncio.Event().wait()
            return await original(target, action, command)

        monkeypatch.setattr(coordinator, "_request", request)
        async with client:
            session = control.Session(
                str(uuid.uuid4()),
                control.StartRequest(mode="LIGHTWEIGHT", duration=1),
                tmp_path,
                start_at_ns=time.time_ns(),
            )
            await asyncio.wait_for(coordinator._drain(session), 0.2)
        assert session.state == "FAILED"
        assert "Timed out waiting for profiling cleanup" in session.errors

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["stop", "close", "recover"])
def test_manifest_quota_failure_does_not_interrupt_cleanup(
    tmp_path, monkeypatch, operation
):
    async def run():
        fleet = FakeFleet()
        coordinator, client = make_coordinator(tmp_path, fleet)
        async with client:
            if operation == "recover":
                session = save_previous_session(coordinator, fleet)
            else:
                session = coordinator.start(
                    control.StartRequest(mode="LIGHTWEIGHT", duration=60)
                )
                while session.state == "STARTING":
                    await asyncio.sleep(0.001)
                assert session.state == "RUNNING"

            def fail_manifest(_session):
                raise OSError(errno.EDQUOT, "Disk quota exceeded")

            monkeypatch.setattr(coordinator, "_persist", fail_manifest)
            if operation == "recover":
                await coordinator.recover()
                session = coordinator.get(session.session_id)
            elif operation == "close":
                await coordinator.close()
            else:
                assert coordinator.stop(session.session_id) is session
                await asyncio.wait_for(session.task, 1)
        assert session.state == "FAILED"
        assert session.data_quality == "PARTIAL"
        assert any("Manifest write failed" in error for error in session.errors)
        assert {role for role, action, _ in fleet.calls if action == "stop"} == {
            "attention",
            "ffn",
        }
        assert all(worker["state"] == "COMPLETED" for worker in session.workers)
        assert coordinator.active is None

    asyncio.run(run())


def test_restart_does_not_stop_workers_owned_by_another_session(tmp_path):
    async def run():
        fleet = FakeFleet()
        coordinator, client = make_coordinator(tmp_path, fleet)
        previous = save_previous_session(coordinator, fleet)
        other_session_id = str(uuid.uuid4())
        for workers in fleet.workers.values():
            for worker in workers:
                worker["session_id"] = other_session_id
        async with client:
            await coordinator.recover()
        assert coordinator.get(previous.session_id).state == "FAILED"
        assert all(action == "status" for _, action, _ in fleet.calls)
        assert all(
            worker["state"] == "RUNNING"
            for workers in fleet.workers.values()
            for worker in workers
        )

    asyncio.run(run())
