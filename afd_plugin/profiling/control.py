# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Coordinate distributed captures through an authenticated HTTP API.

The standalone service discovers endpoint.py workers, calibrates clocks and
commits a shared capture window. It polls and drains sessions and persists
manifests for recovery and analysis; workers own the native backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from afd_plugin.profiling import (
    MAX_DURATION_S,
    MAX_LIGHTWEIGHT_DURATION_S,
    host_clock_id,
)

HTTP_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.1
RUNNING_POLL_INTERVAL_S = 1.0
START_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 15.0
LEASE_GRACE_S = 60.0
NANOSECONDS_PER_SECOND = 1_000_000_000
CLOCK_CALIBRATION_SAMPLES = 3
TERMINAL_STATES = frozenset(("COMPLETED", "FAILED"))


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["LIGHTWEIGHT", "FULL"]
    duration: float = Field(
        gt=0, le=MAX_LIGHTWEIGHT_DURATION_S, allow_inf_nan=False, strict=True
    )

    @model_validator(mode="after")
    def bounded_full_duration(self) -> StartRequest:
        if self.mode == "FULL" and self.duration > MAX_DURATION_S:
            raise ValueError("FULL duration must not exceed one hour")
        return self


class WorkerSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    worker_id: str = Field(min_length=1)
    role: Literal["attention", "ffn"]
    state: Literal[
        "IDLE", "PREPARED", "STARTING", "RUNNING", "STOPPING", "COMPLETED", "FAILED"
    ]
    session_id: str | None
    command_seq: int
    wall_time_ns: int
    monotonic_ns: int
    host_clock_id: str | None = None
    capabilities: list[str]
    error: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    request: StartRequest
    directory: Path
    state: str = "STARTING"
    workers: list[dict[str, Any]] = field(default_factory=list)
    roster: dict[str, tuple[str, ...]] = field(default_factory=dict)
    clock_by_worker: dict[str, dict[str, int]] = field(default_factory=dict)
    clock_method: str = "host_clock_interval_intersection"
    errors: list[str] = field(default_factory=list)
    start_at_ns: int | None = None
    data_quality: str = "INVALID"
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "owner": "afd_cam_profile_coordinator",
            "session_id": self.session_id,
            "mode": self.request.mode,
            "duration": self.request.duration,
            "state": self.state,
            "expected_workers": sum(len(workers) for workers in self.roster.values()),
            "workers": self.workers,
            "roster": self.roster,
            "start_at_ns": self.start_at_ns,
            "clock_method": self.clock_method,
            "clock_by_worker": self.clock_by_worker,
            "data_quality": self.data_quality,
            "errors": self.errors,
        }


class ProfileCoordinator:
    def __init__(
        self,
        targets: list[str],
        token: str,
        output_directory: Path,
        client: httpx.AsyncClient,
    ) -> None:
        if not token:
            raise ValueError("AFD_CAM_PROFILE_TOKEN is required")
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Profiling targets must be nonempty and unique")
        for target in targets:
            url = urlsplit(target)
            if (
                url.scheme not in ("http", "https")
                or not url.netloc
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("Profiling targets must be HTTP(S) URLs without auth")
        self.targets = [target.rstrip("/") for target in targets]
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("Profiling targets must be unique")
        self.token = token
        self.output_directory = output_directory
        self.client = client
        self.host_clock_id = host_clock_id()
        self.sessions: dict[str, Session] = {}
        self.keys: dict[str, str] = {}
        self.active: Session | None = None

    async def recover(self) -> None:
        """Reconcile owned manifests before admitting any new control requests."""
        if not self.output_directory.exists():
            return
        for directory in sorted(self.output_directory.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                if str(uuid.UUID(directory.name)) != directory.name:
                    continue
            except ValueError:
                continue
            manifest = directory / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("owner") != "afd_cam_profile_coordinator":
                continue
            if payload["session_id"] != directory.name:
                raise ValueError(
                    "Coordinator manifest session_id differs from its directory"
                )
            session = Session(
                directory.name,
                StartRequest(mode=payload["mode"], duration=payload["duration"]),
                directory,
                state=payload["state"],
                workers=[
                    WorkerSnapshot.model_validate(w).model_dump()
                    for w in payload["workers"]
                ],
                roster={
                    target: tuple(workers)
                    for target, workers in payload["roster"].items()
                },
                clock_by_worker=payload["clock_by_worker"],
                clock_method=payload["clock_method"],
                errors=payload["errors"],
                start_at_ns=payload["start_at_ns"],
                data_quality=payload["data_quality"],
            )
            self.sessions[session.session_id] = session
            if session.state in TERMINAL_STATES:
                continue
            session.errors.append(
                "Coordinator restarted; previous session was interrupted"
            )
            session.state = "STOPPING"
            self._persist_cleanup(session)
            try:
                await self._exchange(session, "status", {}, discover=True)
            except Exception as error:
                session.errors.append(str(error))
            else:
                if not any(
                    worker["session_id"] == session.session_id
                    and worker["state"] not in TERMINAL_STATES
                    for worker in session.workers
                ):
                    session.state = "FAILED"
                    session.data_quality = (
                        "PARTIAL" if session.start_at_ns else "INVALID"
                    )
                    self._persist_cleanup(session)
                    continue
            # Targets always come from administrator configuration, never a manifest.
            await self._drain(session)

    def start(self, request: StartRequest, key: str | None = None) -> Session:
        if key and key in self.keys:
            session = self.sessions[self.keys[key]]
            if session.request != request:
                raise HTTPException(409, "Idempotency-Key has a different request")
            return session
        if self.active is not None:
            raise HTTPException(409, "A profiling session is already active")
        session_id = str(uuid.uuid4())
        directory = self.output_directory / session_id
        directory.mkdir(parents=True, exist_ok=False)
        session = Session(session_id, request, directory)
        self._persist(session)
        self.sessions[session_id] = session
        if key:
            self.keys[key] = session_id
        self.active = session
        session.task = asyncio.create_task(self._run(session))
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise HTTPException(404, "Unknown profiling session") from error

    def stop(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session.state not in TERMINAL_STATES:
            session.stop_requested.set()
            session.state = "STOPPING"
            self._persist_cleanup(session)
        return session

    @staticmethod
    def _persist(session: Session) -> None:
        temporary = session.directory / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(session.snapshot(), allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(session.directory / "manifest.json")

    def _persist_cleanup(self, session: Session) -> None:
        # A full capture disk must not prevent stopping the owned backends.
        try:
            self._persist(session)
        except OSError as error:
            message = f"Manifest write failed: {error}"
            if message not in session.errors:
                session.errors.append(message)
            if session.state in TERMINAL_STATES:
                session.state = "FAILED"
                session.data_quality = "PARTIAL" if session.start_at_ns else "INVALID"

    async def _request(
        self, target: str, action: str, command: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]], int]:
        wall_before = time.time_ns()
        monotonic_before = time.monotonic_ns()
        response = await asyncio.wait_for(
            self.client.post(
                f"{target}/cam/profile/internal/{action}",
                headers={"Authorization": f"Bearer {self.token}"},
                json=command,
                timeout=HTTP_TIMEOUT_S,
                follow_redirects=False,
            ),
            HTTP_TIMEOUT_S,
        )
        elapsed = time.monotonic_ns() - monotonic_before
        wall_after = time.time_ns()
        response.raise_for_status()
        payload = response.json()
        workers = [WorkerSnapshot.model_validate(item) for item in payload["workers"]]
        if not workers:
            raise ValueError(f"{target}: empty worker roster")
        # A worker reads its clock between request send and response receipt.
        # These bounds include RPC queueing without assuming symmetric latency.
        clocks = {
            worker.worker_id: (
                worker.wall_time_ns - wall_after,
                worker.wall_time_ns - wall_before,
            )
            for worker in workers
        }
        return [worker.model_dump() for worker in workers], clocks, elapsed

    async def _exchange(
        self, session: Session, action: str, command: dict[str, Any], *, discover=False
    ) -> tuple[dict[str, tuple[int, int]], int]:
        results = await asyncio.gather(
            *(self._request(target, action, command) for target in self.targets),
            return_exceptions=True,
        )
        workers: list[dict[str, Any]] = []
        roster: dict[str, tuple[str, ...]] = {}
        clocks: dict[str, tuple[int, int]] = {}
        failures: list[str] = []
        max_rtt = 0
        for target, result in zip(self.targets, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{action} {target}: {result}")
                continue
            snapshots, timing, elapsed = result
            workers.extend(snapshots)
            roster[target] = tuple(sorted(worker["worker_id"] for worker in snapshots))
            clocks.update(timing)
            max_rtt = max(max_rtt, elapsed)
        session.workers = workers
        identities = [worker["worker_id"] for worker in workers]
        if len(set(identities)) != len(identities):
            failures.append("Duplicate worker identity in profiling roster")
        if discover:
            session.roster = roster
            if {worker["role"] for worker in workers} != {"attention", "ffn"}:
                failures.append("Profiling requires both attention and FFN workers")
        elif roster != session.roster:
            failures.append("Worker roster changed or a participant is missing")
        if action != "status" or not discover:
            for worker in workers:
                if worker["session_id"] != session.session_id:
                    failures.append(f"{worker['worker_id']}: session identity mismatch")
                if worker["error"]:
                    failures.append(f"{worker['worker_id']}: {worker['error']}")
        if failures:
            raise RuntimeError("; ".join(failures))
        return clocks, max_rtt

    async def _calibrate_clocks(self, session: Session) -> int:
        hosts = {
            worker["worker_id"]: ("host", worker["host_clock_id"])
            if worker["host_clock_id"]
            else ("worker", worker["worker_id"])
            for worker in session.workers
        }
        bounds: dict[tuple[str, str], tuple[int, int]] = {}
        if self.host_clock_id:
            bounds[("host", self.host_clock_id)] = (0, 0)
        max_rtt = 0
        for _ in range(CLOCK_CALIBRATION_SAMPLES):
            if session.stop_requested.is_set():
                return max_rtt
            samples, elapsed = await self._exchange(
                session, "status", {"session_id": session.session_id}
            )
            max_rtt = max(max_rtt, elapsed)
            for worker_id, (lower, upper) in samples.items():
                host = hosts[worker_id]
                if host in bounds:
                    previous_lower, previous_upper = bounds[host]
                    lower, upper = (
                        max(lower, previous_lower),
                        min(upper, previous_upper),
                    )
                if lower > upper:
                    raise ValueError(
                        "Inconsistent host clock samples during calibration"
                    )
                bounds[host] = (lower, upper)
        session.clock_by_worker = {}
        for worker_id, host in hosts.items():
            lower, upper = bounds[host]
            session.clock_by_worker[worker_id] = {
                "offset_ns": (lower + upper) // 2,
                "uncertainty_ns": (upper - lower + 1) // 2,
            }
        return max_rtt

    async def _run(self, session: Session) -> None:
        try:
            await self._exchange(session, "status", {}, discover=True)
            if session.stop_requested.is_set():
                return
            if any(
                session.request.mode not in w["capabilities"] for w in session.workers
            ):
                raise RuntimeError(f"Worker capability missing: {session.request.mode}")
            lease_s = session.request.duration + LEASE_GRACE_S
            _, max_rtt = await self._exchange(
                session,
                "prepare",
                {
                    "session_id": session.session_id,
                    "command_seq": 1,
                    "mode": session.request.mode,
                    "duration_s": session.request.duration,
                    "lease_s": lease_s,
                },
            )
            if any(worker["state"] != "PREPARED" for worker in session.workers):
                raise RuntimeError("Not all workers reached PREPARED")
            if not session.stop_requested.is_set():
                max_rtt = max(max_rtt, await self._calibrate_clocks(session))
            if not session.stop_requested.is_set():
                await self._commit_and_record(session, max_rtt, lease_s)
        except Exception as error:
            session.errors.append(str(error))
        finally:
            await self._drain(session)
            if self.active is session:
                self.active = None

    async def _commit_and_record(
        self, session: Session, max_rtt: int, lease_s: float
    ) -> None:
        # Leave room for a complete commit exchange using observed control RTT.
        cushion_ns = 4 * max_rtt + int(POLL_INTERVAL_S * NANOSECONDS_PER_SECOND)
        if cushion_ns / NANOSECONDS_PER_SECOND + STOP_TIMEOUT_S >= LEASE_GRACE_S:
            raise RuntimeError("Control RTT exceeds the bounded profiling lease budget")
        session.start_at_ns = time.time_ns() + cushion_ns
        start_deadline = time.monotonic() + cushion_ns / NANOSECONDS_PER_SECOND
        await self._exchange(
            session,
            "commit",
            {
                "session_id": session.session_id,
                "command_seq": 2,
                "start_at_ns": session.start_at_ns,
                "clock_by_worker": session.clock_by_worker,
                "lease_s": lease_s,
            },
        )
        self._persist(session)
        while not session.stop_requested.is_set():
            await self._exchange(session, "status", {"session_id": session.session_id})
            if any(w["state"] == "FAILED" for w in session.workers):
                raise RuntimeError("A worker failed during profiling")
            if session.state == "STARTING":
                if all(
                    w["state"] == "RUNNING" or w["details"].get("started_at_ns")
                    for w in session.workers
                ):
                    session.state = "RUNNING"
                    self._persist(session)
                elif time.monotonic() > start_deadline + START_TIMEOUT_S:
                    raise RuntimeError("Timed out waiting for all profiling backends")
            if time.monotonic() >= start_deadline + session.request.duration:
                if session.state == "STARTING":
                    raise RuntimeError(
                        "Collection ended before all backends became active"
                    )
                return
            with suppress(asyncio.TimeoutError):
                interval = (
                    RUNNING_POLL_INTERVAL_S
                    if session.state == "RUNNING"
                    and session.request.mode == "LIGHTWEIGHT"
                    else POLL_INTERVAL_S
                )
                await asyncio.wait_for(
                    session.stop_requested.wait(),
                    min(
                        interval,
                        max(
                            0,
                            start_deadline
                            + session.request.duration
                            - time.monotonic(),
                        ),
                    ),
                )

    async def _stop_workers(self, session: Session) -> None:
        try:
            await self._exchange(
                session, "stop", {"session_id": session.session_id, "command_seq": 3}
            )
        except Exception as error:
            session.errors.append(str(error))
        while True:
            try:
                await self._exchange(
                    session, "status", {"session_id": session.session_id}
                )
            except Exception as error:
                if str(error) not in session.errors:
                    session.errors.append(str(error))
            else:
                if all(
                    worker["state"] in TERMINAL_STATES for worker in session.workers
                ):
                    if any(worker["state"] == "FAILED" for worker in session.workers):
                        session.errors.append(
                            "A worker failed during profiling cleanup"
                        )
                    return
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _drain(self, session: Session) -> None:
        session.state = "STOPPING"
        self._persist_cleanup(session)
        try:
            await asyncio.wait_for(self._stop_workers(session), STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            session.errors.append("Timed out waiting for profiling cleanup")
        session.state = "FAILED" if session.errors else "COMPLETED"
        if session.state == "COMPLETED":
            incomplete = any(
                worker["details"].get(name, 0)
                for worker in session.workers
                for name in (
                    "dropped_spans",
                    "dropped_metadata",
                    "pending_spans",
                    "unresolved_spans",
                )
            ) or any(
                worker["details"].get("writer", {}).get("dropped_records", 0)
                for worker in session.workers
            )
            session.data_quality = "PARTIAL" if incomplete else "COMPLETE"
        else:
            session.data_quality = "PARTIAL" if session.start_at_ns else "INVALID"
        self._persist_cleanup(session)

    async def close(self) -> None:
        if self.active is not None:
            session = self.stop(self.active.session_id)
            task = session.task
            assert task is not None
            await task


def create_app(coordinator: ProfileCoordinator | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            service = coordinator
            if service is None:
                targets = json.loads(os.environ.get("AFD_CAM_PROFILE_TARGETS", "[]"))
                if not isinstance(targets, list) or any(
                    not isinstance(x, str) for x in targets
                ):
                    raise ValueError(
                        "AFD_CAM_PROFILE_TARGETS must be a JSON list of URLs"
                    )
                client = await stack.enter_async_context(httpx.AsyncClient())
                service = ProfileCoordinator(
                    targets,
                    os.environ.get("AFD_CAM_PROFILE_TOKEN", ""),
                    Path(os.environ.get("AFD_CAM_PROFILE_DIR", "./cam-profiles")),
                    client,
                )
            app.state.coordinator = service
            await service.recover()
            try:
                yield
            finally:
                await service.close()

    app = FastAPI(lifespan=lifespan)

    async def authenticate(request: Request) -> None:
        token = request.app.state.coordinator.token
        if not secrets.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            f"Bearer {token}".encode(),
        ):
            raise HTTPException(401, "Invalid profiling token")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.post(
        "/cam/profile/start", status_code=202, dependencies=[Depends(authenticate)]
    )
    async def start(body: StartRequest, request: Request):
        session = request.app.state.coordinator.start(
            body, request.headers.get("idempotency-key")
        )
        return session.snapshot()

    @app.get("/cam/profile/{session_id}", dependencies=[Depends(authenticate)])
    async def status(session_id: str, request: Request):
        return request.app.state.coordinator.get(session_id).snapshot()

    @app.post(
        "/cam/profile/{session_id}/stop",
        status_code=202,
        dependencies=[Depends(authenticate)],
    )
    async def stop(session_id: str, request: Request):
        return request.app.state.coordinator.stop(session_id).snapshot()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
