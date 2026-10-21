# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Own one worker's capture state, command ordering and admission window.

Worker RPCs prepare/commit/stop leased captures; the control thread enforces
deadlines and writes summaries through storage.py. The model thread advances
native.py; tracing.py controls LIGHTWEIGHT admission. Native failures block reuse.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.profiling import (
    MAX_DURATION_S,
    MAX_LIGHTWEIGHT_DURATION_S,
    host_clock_id,
)
from afd_plugin.profiling.storage import JsonRecordWriter

if TYPE_CHECKING:
    import torch

    from afd_plugin.profiling.native import RuntimeProfiler
    from afd_plugin.profiling.npu import FFNDeviceWaiter
    from afd_plugin.profiling.tracing import DeviceTracer

LEASE_GRACE_S = 60
DRAIN_TIMEOUT_S = 2
PROFILER_OPERATION_TIMEOUT_S = 10


def _bounded_seconds(value: int | float | None, field: str, maximum: float) -> float:
    if value is None or type(value) not in (int, float) or not 0 < value <= maximum:
        raise ValueError(f"{field} must be in (0, {maximum}]")
    return float(value)


@dataclass(slots=True)
class Capture:
    session_id: str
    generation: int
    writer: JsonRecordWriter
    start_ns: int = 0
    deadline_ns: int = 0
    start_host_ns: int = 0
    clock_offset_ns: int = 0
    clock_uncertainty_ns: int = 0
    accepting: bool = False
    stop_requested: bool = False
    drain_deadline_ns: int = 0
    error: str | None = None
    started_at_ns: int | None = None
    flush_deadline_ns: int = 0
    mode: str = "LIGHTWEIGHT"
    profiler: RuntimeProfiler | None = None
    profiler_deadline_ns: int = 0

    def admits(self) -> bool:
        now = time.monotonic_ns()
        return (
            self.mode == "LIGHTWEIGHT"
            and self.accepting
            and self.start_ns <= now < self.deadline_ns
        )


class WorkerSession:
    """Own runtime control independently of either profiling backend."""

    def __init__(
        self,
        worker_id: str,
        role: str,
        output_dir: Path,
        tracer: DeviceTracer | None,
        identity: dict,
        *,
        tracer_factory: Callable[[], DeviceTracer] | None = None,
        profiler_factory: Callable[[Path, str], RuntimeProfiler] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.role = role
        self.output_dir = output_dir
        self.tracer = tracer
        self.device_waiter: FFNDeviceWaiter | None = None
        self.identity = identity
        self.host_clock_id = host_clock_id()
        self.capture: Capture | None = None
        self.state = "IDLE"
        self.command_seq = 0
        self._generation = 0
        self._duration_s = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._last_command: tuple[str, dict] | None = None
        self._profiler_factory = profiler_factory
        self._tracer_factory = tracer_factory
        self._profiler_poisoned = False

    def start(self, device: torch.device) -> None:
        threading.Thread(
            target=self._run, args=(device,), name="afd-cam-control", daemon=True
        ).start()

    def _run(self, device: torch.device) -> None:
        import torch_npu

        torch_npu.npu.set_device(device)
        while True:
            self._wake.clear()
            try:
                self.tick()
            except Exception as exc:
                with self._lock:
                    capture = self.capture
                    if capture is not None:
                        capture.error = f"{type(exc).__name__}: {exc}"
                        self._stop(capture)
                        capture.drain_deadline_ns = time.monotonic_ns()
            with self._lock:
                timeout = self._next_timeout()
            self._wake.wait(timeout)

    def _next_timeout(self) -> float | None:
        """Called under the session lock; state changes wake the control thread."""
        capture = self.capture
        if capture is None or self.state in {"COMPLETED", "FAILED"}:
            return None
        deadlines = []
        now = time.monotonic_ns()
        if not capture.stop_requested:
            deadlines.append(capture.deadline_ns)
            if self.state == "STARTING" and now < capture.start_ns:
                deadlines.append(capture.start_ns)
        if (
            capture.profiler is not None
            and not capture.profiler.finished
            and not self._profiler_poisoned
            and capture.profiler_deadline_ns
            and (capture.started_at_ns is None or capture.stop_requested)
        ):
            deadlines.append(capture.profiler_deadline_ns)
        if capture.stop_requested:
            if now < capture.drain_deadline_ns:
                deadlines.append(capture.drain_deadline_ns)
            if capture.flush_deadline_ns:
                deadlines.append(capture.flush_deadline_ns)
        return max(0, (min(deadlines) - now) / 1e9) if deadlines else None

    @property
    def profiler_reusable(self) -> bool:
        with self._lock:
            profiler = self.capture.profiler if self.capture is not None else None
            return not self._profiler_poisoned and (
                profiler is None or not profiler.busy
            )

    def _command(self, name: str, command: dict) -> bool:
        # RPC payloads cross a process boundary; reject malformed identities.
        if not isinstance(command, dict):
            raise ValueError("command must be an object")
        session_id = command.get("session_id")
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
            raise ValueError("session_id must be a canonical UUID")
        sequence = command.get("command_seq")
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("command_seq must be a positive integer")
        if self.capture is not None and self.capture.session_id == session_id:
            if sequence == self.command_seq and self._last_command == (name, command):
                return False
            if sequence <= self.command_seq:
                raise ValueError("stale or conflicting command")
        elif name != "prepare":
            raise ValueError("unknown session")
        return True

    def prepare(self, command: dict, *, backend_busy: bool) -> dict:
        with self._lock:
            if not self._command("prepare", command):
                return self._status()
            if self.state not in {"IDLE", "COMPLETED", "FAILED"}:
                raise ValueError("a profiling session is already active")
            if self._profiler_poisoned or (
                self.capture is not None
                and self.capture.profiler is not None
                and self.capture.profiler.busy
            ):
                raise ValueError(
                    "native profiling backend is unavailable; restart worker"
                )
            if self.tracer is not None and self.tracer.pending_count():
                raise ValueError("previous generation still owns device resources")
            if (
                self.capture is not None
                and not self.capture.writer.snapshot()["closed"]
            ):
                raise ValueError("previous writer has not closed")
            if backend_busy:
                raise ValueError("an existing NPU profiler is active")
            mode = command.get("mode")
            if mode not in {"LIGHTWEIGHT", "FULL"}:
                raise ValueError("mode must be LIGHTWEIGHT or FULL")
            if self._profiler_factory is None:
                raise ValueError("native profiling backend is not configured")
            duration = _bounded_seconds(
                command.get("duration_s"),
                "duration_s",
                MAX_LIGHTWEIGHT_DURATION_S if mode == "LIGHTWEIGHT" else MAX_DURATION_S,
            )
            lease = _bounded_seconds(
                command.get("lease_s", duration + LEASE_GRACE_S),
                "lease_s",
                duration + LEASE_GRACE_S,
            )
            if mode == "LIGHTWEIGHT":
                if self.tracer is None:
                    if self._tracer_factory is None:
                        raise ValueError("LIGHTWEIGHT backend is not configured")
                    self.tracer = self._tracer_factory()
                assert self.tracer is not None
                self.tracer.session = self
            session_dir = self.output_dir / command["session_id"]
            session_dir.mkdir(parents=True, exist_ok=True)
            profiler = self._profiler_factory(
                session_dir / self.worker_id / mode.lower(), mode
            )
            writer = JsonRecordWriter(
                session_dir / f"{self.worker_id}.jsonl", on_change=self._wake.set
            )
            self._generation += 1
            self.capture = Capture(
                command["session_id"],
                self._generation,
                writer,
                deadline_ns=time.monotonic_ns() + int(lease * 1e9),
                mode=mode,
                profiler=profiler,
            )
            self._duration_s = duration
            if mode == "LIGHTWEIGHT":
                assert self.tracer is not None
                self.tracer.begin_capture()
            self.state = "PREPARED"
            self.command_seq = command["command_seq"]
            self._last_command = ("prepare", dict(command))
            self._wake.set()
            return self._status()

    def commit(self, command: dict) -> dict:
        with self._lock:
            if not self._command("commit", command):
                return self._status()
            if self.state != "PREPARED":
                raise ValueError("session is not prepared")
            capture = self.capture
            assert capture is not None
            start_host_ns = command.get("start_at_ns")
            if type(start_host_ns) is not int:
                raise ValueError("start_at_ns must be an integer")
            clocks = command.get("clock_by_worker")
            if not isinstance(clocks, dict) or not isinstance(
                clock := clocks.get(self.worker_id), dict
            ):
                raise ValueError("missing worker clock estimate")
            offset = clock.get("offset_ns")
            uncertainty = clock.get("uncertainty_ns")
            if (
                type(offset) is not int
                or type(uncertainty) is not int
                or uncertainty < 0
            ):
                raise ValueError("invalid worker clock estimate")
            worker_start = start_host_ns + offset
            now_wall, now_mono = time.time_ns(), time.monotonic_ns()
            start_delay_ns = worker_start - now_wall
            if not 0 < start_delay_ns <= int(LEASE_GRACE_S * 1e9):
                raise ValueError("start_at_ns must be within the next 60 seconds")
            lease = _bounded_seconds(
                command.get("lease_s", self._duration_s + LEASE_GRACE_S),
                "lease_s",
                self._duration_s + LEASE_GRACE_S,
            )
            if int(lease * 1e9) <= start_delay_ns:
                raise ValueError("commit lease_s must extend past start_at_ns")
            capture.start_host_ns = start_host_ns
            capture.clock_offset_ns = offset
            capture.clock_uncertainty_ns = uncertainty
            capture.start_ns = now_mono + start_delay_ns
            capture.deadline_ns = min(
                now_mono + int(lease * 1e9),
                capture.start_ns + int(self._duration_s * 1e9),
            )
            capture.writer.submit(
                {
                    "record_type": "SESSION",
                    "schema_version": 1,
                    "session_id": capture.session_id,
                    "worker_id": self.worker_id,
                    "start_host_ns": start_host_ns,
                    "generation": capture.generation,
                    "identity": self.identity,
                    "host_clock_id": self.host_clock_id,
                    "requires_summary": True,
                    "mode": capture.mode,
                    "view": "pipeline" if capture.mode == "LIGHTWEIGHT" else "full",
                    "role": self.role,
                    "full_trace_dir": str(capture.profiler.trace_dir)
                    if capture.mode == "FULL" and capture.profiler is not None
                    else None,
                    "native_trace_dir": f"{self.worker_id}/lightweight"
                    if capture.mode == "LIGHTWEIGHT"
                    else None,
                    "clock_offset_ns": capture.clock_offset_ns,
                    "clock_uncertainty_ns": capture.clock_uncertainty_ns,
                }
            )
            capture.accepting = False
            self.state = "STARTING"
            self.command_seq = command["command_seq"]
            self._last_command = ("commit", dict(command))
            self._wake.set()
            return self._status()

    def stop(self, command: dict) -> dict:
        with self._lock:
            if not self._command("stop", command):
                return self._status()
            capture = self.capture
            assert capture is not None
            if self.state not in {"COMPLETED", "FAILED"}:
                self._stop(capture)
            self.command_seq = command["command_seq"]
            self._last_command = ("stop", dict(command))
            self._wake.set()
            return self._status()

    def _stop(self, capture: Capture) -> None:
        if capture.mode == "LIGHTWEIGHT":
            assert self.tracer is not None
            self.tracer.close_admission(capture)
        else:
            capture.accepting = False
        if not capture.stop_requested:
            capture.stop_requested = True
            capture.drain_deadline_ns = time.monotonic_ns() + int(DRAIN_TIMEOUT_S * 1e9)
            if capture.profiler is not None:
                capture.profiler.stop()
                capture.profiler_deadline_ns = time.monotonic_ns() + int(
                    PROFILER_OPERATION_TIMEOUT_S * 1e9
                )
        self.state = "STOPPING"

    def status(self, command: dict | None = None) -> dict:
        with self._lock:
            return self._status()

    def advance_profiler(
        self, *, allow_stop: bool = True, stop_ready: Callable[[], bool] | None = None
    ) -> None:
        """Advance native lifecycle on its model thread, outside the session lock."""
        capture = self.capture
        if (
            capture is None
            or capture.profiler is None
            or capture.profiler.state not in {"STARTING", "STOPPING"}
        ):
            return
        with self._lock:
            capture = self.capture
            if capture is None or capture.profiler is None:
                return
            profiler = capture.profiler
            if profiler.state not in {"STARTING", "STOPPING"}:
                return
            if (
                profiler.state == "STARTING"
                and time.monotonic_ns() >= capture.deadline_ns
            ):
                return
            if (
                profiler.state == "STOPPING"
                and capture.mode == "LIGHTWEIGHT"
                and self.tracer is not None
                and self.tracer.pending_count()
                and time.monotonic_ns() < capture.drain_deadline_ns
            ):
                return
        if profiler.state == "STOPPING" and stop_ready is not None and not stop_ready():
            return
        if (
            allow_stop
            and profiler.state == "STOPPING"
            and capture.mode == "LIGHTWEIGHT"
            and self.tracer is not None
        ):
            self.tracer.end_capture(capture)
        profiler.step(startup_only=not allow_stop or profiler.state == "STARTING")
        if (
            profiler.state == "RUNNING"
            and capture.mode == "LIGHTWEIGHT"
            and not capture.stop_requested
        ):
            capture.accepting = True
        self._wake.set()

    def _status(self) -> dict:
        capture = self.capture
        return {
            **self.identity,
            "worker_id": self.worker_id,
            "role": self.role,
            "state": self.state,
            "session_id": capture.session_id if capture else None,
            "command_seq": self.command_seq,
            "host_clock_id": self.host_clock_id,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "capabilities": ["LIGHTWEIGHT", "FULL"]
            if self._profiler_factory is not None
            else [],
            "error": capture.error if capture else None,
            "details": {
                "started_at_ns": capture.started_at_ns if capture else None,
                "writer": capture.writer.snapshot() if capture else None,
                "mode": capture.mode if capture else None,
                "native_profiler": capture.profiler.snapshot()
                if capture is not None and capture.profiler is not None
                else None,
                "profiler_poisoned": self._profiler_poisoned,
                **(
                    self.tracer.snapshot()
                    if capture is not None
                    and capture.mode == "LIGHTWEIGHT"
                    and self.tracer is not None
                    else {}
                ),
            },
        }

    def tick(self) -> None:
        """Called by the device-bound control thread, never an inference thread."""
        with self._lock:
            capture = self.capture
            if capture is None or self.state in {"COMPLETED", "FAILED"}:
                return
            now = time.monotonic_ns()
            if (
                self.state == "STARTING"
                and now >= capture.start_ns
                and capture.profiler is not None
            ):
                if capture.profiler.state == "PREPARED":
                    capture.profiler.start()
                    capture.profiler_deadline_ns = now + int(
                        PROFILER_OPERATION_TIMEOUT_S * 1e9
                    )
                if capture.profiler.started_at_ns is not None:
                    capture.started_at_ns = capture.profiler.started_at_ns
                    self.state = "RUNNING"
            profiler = capture.profiler
            if profiler is not None:
                if profiler.error:
                    capture.error = capture.error or profiler.error
                    self._profiler_poisoned = True
                elif (
                    capture.profiler_deadline_ns
                    and now >= capture.profiler_deadline_ns
                    and not profiler.finished
                    and (profiler.started_at_ns is None or capture.stop_requested)
                ):
                    phase = "stop" if capture.stop_requested else "start"
                    capture.error = f"{capture.mode} profiler {phase} deadline exceeded"
                    self._profiler_poisoned = True
            output_error = capture.writer.snapshot()["error"]
            capture.error = capture.error or output_error
            if now >= capture.deadline_ns or capture.error:
                self._stop(capture)
            tracer = self.tracer if capture.mode == "LIGHTWEIGHT" else None
            if not capture.stop_requested:
                return
            if (
                profiler is not None
                and not profiler.finished
                and not self._profiler_poisoned
            ):
                return
            expired = now >= capture.drain_deadline_ns
            if tracer is not None and tracer.pending_count() and not expired:
                return
            if not capture.flush_deadline_ns:
                if tracer is not None:
                    tracer.finish_capture(capture)
                else:
                    capture.writer.submit(
                        {
                            "record_type": "SUMMARY",
                            "worker_id": self.worker_id,
                            "dropped_spans": 0,
                            "dropped_metadata": 0,
                            "unresolved_spans": 0,
                            "dropped_records": capture.writer.snapshot()[
                                "dropped_records"
                            ],
                            "error": capture.error,
                        }
                    )
                capture.flush_deadline_ns = now + int(DRAIN_TIMEOUT_S * 1e9)
            capture.writer.close(0)
            output = capture.writer.snapshot()
            if output["closed"]:
                capture.error = capture.error or output["error"]
                self.state = "FAILED" if capture.error else "COMPLETED"
            elif now >= capture.flush_deadline_ns:
                capture.error = capture.error or "writer flush deadline exceeded"
                self.state = "FAILED"
