# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Adapt session lifecycle requests to the shared TorchNPU profiling backend.

RuntimeProfiler holds session.py's pending requests and reports backend reuse
state. Only the model thread advances compat.npu.profiler's backend; a separate
thread archives finalized LIGHTWEIGHT output through storage.py.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from afd_plugin.compat.npu.profiler import (
    AFDNPUProfiler,
    AFDNPUProfilerRole,
    create_runtime_npu_profiler,
)
from afd_plugin.profiling.storage import archive_native_trace


class RuntimeProfiler:
    def __init__(
        self,
        role: AFDNPUProfilerRole,
        trace_dir: Path,
        mode: str = "FULL",
        *,
        graph: bool = False,
    ) -> None:
        self.role = role
        self.mode = mode
        self.graph = graph
        self.trace_dir = trace_dir
        self.state = "PREPARED"
        self.started_at_ns: int | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._profiler: AFDNPUProfiler | None = None
        self._stepping = False
        self.archive_state = "PENDING" if mode == "LIGHTWEIGHT" else None
        self.archive_error: str | None = None

    @property
    def finished(self) -> bool:
        with self._lock:
            return self.state in {"COMPLETED", "FAILED"}

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy()

    def _busy(self) -> bool:
        return self._stepping or self.state in {"STARTING", "RUNNING", "STOPPING"}

    def start(self) -> None:
        with self._lock:
            if self.state == "PREPARED":
                self.state = "STARTING"

    def stop(self) -> None:
        with self._lock:
            if self.state in {"COMPLETED", "FAILED"}:
                return
            if self._profiler is None and not self._stepping:
                self.state = "COMPLETED"
                if self.mode == "LIGHTWEIGHT":
                    self.archive_state = "SKIPPED"
            else:
                self.state = "STOPPING"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "trace_dir": str(self.trace_dir),
                "started_at_ns": self.started_at_ns,
                "error": self.error,
                "busy": self._busy(),
                **(
                    {
                        "archive_state": self.archive_state,
                        "archive_error": self.archive_error,
                        "archive_path": str(
                            self.trace_dir.with_name(self.trace_dir.name + ".tar.gz")
                        ),
                    }
                    if self.mode == "LIGHTWEIGHT"
                    else {}
                ),
            }

    def step(self, *, startup_only: bool = False) -> None:
        with self._lock:
            if self.state in {"PREPARED", "COMPLETED", "FAILED"} or self._stepping:
                return
            self._stepping = True
        try:
            if self._profiler is None:
                profiler = create_runtime_npu_profiler(
                    self.role,
                    str(self.trace_dir),
                    lightweight=self.mode == "LIGHTWEIGHT",
                    graph=self.graph,
                )
                with self._lock:
                    self._profiler = profiler
                    self.started_at_ns = time.time_ns()
                    if self.state != "STOPPING":
                        self.state = "RUNNING"
            # A device-wait safe point must not enter synchronized stop.
            if startup_only:
                return
            with self._lock:
                profiler = self._profiler
                stopping = self.state == "STOPPING"
            assert profiler is not None
            if stopping:
                profiler.stop()
                with self._lock:
                    self.state = "COMPLETED"
                if self.mode == "LIGHTWEIGHT":
                    try:
                        threading.Thread(
                            target=self._archive, name="afd-cam-archive", daemon=True
                        ).start()
                    except RuntimeError as error:
                        with self._lock:
                            self.archive_state = "FAILED"
                            self.archive_error = f"{type(error).__name__}: {error}"
            else:
                profiler.step()
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "FAILED"
        finally:
            with self._lock:
                self._stepping = False

    def _archive(self) -> None:
        with self._lock:
            self.archive_state = "WRITING"
        try:
            archive_native_trace(self.trace_dir)
            with self._lock:
                self.archive_state = "COMPLETED"
        except Exception as error:
            # Raw data remains readable if compression or verification fails.
            with self._lock:
                self.archive_error = f"{type(error).__name__}: {error}"
                self.archive_state = "FAILED"
