# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only service state and management HTTP; queries never touch a device."""

from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


class ServicePhase(str, Enum):
    STARTING = "starting"
    LOADING = "loading"
    CONNECTING = "connecting"
    WARMUP = "warmup"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_PHASES: Final = frozenset({ServicePhase.STOPPED, ServicePhase.FAILED})
SIGNAL_STOP_POLL_SECONDS: Final = 0.05


class StopRequest(threading.Event):
    """Signal handlers write one plain flag, without reentering Event locks."""

    def __init__(self) -> None:
        super().__init__()
        self._signal_requested = False

    def request_from_signal(self) -> None:
        self._signal_requested = True

    def is_set(self) -> bool:
        return self._signal_requested or super().is_set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = float("inf") if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            super().wait(min(SIGNAL_STOP_POLL_SECONDS, remaining))
        return True


class FFNService:
    """Own lifecycle state; one external process owns one device runner."""

    def __init__(self, *, rank: int, model_identity: str) -> None:
        self.stop_requested = StopRequest()
        self._lock = threading.Lock()
        self._status: dict[str, str | int | bool] = {
            "rank": rank,
            "pid": os.getpid(),
            "model_identity": model_identity,
            "phase": ServicePhase.STARTING.value,
            "completed_work": 0,
            "loaded_experts": 0,
            "total_experts": 0,
            "resources_released": False,
        }
        self._server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, str | int | bool]:
        with self._lock:
            status = dict(self._status)
            if (
                self.stop_requested.is_set()
                and ServicePhase(status["phase"]) not in TERMINAL_PHASES
            ):
                status["phase"] = ServicePhase.DRAINING.value
            return status

    def phase(self, phase: ServicePhase, **details: str | int | bool) -> None:
        with self._lock:
            current = ServicePhase(self._status["phase"])
            if current in TERMINAL_PHASES:
                raise RuntimeError(
                    f"cannot leave terminal service state {current.value}"
                )
            if self.stop_requested.is_set() and phase == ServicePhase.READY:
                raise InterruptedError("FFN stop was requested before readiness")
            self._status.update(details, phase=phase.value)

    def loading_progress(self, completed: int, total: int) -> None:
        if self.stop_requested.is_set():
            raise InterruptedError("FFN stopped while loading expert weights")
        with self._lock:
            self._status.update(loaded_experts=completed, total_experts=total)

    def completed_work(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("completed work count must be positive")
        with self._lock:
            self._status["completed_work"] = int(self._status["completed_work"]) + count

    def request_stop(self) -> None:
        # A standalone F stop aborts the paired session. Only A can initiate a
        # successful drain after all accepted requests finish every model layer.
        self.stop_requested.set()
        with self._lock:
            if ServicePhase(self._status["phase"]) not in TERMINAL_PHASES:
                self._status["phase"] = ServicePhase.DRAINING.value

    def request_signal_stop(self) -> None:
        # Called on the interrupted main thread, which may already hold either
        # the service lock or Event's condition lock. Never acquire either here.
        self.stop_requested.request_from_signal()

    def start_http(self, host: str, port: int) -> tuple[str, int]:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def respond(self, code: HTTPStatus, body: dict) -> None:
                payload = (json.dumps(body, sort_keys=True) + "\n").encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                status = service.snapshot()
                phase = ServicePhase(status["phase"])
                if self.path == "/status":
                    self.respond(HTTPStatus.OK, status)
                elif self.path in ("/ready", "/health"):
                    healthy = (
                        phase == ServicePhase.READY
                        if self.path == "/ready"
                        else phase not in TERMINAL_PHASES
                    )
                    self.respond(
                        HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                        status,
                    )
                else:
                    self.respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

            def do_POST(self) -> None:
                if self.path != "/stop":
                    self.respond(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return
                service.request_stop()
                self.respond(HTTPStatus.ACCEPTED, service.snapshot())

            def log_message(self, format: str, *args: str) -> None:
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._server.serve_forever, name="ffn-management", daemon=True
        )
        self._http_thread.start()
        address, bound_port = self._server.server_address[:2]
        return str(address), int(bound_port)

    def close_http(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._http_thread is not None:
            self._http_thread.join()
            self._http_thread = None
