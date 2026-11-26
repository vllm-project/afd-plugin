# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host lifecycle for one injected standalone FFN backend."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

from afd_plugin.ffn_faas.interfaces import FFNBackend
from afd_plugin.ffn_faas.service import FFNService, ServicePhase

HOST_STATUS_POLL_SECONDS = 0.1


class FFNModelRunner:
    """Own the backend across stop requests and failed or pending receives."""

    def __init__(
        self,
        service: FFNService,
        backend_factory: Callable[[], FFNBackend],
        *,
        shutdown_timeout: float,
    ) -> None:
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        self.service = service
        self.backend_factory = backend_factory
        self.shutdown_timeout = shutdown_timeout
        self.backend: FFNBackend | None = None
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._failure: BaseException | None = None

    def _serve(self) -> None:
        """Receiver-thread loop: one backend step per control payload.

        The loop ends when the host requests a stop between steps, or when
        ``execute_step()`` reports that ``cancel_receive()`` ended an idle
        receive. Only then does the backend record its receiver exit, which is
        the proof ``close()`` requires before releasing native resources.
        """
        assert self.backend is not None
        try:
            self.backend.start_receiver()
            while not self.service.stop_requested.is_set():
                if not self.backend.execute_step():
                    break
                self.service.completed_batches()
            self.backend.receiver_exited()
        except BaseException as error:
            self._failure = error
        finally:
            self._done.set()

    def run(self) -> None:
        """Prepare the backend, serve until stopped, then release or retain.

        The backend is constructed before native initialization so a failed
        preparation keeps its owners reachable. After READY the receiver runs
        on a daemon thread; on a host stop the backend is asked to cancel an
        idle receive, then the thread is joined with ``shutdown_timeout``.
        Resources are closed only after a clean receiver exit; any failure,
        including a receive that does not end, marks the service FAILED and
        re-raises with every owner retained for process isolation.
        """
        try:
            if self.service.stop_requested.is_set():
                self.service.phase(ServicePhase.STOPPED, resources_released=True)
                return
            self.service.phase(ServicePhase.LOADING)
            # Retain the backend before any native initialization can fail.
            self.backend = self.backend_factory()
            self.backend.prepare(self.service)
            try:
                self.service.phase(ServicePhase.READY)
            except InterruptedError:
                self.backend.receiver_exited()
            else:
                self._thread = threading.Thread(
                    target=self._serve, name="ffn-receiver", daemon=True
                )
                self._thread.start()
                while not self.service.stop_requested.wait(HOST_STATUS_POLL_SECONDS):
                    if self._done.is_set():
                        break
                if self.service.stop_requested.is_set() and not self._done.is_set():
                    # The receiver may be blocked in an idle receive that only
                    # the backend can end; a backend without cancellation keeps
                    # it pending and the join below reports it.
                    self.backend.cancel_receive()
                self._thread.join(timeout=self.shutdown_timeout)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "FFN receive is pending; retain owners and isolate the "
                        "deployment. The backend did not end its receive or "
                        "in-flight device work within the shutdown timeout."
                    )
                if self._failure is not None:
                    raise RuntimeError(
                        "FFN receiver failed; retain resources"
                    ) from self._failure
            self.backend.close()
            self.service.phase(ServicePhase.STOPPED, resources_released=True)
        except BaseException as error:
            # No native close on a failed startup/receive: an enqueued task may
            # still reference weights, Graph pools or communication buffers.
            self.service.phase(ServicePhase.FAILED, error=str(error))
            raise
