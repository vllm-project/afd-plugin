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
        assert self.backend is not None
        try:
            self.backend.start_receiver()
            while not self.service.stop_requested.is_set():
                self.backend.execute_step()
                self.service.completed_batches()
            self.backend.receiver_exited()
        except BaseException as error:
            self._failure = error
        finally:
            self._done.set()

    def run(self) -> None:
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
                self._thread.join(timeout=self.shutdown_timeout)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "FFN receive is pending; retain owners and isolate the "
                        "deployment. A host stop cannot cancel backend device work."
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
