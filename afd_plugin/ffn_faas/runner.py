# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host lifecycle for one standalone CAM FFN process."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.ffn_faas.settings import FFNConfig
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.backend import CAMFFNBackend

HOST_STATUS_POLL_SECONDS = 0.1


def create_backend(
    checkpoint: V4CheckpointDescription, settings: FFNConfig
) -> CAMFFNBackend:
    # Match native startup ordering before importing torch-npu and the runner.
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    from afd_plugin.ffn_faas.backend import CAMFFNBackend

    return CAMFFNBackend(checkpoint, settings)


class FFNModelRunner:
    """Own the backend across stop requests and failed or pending receives."""

    def __init__(
        self,
        service: FFNService,
        checkpoint: V4CheckpointDescription,
        settings: FFNConfig,
    ) -> None:
        self.service, self.checkpoint, self.settings = service, checkpoint, settings
        self.backend: CAMFFNBackend | None = None
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
            self.backend = create_backend(self.checkpoint, self.settings)
            self.backend.prepare(self.service)
            try:
                self.service.phase(ServicePhase.READY, cam_paired=True)
            except InterruptedError:
                self.backend.receiver_exited()
            else:
                self._thread = threading.Thread(
                    target=self._serve, name="ffn-cam-receiver", daemon=True
                )
                self._thread.start()
                while not self.service.stop_requested.wait(HOST_STATUS_POLL_SECONDS):
                    if self._done.is_set():
                        break
                self._thread.join(timeout=self.settings.shutdown_timeout)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "CAM receive is pending; retain owners and terminate the "
                        "paired process group. A host stop cannot cancel device DR."
                    )
                if self._failure is not None:
                    raise RuntimeError(
                        "CAM receiver failed; retain resources"
                    ) from self._failure
            self.backend.close()
            self.service.phase(ServicePhase.STOPPED, resources_released=True)
        except BaseException as error:
            # No native close on a failed startup/receive: an enqueued task may
            # still reference weights, Graph pools or communication buffers.
            self.service.phase(ServicePhase.FAILED, error=str(error))
            raise
