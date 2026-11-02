# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pull service and shutdown handshake for the async CAM connector."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from afd_plugin.validation import cam_receive_capacity

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector


class PullFFNSession:
    requires_drain = True

    def __init__(self, connector: CAMAsyncAFDConnector) -> None:
        self.connector = connector
        self.receiver_finished = False
        self._idle_enabled = False
        self._drained = False

    def keep_serving(self, shutdown_requested: bool) -> bool:
        startup = self.connector.graph_startup
        assert startup._store is not None
        startup.check_failure()
        return not startup._store.check(
            startup._keys("pull_quiet", range(startup.attn_size))
        )

    def before_first_step(self) -> None:
        if not self._idle_enabled:
            self.connector._enable_pull_idle_returns()
            self._idle_enabled = True

    def after_step(self) -> None:
        self.connector._check_pull_status()

    def drain(self, run_one: Callable[[], None]) -> None:
        connector = self.connector
        rounds = (
            connector.max_num_batched_tokens * connector.topk
            + cam_receive_capacity()
            - 1
        ) // cam_receive_capacity()
        for _ in range(rounds):
            if not connector._pull_has_active_source():
                self._drained = True
                return
            run_one()
        if connector._pull_has_active_source():
            raise RuntimeError("Pull source did not drain; retain resources")
        self._drained = True

    def finish_receiver(self) -> None:
        if self.receiver_finished:
            return
        if not self._drained:
            raise RuntimeError("Pull receiver has not drained; retain resources")
        startup = self.connector.graph_startup
        startup._wait(startup._keys("pull_quiet", range(startup.attn_size)))
        startup._publish("pull_stopped")
        self.receiver_finished = True

    def close_attention(self) -> None:
        startup = self.connector.graph_startup
        startup._publish("pull_quiet")
        startup._wait(
            startup._keys("pull_stopped", range(startup.attn_size, startup.world_size))
        )

    def validate_shutdown(self) -> None:
        if not self.receiver_finished or self.connector._pull_has_active_source():
            raise RuntimeError("Pull receiver has not drained; retain resources")
        self.connector._check_pull_status()
