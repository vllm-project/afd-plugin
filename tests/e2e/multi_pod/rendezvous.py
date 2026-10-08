# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""TCPStore-backed rendezvous for the multi-pod AFD E2E runner.

Barriers are polled rather than blocking: ``TCPStore.wait`` cannot notice a peer
dying while it waits, so every barrier re-checks the abort flag and the caller's
own child processes on each pass. A timeout names the pods that did not arrive.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta

from torch.distributed import TCPStore

DEFAULT_STORE_PORT = 29500
STORE_CONNECT_TIMEOUT_S = 600
BARRIER_POLL_INTERVAL_S = 2.0
# Pod 0 must publish the agreement keys before any peer can verify them; this
# only covers process start skew, not scheduling (the driver owns that).
AGREEMENT_TIMEOUT_S = 300

LAYOUT_KEY = "run/layout"
SCENARIO_KEY = "run/scenario"
VERDICT_KEY = "verdict"
ABORT_COUNT_KEY = "count/abort"
PASS_VERDICT = "pass"


class PeerFailureError(RuntimeError):
    """Raised when another pod published an abort."""


class BarrierTimeoutError(TimeoutError):
    """Raised when peers did not reach a barrier before its deadline."""


class Rendezvous:
    """Shared key-value state and polled barriers across the participating pods.

    Pod 0 masters the store. That is unrelated to which pod leads a role or runs
    the evaluation, so the layout axis stays unconstrained.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        pod_index: int,
        num_pods: int,
        connect_timeout_s: float = STORE_CONNECT_TIMEOUT_S,
        poll_interval_s: float = BARRIER_POLL_INTERVAL_S,
    ) -> None:
        if not 0 <= pod_index < num_pods:
            raise ValueError(
                f"pod index {pod_index} is outside 0..{num_pods - 1}",
            )
        self.pod_index = pod_index
        self.num_pods = num_pods
        self.poll_interval_s = poll_interval_s
        self._aborted = False
        self.store = TCPStore(
            host_name=host,
            port=port,
            world_size=num_pods,
            is_master=pod_index == 0,
            timeout=timedelta(seconds=connect_timeout_s),
            wait_for_workers=False,
        )

    # -- primitives ------------------------------------------------------

    def set(self, key: str, value: str) -> None:
        self.store.set(key, value)

    def get(self, key: str) -> str | None:
        """Return a key's value, or None when it is not set yet.

        ``TCPStore.get`` blocks until the key appears, so existence is checked
        first; a polled barrier must never block inside a single read.
        """
        if not self.store.check([key]):
            return None
        return self.store.get(key).decode()

    # -- agreement -------------------------------------------------------

    def verify_agreement(self, layout: str, scenario: str) -> None:
        """Fail fast when pods were launched with mismatched arguments."""
        expected = {LAYOUT_KEY: layout, SCENARIO_KEY: scenario}
        if self.pod_index == 0:
            for key, value in expected.items():
                self.set(key, value)
            return
        for key, value in expected.items():
            published = self._wait_for_value(key, AGREEMENT_TIMEOUT_S)
            if published != value:
                raise PeerFailureError(
                    f"pod {self.pod_index} {key} {value!r} != run {key} {published!r}",
                )

    def _wait_for_value(self, key: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            value = self.get(key)
            if value is not None:
                return value
            if time.monotonic() >= deadline:
                raise BarrierTimeoutError(
                    f"pod 0 did not publish {key} within {timeout_s:.0f}s",
                )
            time.sleep(self.poll_interval_s)

    # -- abort -----------------------------------------------------------

    def publish_abort(self, message: str) -> None:
        """Announce a local failure so every peer unwinds in seconds."""
        if self._aborted:
            return
        self._aborted = True
        self.set(f"abort/{self.pod_index}", message)
        self.store.add(ABORT_COUNT_KEY, 1)

    def poll_abort(self) -> str | None:
        """Return the peers' abort messages, or None when nobody aborted."""
        if not self.store.check([ABORT_COUNT_KEY]):
            return None
        messages = []
        for index in range(self.num_pods):
            if index == self.pod_index:
                continue
            message = self.get(f"abort/{index}")
            if message is not None:
                messages.append(f"pod {index}: {message}")
        if not messages:
            return None
        return "; ".join(messages)

    # -- barriers --------------------------------------------------------

    def barrier(
        self,
        name: str,
        timeout_s: float,
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> None:
        """Wait until every pod reaches ``name``.

        ``on_poll`` is the caller's own liveness check; raising from it is how a
        pod notices its local children died while its peers are still starting.
        """
        self.set(self._arrival_key(name, self.pod_index), "1")
        deadline = time.monotonic() + timeout_s
        while True:
            self._raise_on_abort(name)
            if on_poll is not None:
                on_poll()
            missing = self._missing(name)
            if not missing:
                return
            if time.monotonic() >= deadline:
                arrived = self.num_pods - len(missing)
                raise BarrierTimeoutError(
                    f"barrier {name}: {arrived}/{self.num_pods} after "
                    f"{timeout_s:.0f}s; missing="
                    f"{[f'pod-{index}' for index in missing]}",
                )
            time.sleep(self.poll_interval_s)

    def wait_for_key(
        self,
        key: str,
        timeout_s: float,
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> str:
        """Poll until ``key`` is published, aborting early on a peer failure."""
        deadline = time.monotonic() + timeout_s
        while True:
            self._raise_on_abort(key)
            if on_poll is not None:
                on_poll()
            value = self.get(key)
            if value is not None:
                return value
            if time.monotonic() >= deadline:
                raise BarrierTimeoutError(
                    f"{key} was not published within {timeout_s:.0f}s",
                )
            time.sleep(self.poll_interval_s)

    def _raise_on_abort(self, waiting_on: str) -> None:
        abort = self.poll_abort()
        if abort is not None:
            raise PeerFailureError(
                f"peer aborted while waiting on {waiting_on}: {abort}",
            )

    def _missing(self, name: str) -> list[int]:
        return [
            index
            for index in range(self.num_pods)
            if not self.store.check([self._arrival_key(name, index)])
        ]

    @staticmethod
    def _arrival_key(name: str, pod_index: int) -> str:
        return f"arrived/{name}/{pod_index}"
