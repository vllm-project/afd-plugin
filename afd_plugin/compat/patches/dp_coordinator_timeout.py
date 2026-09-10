# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Raise the DP Coordinator's hardcoded startup ZMQ wait timeout.

``DPCoordinator._wait_for_zmq_addrs`` waits a hardcoded 120 seconds for the
coordinator subprocess to import, bind, and report its ZMQ addresses. On a
CPU-oversubscribed shared box that subprocess can legitimately take longer,
which kills both AFD roles during startup (observed repeatedly on 2A2F
DeepSeek-V4-Flash runs). The wait becomes env-configurable with a 600 second
default; every other behavior matches upstream.
"""

from __future__ import annotations

import multiprocessing
import os

import vllm.v1.engine.coordinator as coordinator_module

DEFAULT_TIMEOUT_S = 600


# Patch reason: the upstream DP Coordinator startup wait is hardcoded to 120
# seconds, which is not enough for the coordinator subprocess to import and
# bind on a CPU-oversubscribed shared machine -- both AFD roles then die
# during startup.
# Patch functionality: identical to upstream, except the wait comes from
# AFD_DP_COORDINATOR_TIMEOUT_S (default 600 seconds).
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0, vllm/v1/engine/coordinator.py
def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
    try:
        timeout = int(
            os.getenv("AFD_DP_COORDINATOR_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)),
        )
        ready = multiprocessing.connection.wait(
            [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
        )
        if not ready:
            raise RuntimeError(
                "DP Coordinator process failed to report ZMQ addresses "
                f"within timeout={timeout} seconds during startup."
            )
        try:
            return zmq_addr_pipe.recv()
        except EOFError:
            raise RuntimeError(
                "DP Coordinator process failed during startup."
            ) from None
    finally:
        zmq_addr_pipe.close()


def apply_dp_coordinator_timeout() -> None:
    coordinator_module.DPCoordinator._wait_for_zmq_addrs = _wait_for_zmq_addrs


apply_dp_coordinator_timeout()

__all__ = ["apply_dp_coordinator_timeout"]
