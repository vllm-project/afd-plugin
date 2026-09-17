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
from vllm.config import get_current_vllm_config

from afd_plugin.config import parse_optional_afd_config

DEFAULT_TIMEOUT_S = 600
# What upstream hardcodes. A process with the plugin installed but no AFD
# configuration is an ordinary vLLM DP run and must keep waiting exactly this
# long; the patch applies at register_afd() for every such process, so the role
# check has to happen per call, the way ffn_local_moe_prepare does it.
UPSTREAM_TIMEOUT_S = 120


def _afd_is_active() -> bool:
    try:
        afd_config = parse_optional_afd_config(
            get_current_vllm_config(),
            validate=False,
        )
    except Exception:
        return False
    return afd_config is not None


# Patch reason: the upstream DP Coordinator startup wait is hardcoded to 120
# seconds, which is not enough for the coordinator subprocess to import and
# bind on a CPU-oversubscribed shared machine -- both AFD roles then die
# during startup.
# Patch functionality: identical to upstream, except that an AFD run takes its
# wait from AFD_DP_COORDINATOR_TIMEOUT_S (default 600 seconds). A non-AFD run
# keeps upstream's 120 seconds.
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0, vllm/v1/engine/coordinator.py
def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
    try:
        default_timeout = DEFAULT_TIMEOUT_S if _afd_is_active() else UPSTREAM_TIMEOUT_S
        timeout = int(
            os.getenv("AFD_DP_COORDINATOR_TIMEOUT_S", str(default_timeout)),
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
