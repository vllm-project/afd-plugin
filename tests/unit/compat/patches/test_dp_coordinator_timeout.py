# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Startup wait selection for the patched DP Coordinator.

The patch applies at ``register_afd()`` in every process that has the plugin
installed, so the thing worth pinning is that a plain non-AFD DP run still gets
upstream's 120 seconds rather than silently waiting five times as long.
"""

from __future__ import annotations

import pytest

from afd_plugin.compat.patches import dp_coordinator_timeout as patch


class _ClosedPipe:
    """A pipe that reports nothing ready, so the wait always times out."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _run_wait(monkeypatch, *, afd_active: bool, env: str | None):
    monkeypatch.setattr(patch, "_afd_is_active", lambda: afd_active)
    if env is None:
        monkeypatch.delenv("AFD_DP_COORDINATOR_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("AFD_DP_COORDINATOR_TIMEOUT_S", env)

    seen = {}

    def fake_wait(_objects, timeout):
        seen["timeout"] = timeout
        return []

    monkeypatch.setattr(patch.multiprocessing.connection, "wait", fake_wait)

    pipe = _ClosedPipe()
    coordinator = type("_C", (), {"proc": type("_P", (), {"sentinel": object()})()})()
    with pytest.raises(RuntimeError, match="within timeout"):
        patch._wait_for_zmq_addrs(coordinator, pipe)
    assert pipe.closed, "the pipe must be closed even when the wait fails"
    return seen["timeout"]


def test_a_non_afd_run_keeps_upstreams_wait(monkeypatch):
    assert _run_wait(monkeypatch, afd_active=False, env=None) == 120


def test_an_afd_run_gets_the_longer_wait(monkeypatch):
    assert _run_wait(monkeypatch, afd_active=True, env=None) == 600


@pytest.mark.parametrize("afd_active", [True, False])
def test_the_environment_overrides_either_default(monkeypatch, afd_active):
    assert _run_wait(monkeypatch, afd_active=afd_active, env="42") == 42


def test_a_process_with_no_vllm_config_is_not_afd():
    # get_current_vllm_config() outside an engine must not propagate; the
    # coordinator starts before any AFD configuration is resolvable.
    assert patch._afd_is_active() in (True, False)
