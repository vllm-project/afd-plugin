# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Management state is observable without loading a device runtime."""

import json
import subprocess
import sys
import textwrap
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from afd_plugin.ffn_faas.service import FFNService, ServicePhase


def test_readiness_requires_paired_warmup_and_stop_revokes_it() -> None:
    service = FFNService(rank=1, model_identity="checkpoint-1")
    host, port = service.start_http("127.0.0.1", 0)
    base = f"http://{host}:{port}"
    try:
        with pytest.raises(HTTPError) as result:
            urlopen(base + "/ready", timeout=2)
        assert result.value.code == 503
        service.phase(ServicePhase.WARMUP)
        service.phase(ServicePhase.READY)
        with urlopen(base + "/ready", timeout=2) as response:
            assert json.load(response)["phase"] == "ready"
        with urlopen(Request(base + "/stop", method="POST"), timeout=2) as response:
            assert response.status == 202
            assert json.load(response)["phase"] == "draining"
        assert service.stop_requested.is_set()
        with pytest.raises(HTTPError) as result:
            urlopen(base + "/ready", timeout=2)
        assert result.value.code == 503
    finally:
        service.close_http()


@pytest.mark.parametrize("phase", [ServicePhase.STOPPED, ServicePhase.FAILED])
def test_terminal_service_cannot_become_ready_again(phase: ServicePhase) -> None:
    service = FFNService(rank=0, model_identity="checkpoint-1")
    service.phase(phase)
    with pytest.raises(RuntimeError, match="terminal"):
        service.phase(ServicePhase.READY)


def test_stop_cancels_checkpoint_loading_at_progress_boundary() -> None:
    service = FFNService(rank=0, model_identity="checkpoint-1")
    service.loading_progress(1, 10)
    service.request_stop()
    with pytest.raises(InterruptedError, match="loading"):
        service.loading_progress(2, 10)
    assert service.snapshot()["loaded_experts"] == 1


def test_completed_work_includes_every_item_in_a_device_batch() -> None:
    service = FFNService(rank=0, model_identity="batch-count")
    service.completed_work(3)
    service.completed_work()
    assert service.snapshot()["completed_work"] == 4


@pytest.mark.parametrize("count", [0, -1])
def test_completed_work_rejects_nonpositive_count(count: int) -> None:
    service = FFNService(rank=0, model_identity="batch-count")
    with pytest.raises(ValueError, match="positive"):
        service.completed_work(count)
    assert service.snapshot()["completed_work"] == 0


@pytest.mark.parametrize("lock", ["service._lock", "service.stop_requested._cond"])
def test_signal_stop_returns_when_service_or_event_lock_is_interrupted(
    lock: str,
) -> None:
    # Fault injection holds each lock at signal delivery; only observable
    # stop/readiness behavior is asserted. A child bounds a deadlock regression.
    script = textwrap.dedent(f"""
        import signal
        from afd_plugin.ffn_faas.service import FFNService, ServicePhase
        service = FFNService(rank=0, model_identity="signal-regression")
        service.phase(ServicePhase.READY)
        signal.signal(
            signal.SIGTERM, lambda signum, frame: service.request_signal_stop()
        )
        with {lock}:
            signal.raise_signal(signal.SIGTERM)
        assert service.stop_requested.is_set()
        assert service.stop_requested.wait(timeout=0)
        assert service.snapshot()["phase"] == "draining"
        try:
            service.phase(ServicePhase.READY)
        except InterruptedError:
            pass
        else:
            raise AssertionError("signal stop failed to revoke readiness")
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=5)
