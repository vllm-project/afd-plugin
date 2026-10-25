# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Management HTTP remains CPU-only and does not promise device drain."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from afd_plugin.ffn_faas.service import FFNService, ServicePhase


def test_readiness_and_host_stop_request():
    service = FFNService(rank=0, model_path="/model")
    host, port = service.start_http("127.0.0.1", 0)
    base = f"http://{host}:{port}"
    try:
        with pytest.raises(HTTPError) as unavailable:
            urlopen(base + "/ready")
        assert unavailable.value.code == 503
        service.phase(ServicePhase.READY, model_loaded=True, cam_paired=True)
        with urlopen(base + "/ready") as response:
            assert json.load(response)["phase"] == "ready"
        with urlopen(Request(base + "/stop", method="POST")) as response:
            status = json.load(response)
            assert response.status == 202
        assert status["phase"] == "stopping"
        assert status["resources_released"] is False
        assert service.stop_requested.is_set()
        with pytest.raises(HTTPError):
            urlopen(base + "/ready")
        service.phase(ServicePhase.FAILED, error="receive pending")
        with pytest.raises(HTTPError) as failed:
            urlopen(base + "/health")
        assert failed.value.code == 503
    finally:
        service.close_http()


def test_signal_stop_keeps_terminal_state():
    service = FFNService(rank=0, model_path="/model")
    service.request_signal_stop()
    assert service.snapshot()["phase"] == "stopping"
    with pytest.raises(InterruptedError):
        service.phase(ServicePhase.READY)
    service.phase(ServicePhase.STOPPED, resources_released=True)
    assert service.snapshot()["phase"] == "stopped"
