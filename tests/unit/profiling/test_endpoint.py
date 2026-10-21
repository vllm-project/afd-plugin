# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from afd_plugin.profiling import endpoint


class FakeEngine:
    def __init__(self):
        self.calls = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, timeout, args, kwargs))
        return [{"worker_id": "attention-dp0"}, {"worker_id": "ffn-dp1"}]


def make_plugin(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    monkeypatch.setenv("AFD_CAM_PROFILE_TOKEN", "test-token")
    monkeypatch.setattr(endpoint, "apply_cam_profile_rpc_patch", lambda: True)
    plugin = endpoint.CAMProfileEndpointPlugin()
    app = FastAPI()
    plugin.attach_router(app)
    engine = FakeEngine()
    asyncio.run(plugin.init_state(engine, app.state, Namespace()))
    return app, engine


def test_disabled_plugin_does_not_install_routes_or_rpc_patch(monkeypatch):
    monkeypatch.delenv("AFD_CAM_PROFILE_ENABLE", raising=False)
    monkeypatch.delenv("AFD_CAM_PROFILE_TOKEN", raising=False)

    def forbidden_patch():
        raise AssertionError("Disabled profiling must not patch vLLM")

    monkeypatch.setattr(endpoint, "apply_cam_profile_rpc_patch", forbidden_patch)
    plugin = endpoint.CAMProfileEndpointPlugin()
    app = FastAPI()
    plugin.attach_router(app)
    asyncio.run(plugin.init_state(None, app.state, Namespace()))
    assert all(not route.path.startswith("/cam/profile") for route in app.routes)


def test_enabled_plugin_requires_token_and_pinned_engine(monkeypatch):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    monkeypatch.delenv("AFD_CAM_PROFILE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TOKEN"):
        endpoint.CAMProfileEndpointPlugin()
    monkeypatch.setenv("AFD_CAM_PROFILE_TOKEN", "test-token")
    plugin = endpoint.CAMProfileEndpointPlugin()
    app = FastAPI()
    with pytest.raises(ValueError, match="generation engine"):
        asyncio.run(plugin.init_state(None, app.state, Namespace()))
    monkeypatch.setattr(endpoint, "apply_cam_profile_rpc_patch", lambda: False)
    with pytest.raises(RuntimeError, match="pinned vLLM"):
        asyncio.run(plugin.init_state(FakeEngine(), app.state, Namespace()))


@pytest.mark.parametrize("action", ["prepare", "commit", "stop", "status"])
def test_every_private_route_authenticates_and_uses_fixed_rpc(monkeypatch, action):
    app, engine = make_plugin(monkeypatch)
    path = f"/cam/profile/internal/{action}"
    command = {} if action == "status" else {"session_id": "session", "command_seq": 1}
    with TestClient(app) as client:
        assert client.post(path, json=command).status_code == 401
        assert (
            client.post(
                path, json=command, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert not engine.calls
        response = client.post(
            path, json=command, headers={"Authorization": "Bearer test-token"}
        )
    assert response.status_code == 200
    assert len(response.json()["workers"]) == 2
    assert response.json()["wall_time_ns"] > 0
    assert engine.calls == [(f"cam_profile_{action}", 5.0, (command,), None)]


@pytest.mark.parametrize("field", ["method", "output_directory", "path"])
def test_private_routes_reject_arbitrary_rpc_and_paths(monkeypatch, field):
    app, engine = make_plugin(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/cam/profile/internal/prepare",
            json={field: "untrusted"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 400
        assert (
            client.post("/cam/profile/internal/arbitrary", json={}).status_code == 404
        )
    assert not engine.calls


@pytest.mark.parametrize("action", ["prepare", "commit", "stop", "status"])
def test_private_rpc_deadline_covers_engine_queue_wait(monkeypatch, action):
    monkeypatch.setattr(endpoint, "RPC_TIMEOUT_S", 0.005)
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1")
    monkeypatch.setenv("AFD_CAM_PROFILE_TOKEN", "test-token")
    plugin = endpoint.CAMProfileEndpointPlugin()
    cancelled = False

    class StalledEngine:
        async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            nonlocal cancelled
            assert method == f"cam_profile_{action}"
            assert timeout == endpoint.RPC_TIMEOUT_S
            try:
                await asyncio.Event().wait()
            finally:
                cancelled = True

    async def run():
        plugin.engine_client = StalledEngine()
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(plugin._handler(action)({}), 0.2)
        assert error.value.status_code == 504
        assert cancelled

    asyncio.run(run())
