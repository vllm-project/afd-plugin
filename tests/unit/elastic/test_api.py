# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def api(monkeypatch, config):
    calls: list[object] = []

    async def native(request):
        calls.append("native")
        return JSONResponse({"native": True})

    async def scale(size, timeout, role="attention"):
        calls.append((role, size, timeout))

    router_module = types.ModuleType("vllm.entrypoints.serve.elastic_ep.api_router")
    router_module.scale_elastic_ep = native  # type: ignore[attr-defined]
    middleware_module = types.ModuleType("vllm.entrypoints.serve.elastic_ep.middleware")

    class ScalingMiddleware:
        def __init__(self, app):
            self.app = app

        def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["app"].state.scaling:
                return JSONResponse({}, 503)(scope, receive, send)
            return self.app(scope, receive, send)

    middleware_module.ScalingMiddleware = ScalingMiddleware  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, router_module.__name__, router_module)
    monkeypatch.setitem(sys.modules, middleware_module.__name__, middleware_module)
    path = Path(__file__).parents[3] / "afd_plugin/elastic/api.py"
    spec = importlib.util.spec_from_file_location("_afd_api_cpu_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app = FastAPI()
    app.state.scaling = False
    app.state.engine_client = SimpleNamespace(
        vllm_config=config, scale_elastic_ep=scale
    )
    dependency_calls = []

    async def dependency():
        dependency_calls.append(True)

    @app.post("/scale_elastic_ep", dependencies=[Depends(dependency)])
    async def original(request: Request):
        return await native(request)

    @app.post("/is_scaling_elastic_ep")
    async def status():
        return {"is_scaling_elastic_ep": app.state.scaling}

    plugin = module.AFDElasticEndpointPlugin()
    plugin.attach_router(app)
    # Preserve v0.26's actual build_app -> init_app_state ordering: endpoint
    # routers are attached before core middleware is installed.
    app.add_middleware(ScalingMiddleware)

    async def initialize():
        return False

    asyncio.run(
        plugin.init_state(
            SimpleNamespace(initialize_afd=initialize), app.state, SimpleNamespace()
        )
    )
    return SimpleNamespace(
        app=app,
        calls=calls,
        config=config,
        plugin=plugin,
        dependency_calls=dependency_calls,
    )


def test_replaces_route_and_preserves_dependencies(api):
    routes = [
        r
        for r in api.app.routes
        if isinstance(r, APIRoute) and r.path == "/scale_elastic_ep"
    ]
    assert len(routes) == 1
    with TestClient(api.app) as client:
        assert (
            client.post(
                "/scale_elastic_ep", json={"new_data_parallel_size": 4}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/scale_elastic_ep", json={"new_data_parallel_size": 2, "role": "ffn"}
            ).status_code
            == 200
        )
    assert api.calls == [("attention", 4, 120), ("ffn", 2, 120)]
    assert len(api.dependency_calls) == 2


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"new_data_parallel_size": True},
        {"new_data_parallel_size": 0},
        {"new_data_parallel_size": 2, "role": "both"},
    ],
)
def test_bad_requests_do_not_reach_engine(api, body):
    with TestClient(api.app) as client:
        assert client.post("/scale_elastic_ep", json=body).status_code == 400
    assert not api.calls


def test_non_afd_delegates_to_existing_handler(api):
    api.config.additional_config = {}
    with TestClient(api.app) as client:
        assert client.post(
            "/scale_elastic_ep", json={"new_data_parallel_size": 4}
        ).json() == {"native": True}
        assert (
            client.post(
                "/scale_elastic_ep", json={"new_data_parallel_size": 2, "role": "ffn"}
            ).status_code
            == 400
        )
    assert api.calls == ["native"]


def test_only_afd_query_is_available_during_scale(api):
    api.app.state.scaling = True
    with TestClient(api.app) as client:
        assert client.post("/is_scaling_elastic_ep").status_code == 503
        api.app.state.afd_elastic_enabled = True
        assert client.post("/is_scaling_elastic_ep").json() == {
            "is_scaling_elastic_ep": True
        }
        assert (
            client.post(
                "/scale_elastic_ep", json={"new_data_parallel_size": 4}
            ).status_code
            == 503
        )


def test_init_state_awaits_engine_initialization(api):
    async def initialize():
        await asyncio.sleep(0)
        api.calls.append("initialized")
        return True

    engine = SimpleNamespace(initialize_afd=initialize)
    asyncio.run(api.plugin.init_state(engine, api.app.state, SimpleNamespace()))
    assert api.calls == ["initialized"]
    assert api.app.state.afd_elastic_enabled
