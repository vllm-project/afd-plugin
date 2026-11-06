# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Official endpoint-plugin entry: one existing scale route with optional role."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Awaitable
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import State
from starlette.middleware import Middleware
from starlette.types import Receive, Scope, Send
from vllm.entrypoints.serve.elastic_ep.api_router import (
    scale_elastic_ep as native_scale_elastic_ep,
)
from vllm.entrypoints.serve.elastic_ep.middleware import ScalingMiddleware

from afd_plugin.elastic.config import is_elastic_attention

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient


async def scale_elastic_ep(raw_request: Request):
    client = raw_request.app.state.engine_client
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid JSON format") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    role = body.get("role", "attention")
    if not is_elastic_attention(client.vllm_config):
        if role != "attention":
            raise HTTPException(400, "role=ffn requires elastic AFD")
        return await native_scale_elastic_ep(raw_request)
    size = body.get("new_data_parallel_size")
    drain_timeout = body.get("drain_timeout", 120)
    if role not in ("attention", "ffn"):
        raise HTTPException(400, "role must be 'attention' or 'ffn'")
    if type(size) is not int or size <= 0:
        raise HTTPException(400, "new_data_parallel_size must be a positive integer")
    if type(drain_timeout) is not int or drain_timeout <= 0:
        raise HTTPException(400, "drain_timeout must be a positive integer")
    try:
        await client.scale_elastic_ep(size, drain_timeout, role=role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(408, str(exc)) from exc
    return JSONResponse({"message": f"Scaled {role} to {size} data parallel engines"})


class AFDScalingMiddleware(ScalingMiddleware):
    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] == "http"
            and scope["path"] == "/is_scaling_elastic_ep"
            and scope["app"].state.afd_elastic_enabled
        ):
            return self.app(scope, receive, send)
        return super().__call__(scope, receive, send)


class AFDElasticEndpointPlugin:
    name = "afd_elastic"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        self.app = app
        app.state.afd_elastic_enabled = False
        for index, route in enumerate(app.router.routes):
            if (
                isinstance(route, APIRoute)
                and route.path == "/scale_elastic_ep"
                and "POST" in route.methods
            ):
                # Replace, do not append a shadowed duplicate. Retain the core
                # route's validation dependencies and OpenAPI response contract.
                app.router.routes[index] = APIRoute(
                    path=route.path,
                    endpoint=scale_elastic_ep,
                    methods=route.methods,
                    dependencies=route.dependencies,
                    responses=route.responses,
                    response_model=route.response_model,
                    status_code=route.status_code,
                    name=route.name,
                    tags=route.tags,
                    summary=route.summary,
                    description=route.description,
                    response_description=route.response_description,
                    response_class=route.response_class,
                    deprecated=route.deprecated,
                    operation_id=route.operation_id,
                    include_in_schema=route.include_in_schema,
                    openapi_extra=route.openapi_extra,
                )
                break
        else:
            raise RuntimeError("The vLLM /scale_elastic_ep route was not installed")

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        assert engine_client is not None
        # Upstream build_app installs ScalingMiddleware *after* attach_router.
        # Phase B runs after build_app and before the ASGI middleware stack is
        # built, so replace it here rather than missing the not-yet-added entry.
        for index, middleware in enumerate(self.app.user_middleware):
            if middleware.cls is ScalingMiddleware:
                self.app.user_middleware[index] = Middleware(AFDScalingMiddleware)
        state.afd_elastic_enabled = await engine_client.initialize_afd()
