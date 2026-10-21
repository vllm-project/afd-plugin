# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expose the coordinator's internal profiling commands on each vLLM server.

The opt-in plugin authenticates and validates prepare/commit/stop/status, then
uses collective RPC and the profile_rpc patch to reach every DP worker. It
returns snapshots to control.py; session.py owns each worker's lifecycle.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.datastructures import State

from afd_plugin.compat.patches.profile_rpc import apply_cam_profile_rpc_patch

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.tasks import SupportedTask

RPC_TIMEOUT_S = 5.0
COMMAND_FIELDS = {
    "prepare": frozenset(
        ("session_id", "command_seq", "mode", "duration_s", "lease_s")
    ),
    "commit": frozenset(
        (
            "session_id",
            "command_seq",
            "start_at_ns",
            "clock_by_worker",
            "lease_s",
        )
    ),
    "stop": frozenset(("session_id", "command_seq")),
    "status": frozenset(("session_id",)),
}


class CAMProfileEndpointPlugin:
    """Implement the endpoint contract pinned by the repository's vLLM version."""

    name = "afd_profile"
    required_tasks: tuple[SupportedTask, ...] | None = ("generate",)

    def __init__(self) -> None:
        self.enabled = os.environ.get("AFD_CAM_PROFILE_ENABLE") == "1"
        self.token = os.environ.get("AFD_CAM_PROFILE_TOKEN", "")
        self.engine_client: EngineClient | None = None
        if self.enabled and not self.token:
            raise ValueError(
                "AFD_CAM_PROFILE_TOKEN is required when profiling is enabled"
            )

    def attach_router(self, app: FastAPI) -> None:
        if not self.enabled:
            return
        for action in COMMAND_FIELDS:
            app.add_api_route(
                f"/cam/profile/internal/{action}",
                self._handler(action),
                methods=["POST"],
                name=f"cam_profile_{action}",
                dependencies=[Depends(self._authenticate)],
            )

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        if not self.enabled:
            return
        if engine_client is None:
            raise ValueError("CAM profiling requires a generation engine")
        if not apply_cam_profile_rpc_patch():
            raise RuntimeError("CAM profiling requires the pinned vLLM version")
        self.engine_client = engine_client

    def _handler(self, action: str):
        async def command(body: dict[str, Any]) -> dict[str, Any]:
            if body.keys() - COMMAND_FIELDS[action]:
                raise HTTPException(
                    status_code=400, detail="Unsupported command fields"
                )
            if self.engine_client is None:
                raise HTTPException(status_code=503, detail="Engine is not initialized")
            try:
                workers = await asyncio.wait_for(
                    self.engine_client.collective_rpc(
                        f"cam_profile_{action}", timeout=RPC_TIMEOUT_S, args=(body,)
                    ),
                    RPC_TIMEOUT_S,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(504, "Profiling worker RPC timed out") from exc
            return {
                "workers": workers,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
            }

        return command

    async def _authenticate(self, request: Request) -> None:
        authorization = request.headers.get("authorization", "").encode("utf-8")
        if not secrets.compare_digest(authorization, f"Bearer {self.token}".encode()):
            raise HTTPException(status_code=401, detail="Invalid profiling token")
