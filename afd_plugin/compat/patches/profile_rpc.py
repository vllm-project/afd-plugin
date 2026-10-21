# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Preserve every DP worker result for CAM profiling control RPCs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from afd_plugin.compat.vllm import is_target_vllm_compatible

_R = TypeVar("_R")


# Upstream source: vllm/v1/engine/core_client.py, AsyncMPClient.collective_rpc_async,
# commit ced6857afa0ea7b2e3f0846a62e1394e90f15607 (vLLM v0.30.0).
# Patch reason: DPLB utilities discard all but the first engine's worker results.
# Patch functionality: retain every result only for the fixed CAM control methods.
# Signature: matches upstream; installed only on DPLBAsyncMPClient.
# Remove this adapter when upstream exposes an all-engine collective RPC result.
async def collective_rpc_async(
    self,
    method: str | Callable[..., _R],
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[_R]:
    # ### PATCH START: collect all CAM profiling participants
    if method in (
        "cam_profile_prepare",
        "cam_profile_commit",
        "cam_profile_stop",
        "cam_profile_status",
    ):
        results = await asyncio.gather(
            *(
                self._call_utility_async(
                    "collective_rpc", method, timeout, args, kwargs, engine=engine
                )
                for engine in self.core_engines
            )
        )
        return [worker for engine_results in results for worker in engine_results]
    # ### PATCH END: collect all CAM profiling participants
    return await self.call_utility_async(
        "collective_rpc", method, timeout, args, kwargs
    )


def apply_cam_profile_rpc_patch() -> bool:
    """Install the frontend adapter when the profiling endpoint is enabled."""
    if not is_target_vllm_compatible():
        return False

    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    DPLBAsyncMPClient.collective_rpc_async = collective_rpc_async
    return True
