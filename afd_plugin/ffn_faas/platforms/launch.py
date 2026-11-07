# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared CPU-only launch-environment checks for standalone FFN backends."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

V2_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
MAX_PORT = 65535
EXTERNAL_LAUNCHER = "external_launcher"

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class FFNLauncherEnvironment:
    """One rank's externally launched F-only world coordinates."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int


def read_ffn_launcher_environment(ffn_ranks: int) -> FFNLauncherEnvironment:
    """Read ``torchrun`` coordinates and require an F-only world."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != ffn_ranks:
        raise ValueError("F-only launcher WORLD_SIZE must equal --ffn-ranks")
    return FFNLauncherEnvironment(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
    )


def require_v2_model_runner() -> None:
    """Require the model-runner generation used by both A and F ranks."""
    if os.environ.get(V2_MODEL_RUNNER_ENV) != "1":
        raise ValueError(f"set {V2_MODEL_RUNNER_ENV}=1 on both A and F")


def validate_common_ffn_launch(
    *,
    attention_ranks: int,
    ffn_ranks: int,
    rank: int,
    local_rank: int,
    host: str,
    port: int,
    shutdown_timeout: float,
) -> None:
    """Validate launch fields shared by every standalone FFN backend."""
    if attention_ranks <= 0 or ffn_ranks <= 0:
        raise ValueError("Attention and FFN rank counts must be positive")
    if not 0 <= rank < ffn_ranks or local_rank < 0:
        raise ValueError("invalid F rank or local rank")
    if not host or not 0 < port <= MAX_PORT:
        raise ValueError("FFN rendezvous requires a host and a valid port")
    if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
        raise ValueError("shutdown timeout must be finite and positive")


def ffn_engine_kwargs(
    *,
    model: Path | str,
    ffn_ranks: int,
    ffn_rank: int,
    additional_config: Mapping[str, Any],
    platform_extras: Mapping[str, Any],
) -> dict[str, Any]:
    """Build common ``EngineArgs`` keywords and merge backend-specific extras."""
    kwargs: dict[str, Any] = {
        "model": str(model),
        "tensor_parallel_size": 1,
        "data_parallel_size": ffn_ranks,
        "data_parallel_rank": ffn_rank,
        "enable_expert_parallel": True,
        "distributed_executor_backend": EXTERNAL_LAUNCHER,
        "additional_config": dict(additional_config),
    }
    kwargs.update(platform_extras)
    return kwargs


def reassert_external_launcher(config: VllmConfig) -> None:
    """Keep an already launched F-only world from borrowing an executor."""
    config.parallel_config.distributed_executor_backend = EXTERNAL_LAUNCHER
