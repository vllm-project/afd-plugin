# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared CPU-only launch-environment checks for standalone FFN backends."""

from __future__ import annotations

import os
from dataclasses import dataclass

V2_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"


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
