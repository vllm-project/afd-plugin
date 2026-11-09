# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch settings for the standalone CAM FFN process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from afd_plugin.ffn_faas.platforms.launch import validate_ffn_rank_settings


@dataclass(frozen=True)
class FFNConfig:
    model: Path
    ffn_ranks: int
    rank: int = 0
    local_rank: int = 0
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        # Engine/connector definitions and validation belong to vLLM/AFD.
        # Only external-launcher and host-shutdown facts live here.
        validate_ffn_rank_settings(
            ffn_ranks=self.ffn_ranks,
            rank=self.rank,
            local_rank=self.local_rank,
            shutdown_timeout=self.shutdown_timeout,
        )


def validate_static_experts(
    *,
    enable_eplb: bool,
    enable_elastic_ep: bool,
    redundant_experts: int,
    dynamic_eplb: bool = False,
) -> None:
    """Keep native expert state resident without enabling unsupported relocation."""
    if enable_eplb or enable_elastic_ep or redundant_experts != 0 or dynamic_eplb:
        raise ValueError(
            "standalone CAM requires static expert placement; EPLB, elastic EP "
            "and redundant experts are not supported"
        )


def validate_native_expert_map(
    *, num_experts: int, ffn_ranks: int, ffn_rank: int, expert_map: list[int] | None
) -> frozenset[int]:
    """Require the native sharding to match CAM's contiguous expert placement."""
    if ffn_ranks <= 0 or num_experts % ffn_ranks or not 0 <= ffn_rank < ffn_ranks:
        raise ValueError("CAM experts require a divisible, valid F topology")
    local_experts = num_experts // ffn_ranks
    if expert_map is None:
        if ffn_ranks != 1:
            raise ValueError("sharded native experts require an explicit map")
        # The native factory omits the map when every expert is local.
        return frozenset(range(num_experts))
    if len(expert_map) != num_experts:
        raise ValueError("native expert map has the wrong global expert count")
    owned = frozenset(expert for expert, local in enumerate(expert_map) if local >= 0)
    if sorted(expert_map[expert] for expert in owned) != list(range(local_experts)):
        raise ValueError("native expert map is not a local bijection")
    first = ffn_rank * local_experts
    end = first + local_experts
    if owned != frozenset(range(first, end)) or expert_map[first:end] != list(
        range(local_experts)
    ):
        raise ValueError("native placement differs from contiguous CAM routing")
    return owned
