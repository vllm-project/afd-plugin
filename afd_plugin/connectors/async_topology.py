# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Rank layout shared by the asynchronous AFD connectors.

Both async connectors -- Ascend CAM and CUDA NVSHMEM -- lay their world out
Attention-first and derive expert placement the same way. Keeping that here lets
the CUDA connector reuse it without importing a backend module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.config import AFDConfig

ASYNC_MOE_REQUEST_SPLIT = "request"
ATTN_RANKS_PER_DP_CONFIG_KEY = "attn_ranks_per_dp"


@dataclass(frozen=True, slots=True)
class AFDAsyncTopology:
    """Role-local and world rank information for one async participant."""

    role: str
    role_rank: int
    world_rank: int
    attn_size: int
    ffn_size: int
    expert_per_rank: int

    @property
    def world_size(self) -> int:
        """Return the total number of Attention and FFN ranks."""
        return self.attn_size + self.ffn_size


def build_async_topology(
    afd_config: AFDConfig,
    role_rank: int,
    *,
    num_routed_experts: int | None = None,
) -> AFDAsyncTopology:
    """Validate role-local rank settings and derive the async world rank.

    The world is Attention-first: Attention role rank ``i`` maps to world rank
    ``i`` and FFN role rank ``j`` maps to ``num_attention_ranks + j``. Routed
    experts are distributed across FFN ranks using a ceiling division;
    production model layouts should keep the routed-expert count divisible by
    the FFN rank count.
    """
    attn_size = afd_config.num_attention_ranks
    ffn_size = afd_config.num_ffn_ranks
    if attn_size <= 0 or ffn_size <= 0:
        raise ValueError("AFD async topology sizes must be positive")
    if role_rank < 0:
        raise ValueError(f"AFD async role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attn_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attn_size})",
            )
        world_rank = role_rank
    elif afd_config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = attn_size + role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    expert_count = num_routed_experts or 1
    expert_per_rank = (expert_count + ffn_size - 1) // ffn_size
    return AFDAsyncTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        attn_size=attn_size,
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )


__all__ = [
    "ASYNC_MOE_REQUEST_SPLIT",
    "ATTN_RANKS_PER_DP_CONFIG_KEY",
    "AFDAsyncTopology",
    "build_async_topology",
]
