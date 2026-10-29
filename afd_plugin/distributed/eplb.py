# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared helpers for expert-parallel load balancing (EPLB) observability.

Used by both EPLB paths: the P2P connector's FFN runner (routing and load
recording run on the FFN, so vLLM's ``EplbState`` is stepped natively) and
the DeepEP M2N connector's coordinators (routing runs on Attention, so the
plugin drives the rearrangement itself).
"""

from __future__ import annotations

import torch


def describe_eplb_placement(
    physical_to_logical_map: torch.Tensor,
    *,
    ffn_size: int,
) -> list[str]:
    """One line per MoE layer describing the placement EPLB committed.

    Each line names, per FFN rank, the logical experts its physical slots
    hold, and which logical experts are replicated (hold more than one slot).
    ``physical_to_logical_map`` is the committed map, shape
    ``(num_moe_layers, num_physical_experts)``; FFN rank ``g`` owns the
    contiguous slot block ``[g * P / F, (g + 1) * P / F)``.
    """
    num_physical = physical_to_logical_map.shape[1]
    slots_per_rank = num_physical // ffn_size
    lines: list[str] = []
    for layer_idx, layer_map in enumerate(physical_to_logical_map.tolist()):
        parts = []
        for rank in range(ffn_size):
            slots = layer_map[rank * slots_per_rank : (rank + 1) * slots_per_rank]
            experts = ", ".join(map(str, sorted(slots)))
            parts.append(f"rank {rank}=[{experts}]")
        replica_counts: dict[int, int] = {}
        for logical in layer_map:
            if logical >= 0:
                replica_counts[logical] = replica_counts.get(logical, 0) + 1
        replicated = {
            logical: count
            for logical, count in sorted(replica_counts.items())
            if count > 1
        }
        replicas = (
            "none"
            if not replicated
            else ", ".join(
                f"{logical}x{count}" for logical, count in replicated.items()
            )
        )
        lines.append(f"layer {layer_idx}: {' | '.join(parts)} | replicas: {replicas}")
    return lines


def describe_eplb_rank_load(
    expert_load_window: torch.Tensor,
    *,
    ffn_size: int,
) -> str:
    """Per-FFN-rank token load recorded in one EPLB load window.

    ``expert_load_window`` is the recorded load, shape
    ``(window_size, num_moe_layers, num_physical_experts)``; a rank's load is
    the sum over its contiguous slot block.
    """
    slots_per_rank = expert_load_window.shape[2] // ffn_size
    rank_loads = expert_load_window.sum(dim=(0, 1)).tolist()
    parts = []
    for rank in range(ffn_size):
        slots = rank_loads[rank * slots_per_rank : (rank + 1) * slots_per_rank]
        parts.append(f"rank {rank}={int(sum(slots))} tokens")
    return ", ".join(parts)


__all__ = [
    "describe_eplb_placement",
    "describe_eplb_rank_load",
]
