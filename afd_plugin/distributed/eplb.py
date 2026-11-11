# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared helpers for expert-parallel load balancing (EPLB) observability.

Used by both EPLB paths: the P2P connector's FFN runner (routing and load
recording run on the FFN, so vLLM's ``EplbState`` is stepped natively) and
the DeepEP M2N connector's coordinators (routing runs on Attention, so the
plugin drives the rearrangement itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EplbLoadSummary:
    """Expert-parallel load figures, summed over MoE layers.

    ``avg_tokens`` and ``max_tokens`` are per-layer means and maxima across
    ranks, summed over layers; ``balancedness`` is their ratio, so 1.0 is
    perfectly balanced. The window the figures describe is whatever the caller
    scored: one step, or an EPLB load window.
    """

    avg_tokens: float
    max_tokens: float
    balancedness: float


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


def summarize_rank_load(num_tokens_per_rank: torch.Tensor) -> EplbLoadSummary:
    """Balancedness of a per-rank load table, as vLLM's EPLB computes it.

    ``num_tokens_per_rank`` has shape ``(num_moe_layers, ffn_size)``. Per layer
    the mean and the max rank load are taken across ranks and both are summed
    over layers; balancedness is their ratio, so 1.0 is perfectly balanced.
    Recorded load is integer, so the mean is taken in float, as vLLM does.
    """
    num_tokens_per_rank = num_tokens_per_rank.float()
    avg_tokens = float(num_tokens_per_rank.mean(dim=1).sum())
    max_tokens = float(num_tokens_per_rank.max(dim=1).values.sum())
    balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0
    return EplbLoadSummary(
        avg_tokens=avg_tokens,
        max_tokens=max_tokens,
        balancedness=balancedness,
    )


def logical_load_from_window(
    expert_load_window: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor:
    """Aggregate a recorded load window into logical experts.

    ``expert_load_window`` is the recorded load, shape
    ``(window_size, num_moe_layers, num_physical_experts)``, indexed by the
    slots of ``logical_to_physical_map`` (``(num_moe_layers, num_logical,
    max_replicas)``, valid entries designated by ``logical_replica_count``).
    The window is summed over its steps and gathered through the map, giving
    ``(num_moe_layers, num_logical)`` tokens per logical expert, which is the
    only load EPLB's policy looks at.
    """
    num_layers, num_logical, max_replicas = logical_to_physical_map.shape
    slot_is_used = torch.arange(
        max_replicas, device=logical_to_physical_map.device
    ).unsqueeze(0).unsqueeze(1) < logical_replica_count.unsqueeze(-1)
    used_slots = logical_to_physical_map.masked_fill(~slot_is_used, 0).long()
    window_load = expert_load_window.sum(dim=0)
    per_slot = window_load.unsqueeze(1).expand(num_layers, num_logical, -1)
    per_replica = per_slot.gather(2, used_slots).masked_fill(~slot_is_used, 0)
    return per_replica.sum(dim=-1).to(expert_load_window.dtype)


def eplb_map_loss(
    logical_load: torch.Tensor,
    candidate_map: torch.Tensor,
    *,
    ffn_size: int,
) -> EplbLoadSummary:
    """Score one placement with the quantity ``DefaultEplbPolicy`` minimises.

    The policy never sees realised per-slot counts: it aggregates the window
    into logical experts, divides each expert's load evenly over the replicas
    the *candidate* map gives it, and packs those per-replica loads into ranks.
    This function evaluates that model in the per-layer mean/max form EPLB logs
    balancedness in, so a deployed map and the policy's proposal are directly
    comparable on one window.

    ``logical_load`` is ``(num_moe_layers, num_logical)`` tokens per logical
    expert, as ``logical_load_from_window`` produces. ``candidate_map`` is the
    placement to score, ``(num_moe_layers, num_physical_experts)``; slots
    mapped to ``-1`` carry no load and no replica.
    """
    logical_load = logical_load.cpu().float()
    num_layers, num_logical = logical_load.shape
    num_slots = candidate_map.shape[1]
    # Unused slots are gathered into a trailing guard column so they neither
    # add load nor inflate an expert's replica count.
    slot_is_used = candidate_map.cpu() >= 0
    candidate_index = torch.where(
        slot_is_used,
        candidate_map.cpu().long(),
        torch.full_like(candidate_map.cpu().long(), num_logical),
    )
    logical_load = torch.cat(
        [logical_load, torch.zeros((num_layers, 1), dtype=logical_load.dtype)], dim=1
    )
    replica_count = torch.zeros_like(logical_load)
    replica_count.scatter_add_(1, candidate_index, slot_is_used.float())
    load_per_replica = logical_load.div(replica_count.clamp_min(1)).gather(
        1, candidate_index
    )
    load_per_rank = (
        load_per_replica.mul(slot_is_used.float())
        .reshape(num_layers, ffn_size, num_slots // ffn_size)
        .sum(dim=-1)
    )
    return summarize_rank_load(load_per_rank)


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
    "EplbLoadSummary",
    "describe_eplb_placement",
    "describe_eplb_rank_load",
    "eplb_map_loss",
    "logical_load_from_window",
    "summarize_rank_load",
]
