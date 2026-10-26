# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe AFD rank topology helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True, slots=True)
class AFDRankMapping:
    """Rank mapping for the P2P connector.

    The P2P world always places FFN ranks first, followed by Attention ranks:
    ``[F0, F1, ..., A0, A1, ...]``. Each FFN rank owns one subgroup containing
    itself at subgroup rank 0 and one or more consecutive Attention ranks.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    subgroup_index: int
    rank_in_subgroup: int
    subgroup_ranks: tuple[int, ...]
    dp_metadata_destinations: tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_attention_top_min_size_rank(self) -> bool:
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size

    @property
    def participates_in_dp_metadata_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attention_top_min_size_rank


def topology_from_config(config: AFDConfig) -> tuple[int, int]:
    """Return ``(attention_size, ffn_size)`` for an AFD config."""

    return config.num_attention_ranks, config.num_ffn_ranks


def validate_p2p_topology(config: AFDConfig) -> None:
    attention_size, ffn_size = topology_from_config(config)
    if attention_size < ffn_size:
        raise ValueError(
            "P2pNcclAFDConnector currently requires num_attention_ranks >= "
            f"num_ffn_ranks, got {attention_size} < {ffn_size}",
        )


def resolve_role_rank(vllm_config: VllmConfig, config: AFDConfig) -> int:
    """Resolve this worker's connector-independent AFD role rank.

    The resolver linearizes vLLM's global DP rank and local PCP/TP coordinates.
    Connectors receive the result as runtime state and map it to their own
    communication-world rank.
    """

    parallel_config = vllm_config.parallel_config
    dp_size = int(parallel_config.data_parallel_size)
    pcp_size = int(parallel_config.prefill_context_parallel_size)
    tp_size = int(parallel_config.tensor_parallel_size)

    # vLLM's data_parallel_rank is global and already includes any configured
    # data_parallel_start_rank.
    dp_rank = int(parallel_config.data_parallel_rank) if dp_size > 1 else 0
    # Import rank accessors lazily so this topology module remains importable in
    # CPU-only configuration and documentation tests.
    if pcp_size > 1:
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_rank = int(get_pcp_group().rank_in_group)
    else:
        pcp_rank = 0
    if tp_size > 1:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
    else:
        tp_rank = 0

    role_rank = (dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
    if config.role == "attention":
        role_size = config.num_attention_ranks
    elif config.role == "ffn":
        role_size = config.num_ffn_ranks
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")
    if not 0 <= role_rank < role_size:
        raise ValueError(
            "AFD role rank derived from distributed ranks is out of range: "
            f"role={config.role!r}, dp_rank={dp_rank}, pcp_rank={pcp_rank}, "
            f"tp_rank={tp_rank}, role_size={role_size}",
        )
    return role_rank


def subgroup_attention_block(
    subgroup_index: int, attention_size: int, ffn_size: int
) -> range:
    """Attention role ranks owned by one FFN rank: contiguous, sizes within one."""
    return range(
        (subgroup_index * attention_size + ffn_size - 1) // ffn_size,
        ((subgroup_index + 1) * attention_size + ffn_size - 1) // ffn_size,
    )


def build_rank_mapping(
    config: AFDConfig,
    role_rank: int,
) -> AFDRankMapping:
    """Build the P2P rank mapping for one Attention or FFN process."""

    validate_p2p_topology(config)
    attention_size, ffn_size = topology_from_config(config)
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank * ffn_size // attention_size
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    # Balanced block distribution: Attention rank ``a`` joins the subgroup of
    # FFN rank ``a * F // A``, and ``subgroup_attention_block`` is the reverse
    # lookup that lists a subgroup's Attention ranks. The blocks are contiguous and
    # differ in size by at most one, so an integral ratio is no longer
    # required; when A is a multiple of F this is the same grouping as before.
    min_size = min(ffn_size, attention_size)
    subgroup_attention_ranks = [
        ffn_size + attention_rank
        for attention_rank in subgroup_attention_block(
            subgroup_index, attention_size, ffn_size
        )
    ]
    subgroup_ranks = tuple([subgroup_index] + subgroup_attention_ranks)
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    p2p_rank = role_rank + min_size if config.role == "attention" else role_rank

    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        destination = local_attention_rank
        while destination < ffn_size:
            destinations.append(destination)
            destination += min_size

    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=tuple(destinations),
    )


__all__ = [
    "AFDRankMapping",
    "build_rank_mapping",
    "resolve_role_rank",
    "topology_from_config",
    "validate_p2p_topology",
]
