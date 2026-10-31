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
    ``[F0, F1, ..., A0, A1, ...]``. Each subgroup contains the consecutive
    ranks on the larger side and one rank on the smaller side.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    ratio: int
    subgroup_index: int
    rank_in_subgroup: int
    subgroup_ranks: tuple[int, ...]
    ffn_peer_ranks: tuple[int, ...]
    attention_peer_ranks: tuple[int, ...]
    dp_metadata_destinations: tuple[int, ...] = field(default_factory=tuple)

    @property
    def attention_fans_out(self) -> bool:
        return self.ffn_size > self.attention_size

    @property
    def is_attention_top_min_size_rank(self) -> bool:
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size

    @property
    def participates_in_dp_metadata_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attention_top_min_size_rank


@dataclass(frozen=True, slots=True)
class AFDWindowRankMapping:
    """M2N rank mapping used by the Window connector."""

    world_rank: int
    attention_size: int
    ffn_size: int
    peer_ranks: tuple[int, ...]

    @property
    def world_size(self) -> int:
        return self.attention_size + self.ffn_size


@dataclass(frozen=True, slots=True)
class AFDWindowExpertLayout:
    """Fixed expert slots owned by one Window FFN role rank."""

    kind: str
    local_expert_start: int
    local_expert_count: int

    @property
    def is_shared(self) -> bool:
        return self.kind == "shared"


def build_window_expert_layout(
    *,
    routed_expert_num: int,
    ffn_size: int,
    ffn_rank: int,
) -> AFDWindowExpertLayout:
    """Return the deterministic shared-first Window expert layout."""

    if routed_expert_num <= 0:
        raise ValueError(
            f"Window AFD requires routed_expert_num > 0, got {routed_expert_num}"
        )
    if ffn_size < 2:
        raise ValueError(
            "Window AFD requires one shared and at least one routed FFN rank, "
            f"got ffn_size={ffn_size}"
        )
    if not 0 <= ffn_rank < ffn_size:
        raise ValueError(
            f"Window FFN rank {ffn_rank} is outside configured size {ffn_size}"
        )

    routed_rank_num = ffn_size - 1
    if routed_rank_num > routed_expert_num:
        raise ValueError(
            "Window AFD requires every routed FFN rank to own at least one "
            f"expert, got {routed_rank_num} ranks for {routed_expert_num} experts"
        )
    if ffn_rank == 0:
        return AFDWindowExpertLayout(
            kind="shared",
            local_expert_start=routed_expert_num,
            local_expert_count=1,
        )

    routed_rank = ffn_rank - 1
    base_count, extra_rank_num = divmod(routed_expert_num, routed_rank_num)
    local_expert_count = base_count + int(routed_rank < extra_rank_num)
    local_expert_start = routed_rank * base_count + min(routed_rank, extra_rank_num)
    return AFDWindowExpertLayout(
        kind="routed",
        local_expert_start=local_expert_start,
        local_expert_count=local_expert_count,
    )


def topology_from_config(config: AFDConfig) -> tuple[int, int]:
    """Return ``(attention_size, ffn_size)`` for an AFD config."""

    return config.num_attention_ranks, config.num_ffn_ranks


def build_window_rank_mapping(
    config: AFDConfig,
    role_rank: int,
) -> AFDWindowRankMapping:
    """Build FFN-first global ranks and opposite-role Window peers."""

    attention_size, ffn_size = topology_from_config(config)
    if attention_size <= 0 or ffn_size <= 0:
        raise ValueError(
            "Window AFD requires positive num_attention_ranks and "
            f"num_ffn_ranks, got {attention_size} and {ffn_size}"
        )
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    if config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                f"FFN role rank {role_rank} is outside configured size {ffn_size}"
            )
        world_rank = role_rank
        peer_ranks = tuple(range(ffn_size, ffn_size + attention_size))
    elif config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank "
                f"{role_rank} is outside configured size {attention_size}"
            )
        world_rank = ffn_size + role_rank
        peer_ranks = tuple(range(ffn_size))
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    return AFDWindowRankMapping(
        world_rank=world_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        peer_ranks=peer_ranks,
    )


def validate_p2p_topology(config: AFDConfig) -> None:
    attention_size, ffn_size = topology_from_config(config)
    if attention_size <= 0:
        raise ValueError(
            "P2P AFD connectors require num_attention_ranks to be positive, "
            f"got {attention_size}",
        )
    if ffn_size <= 0:
        raise ValueError(
            f"P2P AFD connectors require num_ffn_ranks to be positive, got {ffn_size}",
        )
    if attention_size < ffn_size and config.connector != "P2pHcclAFDConnector":
        raise ValueError(
            "P2P AFD connectors other than P2pHcclAFDConnector require "
            "num_attention_ranks >= num_ffn_ranks, got "
            f"{attention_size} < {ffn_size}",
        )
    if attention_size >= ffn_size and attention_size % ffn_size != 0:
        raise ValueError(
            "P2P AFD connectors require num_attention_ranks to be a "
            "multiple of num_ffn_ranks, got "
            f"{attention_size} and {ffn_size}",
        )
    if ffn_size > attention_size and ffn_size % attention_size != 0:
        raise ValueError(
            "P2pHcclAFDConnector requires num_ffn_ranks to be a multiple "
            "of num_attention_ranks, got "
            f"{ffn_size} and {attention_size}",
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


def build_rank_mapping(
    config: AFDConfig,
    role_rank: int,
) -> AFDRankMapping:
    """Build the P2P rank mapping for one Attention or FFN process."""

    validate_p2p_topology(config)
    attention_size, ffn_size = topology_from_config(config)
    ratio = max(attention_size, ffn_size) // min(attention_size, ffn_size)
    attention_fans_out = ffn_size > attention_size
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank if attention_fans_out else role_rank // ratio
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank // ratio if attention_fans_out else role_rank
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    min_size = min(ffn_size, attention_size)
    ffn_ranks = list(range(ffn_size))
    attention_ranks = list(range(ffn_size, ffn_size + attention_size))
    if attention_fans_out:
        ffn_peer_ranks = tuple(
            ffn_ranks[subgroup_index * ratio + offset] for offset in range(ratio)
        )
        attention_peer_ranks = (attention_ranks[subgroup_index],)
    else:
        ffn_peer_ranks = (ffn_ranks[subgroup_index],)
        attention_peer_ranks = tuple(
            attention_ranks[subgroup_index * ratio + offset] for offset in range(ratio)
        )
    subgroup_ranks = (*ffn_peer_ranks, *attention_peer_ranks)
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    p2p_rank = world_rank

    destinations: list[int] = []
    if config.role == "attention" and attention_fans_out:
        destinations.extend(ffn_peer_ranks)
    elif config.role == "attention" and ffn_size <= world_rank < ffn_size + min_size:
        destination = world_rank - ffn_size
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
        ratio=ratio,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        ffn_peer_ranks=ffn_peer_ranks,
        attention_peer_ranks=attention_peer_ranks,
        dp_metadata_destinations=tuple(destinations),
    )


__all__ = [
    "AFDRankMapping",
    "AFDWindowExpertLayout",
    "AFDWindowRankMapping",
    "build_rank_mapping",
    "build_window_expert_layout",
    "build_window_rank_mapping",
    "resolve_role_rank",
    "topology_from_config",
    "validate_p2p_topology",
]
