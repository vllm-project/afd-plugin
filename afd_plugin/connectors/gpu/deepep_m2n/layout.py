# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expert placement over the DeepEP M2N union world.

Adapted from FastAFD ``python/minisgl/moe/deepep_m2n_union.py`` (``3c71619``).

DeepEP treats the union of Attention and FFN ranks as one symmetric EP group
in which every rank owns ``experts_per_rank`` experts. FastAFD orders the
union ``[Attention.., FFN..]`` and remaps real expert ids by an offset. The
plugin orders the AFD role world FFN first (``[F0.., A0..]``, as the P2P
connector does), so FFN rank ``g`` owns real experts
``[g * E_local, (g + 1) * E_local)``. Attention ranks own dummy expert slots
that routing never selects.

The transport world (``plan_transport_layout``) holds the same ranks in
physical order: NVLink domain by domain, as DeepEP numbers ranks, with each
domain's FFN ranks first. When the FFN ranks fill the first domains, it equals
the role world and DeepEP expert ids are the real ones (afdEP-comm's FFN
prefix). When every domain holds ``f`` FFN ranks, afdEP-comm's per-domain mode
places FFN rank ``j`` at transport rank ``(j // f) * U + j % f`` (``U`` ranks
per domain) and translates the real expert ids itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

FFN_ROLE = "ffn"
ATTENTION_ROLE = "attention"


@dataclass(frozen=True)
class DeepEPM2NLayout:
    ffn_size: int
    attn_size: int
    num_experts: int

    def __post_init__(self) -> None:
        if self.ffn_size <= 0 or self.attn_size <= 0:
            raise ValueError(
                "DeepEP M2N needs at least one FFN and one Attention rank, got "
                f"ffn_size={self.ffn_size} attn_size={self.attn_size}"
            )
        if self.num_experts % self.ffn_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"ffn_size={self.ffn_size}"
            )

    @property
    def union_size(self) -> int:
        return self.ffn_size + self.attn_size

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ffn_size

    @property
    def deepep_num_experts(self) -> int:
        """Expert count DeepEP sees: real experts plus Attention dummy slots."""
        return self.union_size * self.experts_per_rank

    def world_rank(self, role: str, role_rank: int) -> int:
        """Rank in the role world ``[F0.., A0..]``."""
        if role == FFN_ROLE:
            _check_rank("ffn role rank", role_rank, self.ffn_size)
            return role_rank
        if role == ATTENTION_ROLE:
            _check_rank("attention role rank", role_rank, self.attn_size)
            return self.ffn_size + role_rank
        raise ValueError(f"unknown AFD role {role!r}")

    def expert_range(self, ffn_rank: int) -> range:
        """Real expert ids owned by FFN rank ``ffn_rank``."""
        _check_rank("ffn rank", ffn_rank, self.ffn_size)
        start = ffn_rank * self.experts_per_rank
        return range(start, start + self.experts_per_rank)


def _check_rank(name: str, rank: int, size: int) -> None:
    if not 0 <= rank < size:
        raise ValueError(f"{name} must be in [0, {size}), got {rank}")


@dataclass(frozen=True)
class RankPlacement:
    """Where one rank runs, as it reports in the handshake."""

    role: str
    role_rank: int
    nvlink_domain: str


@dataclass(frozen=True)
class TransportLayout:
    """The physical rank order afdEP-comm's transport sees.

    Attributes:
        order: ``(role, role_rank)`` of each transport rank.
        ffn_ranks_per_domain: afdEP-comm's ``ffn_ranks_per_domain``: 0 when the
            FFN ranks are transport ranks ``[0, F)`` (FFN prefix), else the FFN
            rank count of every NVLink domain (per-domain mode).
        num_domains: NVLink domains.
        domain_size: ranks per NVLink domain.
        shape: ``<a>A<f>F`` per domain in transport order, for logs and errors.
    """

    order: tuple[tuple[str, int], ...]
    ffn_ranks_per_domain: int
    num_domains: int
    domain_size: int
    shape: str

    def transport_rank(self, role: str, role_rank: int) -> int:
        return self.order.index((role, role_rank))

    @property
    def is_identity(self) -> bool:
        """The transport order is the role world ``[F0.., A0..]``."""
        ffn_size = sum(role == FFN_ROLE for role, _ in self.order)
        return all(
            rank == (role_rank if role == FFN_ROLE else ffn_size + role_rank)
            for rank, (role, role_rank) in enumerate(self.order)
        )

    @property
    def mode(self) -> str:
        return "per_domain" if self.ffn_ranks_per_domain else "ffn_prefix"

    def summary(self) -> str:
        per_domain = (
            f" f={self.ffn_ranks_per_domain}" if self.ffn_ranks_per_domain else ""
        )
        return (
            f"{self.mode}{per_domain} "
            f"domains={self.num_domains}x{self.domain_size} placement={self.shape}"
        )


def plan_transport_layout(
    placements: Sequence[RankPlacement], ffn_size: int
) -> TransportLayout:
    """Order the ranks for afdEP-comm; every rank computes the same layout.

    Each NVLink domain is a contiguous block (DeepEP numbers ranks domain by
    domain) with its FFN ranks first. Domains follow their lowest role-world
    rank, so the domains holding FFN ranks come first. afdEP-comm takes two
    shapes: the FFN ranks fill the first domains (FFN prefix: one domain, or
    role-split pods), or every domain holds the same number of FFN ranks, in
    pod order (per-domain mode). Every domain must hold the same number of
    ranks.
    """

    def role_world_rank(placement: RankPlacement) -> int:
        if placement.role == FFN_ROLE:
            return placement.role_rank
        return ffn_size + placement.role_rank

    domains: dict[str, list[RankPlacement]] = {}
    for placement in placements:
        domains.setdefault(placement.nvlink_domain, []).append(placement)
    ordered = sorted(
        domains.values(), key=lambda ranks: min(map(role_world_rank, ranks))
    )
    order: list[tuple[str, int]] = []
    shapes: list[str] = []
    ffn_counts: set[int] = set()
    for ranks in ordered:
        ffn = sorted(p.role_rank for p in ranks if p.role == FFN_ROLE)
        attention = sorted(p.role_rank for p in ranks if p.role != FFN_ROLE)
        order += [(FFN_ROLE, rank) for rank in ffn]
        order += [(ATTENTION_ROLE, rank) for rank in attention]
        shapes.append(f"{len(attention)}A{len(ffn)}F")
        ffn_counts.add(len(ffn))
    shape = ",".join(shapes)
    domain_sizes = {len(ranks) for ranks in ordered}
    if len(domain_sizes) != 1:
        raise RuntimeError(
            f"NVLink domains hold {shape}: DeepEP needs the same number of ranks "
            "in every NVLink domain (pod)"
        )
    ffn_order = [rank for role, rank in order if role == FFN_ROLE]
    if ffn_order != list(range(ffn_size)):
        raise RuntimeError(
            f"NVLink domains ({shape}) hold FFN ranks in the order {ffn_order}: "
            "FFN DP ranks must rise in pod order (--data-parallel-start-rank)"
        )
    if order[:ffn_size] == [(FFN_ROLE, rank) for rank in range(ffn_size)]:
        ffn_ranks_per_domain = 0
    elif len(ffn_counts) == 1:
        ffn_ranks_per_domain = ffn_counts.pop()
    else:
        raise RuntimeError(
            f"NVLink domains hold {shape}: every domain (pod) must hold the same "
            "number of FFN ranks, or the FFN ranks must fill the first domains"
        )
    return TransportLayout(
        order=tuple(order),
        ffn_ranks_per_domain=ffn_ranks_per_domain,
        num_domains=len(ordered),
        domain_size=domain_sizes.pop(),
        shape=shape,
    )
