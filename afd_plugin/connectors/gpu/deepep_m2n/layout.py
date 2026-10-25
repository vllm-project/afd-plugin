# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expert placement over the DeepEP M2N union world.

Adapted from FastAFD ``python/minisgl/moe/deepep_m2n_union.py`` (``3c71619``).

DeepEP treats the union of Attention and FFN ranks as one symmetric EP group
in which every rank owns ``experts_per_rank`` experts. FastAFD orders the
union ``[Attention.., FFN..]`` and remaps real expert ids by an offset. The
plugin orders AFD worlds FFN first (``[F0.., A0..]``, as the P2P connector
does), so FFN rank ``g`` owns DeepEP experts ``[g * E_local, (g + 1) * E_local)``,
which are exactly the real expert ids: no remap. Attention ranks own dummy
expert slots after the real ones that routing never selects.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        if role == "ffn":
            _check_rank("ffn role rank", role_rank, self.ffn_size)
            return role_rank
        if role == "attention":
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
