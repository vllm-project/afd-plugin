# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe wire geometry and configuration for the async GPU window.

Version 5 extends the single-node v4 format with 64-bit arrival signals for
NVSHMEM remote transfers. Sender rank and
stage are identified by the region and ring; the header carries only layer,
batch size and routing. Replies need no header because their shape is known.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from afd_plugin.config_utils import coerce_extra_positive_int
from afd_plugin.connectors.base import ConnectorExtraInfo

HEADER_MAGIC = 0x41464447
HEADER_VERSION = 5
HEADER_HOST_WORDS = 5
H_ROUTED_TOKENS = 5
H_SEGMENT_START = 6
HEADER_FIXED_WORDS = 7
FLAG_EMPTY = 0
FLAG_REPLY_READY = 1
FLAG_SHUTDOWN_BIT = 1 << 1
FIELD_ALIGNMENT = 256
HEADER_WORD_BYTES = 4
SIGNAL_WORD_BYTES = 8


def nvshmem_pe_mapping(hosts: Sequence[str]) -> tuple[int, ...]:
    """Map A-first AFD ranks to node-contiguous NVSHMEM PEs.

    NVSHMEM's built-in same-local-rank teams assume block placement. Keep
    routing ranks stable while arranging only the transport's bootstrap IDs.
    """
    ranks_by_host: dict[str, list[int]] = {}
    for rank, host in enumerate(hosts):
        ranks_by_host.setdefault(host, []).append(rank)
    counts = [len(ranks) for ranks in ranks_by_host.values()]
    if len(set(counts)) != 1:
        raise ValueError(
            "NVSHMEM requires equal PE counts on every node; "
            f"AFD placement has {dict(zip(ranks_by_host, counts, strict=True))}"
        )
    mapping = [0] * len(hosts)
    for pe, rank in enumerate(
        rank for ranks in ranks_by_host.values() for rank in ranks
    ):
        mapping[rank] = pe
    return tuple(mapping)


def align_bytes(offset: int) -> int:
    return (offset + FIELD_ALIGNMENT - 1) // FIELD_ALIGNMENT * FIELD_ALIGNMENT


@dataclass(frozen=True, slots=True)
class SlotLayout:
    header_words: int
    partial_cap: int
    token_cap: int
    shared_cap: int
    hidden_size: int
    payload_itemsize: int
    header_off: int
    expand_idx_off: int
    weights_off: int
    routed_x_off: int
    shared_x_off: int
    slot_bytes: int

    @classmethod
    def build(
        cls,
        *,
        expert_per_rank: int,
        partial_cap: int,
        token_cap: int,
        shared_cap: int,
        hidden_size: int,
        payload_itemsize: int,
    ) -> SlotLayout:
        if min(expert_per_rank, partial_cap, token_cap, hidden_size) <= 0:
            raise ValueError(
                "Window experts, capacities and hidden size must be positive"
            )
        if shared_cap < 0 or shared_cap > token_cap:
            raise ValueError("Window shared capacity must lie within token capacity")
        if payload_itemsize not in (2, 4):
            raise ValueError(
                "Async GPU payloads require 2-byte or 4-byte floating point"
            )
        header_words = HEADER_FIXED_WORDS + expert_per_rank
        expand_idx_off = align_bytes(header_words * HEADER_WORD_BYTES)
        weights_off = align_bytes(expand_idx_off + partial_cap * HEADER_WORD_BYTES)
        routed_x_off = align_bytes(weights_off + partial_cap * HEADER_WORD_BYTES)
        shared_x_off = align_bytes(
            routed_x_off + token_cap * hidden_size * payload_itemsize,
        )
        slot_bytes = align_bytes(
            shared_x_off + shared_cap * hidden_size * payload_itemsize,
        )
        return cls(
            header_words,
            partial_cap,
            token_cap,
            shared_cap,
            hidden_size,
            payload_itemsize,
            0,
            expand_idx_off,
            weights_off,
            routed_x_off,
            shared_x_off,
            slot_bytes,
        )


@dataclass(frozen=True, slots=True)
class SlotHeader:
    layer_idx: int
    num_tokens: int
    routed_tokens: int
    flags: int
    segment_start: int
    expert_counts: tuple[int, ...]

    @property
    def is_shutdown(self) -> bool:
        return bool(self.flags & FLAG_SHUTDOWN_BIT)

    def validate(self, layout: SlotLayout) -> None:
        if self.flags not in (0, FLAG_SHUTDOWN_BIT):
            raise RuntimeError("Async GPU header has unsupported flags")
        if self.layer_idx < 0:
            raise RuntimeError("Async GPU layer index must be non-negative")
        if not 0 <= self.num_tokens <= layout.token_cap:
            raise RuntimeError("Async GPU token count exceeds window capacity")
        if self.routed_tokens < 0 or self.segment_start < 0:
            raise RuntimeError(
                "Async GPU partial counts and offsets must be non-negative"
            )
        if self.segment_start + self.routed_tokens > layout.partial_cap:
            raise RuntimeError("Async GPU routed segment exceeds window capacity")
        max_partials = self.num_tokens * layout.partial_cap // layout.token_cap
        if self.segment_start + self.routed_tokens > max_partials:
            raise RuntimeError("Async GPU routed segment exceeds this batch's partials")
        if len(self.expert_counts) != layout.header_words - HEADER_FIXED_WORDS:
            raise RuntimeError(
                "Async GPU expert count width differs from window geometry"
            )
        if any(count < 0 for count in self.expert_counts):
            raise RuntimeError("Async GPU expert counts must be non-negative")
        if sum(self.expert_counts) != self.routed_tokens:
            raise RuntimeError("Async GPU expert counts do not sum to routed tokens")
        if self.is_shutdown and (self.num_tokens or self.routed_tokens):
            raise RuntimeError("Async GPU terminal message must carry no token rows")


def encode_header_words(header: SlotHeader, layout: SlotLayout) -> tuple[int, ...]:
    header.validate(layout)
    return (
        HEADER_MAGIC,
        HEADER_VERSION,
        header.layer_idx,
        header.num_tokens,
        header.flags,
        header.routed_tokens,
        header.segment_start,
        *header.expert_counts,
    )


def decode_header_words(words: Sequence[int], layout: SlotLayout) -> SlotHeader:
    if len(words) != layout.header_words:
        raise RuntimeError("Async GPU header byte length differs from window geometry")
    if words[0] != HEADER_MAGIC:
        raise RuntimeError("Async GPU header magic mismatch")
    if words[1] != HEADER_VERSION:
        raise RuntimeError(f"Async GPU header version {words[1]} is not supported")
    header = SlotHeader(
        words[2], words[3], words[5], words[4], words[6], tuple(words[7:])
    )
    header.validate(layout)
    return header


@dataclass(frozen=True)
class GpuAsyncExtraInfo(ConnectorExtraInfo):
    attn_ranks_per_dp: int = 1
    ring_depth: int = 0
    ring_depth_pinned: bool = False
    recv_poll_timeout_ms: int = 50

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> GpuAsyncExtraInfo:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise TypeError("Async GPU connector_extra_config must be a mapping")
        allowed = {"attn_ranks_per_dp", "ring_depth", "recv_poll_timeout_ms"}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "Unknown async GPU connector field(s): " + ", ".join(unknown)
            )
        return cls(
            attn_ranks_per_dp=coerce_extra_positive_int(
                raw.get("attn_ranks_per_dp", 1),
                field_name="attn_ranks_per_dp",
            ),
            ring_depth=coerce_extra_positive_int(
                raw.get("ring_depth", 1),
                field_name="ring_depth",
            ),
            ring_depth_pinned="ring_depth" in raw,
            recv_poll_timeout_ms=coerce_extra_positive_int(
                raw.get("recv_poll_timeout_ms", 50),
                field_name="recv_poll_timeout_ms",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
            "recv_poll_timeout_ms": self.recv_poll_timeout_ms,
        }
        if self.ring_depth_pinned:
            result["ring_depth"] = self.ring_depth
        return result


def shared_token_slice(
    ffn_rank: int,
    ffn_size: int,
    num_tokens: int,
    has_shared_experts: bool,
) -> slice:
    if not has_shared_experts:
        return slice(0, 0)
    return slice(
        ffn_rank * num_tokens // ffn_size,
        (ffn_rank + 1) * num_tokens // ffn_size,
    )


def stage_rings(stage_idx: int, num_stages: int, ring_depth: int) -> list[int]:
    if not 0 <= stage_idx < num_stages:
        raise ValueError(f"Async GPU stage {stage_idx} is outside [0, {num_stages})")
    if ring_depth < num_stages:
        raise ValueError("Async GPU ring_depth must cover every live stage")
    return list(range(stage_idx, ring_depth, num_stages))
