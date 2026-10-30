# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Portable DSpark prompt-tail descriptors and actual-allocation slot mapping.

Positions are absolute prompt positions, including when only an SWA tail is
transferred. ``prompt_tokens`` counts rows actually computed on P: for the
native DSV4 Mooncake N-1 cut, it is N-1 rather than the original N-token prompt.
Block tables must retain their leading null entries; compact transfer-block
lists cannot map absolute positions. Transport, device writes and readiness
remain owned by the connector and model worker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

NULL_BLOCK_ID = 0
PAD_SLOT_ID = -1

DSparkContextValue = str | int | list[int]


class DSparkContextMetadata(TypedDict):
    request_id: str
    generation: str
    prompt_tokens: int
    window_start: int
    window_end: int
    aux_layer_ids: list[int]
    hidden_size: int


@dataclass(frozen=True)
class DSparkContextDescriptor:
    """Identify one allocation and its contiguous P-side auxiliary tail."""

    request_id: str
    generation: str
    prompt_tokens: int
    window_start: int
    window_end: int
    aux_layer_ids: tuple[int, ...]
    hidden_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("DSpark context requires a nonempty request_id")
        if not isinstance(self.generation, str) or not self.generation:
            raise ValueError("DSpark context requires a nonempty allocation generation")
        if type(self.prompt_tokens) is not int or self.prompt_tokens <= 0:
            raise ValueError("DSpark prompt_tokens must be a positive integer")
        if (
            type(self.window_start) is not int
            or type(self.window_end) is not int
            or not 0 <= self.window_start < self.window_end <= self.prompt_tokens
        ):
            raise ValueError(
                "DSpark window must be a nonempty range within P prompt rows"
            )
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("DSpark hidden_size must be a positive integer")
        if (
            not isinstance(self.aux_layer_ids, tuple)
            or not self.aux_layer_ids
            or any(type(layer) is not int or layer < 0 for layer in self.aux_layer_ids)
            or tuple(sorted(set(self.aux_layer_ids))) != self.aux_layer_ids
        ):
            raise ValueError(
                "DSpark auxiliary layer IDs must be ordered, unique integers"
            )

    @property
    def num_tokens(self) -> int:
        return self.window_end - self.window_start

    @property
    def feature_width(self) -> int:
        return self.hidden_size * len(self.aux_layer_ids)

    def to_dict(self) -> DSparkContextMetadata:
        """Return a JSON/MessagePack-compatible, independently owned descriptor."""
        return {
            "request_id": self.request_id,
            "generation": self.generation,
            "prompt_tokens": self.prompt_tokens,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "aux_layer_ids": list(self.aux_layer_ids),
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_dict(
        cls, metadata: DSparkContextMetadata | Mapping[str, DSparkContextValue]
    ) -> DSparkContextDescriptor:
        """Validate a decoded descriptor; no fields may be silently omitted."""
        required_fields = {
            "request_id",
            "generation",
            "prompt_tokens",
            "window_start",
            "window_end",
            "aux_layer_ids",
            "hidden_size",
        }
        if set(metadata) != required_fields:
            raise ValueError("DSpark context metadata fields do not match the schema")
        aux_layer_ids = metadata["aux_layer_ids"]
        if not isinstance(aux_layer_ids, list):
            raise ValueError("DSpark auxiliary layer IDs must be a serialized list")
        # Casts describe decoded wire types; __post_init__ validates all values.
        return cls(
            request_id=cast(str, metadata["request_id"]),
            generation=cast(str, metadata["generation"]),
            prompt_tokens=cast(int, metadata["prompt_tokens"]),
            window_start=cast(int, metadata["window_start"]),
            window_end=cast(int, metadata["window_end"]),
            aux_layer_ids=tuple(aux_layer_ids),
            hidden_size=cast(int, metadata["hidden_size"]),
        )


def build_draft_context_slots(
    positions: Sequence[int],
    block_ids_by_group: Mapping[int, Sequence[int]],
    block_sizes_by_group: Mapping[int, int],
    layer_group_ids: Sequence[int],
) -> list[list[int]]:
    """Map absolute rows to actual draft slots in loaded attention-layer order.

    A table includes null blocks for evicted SWA positions. Such rows produce
    ``PAD_SLOT_ID`` and must not write physical block zero, matching native
    DSpark input preparation. Group IDs are allocation IDs, never CAM layers.
    """
    group_ids = set(block_ids_by_group)
    if (
        not group_ids
        or group_ids != set(block_sizes_by_group)
        or any(type(group_id) is not int or group_id < 0 for group_id in group_ids)
        or any(
            type(group_id) is not int or group_id < 0
            for group_id in block_sizes_by_group
        )
    ):
        raise ValueError(
            "DSpark block tables and sizes require matching real group IDs"
        )
    if not layer_group_ids or any(
        type(group_id) is not int or group_id not in group_ids
        for group_id in layer_group_ids
    ):
        raise ValueError("Every DSpark attention layer must map to an allocated group")
    if any(type(position) is not int or position < 0 for position in positions):
        raise ValueError("DSpark context positions must be nonnegative integers")

    slots_by_group: dict[int, list[int]] = {}
    for group_id, block_ids in block_ids_by_group.items():
        block_size = block_sizes_by_group[group_id]
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("DSpark block sizes must be positive integers")
        if any(type(block_id) is not int or block_id < 0 for block_id in block_ids):
            raise ValueError("DSpark block IDs must be nonnegative integers")
        slots = []
        for position in positions:
            block_index, block_offset = divmod(position, block_size)
            if block_index >= len(block_ids):
                raise ValueError(
                    f"DSpark group {group_id} does not cover position {position}"
                )
            block_id = block_ids[block_index]
            slots.append(
                PAD_SLOT_ID
                if block_id == NULL_BLOCK_ID
                else block_id * block_size + block_offset
            )
        slots_by_group[group_id] = slots
    return [list(slots_by_group[group_id]) for group_id in layer_group_ids]


@dataclass(frozen=True)
class DSparkKVPage:
    """A target layer's logical page inside its actual physical allocation."""

    layer_name: str
    group_id: int
    base_address: int
    block_stride: int
    page_size: int
    block_size: int
    num_blocks: int
    format_tag: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.layer_name, str)
            or not self.layer_name
            or not isinstance(self.format_tag, str)
            or not self.format_tag
        ):
            raise ValueError("KV page requires layer identity and format")
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("KV page requires a real group ID")
        for value in (
            self.base_address,
            self.block_stride,
            self.page_size,
            self.block_size,
            self.num_blocks,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "KV page addresses, sizes and capacity must be positive"
                )
        if self.page_size > self.block_stride:
            raise ValueError("KV page payload exceeds its physical block stride")


def build_target_page_reads(
    local_pages: Sequence[DSparkKVPage],
    remote_pages: Sequence[DSparkKVPage],
    local_block_ids: Sequence[Sequence[int]],
    remote_block_ids: Sequence[Sequence[int]],
    prompt_tokens: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build engine reads with independent P/D strides, excluding draft pages.

    Block tables retain absolute logical block indices, including evicted null
    blocks. The first pointer list is local, matching Mooncake's sync-read API.
    Read each layer's payload rather than the packed block containing it.
    """
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("KV page reads require actual positive P prompt length")
    remote_by_name = {page.layer_name: page for page in remote_pages}
    local_names = {page.layer_name for page in local_pages}
    if (
        not local_names
        or len(local_names) != len(local_pages)
        or len(remote_by_name) != len(remote_pages)
        or local_names != remote_by_name.keys()
    ):
        raise ValueError("P/D target KV layer identities must match uniquely")
    local_ptrs: list[int] = []
    remote_ptrs: list[int] = []
    lengths: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for local in local_pages:
        remote = remote_by_name[local.layer_name]
        if (
            local.block_size != remote.block_size
            or local.page_size != remote.page_size
            or local.format_tag != remote.format_tag
        ):
            raise ValueError(f"P/D target KV page formats differ: {local.layer_name}")
        if local.group_id >= len(local_block_ids) or remote.group_id >= len(
            remote_block_ids
        ):
            raise ValueError("Target KV page references an absent allocation group")
        local_blocks = local_block_ids[local.group_id]
        remote_blocks = remote_block_ids[remote.group_id]
        num_prompt_blocks = (prompt_tokens + local.block_size - 1) // local.block_size
        if min(len(local_blocks), len(remote_blocks)) < num_prompt_blocks:
            raise ValueError("Target KV allocation does not cover P prompt positions")
        for block_index in range(num_prompt_blocks):
            local_block, remote_block = (
                local_blocks[block_index],
                remote_blocks[block_index],
            )
            if (
                type(local_block) is not int
                or type(remote_block) is not int
                or not 0 <= local_block < local.num_blocks
                or not 0 <= remote_block < remote.num_blocks
            ):
                raise ValueError("Target KV block ID is outside its allocation")
            if local_block == NULL_BLOCK_ID:
                continue
            if remote_block == NULL_BLOCK_ID:
                raise ValueError("P evicted a target KV block still required by D")
            read = (
                local.base_address + local_block * local.block_stride,
                remote.base_address + remote_block * remote.block_stride,
                local.page_size,
            )
            if read not in seen:
                local_ptrs.append(read[0])
                remote_ptrs.append(read[1])
                lengths.append(read[2])
                seen.add(read)
    return local_ptrs, remote_ptrs, lengths
