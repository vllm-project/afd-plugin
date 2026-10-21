# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build request-local identity fragments from immutable CPU batch snapshots.

InputSnapshot freezes runner metadata; build_fragments intersects token slices
with real queries, excluding padding. These helpers serve the older format
read by analyze.py; current LIGHTWEIGHT uses model.py's scalar annotations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    req_ids: tuple[str, ...]
    query_offsets: tuple[int, ...]
    computed_before: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    execution_modes: tuple[str, ...]
    real_extent: int

    @classmethod
    def from_cpu(
        cls,
        req_ids: Sequence[str],
        query_offsets: Iterable[int],
        computed_before: Iterable[int],
        prompt_lengths: Iterable[int],
        is_prefilling: Iterable[bool],
        real_extent: int,
    ) -> InputSnapshot:
        # Callers pass native CPU views before reusable runner buffers advance.
        return cls(
            tuple(req_ids),
            tuple(int(value) for value in query_offsets),
            tuple(int(value) for value in computed_before),
            tuple(int(value) for value in prompt_lengths),
            tuple("PREFILL" if value else "DECODE" for value in is_prefilling),
            int(real_extent),
        )


def build_fragments(
    snapshot: InputSnapshot,
    group_id: str,
    stage_idx: int,
    token_begin: int,
    token_end: int,
) -> list[dict]:
    """Intersect a physical full-batch slice with real request query intervals."""
    if token_begin < 0 or token_end < token_begin:
        raise ValueError("invalid full-batch token interval")
    fragments = []
    for index, request_id in enumerate(snapshot.req_ids):
        query_begin = snapshot.query_offsets[index]
        begin = max(token_begin, query_begin)
        end = min(token_end, snapshot.query_offsets[index + 1], snapshot.real_extent)
        if begin >= end:
            continue
        position_begin = snapshot.computed_before[index] + begin - query_begin
        position_end = position_begin + end - begin
        prompt_length = snapshot.prompt_lengths[index]
        boundaries = [position_begin]
        if position_begin < prompt_length < position_end:
            boundaries.append(prompt_length)
        boundaries.append(position_end)
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            fragments.append(
                {
                    "record_type": "FRAGMENT",
                    "fragment_id": (
                        f"{group_id}:{stage_idx}:{begin}-{end}:{index}:{start}-{stop}"
                    ),
                    "request_id": request_id,
                    "group_id": group_id,
                    "stage_idx": stage_idx,
                    "token_start": start,
                    "token_end": stop,
                    "mode": snapshot.execution_modes[index],
                    "token_origin": "PROMPT" if start < prompt_length else "GENERATED",
                }
            )
    return fragments
