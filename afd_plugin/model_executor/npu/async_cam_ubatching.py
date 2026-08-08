# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure execution planning for NPU Async CAM MoE stages."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

_ASYNC_MOE_STAGE_COUNT = 2


@dataclass(frozen=True)
class AsyncMoeStage:
    """One ordered stage in the flattened Attention token layout.

    ``token_slice`` describes the stage's ordered real-token range in the
    parent batch. ``input_tokens`` includes only the minimum stage-local
    padding required by the Attention TP/SP layout.
    """

    request_slice: slice
    token_slice: slice
    input_tokens: int

    @property
    def actual_tokens(self) -> int:
        return int(self.token_slice.stop) - int(self.token_slice.start)


def plan_async_moe_stages(
    num_scheduled_tokens: Sequence[int],
    *,
    split: str,
    use_sequence_parallel: bool,
    tensor_parallel_size: int,
) -> tuple[AsyncMoeStage, ...] | None:
    """Plan two ordered CAM stages without importing a device runtime.

    Returns ``None`` when the selected policy cannot form two non-empty stages.
    """

    scheduled_tokens = tuple(int(token_count) for token_count in num_scheduled_tokens)
    if not scheduled_tokens or any(
        token_count <= 0 for token_count in scheduled_tokens
    ):
        raise ValueError(
            "Scheduled token counts must all be positive: "
            f"scheduled_tokens={scheduled_tokens}",
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            "tensor_parallel_size must be positive: "
            f"tensor_parallel_size={tensor_parallel_size}",
        )

    cumulative_tokens = tuple(accumulate(scheduled_tokens, initial=0))
    num_tokens = cumulative_tokens[-1]
    input_alignment = tensor_parallel_size if use_sequence_parallel else 1
    if split == "request":
        if len(scheduled_tokens) < _ASYNC_MOE_STAGE_COUNT:
            return None
        split_request = min(
            range(1, len(scheduled_tokens)),
            key=lambda request_index: (
                abs(
                    cumulative_tokens[request_index] * _ASYNC_MOE_STAGE_COUNT
                    - num_tokens
                ),
                abs(request_index * _ASYNC_MOE_STAGE_COUNT - len(scheduled_tokens)),
            ),
        )
        split_token = cumulative_tokens[split_request]
        stage_bounds = (
            (slice(0, split_request), 0, split_token),
            (
                slice(split_request, len(scheduled_tokens)),
                split_token,
                num_tokens,
            ),
        )
    elif split == "token":
        if tensor_parallel_size <= 1 or num_tokens < _ASYNC_MOE_STAGE_COUNT:
            return None
        split_token = (num_tokens + 1) // _ASYNC_MOE_STAGE_COUNT
        stage_bounds = tuple(
            (
                slice(
                    bisect_right(cumulative_tokens, token_start) - 1,
                    bisect_left(cumulative_tokens, token_stop),
                ),
                token_start,
                token_stop,
            )
            for token_start, token_stop in (
                (0, split_token),
                (split_token, num_tokens),
            )
        )
    else:
        raise ValueError(f"Unsupported Async CAM MoE split policy: {split!r}")

    return tuple(
        AsyncMoeStage(
            request_slice=request_slice,
            token_slice=slice(token_start, token_stop),
            input_tokens=_align_tokens(
                token_stop - token_start,
                input_alignment,
            ),
        )
        for request_slice, token_start, token_stop in stage_bounds
    )


def _align_tokens(num_tokens: int, alignment: int) -> int:
    return ((num_tokens + alignment - 1) // alignment) * alignment


__all__ = [
    "AsyncMoeStage",
    "plan_async_moe_stages",
]
