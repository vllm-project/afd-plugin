# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request-aligned ubatch splitting for AFD DBO.

Upstream vLLM's dual-batch overlap splits a batch at an even token count,
which cuts whichever request straddles that point into both ubatches. AFD's
DBO story is overlap between whole requests: one request runs in the first
ubatch while the other runs in the second, so the split must land on a
request boundary -- and a batch that cannot be split without cutting a
request must run whole instead of being divided.
"""

from __future__ import annotations

import numpy as np
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    _pad_out_ubatch_slices,
)

_DBO_UBATCH_COUNT = 2

# A ubatch's per-token tensors (positions, slot_mapping) are views into the
# step's buffers starting at the split point, so the split point decides their
# data pointers' alignment. DeepSeek-V4's CuTeDSL compressor kernel rejects any
# input below 64-byte alignment ("Misaligned Tensor data on argument #2"), and
# 16 four-byte tokens is the coarsest element stride that guarantees it for
# every per-token dtype in play. This is a preference, not a requirement: the
# splitter cannot see which kernels the model will run, and refusing to split
# every batch without an aligned boundary would disable DBO for uniform decode
# outright, where the boundaries are 1, 2, 3, ... by construction.
_UBATCH_SPLIT_TOKEN_ALIGNMENT = 16


def request_aligned_split_token(num_scheduled_tokens: np.ndarray) -> int | None:
    """Token index of the request boundary nearest the half-way point.

    Boundaries that leave every ubatch's per-token views aligned to
    ``_UBATCH_SPLIT_TOKEN_ALIGNMENT`` win over closer unaligned ones, because
    some attention kernels reject a misaligned view outright.

    Returns ``None`` when the batch has no interior request boundary -- fewer
    than two requests carrying tokens -- meaning it cannot be divided without
    cutting a request.
    """
    cumulative = np.cumsum(np.asarray(num_scheduled_tokens, dtype=np.int64))
    total = int(cumulative[-1]) if cumulative.size else 0
    # Interior boundaries: every request edge except the batch start and the
    # batch end. A boundary at either end would empty one ubatch.
    boundaries = np.unique(cumulative[:-1])
    boundaries = boundaries[(boundaries > 0) & (boundaries < total)]
    if boundaries.size == 0:
        return None
    aligned = boundaries[boundaries % _UBATCH_SPLIT_TOKEN_ALIGNMENT == 0]
    candidates = aligned if aligned.size else boundaries
    nearest = int(np.argmin(np.abs(candidates - total / 2)))
    return int(candidates[nearest])


# Patch reason: upstream maybe_create_ubatch_slices splits at an even token
# count, cutting the straddling request into both ubatches. AFD overlaps whole
# requests, so the split must fall on a request boundary, and a batch with no
# interior boundary (a single request) must not be split at all.
# Patch functionality: with no explicit split point and exactly two ubatches,
# split at the request boundary nearest the even token split; return
# (None, None) -- vLLM's no-ubatch state -- when there is no such boundary.
# Explicit split points and other ubatch counts keep upstream behavior.
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0, vllm/v1/worker/ubatch_utils.py
def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[list[UBatchSlice] | None, list[UBatchSlice] | None]:
    if not should_ubatch:
        return None, None

    # ### PATCH START: request-aligned ubatch split
    if split_point is None and num_ubatches == _DBO_UBATCH_COUNT:
        aligned = request_aligned_split_token(num_scheduled_tokens)
        if aligned is None:
            # No interior request boundary: dividing would cut a request in
            # half. Run the batch whole; vLLM treats absent slices as the
            # single-batch path.
            return None, None
        split_point = aligned
    # ### PATCH END: request-aligned ubatch split
    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    # ### PATCH START: keep the final split point a Python int
    # Upstream appends the numpy int32 straight off cu_num_tokens, which makes
    # the last ubatch's token_slice.stop -- and therefore its
    # num_actual_tokens, and every token count derived from it -- a
    # numpy.int32. Triton refuses to specialize a numpy scalar, so DeepSeek-V4
    # dies in _build_c128a_topk_metadata_kernel on the last ubatch.
    all_points = token_split_points + [int(cu_num_tokens[-1])]
    # ### PATCH END: keep the final split point a Python int

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def apply_request_aligned_ubatch_split() -> None:
    """Install the request-aligned splitter into vLLM's GPU runner.

    Both the execution and dummy-run call sites resolve the function through
    the ``gpu_model_runner`` namespace, so patching that alias (and the
    source module for any later importer) covers every caller. Idempotent via
    a marker attribute on the installed function.
    """
    from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
    from vllm.v1.worker import ubatch_utils as ubatch_utils_module

    if getattr(maybe_create_ubatch_slices, "_afd_request_aligned", False):
        return
    maybe_create_ubatch_slices._afd_request_aligned = True  # type: ignore[attr-defined]
    gpu_model_runner_module.maybe_create_ubatch_slices = maybe_create_ubatch_slices
    ubatch_utils_module.maybe_create_ubatch_slices = maybe_create_ubatch_slices


apply_request_aligned_ubatch_split()

__all__ = [
    "apply_request_aligned_ubatch_split",
    "maybe_create_ubatch_slices",
    "request_aligned_split_token",
]
