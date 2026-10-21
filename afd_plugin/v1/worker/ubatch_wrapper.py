# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared AFD metadata for native microbatch contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afd_plugin.connectors import AFDForwardContextMetadata
from afd_plugin.profiling.model import set_attention_profile_batch
from afd_plugin.v1.worker.attention_metadata import build_ubatch_dp_metadata_list

if TYPE_CHECKING:
    from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


def build_ubatch_afd_metadata(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: UBatchSlices,
    ubatch_idx: int,
) -> AFDForwardContextMetadata:
    """Clone parent AFD metadata for one vLLM ubatch."""

    if ubatch_idx < 0 or ubatch_idx >= len(ubatch_slices):
        raise IndexError(f"ubatch_idx {ubatch_idx} out of range")

    ubatch_slice = ubatch_slices[ubatch_idx]
    clone = afd_metadata.clone()
    clone.stage_idx = ubatch_idx
    clone.num_stages = len(ubatch_slices)
    clone.tokens_start_loc = [int(ubatch_slice.token_slice.start)]
    clone.requests_start_loc = [int(ubatch_slice.request_slice.start)]
    clone.tokens_lens = [int(ubatch_slice.num_tokens)]
    clone.tokens_unpadded_lens = [
        _resolve_ubatch_unpadded_tokens(afd_metadata, ubatch_slice, ubatch_idx),
    ]
    if len(clone.profile_batch) == 1 and len(ubatch_slices) > 1:
        # V1 can supply whole-batch counts before native ubatch slicing.
        _, requests, tokens = clone.profile_batch[0]
        set_attention_profile_batch(clone, requests, tokens, ubatch_slices)
    if clone.profile_batch:
        clone.profile_batch = [clone.profile_batch[ubatch_idx]]
    return clone


def build_ubatch_additional_kwargs(
    parent_additional_kwargs: dict[str, Any],
    afd_metadata: AFDForwardContextMetadata,
) -> dict[str, Any]:
    child_kwargs = dict(parent_additional_kwargs)
    child_kwargs["afd_metadata"] = afd_metadata
    return child_kwargs


def _resolve_ubatch_unpadded_tokens(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slice: UBatchSlice,
    ubatch_idx: int,
) -> int:
    unpadded_lens = afd_metadata.tokens_unpadded_lens
    if ubatch_idx < len(unpadded_lens):
        return int(unpadded_lens[ubatch_idx])
    return int(ubatch_slice.num_tokens)


__all__ = [
    "build_ubatch_additional_kwargs",
    "build_ubatch_afd_metadata",
    "build_ubatch_dp_metadata_list",
]
