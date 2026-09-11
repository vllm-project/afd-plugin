# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Tests for the request-aligned ubatch split patch."""

from __future__ import annotations

import numpy as np
from vllm.v1.worker import gpu_model_runner, ubatch_utils
from vllm.v1.worker.ubatch_utils import UBatchSlice

from afd_plugin.compat.patches import ubatch_split
from afd_plugin.compat.patches.ubatch_split import (
    maybe_create_ubatch_slices,
    request_aligned_split_token,
)


def test_request_aligned_split_token_prefers_halfway_boundary() -> None:
    # Requests of 200, 100, 100 tokens: boundaries at 200 and 300; the even
    # split of 200 sits exactly on the first one.
    assert request_aligned_split_token(np.array([200, 100, 100])) == 200


def test_request_aligned_split_token_takes_nearest_boundary() -> None:
    # Boundaries at 300 and 301 for a 302-token batch; 151 is much closer
    # to 300.
    assert request_aligned_split_token(np.array([300, 1, 1])) == 300


def test_request_aligned_split_token_single_request_has_no_boundary() -> None:
    assert request_aligned_split_token(np.array([512])) is None


def test_request_aligned_split_token_prefers_an_aligned_boundary() -> None:
    # Boundaries at 33 and 64 for a 100-token batch. 33 is nearer the halfway
    # point, but only 64 leaves both ubatches' per-token views aligned, and
    # some attention kernels reject a misaligned view outright.
    assert request_aligned_split_token(np.array([33, 31, 36])) == 64


def test_request_aligned_split_token_falls_back_when_none_aligned() -> None:
    # Uniform decode: boundaries are 1, 2, 3, ... so none is ever aligned.
    # Refusing here would disable DBO for decode entirely, so the nearest
    # unaligned boundary is still taken.
    assert request_aligned_split_token(np.array([1] * 6)) == 3


def test_request_aligned_split_token_ignores_batch_edges() -> None:
    # A zero-token request puts a cumulative sum at 0; a boundary there would
    # empty the first ubatch.
    assert request_aligned_split_token(np.array([0, 100])) is None


def test_request_aligned_split_lands_on_request_boundary() -> None:
    # Two prefills of 300 and 100 tokens: the even token split (200) would cut
    # the first request; the patch must split at 300 instead.
    slices, slices_padded = maybe_create_ubatch_slices(
        True,
        np.array([300, 100]),
        num_tokens_padded=400,
        num_reqs_padded=2,
        num_ubatches=2,
    )
    assert slices is not None
    assert slices_padded is not None
    assert slices[0] == UBatchSlice(slice(0, 1), slice(0, 300))
    assert slices[1] == UBatchSlice(slice(1, 2), slice(300, 400))
    assert sum(s.num_tokens for s in slices_padded) == 400


def test_request_aligned_split_uniform_decode_matches_upstream() -> None:
    # One token per request: every boundary is a request edge, so the aligned
    # split coincides with the even token split.
    slices, _ = maybe_create_ubatch_slices(
        True,
        np.array([1, 1, 1, 1]),
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_ubatches=2,
    )
    assert slices is not None
    assert slices[0] == UBatchSlice(slice(0, 2), slice(0, 2))
    assert slices[1] == UBatchSlice(slice(2, 4), slice(2, 4))


def test_single_request_is_not_split() -> None:
    # A lone 512-token prefill has no interior request boundary: DBO must not
    # cut it in half.
    assert maybe_create_ubatch_slices(
        True,
        np.array([512]),
        num_tokens_padded=512,
        num_reqs_padded=1,
        num_ubatches=2,
    ) == (None, None)


def test_no_split_when_disabled() -> None:
    assert maybe_create_ubatch_slices(
        False,
        np.array([100, 100]),
        num_tokens_padded=200,
        num_reqs_padded=2,
        num_ubatches=2,
    ) == (None, None)


def test_explicit_split_point_keeps_upstream_behavior() -> None:
    slices, _ = maybe_create_ubatch_slices(
        True,
        np.array([300, 100]),
        num_tokens_padded=400,
        num_reqs_padded=2,
        num_ubatches=2,
        split_point=200,
    )
    # Upstream cuts at the given token count regardless of request edges, so
    # the first request straddles both ubatches.
    assert slices is not None
    assert slices[0] == UBatchSlice(slice(0, 1), slice(0, 200))
    assert slices[1] == UBatchSlice(slice(0, 2), slice(200, 400))


def test_non_two_ubatch_count_keeps_upstream_behavior() -> None:
    slices, _ = maybe_create_ubatch_slices(
        True,
        np.array([300, 100]),
        num_tokens_padded=400,
        num_reqs_padded=2,
        num_ubatches=4,
    )
    # Upstream's even token split, cutting through the first request.
    assert slices is not None
    assert [s.token_slice for s in slices] == [
        slice(0, 100),
        slice(100, 200),
        slice(200, 300),
        slice(300, 400),
    ]


def test_patch_installed_on_vllm_modules() -> None:
    assert gpu_model_runner.maybe_create_ubatch_slices is (
        ubatch_split.maybe_create_ubatch_slices
    )
    assert ubatch_utils.maybe_create_ubatch_slices is (
        ubatch_split.maybe_create_ubatch_slices
    )
