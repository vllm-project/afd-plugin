# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from afd_plugin.v1.worker.cuda_graph import (
    FULL_DECODE_ONLY,
    AFDGraphRunMode,
    cudagraph_mode_name,
    graph_run_mode,
    make_ffn_graph_key,
    pad_counts_to_shape,
    padded_ffn_graph_buckets,
    padded_ffn_graph_shape,
    select_padded_ffn_bucket,
    shared_rows_for_bucket,
    validate_cuda_graph_mode,
)


def _config(
    *,
    enforce_eager,
    cudagraph_mode=None,
    use_ubatching=False,
    num_ubatches=1,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        parallel_config=SimpleNamespace(
            use_ubatching=use_ubatching,
            num_ubatches=num_ubatches,
        ),
    )


def test_cuda_graph_policy_allows_eager():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=True, cudagraph_mode="FULL"),
        role="attention",
    )

    assert policy.enabled is False
    assert policy.allow_attention_full_decode_only is False


def test_cuda_graph_policy_allows_full_decode_only_for_attention():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="attention",
    )

    assert policy.enabled is True
    assert policy.mode_name == FULL_DECODE_ONLY
    assert policy.allow_attention_full_decode_only is True
    assert policy.enable_ffn_graph_cache is False


def test_cuda_graph_policy_allows_full_decode_only_for_ffn():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="ffn",
    )

    assert policy.enabled is True
    assert policy.allow_attention_full_decode_only is False
    assert policy.enable_ffn_graph_cache is True


@pytest.mark.parametrize(
    "mode",
    [None, "NONE", "FULL", "PIECEWISE", "FULL_AND_PIECEWISE"],
)
def test_cuda_graph_policy_rejects_non_full_decode_only_graph_modes(mode):
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        validate_cuda_graph_mode(
            _config(enforce_eager=False, cudagraph_mode=mode),
            role="attention",
        )


def test_cuda_graph_policy_allows_two_way_ubatching_with_full_decode_only_graph():
    policy = validate_cuda_graph_mode(
        _config(
            enforce_eager=False,
            cudagraph_mode=FULL_DECODE_ONLY,
            use_ubatching=True,
            num_ubatches=2,
        ),
        role="attention",
    )

    assert policy.enabled is True
    assert policy.allow_cuda_graph_with_ubatching is True


def test_cuda_graph_policy_rejects_unsupported_ubatch_count_with_graph():
    with pytest.raises(RuntimeError, match="ubatching"):
        validate_cuda_graph_mode(
            _config(
                enforce_eager=False,
                cudagraph_mode=FULL_DECODE_ONLY,
                use_ubatching=True,
                num_ubatches=4,
            ),
            role="attention",
        )


def test_cudagraph_mode_name_handles_enum_like_values():
    mode = SimpleNamespace(name=FULL_DECODE_ONLY)

    assert cudagraph_mode_name(_config(enforce_eager=False, cudagraph_mode=mode)) == (
        FULL_DECODE_ONLY
    )


def test_make_ffn_graph_key_matches_original_shape():
    metadata_0 = SimpleNamespace(num_tokens_across_dp_cpu=[3, 5])
    metadata_1 = SimpleNamespace(num_tokens_across_dp_cpu=[7, 11])

    assert make_ffn_graph_key({1: metadata_1, 0: metadata_0}) == (
        (0, (3, 5)),
        (1, (7, 11)),
    )


def test_make_ffn_graph_key_can_aggregate_attention_counts_to_ffn_counts():
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[12] * 8)

    assert make_ffn_graph_key(
        {0: metadata},
        attention_size=8,
        ffn_size=4,
        fallback=24,
    ) == ((0, (24, 24, 24, 24)),)


# --- TP expansion tests ---


def test_make_ffn_graph_key_expands_dp1_tp2():
    """DP=1 has 1 entry in num_tokens_across_dp_cpu; with TP=2 the
    attention_size=2 so the single DP entry must be replicated."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )
    # OLD (buggy) behaviour returned (32, 32) via the fallback path.
    # Correct behaviour: replicate [8] -> (8, 8), then aggregate.
    assert key == ((0, (8, 8)),)


def test_make_ffn_graph_key_different_tokens_for_dp1_tp2():
    """Prefill (4 tokens) and decode (8 tokens) must produce different keys
    so that the FFN correctly distinguishes EAGER from REPLAY."""
    prefill_meta = SimpleNamespace(num_tokens_across_dp_cpu=[4])
    decode_meta = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    prefill_key = make_ffn_graph_key(
        {0: prefill_meta},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )
    decode_key = make_ffn_graph_key(
        {0: decode_meta},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )

    assert prefill_key == ((0, (4, 4)),)
    assert decode_key == ((0, (8, 8)),)
    assert prefill_key != decode_key


def test_make_ffn_graph_key_expands_dp2_tp2():
    """DP=2, TP=2: two DP entries replicated to 4 AFD entries."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[4, 8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=4,
        ffn_size=4,
        fallback=32,
    )
    # [4, 8] -> replicate tp_size=2 -> [4, 4, 8, 8] -> aggregate group_size=1
    assert key == ((0, (4, 4, 8, 8)),)


def test_make_ffn_graph_key_dp1_tp1_unchanged():
    """TP=1 should not trigger expansion; behaviour unchanged from before."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=1,
        ffn_size=1,
        fallback=32,
    )
    assert key == ((0, (8,)),)


@pytest.mark.parametrize(
    ("is_graph_replaying", "graph_exists", "expected"),
    [
        (False, True, AFDGraphRunMode.EAGER),
        (True, True, AFDGraphRunMode.REPLAY),
        (True, False, AFDGraphRunMode.EAGER),
    ],
)
def test_graph_run_mode_requires_attention_replaying(
    is_graph_replaying,
    graph_exists,
    expected,
):
    assert (
        graph_run_mode(
            is_warmup=False,
            is_graph_capturing=False,
            is_graph_replaying=is_graph_replaying,
            graph_enabled=True,
            graph_exists=graph_exists,
        )
        is expected
    )


# ----------------------------------------------------------------------
# Padded FFN graph shape
#
# A grouped GEMM reads its grouping from a device-side count vector, not from
# its row count, so one row count can be captured and smaller items padded up
# to it. These pin what "big enough" means and where the padding lands.
# ----------------------------------------------------------------------


def test_padded_shape_bounds_the_largest_batch():
    # Worst case for one FFN rank: every token sends every one of its topk
    # slots here. Shared rows are split contiguously, so a rank holds a share.
    routed, shared = padded_ffn_graph_shape(
        num_tokens=8,
        topk=6,
        ffn_size=2,
        has_shared_experts=True,
    )
    assert routed == 48
    assert shared == 4


def test_padded_shape_rounds_the_shared_split_up():
    # 7 tokens over 2 ranks is 4 and 3; the buffer has to hold the larger.
    _, shared = padded_ffn_graph_shape(
        num_tokens=7,
        topk=2,
        ffn_size=2,
        has_shared_experts=True,
    )
    assert shared == 4


def test_padded_shape_has_no_shared_rows_without_shared_experts():
    routed, shared = padded_ffn_graph_shape(
        num_tokens=8,
        topk=6,
        ffn_size=2,
        has_shared_experts=False,
    )
    assert routed == 48
    assert shared == 0


@pytest.mark.parametrize(
    ("num_tokens", "topk", "ffn_size"),
    [(0, 6, 2), (8, 0, 2), (8, 6, 0)],
)
def test_padded_shape_rejects_nonpositive_inputs(num_tokens, topk, ffn_size):
    with pytest.raises(ValueError):
        padded_ffn_graph_shape(
            num_tokens=num_tokens,
            topk=topk,
            ffn_size=ffn_size,
            has_shared_experts=True,
        )


def test_padding_lands_on_the_last_expert():
    # Real rows are grouped by expert in ascending order, so the tail of the
    # row range is the last expert's either way -- charging the padding there
    # leaves every real row's expert assignment untouched.
    counts = torch.tensor([3, 2, 1], dtype=torch.int32)
    pad_counts_to_shape(counts, padded_rows=10, actual_rows=6)
    assert counts.tolist() == [3, 2, 5]
    assert int(counts.sum()) == 10


def test_padding_an_exact_fit_changes_nothing():
    counts = torch.tensor([3, 2, 1], dtype=torch.int32)
    pad_counts_to_shape(counts, padded_rows=6, actual_rows=6)
    assert counts.tolist() == [3, 2, 1]


def test_padding_refuses_rows_that_do_not_fit():
    counts = torch.tensor([4, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="do not fit"):
        pad_counts_to_shape(counts, padded_rows=6, actual_rows=8)


def test_bucket_ladder_puts_exact_stops_on_the_common_item_sizes():
    # The sizes items actually cluster at: a whole item is max_routed/ffn_size,
    # a DBO ubatch is half of that. A boundary sitting exactly on either sends
    # every above-average item a full step up, which is what made the first
    # bucketed run only 7% faster.
    buckets = padded_ffn_graph_buckets(12288, ffn_size=2)
    assert 6144 in buckets, "whole item size needs its own bucket"
    assert 3072 in buckets, "DBO ubatch size needs its own bucket"
    # And an item just above either pays a small step, not a doubling.
    assert select_padded_ffn_bucket(buckets, 6200) / 6200 < 1.15
    assert select_padded_ffn_bucket(buckets, 3100) / 3100 < 1.15


def test_bucket_ladder_scales_with_the_step_size():
    small = padded_ffn_graph_buckets(12288, ffn_size=2)
    large = padded_ffn_graph_buckets(24576, ffn_size=2)
    assert len(small) == len(large)
    assert max(large) == 2 * max(small)


def test_oversized_item_has_no_bucket_and_runs_eager():
    buckets = padded_ffn_graph_buckets(12288, ffn_size=2)
    assert select_padded_ffn_bucket(buckets, max(buckets) + 1) is None


def test_bucket_ladder_stays_within_the_row_bound():
    # The ladder no longer has to reach max_routed -- the workspace ceiling is
    # pinned by an eager warm-up in capture_padded_ffn_graphs instead, which
    # costs one forward rather than the largest graph per layer -- but no bucket
    # may exceed the buffers.
    for max_routed, ffn_size in ((12288, 2), (24576, 2), (4096, 4)):
        buckets = padded_ffn_graph_buckets(max_routed, ffn_size=ffn_size)
        assert max(buckets) <= max_routed
        assert min(buckets) > 0


def test_shared_rows_scale_with_the_bucket():
    # Both row counts scale with the item's tokens, so a half-sized bucket
    # captures half the shared rows. Capturing every bucket at max_shared made
    # a DBO ubatch pay double on the shared expert, which is what kept graphs
    # losing to eager under DBO after the routed rows were already bucketed.
    assert shared_rows_for_bucket(12288, max_routed=24576, max_shared=4096) == 2048
    assert shared_rows_for_bucket(24576, max_routed=24576, max_shared=4096) == 4096
    # No shared experts stays zero.
    assert shared_rows_for_bucket(1024, max_routed=2048, max_shared=0) == 0


def test_bucket_selection_escalates_when_the_shared_slice_does_not_fit():
    buckets = (12288, 24576)
    kwargs = {"max_routed": 24576, "max_shared": 4096}
    # Routed fits the small bucket and so does its share of the shared rows.
    assert select_padded_ffn_bucket(buckets, 12288, 2048, **kwargs) == 12288
    # Same routed rows but more shared rows than the small bucket captured.
    assert select_padded_ffn_bucket(buckets, 12288, 4000, **kwargs) == 24576
    # Beyond every bucket's shared capacity: eager.
    assert select_padded_ffn_bucket(buckets, 12288, 9999, **kwargs) is None
