# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.v1.worker.cuda_graph import (
    FULL_DECODE_ONLY,
    AFDGraphRunMode,
    cudagraph_mode_name,
    graph_run_mode,
    make_ffn_graph_key,
    pad_counts_to_shape,
    padded_ffn_graph_buckets,
    padded_ffn_graph_shape,
    resolve_padded_ffn_graph_reserve_bytes,
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


@pytest.mark.parametrize(
    "counts,attention,ffn,expected",
    [
        ([4], 2, 2, (4, 4)),
        ([8], 2, 2, (8, 8)),
        ([4, 8], 4, 4, (4, 4, 8, 8)),
        ([8], 1, 1, (8,)),
    ],
)
def test_graph_keys_expand_dp_counts_by_attention_tp(counts, attention, ffn, expected):
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=counts)
    assert make_ffn_graph_key(
        {0: metadata}, attention_size=attention, ffn_size=ffn, fallback=32
    ) == ((0, expected),)


@pytest.mark.parametrize(
    "topk, ffn_size, shared, expected",
    [(2, 2, True, (32, 8)), (8, 3, True, (128, 6)), (2, 2, False, (32, 0))],
)
def test_async_ffn_graph_shape_bounds_one_producers_routed_and_shared_rows(
    topk, ffn_size, shared, expected
):
    assert (
        padded_ffn_graph_shape(
            num_tokens=16,
            topk=topk,
            ffn_size=ffn_size,
            has_shared_experts=shared,
        )
        == expected
    )


@pytest.mark.parametrize(
    "fractions", [(), (0,), (-1,), (float("nan"),), (float("inf"),)]
)
def test_async_ffn_graph_buckets_reject_invalid_calibration(fractions):
    with pytest.raises(ValueError, match="multiples"):
        padded_ffn_graph_buckets(1024, fractions=fractions)


def test_async_ffn_graph_bucket_preserves_both_routed_and_shared_capacity():
    buckets = padded_ffn_graph_buckets(1024, ffn_size=2, fractions=(0.25, 0.5, 1))
    assert buckets == (128, 256, 512)
    assert (
        select_padded_ffn_bucket(
            buckets,
            200,
            20,
            max_routed=1024,
            max_shared=64,
        )
        == 512
    )
    assert shared_rows_for_bucket(512, max_routed=1024, max_shared=64) == 32
    assert select_padded_ffn_bucket(buckets, 600) is None


@pytest.mark.parametrize("ffn_size", [1, 2, 4])
def test_async_ffn_bucket_fits_balanced_routed_and_shared_experts(ffn_size):
    # A batch of 128 * ffn_size tokens routes six experts per token. Each FFN
    # receives 768 routed rows and 128 shared rows under balanced routing.
    options = dict(max_routed=6144, max_shared=1024 // ffn_size, ffn_size=ffn_size)
    assert shared_rows_for_bucket(768, **options) == 128
    assert select_padded_ffn_bucket((768,), 768, 128, **options) == 768


def test_async_ffn_graph_padding_keeps_real_expert_assignments():
    counts = [2, 1, 3]
    pad_counts_to_shape(counts, padded_rows=10, actual_rows=6)
    assert counts == [2, 1, 7]
    with pytest.raises(ValueError, match="do not fit"):
        pad_counts_to_shape(counts, padded_rows=4, actual_rows=6)


@pytest.mark.parametrize("reserve", ["-1", "nan", "inf"])
def test_async_ffn_graph_memory_reserve_rejects_invalid_sizes(reserve):
    with pytest.raises(ValueError, match="MEM_RESERVE"):
        resolve_padded_ffn_graph_reserve_bytes(
            {"AFD_FFN_GRAPH_MEM_RESERVE_GIB": reserve}
        )
