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


def test_make_ffn_graph_key_uniform_padding_matches_camp2p_rows():
    """NPU CAMP2p: every FFN rank receives uniform * group_size rows.

    The Attention side pads every rank to the largest count, so uneven
    counts (1, 5, 1, 5) give every FFN rank 5 * 2 = 10 rows, not the
    contiguous-group sums (6, 6) used by the GPU mapping.
    """
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[1, 5, 1, 5])

    assert make_ffn_graph_key(
        {0: metadata},
        attention_size=4,
        ffn_size=2,
        fallback=24,
        uniform_padding=True,
    ) == ((0, (10, 10)),)


def test_make_ffn_graph_key_uniform_padding_keeps_ratio_one_identity():
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[3, 7])

    assert make_ffn_graph_key(
        {0: metadata},
        attention_size=2,
        ffn_size=2,
        fallback=24,
        uniform_padding=True,
    ) == ((0, (3, 7)),)


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
