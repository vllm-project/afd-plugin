from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.v1.worker.cuda_graph import (
    FULL_DECODE_ONLY,
    cudagraph_mode_name,
    make_ffn_graph_key,
    make_mtp_ffn_graph_key,
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


def test_make_ffn_graph_key_retains_attention_peer_counts():
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[12] * 8)

    assert make_ffn_graph_key(
        {0: metadata},
        attention_size=8,
        ffn_size=4,
        fallback=24,
    ) == ((0, (12, 12, 12, 12, 12, 12, 12, 12)),)


def test_make_ffn_graph_key_separates_equal_aggregates_with_different_peers():
    first = SimpleNamespace(num_tokens_across_dp_cpu=[2, 3, 4, 5])
    second = SimpleNamespace(num_tokens_across_dp_cpu=[1, 4, 3, 6])

    first_key = make_ffn_graph_key(
        {0: first},
        attention_size=4,
        ffn_size=2,
    )
    second_key = make_ffn_graph_key(
        {0: second},
        attention_size=4,
        ffn_size=2,
    )

    assert first_key == ((0, (2, 3, 4, 5)),)
    assert second_key == ((0, (1, 4, 3, 6)),)
    assert first_key != second_key


def test_make_mtp_ffn_graph_key_merges_u2_by_attention_peer():
    stage_0 = SimpleNamespace(num_tokens_across_dp_cpu=[2, 3, 4, 5])
    stage_1 = SimpleNamespace(num_tokens_across_dp_cpu=[7, 11, 13, 17])

    assert make_mtp_ffn_graph_key(
        {0: stage_0, 1: stage_1},
        attention_size=4,
        ffn_size=2,
        fallback=32,
    ) == (9, 14, 17, 22)


def test_make_mtp_ffn_graph_key_separates_equal_ffn_aggregates():
    first = SimpleNamespace(num_tokens_across_dp_cpu=[2, 3, 4, 5])
    second = SimpleNamespace(num_tokens_across_dp_cpu=[1, 4, 3, 6])

    first_key = make_mtp_ffn_graph_key(
        {0: first},
        attention_size=4,
        ffn_size=2,
        fallback=32,
    )
    second_key = make_mtp_ffn_graph_key(
        {0: second},
        attention_size=4,
        ffn_size=2,
        fallback=32,
    )

    assert first_key == (2, 3, 4, 5)
    assert second_key == (1, 4, 3, 6)
    assert first_key != second_key


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
