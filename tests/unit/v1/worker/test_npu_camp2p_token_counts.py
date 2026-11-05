# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Row-count contracts for the NPU CAMP2p uniform-padding protocol.

The a2e/e2a kernels move ``batch_size / (attention_size / ffn_size)`` rows
to or from every Attention rank's shared-memory window, so the Attention
side pads every rank up to the largest count across all of them and each
FFN rank receives ``uniform * group_size`` rows.  These tests pin the
Python-side helpers to that contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.npu.camp2p import (  # noqa: E402
    _num_tokens_for_ffn_rank,
)


class _FakeDPMetadata:
    def __init__(self, counts):
        self.num_tokens_across_dp_cpu = torch.tensor(counts, dtype=torch.int64)


def _dp_metadata_list(counts):
    return {0: _FakeDPMetadata(counts)}


def test_ffn_rank_expects_uniform_rows_from_every_attention_rank():
    # 4A2F with counts (1, 5, 1, 5): the Attention side pads every rank to
    # 5 rows, so each FFN rank receives 5 * 2 = 10 rows regardless of which
    # Attention ranks happen to be busiest.
    dp_metadata_list = _dp_metadata_list([1, 5, 1, 5])

    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=0,
            attention_size=4,
            ffn_size=2,
            fallback=24,
        )
        == 10
    )
    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=1,
            attention_size=4,
            ffn_size=2,
            fallback=24,
        )
        == 10
    )


def test_ffn_rank_uses_own_attention_twin_at_one_to_one():
    # 2A2F: no padding happens, so each FFN rank receives exactly its
    # twin's count.
    dp_metadata_list = _dp_metadata_list([3, 7])

    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=0,
            attention_size=2,
            ffn_size=2,
            fallback=24,
        )
        == 3
    )
    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=1,
            attention_size=2,
            ffn_size=2,
            fallback=24,
        )
        == 7
    )


def test_ffn_rank_matches_attention_side_publication_for_2a1f():
    # 2A1F matches the validated F 2->1 resize: counts (6, 1) pad the small
    # Attention rank up to 6, and the single FFN rank receives 12 rows.
    dp_metadata_list = _dp_metadata_list([6, 1])

    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=0,
            attention_size=2,
            ffn_size=1,
            fallback=24,
        )
        == 12
    )


def test_ffn_rank_expands_dp_level_counts_for_tensor_parallel():
    # DP=2 with TP=2 gives attn_size=4 but only 2 entries; each DP count is
    # replicated to its TP workers before the uniform is taken.
    dp_metadata_list = _dp_metadata_list([6, 1])

    assert (
        _num_tokens_for_ffn_rank(
            dp_metadata_list,
            0,
            ffn_rank=0,
            attention_size=4,
            ffn_size=2,
            fallback=24,
        )
        == 12
    )


def test_ffn_rank_falls_back_when_metadata_is_missing():
    assert (
        _num_tokens_for_ffn_rank(
            {},
            0,
            ffn_rank=0,
            attention_size=4,
            ffn_size=2,
            fallback=24,
        )
        == 24
    )


def test_ffn_rank_matches_camp2p_graph_key_uniform():
    """The graph key and the a2e batch must agree, or replay picks the
    graph captured for the wrong batch shape."""
    from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key

    counts = [1, 5, 1, 5]
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=counts)
    expected = _num_tokens_for_ffn_rank(
        _dp_metadata_list(counts),
        0,
        ffn_rank=0,
        attention_size=4,
        ffn_size=2,
        fallback=24,
    )

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=4,
        ffn_size=2,
        fallback=24,
        uniform_padding=True,
    )

    assert key == ((0, (expected, expected)),)
