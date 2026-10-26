# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side token counts feed the EP dispatch sizes, so they must follow the
same Attention-rank blocks the connector receives from."""

from __future__ import annotations

import pytest

from afd_plugin.v1.worker.ffn_metadata import (
    aggregate_ffn_token_counts,
    project_ffn_token_counts_to_dp,
    real_attention_token_total,
)


@pytest.mark.parametrize(
    ("attention_counts", "attention", "ffn", "expected"),
    [
        # Historical A >= F layouts: unchanged (docstring example and 1:1).
        ((0, 4, 5, 6), 4, 2, (5, 11)),
        ((3, 5), 2, 2, (3, 5)),
        ((), 4, 2, (2, 2)),
        # TP-expanded counts (one count per DP rank, two TP ranks each).
        ((4, 6), 4, 2, (8, 12)),
        # F does not divide A: uneven subgroups {A0,A1}{A2}.
        ((1, 2, 3), 3, 2, (3, 3)),
        ((1, 2, 3, 4, 5, 6), 6, 4, (3, 3, 9, 6)),
    ],
)
def test_aggregate_follows_the_subgroup_partition(
    attention_counts, attention, ffn, expected
):
    assert (
        aggregate_ffn_token_counts(
            attention_counts, attention_size=attention, ffn_size=ffn
        )
        == expected
    )


def test_project_collapses_tp_ranks_to_dp_ranks():
    assert project_ffn_token_counts_to_dp((8, 8, 13, 13), dp_size=2) == (8, 13)
    assert project_ffn_token_counts_to_dp((3, 2), dp_size=2) == (3, 2)


@pytest.mark.parametrize(
    ("attention_counts", "attention", "expected"),
    [
        # Zero-token peers transfer placeholder rows but contribute no
        # real tokens (docstring example counterpart of the aggregation).
        ((0, 4, 5, 6), 4, 15),
        ((3, 5), 2, 8),
        ((), 4, 0),
        ((), 1, 0),
        # TP-expanded counts: one count per DP rank, two TP ranks each.
        ((4, 6), 4, 20),
        ((7,), 2, 14),
        # Counts already covering every Attention rank are kept as-is.
        ((1, 2, 3), 3, 6),
    ],
)
def test_real_token_total_excludes_placeholder_rows(
    attention_counts, attention, expected
):
    assert (
        real_attention_token_total(attention_counts, attention_size=attention)
        == expected
    )
