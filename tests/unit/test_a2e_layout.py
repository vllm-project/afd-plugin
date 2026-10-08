# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the A2E tile layout.

The layout is plain integer arithmetic that mirrors the A2E/E2A kernels, so it is
testable without an Ascend device. The properties worth pinning are the peer set
(the kernel pairs an FFN rank with strided Attention ranks), the tile every peer
of a group derives identically, and the rows an FFN rank therefore reads.
"""

from __future__ import annotations

import pytest

from afd_plugin.a2e_layout import (
    attention_peer_ranks,
    attention_rank_token_counts,
    ffn_rank_for_attention_rank,
    ffn_receive_rows,
    padded_tile_rows,
)


class TestAttentionPeerRanks:
    def test_pairs_strided_attention_ranks_with_each_ffn_rank(self):
        # 4A2F: the kernel reads peers rank + (i + 1) * ffn_size.
        assert list(attention_peer_ranks(0, attention_size=4, ffn_size=2)) == [0, 2]
        assert list(attention_peer_ranks(1, attention_size=4, ffn_size=2)) == [1, 3]

    def test_keeps_one_peer_per_ffn_rank_when_the_sizes_match(self):
        assert list(attention_peer_ranks(0, attention_size=2, ffn_size=2)) == [0]
        assert list(attention_peer_ranks(1, attention_size=2, ffn_size=2)) == [1]

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (0, 2), (4, 0)],
    )
    def test_rejects_a_topology_it_cannot_describe(self, attention_size, ffn_size):
        assert (
            attention_peer_ranks(0, attention_size=attention_size, ffn_size=ffn_size)
            is None
        )

    def test_rejects_a_rank_outside_the_role(self):
        assert attention_peer_ranks(2, attention_size=4, ffn_size=2) is None


class TestAttentionRankTokenCounts:
    def test_keeps_one_count_per_attention_rank(self):
        assert attention_rank_token_counts([4, 8, 16, 16], attention_size=4) == [
            4,
            8,
            16,
            16,
        ]

    def test_expands_dp_counts_over_the_tp_workers_that_share_them(self):
        # DP=2, TP=2: AFD rank 0/1 hold DP rank 0's count, 2/3 hold DP rank 1's.
        assert attention_rank_token_counts([4, 8], attention_size=4) == [4, 4, 8, 8]

    def test_returns_none_without_counts(self):
        assert attention_rank_token_counts([], attention_size=4) is None

    def test_returns_none_when_the_counts_cannot_cover_every_rank(self):
        assert attention_rank_token_counts([4, 8, 16], attention_size=4) is None


class TestFfnRankForAttentionRank:
    def test_maps_an_attention_rank_to_the_ffn_rank_it_feeds(self):
        assert ffn_rank_for_attention_rank(0, attention_size=4, ffn_size=2) == 0
        assert ffn_rank_for_attention_rank(1, attention_size=4, ffn_size=2) == 1
        assert ffn_rank_for_attention_rank(2, attention_size=4, ffn_size=2) == 0
        assert ffn_rank_for_attention_rank(3, attention_size=4, ffn_size=2) == 1

    def test_rejects_a_topology_it_cannot_describe(self):
        assert ffn_rank_for_attention_rank(0, attention_size=3, ffn_size=2) is None


class TestPaddedTileRows:
    def test_takes_the_largest_count_of_the_peer_group(self):
        # F0 owns {A0, A2} and F1 owns {A1, A3}.
        assert (
            padded_tile_rows([2, 2, 5, 7], ffn_rank=0, attention_size=4, ffn_size=2)
            == 5
        )
        assert (
            padded_tile_rows([2, 2, 5, 7], ffn_rank=1, attention_size=4, ffn_size=2)
            == 7
        )

    def test_keeps_an_even_group_at_its_own_count(self):
        assert (
            padded_tile_rows([6, 6, 6, 6], ffn_rank=0, attention_size=4, ffn_size=2)
            == 6
        )

    def test_falls_back_to_the_all_rank_count_both_roles_pass(self):
        # A step without usable counts sizes both sides by the same run-level
        # value instead of letting the receiver invent a larger tile.
        assert (
            padded_tile_rows([], ffn_rank=0, attention_size=4, ffn_size=2, fallback=64)
            == 64
        )

    def test_never_returns_zero_rows(self):
        counts = [0, 0]
        assert (
            padded_tile_rows(
                counts, ffn_rank=0, attention_size=2, ffn_size=2, fallback=0
            )
            == 1
        )

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (4, 0)],
    )
    def test_falls_back_for_a_topology_it_cannot_describe(
        self,
        attention_size,
        ffn_size,
    ):
        assert (
            padded_tile_rows(
                [4, 4],
                ffn_rank=0,
                attention_size=attention_size,
                ffn_size=ffn_size,
                fallback=16,
            )
            == 16
        )


class TestFfnReceiveRows:
    def test_counts_one_tile_per_attention_peer(self):
        # 4A2F: F0 owns {A0, A2} and F1 owns {A1, A3}, each receiving two tiles
        # sized by its own group maximum.
        assert ffn_receive_rows([4, 8, 16, 16], 0, attention_size=4, ffn_size=2) == 32
        assert ffn_receive_rows([2, 2, 5, 7], 1, attention_size=4, ffn_size=2) == 14

    def test_keeps_even_groups_at_their_real_total(self):
        assert ffn_receive_rows([6, 6, 6, 6], 0, attention_size=4, ffn_size=2) == 12

    def test_matches_the_single_tile_layout_when_the_sizes_match(self):
        # 2A2F: one tile per FFN rank, so the receive is the peer's own count.
        assert ffn_receive_rows([5, 7], 0, attention_size=2, ffn_size=2) == 5
        assert ffn_receive_rows([5, 7], 1, attention_size=2, ffn_size=2) == 7

    def test_expands_dp_counts_before_aggregating(self):
        # DP=1 with TP=2 replicates the single count over both Attention ranks.
        assert ffn_receive_rows([8], 0, attention_size=2, ffn_size=2) == 8

    def test_sizes_a_missing_counts_step_by_the_shared_fallback(self):
        # No counts: two tiles of the fallback, which is what the Attention ranks
        # pad their payloads up to for the same step.
        assert ffn_receive_rows([], 0, attention_size=4, ffn_size=2, fallback=64) == 128

    @pytest.mark.parametrize(
        ("attention_size", "ffn_size"),
        [(2, 4), (3, 2), (4, 0)],
    )
    def test_uses_one_tile_for_a_topology_it_cannot_describe(
        self,
        attention_size,
        ffn_size,
    ):
        assert (
            ffn_receive_rows(
                [4, 4],
                0,
                attention_size=attention_size,
                ffn_size=ffn_size,
                fallback=16,
            )
            == 16
        )

    def test_the_receive_total_is_the_senders_tile_times_the_peers(self):
        # Both roles have to agree on one number: the Attention rank writes
        # ``tile`` rows and the FFN rank reads ``tiles`` of them.
        counts = [24, 24]
        tile = padded_tile_rows(counts, ffn_rank=0, attention_size=4, ffn_size=2)
        assert tile == 24
        assert ffn_receive_rows(counts, 0, attention_size=4, ffn_size=2) == 2 * tile
