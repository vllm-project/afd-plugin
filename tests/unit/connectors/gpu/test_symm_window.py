# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Slot layout and header codec for the NVSHMEM symmetric window.

The window is the transport under the async GPU connector: these pin the wire
format itself -- field offsets, and what a header does and does not carry --
independently of any AFD connector that writes into it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from afd_plugin.connectors.gpu.symm_window import (  # noqa: E402
    FLAG_SHUTDOWN_BIT,
    HEADER_FIXED_WORDS,
    HEADER_HOST_WORDS,
    SlotLayout,
    decode_header,
    encode_header,
)


@pytest.fixture
def layout() -> SlotLayout:
    # shared_cap is deliberately not token_cap: the shared field is sized by the
    # per-rank split, so a layout that quietly reused token_cap for it would
    # otherwise still satisfy every assertion below.
    return SlotLayout.build(
        expert_per_rank=4,
        partial_cap=100,
        token_cap=32,
        shared_cap=8,
        hidden_size=8,
        payload_itemsize=2,
    )


def test_slot_fields_are_disjoint_and_fit_inside_the_slot(layout: SlotLayout):
    assert layout.header_words == HEADER_FIXED_WORDS + 4
    assert layout.expand_idx_off >= layout.header_off + layout.header_words * 4
    assert layout.weights_off >= layout.expand_idx_off + 100 * 4
    assert layout.routed_x_off >= layout.weights_off + 100 * 4
    # The payload is sized by distinct tokens, not by partials, and the shared
    # rows are a contiguous range so no index rides along with them.
    assert layout.shared_x_off >= layout.routed_x_off + 32 * 8 * 2
    # The shared field is sized by the per-rank split, not by the batch: a slot
    # can never hold a whole batch of shared rows, and reserving room for one
    # doubled the payload half of every slot.
    assert layout.slot_bytes >= layout.shared_x_off + 8 * 8 * 2
    assert layout.slot_bytes < layout.shared_x_off + 32 * 8 * 2


def test_every_field_offset_is_viewable_as_int32_and_payload(layout: SlotLayout):
    # get_buffer takes an element offset, so a byte offset that is not a
    # multiple of the element size would silently land on the wrong address.
    for offset in (
        layout.header_off,
        layout.expand_idx_off,
        layout.weights_off,
        layout.routed_x_off,
        layout.shared_x_off,
    ):
        assert offset % 4 == 0
        assert offset % layout.payload_itemsize == 0


def test_header_round_trip(layout: SlotLayout):
    header = encode_header(
        layout,
        layer_idx=11,
        num_tokens=32,
        routed_tokens=90,
        flags=0,
        expert_counts=[10, 20, 30, 30],
        segment_start=25,
    )
    decoded = decode_header(header)
    assert decoded.layer_idx == 11
    assert decoded.num_tokens == 32
    assert decoded.routed_tokens == 90
    assert decoded.expert_counts == [10, 20, 30, 30]
    assert decoded.segment_start == 25
    assert sum(decoded.expert_counts) == decoded.routed_tokens
    assert not decoded.is_shutdown


def test_header_carries_nothing_the_receiver_could_derive(layout: SlotLayout):
    # The sender's rank, the stage and the shared-token count all come off the
    # slot a message arrived in, so putting them on the wire would ship a value
    # the reader already has -- and a captured graph would ship it forever.
    header = encode_header(
        layout,
        layer_idx=11,
        num_tokens=32,
        routed_tokens=90,
        flags=0,
        expert_counts=[10, 20, 30, 30],
    )
    decoded = decode_header(header)
    for dropped in ("seq", "src_role_rank", "stage_idx", "shared_tokens", "topk"):
        assert not hasattr(decoded, dropped), dropped
    # Five host words plus the routing tail; nothing else earns a slot.
    assert HEADER_HOST_WORDS == 5
    assert header.numel() == HEADER_FIXED_WORDS + len(decoded.expert_counts)


def test_shutdown_flag_survives_the_round_trip(layout: SlotLayout):
    header = encode_header(
        layout,
        layer_idx=0,
        num_tokens=0,
        routed_tokens=0,
        flags=FLAG_SHUTDOWN_BIT,
        expert_counts=[0, 0, 0, 0],
    )
    assert decode_header(header).is_shutdown


def test_corrupt_magic_is_rejected(layout: SlotLayout):
    header = encode_header(
        layout,
        layer_idx=0,
        num_tokens=1,
        routed_tokens=0,
        flags=0,
        expert_counts=[0, 0, 0, 0],
    )
    header[0] = 0
    with pytest.raises(RuntimeError, match="magic mismatch"):
        decode_header(header)


def test_expert_counts_length_must_match_the_layout(layout: SlotLayout):
    with pytest.raises(ValueError, match="expert_counts"):
        encode_header(
            layout,
            layer_idx=0,
            num_tokens=1,
            routed_tokens=0,
            flags=0,
            expert_counts=[1, 2],
        )
