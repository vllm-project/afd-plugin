# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Wire contracts tested independently from torch, CUDA and vLLM."""

from dataclasses import replace

import pytest

from afd_plugin.connectors.gpu.async_gpu_protocol import (
    FLAG_SHUTDOWN_BIT,
    GpuAsyncExtraInfo,
    SlotHeader,
    SlotLayout,
    decode_header_words,
    encode_header_words,
    nvshmem_pe_mapping,
    shared_token_slice,
    stage_rings,
)


@pytest.mark.parametrize(
    ("hosts", "expected"),
    [
        (("a", "a", "a"), (0, 1, 2)),
        (("a", "a", "b", "b"), (0, 1, 2, 3)),
        (("a", "b", "a", "b"), (0, 2, 1, 3)),
        (("z", "a", "z", "a"), (0, 2, 1, 3)),
    ],
)
def test_nvshmem_pe_mapping_keeps_each_nodes_ids_contiguous(hosts, expected):
    assert nvshmem_pe_mapping(hosts) == expected


def test_nvshmem_pe_mapping_rejects_unequal_node_counts():
    with pytest.raises(ValueError, match="equal PE counts.*'a': 2.*'b': 1"):
        nvshmem_pe_mapping(("a", "a", "b"))


@pytest.fixture
def layout():
    return SlotLayout.build(
        expert_per_rank=4,
        partial_cap=64,
        token_cap=16,
        shared_cap=4,
        hidden_size=8,
        payload_itemsize=2,
    )


def test_slot_fields_do_not_overlap_or_waste_a_whole_shared_batch(layout):
    assert layout.expand_idx_off >= layout.header_words * 4
    assert layout.weights_off >= layout.expand_idx_off + layout.partial_cap * 4
    assert layout.routed_x_off >= layout.weights_off + layout.partial_cap * 4
    assert layout.shared_x_off >= layout.routed_x_off + 16 * 8 * 2
    assert layout.slot_bytes >= layout.shared_x_off + 4 * 8 * 2
    for offset in (
        layout.expand_idx_off,
        layout.weights_off,
        layout.routed_x_off,
        layout.shared_x_off,
        layout.slot_bytes,
    ):
        assert offset % 256 == 0


@pytest.mark.parametrize("shutdown", [False, True])
def test_version_four_header_round_trip(layout, shutdown):
    header = (
        SlotHeader(0, 0, 0, FLAG_SHUTDOWN_BIT, 0, (0, 0, 0, 0))
        if shutdown
        else SlotHeader(3, 16, 11, 0, 8, (1, 2, 3, 5))
    )
    assert decode_header_words(encode_header_words(header, layout), layout) == header


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("num_tokens", -1, "token count"),
        ("num_tokens", 17, "token count"),
        ("routed_tokens", -1, "non-negative"),
        ("segment_start", 64, "routed segment"),
        ("layer_idx", -1, "layer index"),
        ("expert_counts", (-1, 0, 0, 12), "non-negative"),
        ("expert_counts", (0, 0, 0, 0), "sum"),
        ("expert_counts", (11,), "width"),
        ("flags", 8, "unsupported flags"),
        ("flags", FLAG_SHUTDOWN_BIT, "terminal message"),
    ],
)
def test_invalid_peer_headers_are_rejected(layout, field, value, error):
    header = replace(SlotHeader(3, 16, 11, 0, 8, (1, 2, 3, 5)), **{field: value})
    with pytest.raises(RuntimeError, match=error):
        encode_header_words(header, layout)


@pytest.mark.parametrize(
    ("index", "value", "error"),
    [
        (0, 0, "magic"),
        (1, 99, "version"),
    ],
)
def test_invalid_wire_identity_is_rejected(layout, index, value, error):
    words = list(encode_header_words(SlotHeader(0, 0, 0, 0, 0, (0, 0, 0, 0)), layout))
    words[index] = value
    with pytest.raises(RuntimeError, match=error):
        decode_header_words(words, layout)


@pytest.mark.parametrize("ring_depth", [2, 3, 4, 7])
def test_live_stages_own_disjoint_rings(ring_depth):
    first = stage_rings(0, 2, ring_depth)
    second = stage_rings(1, 2, ring_depth)
    assert not set(first) & set(second)
    assert sorted(first + second) == list(range(ring_depth))


@pytest.mark.parametrize("stage", [-1, 2, 4])
def test_out_of_range_stage_cannot_alias_a_live_ring(stage):
    with pytest.raises(ValueError, match="stage"):
        stage_rings(stage, 2, 4)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 13, 513])
@pytest.mark.parametrize("ffn_size", [1, 2, 3, 6])
def test_shared_rows_tile_the_batch_exactly_once(num_tokens, ffn_size):
    slices = [
        shared_token_slice(rank, ffn_size, num_tokens, True) for rank in range(ffn_size)
    ]
    assert [index for part in slices for index in range(part.start, part.stop)] == list(
        range(num_tokens),
    )
    assert max(part.stop - part.start for part in slices) <= -(-num_tokens // ffn_size)


@pytest.mark.parametrize(
    "raw",
    [None, {}, {"ring_depth": 4}, {"attn_ranks_per_dp": 2, "recv_poll_timeout_ms": 80}],
)
def test_connector_configuration_round_trips_explicit_ring_intent(raw):
    config = GpuAsyncExtraInfo.from_mapping(raw)
    assert GpuAsyncExtraInfo.from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    "raw",
    [
        {"ring_depth": 0},
        {"attn_ranks_per_dp": -1},
        {"recv_poll_timeout_ms": False},
        {"old_split": True},
    ],
)
def test_invalid_connector_configuration_is_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        GpuAsyncExtraInfo.from_mapping(raw)
