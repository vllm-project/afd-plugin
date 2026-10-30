# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from afd_plugin.distributed.kv_transfer.dspark_context import (
    DSparkContextDescriptor,
    DSparkKVPage,
    build_draft_context_slots,
    build_target_page_reads,
)


@pytest.fixture
def descriptor():
    return DSparkContextDescriptor(
        request_id="external-request",
        generation="allocation-1",
        prompt_tokens=257,
        window_start=129,
        window_end=257,
        aux_layer_ids=(8, 21, 34),
        hidden_size=4096,
    )


def test_descriptor_round_trips_portably_with_explicit_prompt_tail(descriptor):
    decoded = json.loads(json.dumps(descriptor.to_dict()))

    assert DSparkContextDescriptor.from_dict(decoded) == descriptor
    assert descriptor.num_tokens == 128
    assert descriptor.feature_width == 12288
    assert decoded["window_end"] == decoded["prompt_tokens"] == 257


def test_serialized_layer_list_cannot_mutate_descriptor(descriptor):
    serialized = descriptor.to_dict()
    serialized["aux_layer_ids"].append(40)

    assert descriptor.aux_layer_ids == (8, 21, 34)
    with pytest.raises(FrozenInstanceError):
        descriptor.generation = "replacement"


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"request_id": ""}, "request_id"),
        ({"generation": ""}, "generation"),
        ({"prompt_tokens": 0}, "prompt_tokens"),
        ({"prompt_tokens": True}, "prompt_tokens"),
        ({"window_start": -1}, "window"),
        ({"window_start": 257}, "window"),
        ({"window_end": 258}, "window"),
        ({"window_end": False}, "window"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"aux_layer_ids": ()}, "layer IDs"),
        ({"aux_layer_ids": (8, 8)}, "layer IDs"),
        ({"aux_layer_ids": (21, 8)}, "layer IDs"),
        ({"aux_layer_ids": (-1, 8)}, "layer IDs"),
        ({"aux_layer_ids": (True, 8)}, "layer IDs"),
    ],
)
def test_descriptor_rejects_invalid_identity_and_ranges(descriptor, changes, error):
    with pytest.raises(ValueError, match=error):
        replace(descriptor, **changes)


@pytest.mark.parametrize("field", ["request_id", "generation", "window_end"])
def test_wire_descriptor_rejects_missing_identity_or_range(descriptor, field):
    metadata = descriptor.to_dict()
    del metadata[field]

    with pytest.raises(ValueError, match="schema"):
        DSparkContextDescriptor.from_dict(metadata)


def test_wire_descriptor_rejects_unknown_fields(descriptor):
    metadata = descriptor.to_dict() | {"original_prompt_tokens": 258}

    with pytest.raises(ValueError, match="schema"):
        DSparkContextDescriptor.from_dict(metadata)


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("prompt_tokens", "257", "prompt_tokens"),
        ("window_start", False, "window"),
        ("aux_layer_ids", "8,21,34", "serialized list"),
        ("aux_layer_ids", [8, 8], "layer IDs"),
    ],
)
def test_wire_descriptor_validates_decoded_values(descriptor, field, value, error):
    metadata = descriptor.to_dict()
    metadata[field] = value

    with pytest.raises(ValueError, match=error):
        DSparkContextDescriptor.from_dict(metadata)


def test_slots_keep_absolute_swa_positions_and_null_blocks():
    slots = build_draft_context_slots(
        positions=[7, 8, 15, 16],
        block_ids_by_group={2: [0, 9, 4], 5: [0, 0, 6, 8, 7]},
        block_sizes_by_group={2: 8, 5: 4},
        layer_group_ids=[5, 2, 5],
    )

    assert slots == [[-1, 24, 35, 28], [-1, 72, 79, 32], [-1, 24, 35, 28]]


def test_slots_do_not_renumber_a_tail_or_require_sorted_positions():
    assert build_draft_context_slots(
        positions=[257, 255, 256],
        block_ids_by_group={3: [0] * 15 + [9, 4]},
        block_sizes_by_group={3: 16},
        layer_group_ids=[3],
    ) == [[65, 159, 64]]


def test_empty_positions_produce_empty_slots_for_each_layer():
    assert build_draft_context_slots([], {3: []}, {3: 16}, [3, 3]) == [[], []]


@pytest.mark.parametrize(
    "positions, blocks, sizes, layers, error",
    [
        ([0], {}, {}, [0], "matching real group IDs"),
        ([0], {1: [2]}, {2: 8}, [1], "matching real group IDs"),
        ([0], {-1: [2]}, {-1: 8}, [-1], "matching real group IDs"),
        ([0], {True: [2]}, {True: 8}, [True], "matching real group IDs"),
        ([0], {1: [2]}, {True: 8}, [1], "matching real group IDs"),
        ([0], {1: [2]}, {1: 8}, [], "allocated group"),
        ([0], {1: [2]}, {1: 8}, [2], "allocated group"),
        ([0], {1: [2]}, {1: 8}, [True], "allocated group"),
        ([-1], {1: [2]}, {1: 8}, [1], "positions"),
        ([True], {1: [2]}, {1: 8}, [1], "positions"),
        ([0], {1: [2]}, {1: 0}, [1], "block sizes"),
        ([0], {1: [2]}, {1: True}, [1], "block sizes"),
        ([0], {1: [-1]}, {1: 8}, [1], "block IDs"),
        ([0], {1: [True]}, {1: 8}, [1], "block IDs"),
        ([8], {1: [2]}, {1: 8}, [1], "does not cover position 8"),
    ],
)
def test_slots_reject_invalid_allocation_and_position_data(
    positions, blocks, sizes, layers, error
):
    with pytest.raises(ValueError, match=error):
        build_draft_context_slots(positions, blocks, sizes, layers)


@pytest.fixture
def remote_page():
    return DSparkKVPage(
        layer_name="model.layers.0.swa_cache",
        group_id=0,
        base_address=1000,
        block_stride=256,
        page_size=64,
        block_size=4,
        num_blocks=20,
        format_tag="bf16:1:8",
    )


def test_target_reads_use_both_strides_and_only_the_layer_payload(remote_page):
    local_page = replace(remote_page, group_id=1, base_address=10000, block_stride=512)

    assert build_target_page_reads(
        [local_page], [remote_page], [[], [0, 7, 8, 10]], [[0, 3, 4, 9]], 9
    ) == ([13584, 14096], [1768, 2024], [64, 64])


def test_target_reads_skip_evicted_local_history_without_compacting_tables(remote_page):
    local_page = replace(remote_page, base_address=10000)

    assert build_target_page_reads(
        [local_page], [remote_page], [[0, 0, 8]], [[2, 3, 4]], 12
    ) == ([12048], [2024], [64])


def test_target_reads_preserve_aliases_without_repeating_the_same_copy(remote_page):
    alias = replace(remote_page, layer_name="model.layers.0.alias")

    assert build_target_page_reads(
        [remote_page, alias], [remote_page, alias], [[2]], [[3]], 1
    ) == ([1512], [1768], [64])


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"layer_name": ""}, "identity"),
        ({"group_id": -1}, "group ID"),
        ({"base_address": 0}, "positive"),
        ({"num_blocks": True}, "positive"),
        ({"page_size": 257}, "payload"),
    ],
)
def test_page_descriptor_rejects_invalid_wire_geometry(remote_page, changes, error):
    with pytest.raises(ValueError, match=error):
        replace(remote_page, **changes)


@pytest.mark.parametrize(
    "remote_changes, local_blocks, remote_blocks, prompt_tokens, error",
    [
        ({"layer_name": "draft_model.cache"}, [[2]], [[3]], 1, "identities"),
        ({"block_size": 8}, [[2]], [[3]], 1, "formats"),
        ({"format_tag": "fp8:1:8"}, [[2]], [[3]], 1, "formats"),
        ({"group_id": 1}, [[2]], [[3]], 1, "absent allocation"),
        ({}, [[2]], [[3]], 0, "positive P prompt"),
        ({}, [[2]], [[3]], 5, "cover P prompt"),
        ({}, [[20]], [[3]], 1, "outside"),
        ({}, [[2]], [[20]], 1, "outside"),
        ({}, [[True]], [[3]], 1, "outside"),
        ({}, [[2]], [[0]], 1, "P evicted"),
    ],
)
def test_target_reads_reject_incompatible_or_stale_allocations(
    remote_page, remote_changes, local_blocks, remote_blocks, prompt_tokens, error
):
    with pytest.raises(ValueError, match=error):
        build_target_page_reads(
            [remote_page],
            [replace(remote_page, **remote_changes)],
            local_blocks,
            remote_blocks,
            prompt_tokens,
        )
