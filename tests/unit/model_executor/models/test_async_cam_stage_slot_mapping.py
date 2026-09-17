# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-stage KV slot mapping for the async-CAM ubatch path.

The bug this pins is silent: with the full-batch mapping left in place, every
stage writes its rows into the whole batch's KV slots, so the cache is corrupt
but nothing raises.
"""

import torch

from afd_plugin.model_executor.models.npu.deepseek_v2_async_cam_forward import (
    build_stage_slot_mapping,
)


def test_each_stage_gets_its_own_slice():
    slot_mapping = {"layer.0": torch.arange(8), "layer.1": torch.arange(8) + 100}

    first = build_stage_slot_mapping(slot_mapping, slice(0, 3))
    second = build_stage_slot_mapping(slot_mapping, slice(3, 8))

    assert torch.equal(first["layer.0"], torch.tensor([0, 1, 2]))
    assert torch.equal(second["layer.0"], torch.tensor([3, 4, 5, 6, 7]))
    # Every layer is sliced, not just the first.
    assert torch.equal(first["layer.1"], torch.tensor([100, 101, 102]))
    assert torch.equal(second["layer.1"], torch.tensor([103, 104, 105, 106, 107]))


def test_stages_partition_the_batch_without_overlap():
    slot_mapping = {"layer.0": torch.arange(6)}
    slices = [slice(0, 2), slice(2, 4), slice(4, 6)]

    rows = torch.cat(
        [build_stage_slot_mapping(slot_mapping, s)["layer.0"] for s in slices],
    )

    # Concatenating the stages must reproduce the batch exactly: no row written
    # twice, none dropped.
    assert torch.equal(rows, slot_mapping["layer.0"])


def test_the_parent_mapping_is_left_alone():
    slot_mapping = {"layer.0": torch.arange(4)}

    build_stage_slot_mapping(slot_mapping, slice(0, 2))

    assert torch.equal(slot_mapping["layer.0"], torch.arange(4))
    assert list(slot_mapping) == ["layer.0"]
