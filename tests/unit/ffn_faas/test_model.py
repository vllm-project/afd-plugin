# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU model-resource boundary; native loading is qualified on the NPU."""

from __future__ import annotations

import pytest

from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel
from tests.unit.fakes.ffn_faas import MetadataModelAdapter


def test_resource_uses_adapter_routed_layers_and_loading_estimate(tmp_path):
    model = CAMRoutedModel(MetadataModelAdapter(tmp_path), ffn_ranks=2, ffn_rank=1)
    assert model.root == tmp_path
    assert model.get_experts_layer_indices() == (2, 7)
    assert model.parameter_bytes == 2048
    with pytest.raises(RuntimeError, match="not been loaded"):
        model.get_async_cam_w4a8_layers()
    with pytest.raises(RuntimeError, match="not finished loading"):
        _ = model.native_owner
    model.close()
    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = model.native_view


@pytest.mark.parametrize("world,rank", [(0, 0), (3, 0), (2, 2), (2, -1)])
def test_resource_rejects_incompatible_f_topology_before_native_loading(
    tmp_path, world, rank
):
    with pytest.raises(ValueError):
        CAMRoutedModel(MetadataModelAdapter(tmp_path), ffn_ranks=world, ffn_rank=rank)
