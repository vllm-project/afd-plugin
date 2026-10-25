# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native routed checkpoint metadata and contiguous placement contracts."""

from __future__ import annotations

import json

import pytest

from afd_plugin.ffn_faas.platforms.npu.environment import AscendFFNEnvironment
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig
from afd_plugin.ffn_faas.platforms.npu.v4_model import (
    V4CheckpointDescription,
    validate_static_experts,
)
from afd_plugin.ffn_faas.platforms.npu.v4_runtime import (
    V4FFNModel,
    validate_native_expert_map,
)


def test_main_routed_metadata_without_parameter_allocation(checkpoint_dir):
    checkpoint = V4CheckpointDescription.read(checkpoint_dir)
    assert checkpoint.layer_indices == (0, 1)
    assert len(checkpoint.routed_weight_map) == 2 * 4 * 3 * 4
    model = V4FFNModel(checkpoint, ffn_ranks=2, ffn_rank=1)
    assert model.get_experts_layer_indices() == (0, 1)
    assert model.parameter_bytes > 0
    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = model.native_view


@pytest.mark.parametrize(
    "rank,entries,owned",
    [
        (0, [0, 1, -1, -1], {0, 1}),
        (1, [-1, -1, 0, 1], {2, 3}),
    ],
)
def test_native_expert_map_matches_cam(rank, entries, owned):
    assert (
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=rank, expert_map=entries
        )
        == owned
    )


@pytest.mark.parametrize(
    "entries",
    [
        [0, 0, -1, -1],
        [-1, -1, 0, 1],
        [1, 0, -1, -1],
        [0, 1],
    ],
)
def test_native_expert_map_mismatch_rejected(entries):
    with pytest.raises(ValueError):
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=entries
        )


@pytest.mark.parametrize("ffn_ranks", [1, 2])
def test_implicit_native_map_only_when_all_experts_are_local(ffn_ranks):
    if ffn_ranks == 1:
        assert validate_native_expert_map(
            num_experts=4, ffn_ranks=1, ffn_rank=0, expert_map=None
        ) == frozenset(range(4))
    else:
        with pytest.raises(ValueError, match="explicit map"):
            validate_native_expert_map(
                num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=None
            )


@pytest.mark.parametrize(
    "name,value",
    [
        ("hidden_act", "gelu"),
        ("swiglu_limit", -1),
        ("hidden_size", 0),
        ("num_experts_per_tok", 5),
        ("num_hash_layers", 3),
    ],
)
def test_incompatible_model_metadata_rejected(checkpoint_dir, name, value):
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text())
    config[name] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        V4CheckpointDescription.read(checkpoint_dir)


def test_missing_routed_source_rejected_before_loading(checkpoint_dir):
    index_path = checkpoint_dir / "quant_model_weights.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["layers.1.ffn.experts.3.w3.scale_bias"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing"):
        V4CheckpointDescription.read(checkpoint_dir)


def test_wrong_compute_world_rejected_before_native_imports(
    checkpoint_dir, monkeypatch
):
    settings = FFNConfig(checkpoint_dir, attention_ranks=1, attention_tp=1, ffn_ranks=2)
    environment = AscendFFNEnvironment(settings)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="differs from the F-only"):
        environment.initialize_device()


@pytest.mark.parametrize(
    "eplb,elastic,redundant,dynamic",
    [
        (True, False, 0, False),
        (False, True, 0, False),
        (False, False, 1, False),
        (False, False, 0, True),
    ],
)
def test_static_cam_adapter_rejects_expert_relocation(
    eplb, elastic, redundant, dynamic
):
    with pytest.raises(ValueError, match="static expert placement"):
        validate_static_experts(
            enable_eplb=eplb,
            enable_elastic_ep=elastic,
            redundant_experts=redundant,
            dynamic_eplb=dynamic,
        )


def test_unused_compute_owner_can_close_without_importing_runtime(checkpoint_dir):
    settings = FFNConfig(checkpoint_dir, attention_ranks=1, attention_tp=1, ffn_ranks=1)
    environment = AscendFFNEnvironment(settings)
    environment.close()
    with pytest.raises(RuntimeError, match="device has not been initialized"):
        _ = environment.device
    with pytest.raises(RuntimeError, match="expert group has not been initialized"):
        _ = environment.ep_group
