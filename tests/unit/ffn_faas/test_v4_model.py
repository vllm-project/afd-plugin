# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native routed checkpoint metadata and contiguous placement contracts."""

from __future__ import annotations

import json

import pytest

from afd_plugin.ffn_faas.platforms.npu import get_model
from afd_plugin.ffn_faas.platforms.npu.environment import AscendFFNEnvironment
from afd_plugin.ffn_faas.platforms.npu.settings import (
    FFNConfig,
)
from afd_plugin.ffn_faas.platforms.npu.v4_model import (
    V4CheckpointDescription,
    V4ModelAdapter,
)


def test_main_routed_metadata_without_parameter_allocation(checkpoint_dir):
    checkpoint = V4CheckpointDescription.read(checkpoint_dir)
    assert checkpoint.layer_indices == (0, 1)
    assert len(checkpoint.routed_weight_map) == 2 * 4 * 3 * 4
    model = get_model(checkpoint_dir, ffn_ranks=2, ffn_rank=1)
    assert model.get_experts_layer_indices() == (0, 1)
    assert model.parameter_bytes > 0
    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = model.native_view


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


def test_v4_adapter_selects_only_local_checkpoint_sources(checkpoint_dir):
    adapter = V4ModelAdapter(V4CheckpointDescription.read(checkpoint_dir))
    sources = adapter.local_sources(1, frozenset({2, 3}))
    assert len(sources) == 24
    assert ("layers.1.ffn.experts.2.w1.weight_scale", "weights.safetensors") in sources
    assert all(name.startswith("layers.1.ffn.experts.") for name, _ in sources)
    assert not any("experts.0." in name or "experts.1." in name for name, _ in sources)


@pytest.mark.parametrize("architecture", ["DeepseekV3ForCausalLM", "UnknownModel"])
def test_model_selection_rejects_unqualified_architectures(
    checkpoint_dir, architecture
):
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = [architecture]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="DeepseekV4ForCausalLM"):
        get_model(checkpoint_dir, ffn_ranks=2, ffn_rank=0)


def test_model_selection_rejects_other_quantization_before_loading(checkpoint_dir):
    quant_path = checkpoint_dir / "quant_model_description.json"
    quant = json.loads(quant_path.read_text())
    quant["group_size"] = 128
    quant_path.write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="per-channel"):
        get_model(checkpoint_dir, ffn_ranks=2, ffn_rank=0)


def test_wrong_compute_world_rejected_before_native_imports(
    checkpoint_dir, monkeypatch
):
    settings = FFNConfig(checkpoint_dir, attention_ranks=1, attention_tp=1, ffn_ranks=2)
    environment = AscendFFNEnvironment(settings)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="differs from the F-only"):
        environment.initialize_device()


def test_unused_compute_owner_can_close_without_importing_runtime(checkpoint_dir):
    settings = FFNConfig(checkpoint_dir, attention_ranks=1, attention_tp=1, ffn_ranks=1)
    environment = AscendFFNEnvironment(settings)
    environment.close()
    with pytest.raises(RuntimeError, match="device has not been initialized"):
        _ = environment.device
    with pytest.raises(RuntimeError, match="expert group has not been initialized"):
        _ = environment.ep_group
