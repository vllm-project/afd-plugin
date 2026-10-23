# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint metadata planning, without a tensor or device runtime."""

import json
from pathlib import Path

import pytest

from afd_plugin.ffn_faas.model import (
    CheckpointDescription,
    FFNModel,
    is_v4_checkpoint,
)


@pytest.fixture
def checkpoint(tmp_path: Path) -> CheckpointDescription:
    config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
    }
    names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in (1, 2)
        for expert in range(4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ] + [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
    ]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "shard.safetensors")})
    )
    # Planning validates index keys and file presence, never reads tensors.
    (tmp_path / "shard.safetensors").touch()
    return CheckpointDescription.read(tmp_path)


@pytest.mark.parametrize("rank", [0, 1])
def test_plan_contains_only_owned_routed_experts(
    checkpoint: CheckpointDescription, rank: int
) -> None:
    spec = checkpoint.session_spec(attention_ranks=4, ffn_ranks=2, attention_tp=2)
    plan = tuple(checkpoint.plan(spec, rank))
    expected = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in (1, 2)
        for expert in range(rank * 2, rank * 2 + 2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert {
        name for item in plan for name in (item.gate, item.up, item.down)
    } == expected
    assert [(item.compact_layer, item.local_expert) for item in plan] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert FFNModel(checkpoint, spec, rank).parameter_bytes == 49152


def test_missing_expert_key_fails_before_device_loading(
    checkpoint: CheckpointDescription,
) -> None:
    path = checkpoint.root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    del index["weight_map"]["model.layers.1.mlp.experts.0.gate_proj.weight"]
    path.write_text(json.dumps(index))
    candidate = CheckpointDescription.read(checkpoint.root)
    spec = candidate.session_spec(attention_ranks=2, ffn_ranks=2, attention_tp=2)
    with pytest.raises(KeyError, match="gate_proj"):
        FFNModel(candidate, spec, 0)


def test_dense_layer_cannot_be_bound_as_routed_experts(
    checkpoint: CheckpointDescription,
) -> None:
    with pytest.raises(ValueError, match="non-routed"):
        checkpoint.session_spec(
            attention_ranks=2, ffn_ranks=2, attention_tp=2, layer_indices=(0,)
        )


def test_unloaded_model_has_no_native_view(checkpoint: CheckpointDescription) -> None:
    spec = checkpoint.session_spec(attention_ranks=2, ffn_ranks=2, attention_tp=2)
    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = FFNModel(checkpoint, spec, 0).native_view


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["DeepseekV4ForCausalLM"], True),
        (["AFDDeepseekV4ForCausalLM"], True),
        (["DeepseekV2ForCausalLM"], False),
        (["AFDDeepseekV2ForCausalLM"], False),
        ([], False),
    ],
)
def test_architecture_selects_the_v4_adapter(
    tmp_path: Path, architectures: list[str], expected: bool
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": architectures}))
    assert is_v4_checkpoint(tmp_path) is expected
