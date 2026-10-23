# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 source-checkpoint metadata, without loading tensors or device libraries."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription


@pytest.fixture
def v4_checkpoint(tmp_path: Path) -> Path:
    # Geometry and raw ModelSlim names match the archived Flash checkpoint.
    # Empty shard files exercise metadata only; they are never tensor evidence.
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_act": "silu",
        "scoring_func": "sqrtsoftplus",
        "torch_dtype": "bfloat16",
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
    }
    names = [
        f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
        for layer in range(43)
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
        for suffix in ("weight", "weight_scale", "weight_offset", "scale_bias")
    ]
    quant = dict.fromkeys(names, "W4A8_DYNAMIC") | {"group_size": 0}
    # Attention/shared and extra MTP parameters must stay outside routed F.
    extras = (
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.43.ffn.experts.0.w1.weight",
    )
    quant.update(dict.fromkeys(extras, "W8A8_DYNAMIC"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "quant_model_description.json").write_text(json.dumps(quant))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": dict.fromkeys([*names, *extras], "shard.safetensors")}
        )
    )
    (tmp_path / "shard.safetensors").touch()
    return tmp_path


def test_v4_metadata_retains_raw_quantized_names_and_main_layer_schedule(
    v4_checkpoint: Path,
) -> None:
    checkpoint = V4CheckpointDescription.read(v4_checkpoint)
    assert checkpoint.layer_indices == tuple(range(43))
    assert (checkpoint.hidden_size, checkpoint.intermediate_size) == (4096, 2048)
    assert (checkpoint.num_experts, checkpoint.top_k, checkpoint.hash_layers) == (
        256,
        6,
        3,
    )
    assert (checkpoint.routed_scaling_factor, checkpoint.swiglu_limit) == (1.5, 10.0)
    assert len(checkpoint.routed_weight_map) == 43 * 256 * 3 * 4
    assert "layers.42.ffn.experts.255.w3.scale_bias" in checkpoint.routed_weight_map
    assert "layers.43.ffn.experts.0.w1.weight" not in checkpoint.routed_weight_map
    assert "layers.0.ffn.shared_experts.w1.weight" not in checkpoint.routed_weight_map
    assert len(checkpoint.metadata_sha256) == 3
    spec = checkpoint.session_spec(
        attention_ranks=12, attention_tp=4, ffn_ranks=4, capacity=8, slots=2
    )
    assert (spec.source_count, spec.local_experts) == (3, 64)
    assert spec.weight_quantization == "w4a8_per_channel"
    assert spec.swiglu_limit == 10.0
    assert spec.dtype == "bfloat16"
    assert len(spec.native_config(rank=12, poll_budget=16)) == 13


@pytest.mark.parametrize("quant_type", ["BF16", "W8A8_DYNAMIC"])
def test_v4_rejects_wrong_routed_quantization(
    v4_checkpoint: Path, quant_type: str
) -> None:
    path = v4_checkpoint / "quant_model_description.json"
    quant = json.loads(path.read_text())
    quant["layers.42.ffn.experts.255.w3.weight"] = quant_type
    path.write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="requires W4A8_DYNAMIC"):
        V4CheckpointDescription.read(v4_checkpoint)


def test_v4_rejects_missing_compensation_before_device_loading(
    v4_checkpoint: Path,
) -> None:
    path = v4_checkpoint / "quant_model_weights.safetensors.index.json"
    index = json.loads(path.read_text())
    del index["weight_map"]["layers.42.ffn.experts.255.w3.scale_bias"]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="tensor is missing:.*w3.scale_bias"):
        V4CheckpointDescription.read(v4_checkpoint)


def test_v4_rejects_unsupported_grouped_weights(v4_checkpoint: Path) -> None:
    path = v4_checkpoint / "quant_model_description.json"
    quant = json.loads(path.read_text())
    quant["group_size"] = 128
    path.write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="per-channel"):
        V4CheckpointDescription.read(v4_checkpoint)


def test_v4_metadata_import_needs_no_tensor_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import afd_plugin.ffn_faas.v4_model; "
            "assert {'torch','torch_npu','vllm','vllm_ascend'}.isdisjoint(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
