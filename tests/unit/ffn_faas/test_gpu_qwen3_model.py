# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The GPU FFN service accepts only the routed-experts Qwen3 MoE shape."""

from __future__ import annotations

import pytest

from afd_plugin.ffn_faas.platforms.gpu.qwen3_model import Qwen3CheckpointDescription
from tests.unit.fakes.ffn_faas import write_qwen3_checkpoint

# Qwen3-30B-A3B at 2F: 48 layers x 64 experts x 3 x 2048 x 768 x 2 B.
QWEN3_30B_2F_EXPERT_BYTES = 48 * 64 * 3 * 2048 * 768 * 2


def test_stock_qwen3_30b_checkpoint_is_described(tmp_path):
    checkpoint = Qwen3CheckpointDescription.read(write_qwen3_checkpoint(tmp_path))

    assert checkpoint.root == tmp_path.resolve()
    assert checkpoint.layer_indices == tuple(range(48))
    assert (checkpoint.num_experts, checkpoint.top_k) == (128, 8)
    assert checkpoint.local_experts(2) == 64
    assert checkpoint.estimated_local_expert_bytes(2) == QWEN3_30B_2F_EXPERT_BYTES


@pytest.mark.parametrize(
    "overrides",
    [
        {"shared_expert_intermediate_size": 0},
        {"shared_expert_intermediate_size": None},
        {"torch_dtype": None, "dtype": "bfloat16"},
    ],
    ids=["shared-zero", "shared-absent", "dtype-key"],
)
def test_equivalent_config_spellings_are_accepted(tmp_path, overrides):
    checkpoint = Qwen3CheckpointDescription.read(
        write_qwen3_checkpoint(tmp_path, **overrides)
    )
    assert checkpoint.num_layers == 48


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"architectures": ["DeepseekV2ForCausalLM"]}, "Qwen3MoeForCausalLM"),
        ({"torch_dtype": "float16"}, "BF16"),
        (
            {"quantization_config": {"quant_method": "fp8"}},
            "quantized",
        ),
        ({"mlp_only_layers": [0]}, "every layer to be MoE"),
        ({"decoder_sparse_step": 2}, "decoder_sparse_step=1"),
        ({"shared_expert_intermediate_size": 512}, "shared experts"),
        ({"num_experts": 0}, "num_experts must be a positive integer"),
    ],
    ids=[
        "architecture",
        "dtype",
        "fp8",
        "mlp-only-layers",
        "sparse-step",
        "shared-experts",
        "no-experts",
    ],
)
def test_unsupported_qwen3_shapes_fail_on_cpu(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        Qwen3CheckpointDescription.read(write_qwen3_checkpoint(tmp_path, **overrides))


def test_experts_must_divide_across_ffn_ranks(tmp_path):
    checkpoint = Qwen3CheckpointDescription.read(write_qwen3_checkpoint(tmp_path))
    with pytest.raises(ValueError, match="do not divide across 3 FFN ranks"):
        checkpoint.local_experts(3)
