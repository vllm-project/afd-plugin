# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint weight-role policy tests for the Kimi K3 AFD adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm.models.kimi_k3")

from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig  # noqa: E402

from afd_plugin.model_executor.models import kimi_k3 as adapter  # noqa: E402


@pytest.fixture
def kimi_config() -> KimiLinearConfig:
    """Only the MoE-layer predicate fields are read by the classifier."""
    return KimiLinearConfig(
        num_experts=8,
        first_k_dense_replace=1,
    )


@pytest.mark.parametrize(
    ("name", "expected_roles"),
    [
        # Vision paths are never loaded: AFD runs text-only.
        ("vision_tower.encoder.blocks.0.attn.qkv.weight", frozenset()),
        ("mm_projector.linear_1.weight", frozenset()),
        # Non-layer paths are Attention-owned.
        (
            "language_model.model.embed_tokens.weight",
            frozenset({"attention"}),
        ),
        ("language_model.model.norm.weight", frozenset({"attention"})),
        (
            "language_model.model.output_attn_res_proj.weight",
            frozenset({"attention"}),
        ),
        ("language_model.lm_head.weight", frozenset({"attention"})),
        ("model.embed_tokens.weight", frozenset({"attention"})),
        # Attention stages.
        (
            "language_model.model.layers.3.self_attn.fused_qkv_a_proj.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.self_attn.in_proj_qkvgfab.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.input_layernorm.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.post_attention_layernorm.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.self_attention_res_norm.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.mlp_res_proj.weight",
            frozenset({"attention"}),
        ),
        # Dense layer 0 runs on Attention under gate-on-attention.
        (
            "language_model.model.layers.0.mlp.gate_up_proj.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.0.mlp.down_proj.weight",
            frozenset({"attention"}),
        ),
        # Latent MoE: Attention computes routing on both roles' native gate
        # paths; latent projections and shared experts are Attention-owned.
        (
            "language_model.model.layers.3.block_sparse_moe.gate.weight",
            frozenset({"attention", "ffn"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.gate."
            "e_score_correction_bias",
            frozenset({"attention", "ffn"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe."
            "routed_expert_down_proj.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.routed_expert_norm.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe."
            "routed_expert_up_proj.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.shared_experts."
            "gate_up_proj.weight",
            frozenset({"attention"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.shared_experts."
            "down_proj.weight",
            frozenset({"attention"}),
        ),
        # Routed experts are FFN-owned.
        (
            "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
            frozenset({"ffn"}),
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.experts.5.w2.weight_scale",
            frozenset({"ffn"}),
        ),
    ],
)
def test_checkpoint_weight_roles(kimi_config, name, expected_roles):
    assert adapter._checkpoint_weight_roles(name, kimi_config) == expected_roles


def test_unclassified_layer_stage_raises(kimi_config):
    with pytest.raises(RuntimeError, match="unclassified"):
        adapter._checkpoint_weight_roles(
            "language_model.model.layers.3.unknown_stage.weight",
            kimi_config,
        )


def test_unclassified_moe_stage_raises(kimi_config):
    with pytest.raises(RuntimeError, match="unclassified"):
        adapter._checkpoint_weight_roles(
            "language_model.model.layers.3.block_sparse_moe.unknown.weight",
            kimi_config,
        )


def test_dense_layer_block_sparse_moe_path_raises(kimi_config):
    with pytest.raises(RuntimeError, match="unexpected block_sparse_moe"):
        adapter._checkpoint_weight_roles(
            "language_model.model.layers.0.block_sparse_moe.gate.weight",
            kimi_config,
        )


def test_iter_role_weights_filters_once_and_preserves_tensors(kimi_config):
    weights = [
        ("vision_tower.encoder.blocks.0.attn.qkv.weight", torch.zeros(1)),
        (
            "language_model.model.layers.1.self_attn.in_proj_qkvgfab.weight",
            torch.zeros(2),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.gate.weight",
            torch.zeros(3),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
            torch.zeros(4),
        ),
    ]

    attention_names = [
        name
        for name, _tensor in adapter._iter_role_weights(
            weights,
            role="attention",
            config=kimi_config,
        )
    ]
    ffn_names = [
        name
        for name, _tensor in adapter._iter_role_weights(
            weights,
            role="ffn",
            config=kimi_config,
        )
    ]

    assert attention_names == [
        "language_model.model.layers.1.self_attn.in_proj_qkvgfab.weight",
        "language_model.model.layers.1.block_sparse_moe.gate.weight",
    ]
    assert ffn_names == [
        "language_model.model.layers.1.block_sparse_moe.gate.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
    ]
    # The iterator is consumed once; tensor objects pass through unchanged.
    tensors = [
        tensor
        for _name, tensor in adapter._iter_role_weights(
            weights,
            role="ffn",
            config=kimi_config,
        )
    ]
    assert all(
        tensor is original
        for tensor, (_name, original) in zip(tensors, weights[2:], strict=False)
    )
