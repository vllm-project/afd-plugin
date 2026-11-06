# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checkpoint ownership contracts; no native runtime or device execution."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def weight_policy():
    # Match the V4 harness: execute the real pure helpers and constants,
    # without importing the native model or duplicating its prefix policy.
    path = (
        Path(__file__).resolve().parents[5]
        / "afd_plugin/model_executor/models/npu/deepseek_v41.py"
    )
    names = {"_weight_layer_path", "_checkpoint_weight_roles", "_iter_role_weights"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
        or isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id.isupper()
    ]
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDDeepseekV41DecoderLayer"
    )
    helpers.append(
        next(
            node
            for node in decoder.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_ffn_output"
        )
    )
    # Only the Python rejection branch uses this nominal MoE type. No native
    # constructor, tensor computation or connector execution is simulated.
    namespace = {"native": SimpleNamespace(DeepseekV41MoE=SimpleNamespace)}
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def _checkpoint_weight_roles(weight_policy):
    return weight_policy["_checkpoint_weight_roles"]


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.engram.embed.weight",
        "model.layers.14.engram.embed.weight",
        "model.layers.14.engram.embed.scale",
    ],
)
def test_engram_paths_belong_to_the_attention_role(
    _checkpoint_weight_roles, name: str
) -> None:
    """Only Attention allocates n-gram tables, so only it may receive them."""

    assert _checkpoint_weight_roles(
        name,
        attention_owns_engram=True,
    ) == frozenset({"attention"})
    assert (
        _checkpoint_weight_roles(
            name,
            attention_owns_engram=False,
        )
        == frozenset()
    )


@pytest.mark.parametrize(
    "name",
    [
        "vision.blocks.0.attn.qkv.weight",
        "aligner.mlp.0.weight",
        "image_newline",
        "model.image_end",
        "model.image_start",
        "model.image_newline",
        "model.vision.blocks.0.attn.qkv.weight",
        "model.aligner.mlp.0.weight",
        "model.mtp.layers.0.ffn.gate.weight",
        "mtp.layers.0.ffn.gate.weight",
    ],
)
def test_non_text_paths_belong_to_no_afd_role(
    _checkpoint_weight_roles, name: str
) -> None:
    """The text-only split never executes vision or draft tensors."""

    for attn_owns_gate in (True, False):
        assert (
            _checkpoint_weight_roles(
                name,
                attn_owns_gate=attn_owns_gate,
                attention_owns_engram=True,
            )
            == frozenset()
        )


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.attn.q_a_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ],
)
def test_attention_paths_are_attention_owned(
    _checkpoint_weight_roles, name: str
) -> None:
    assert _checkpoint_weight_roles(name) == frozenset({"attention"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.ffn.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ],
)
def test_expert_paths_are_ffn_owned(_checkpoint_weight_roles, name: str) -> None:
    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.ffn.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_follow_the_router_owner(
    _checkpoint_weight_roles, name: str
) -> None:
    """A role that built no router raises ``KeyError`` on a foreign gate path."""

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=True,
    ) == frozenset({"attention", "ffn"})
    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=False,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_ffn_base",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
)
def test_layer_scaffolding_is_shared_by_both_roles(
    _checkpoint_weight_roles, name: str
) -> None:
    assert _checkpoint_weight_roles(name) == frozenset({"attention", "ffn"})


def test_vision_router_bias_stays_on_the_ffn_role(_checkpoint_weight_roles) -> None:
    """Only the native MoE registers ``gate.bias_vl``, so Attention has no slot."""

    assert _checkpoint_weight_roles(
        "model.layers.0.ffn.gate.bias_vl",
        attn_owns_gate=True,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize("stage", ["ffn", "mlp"])
def test_async_shared_experts_belong_to_the_payload_owner(
    _checkpoint_weight_roles, stage: str
) -> None:
    name = f"model.layers.0.{stage}.shared_experts.gate_proj.weight"

    assert _checkpoint_weight_roles(
        name,
        attention_shared_experts=True,
    ) == frozenset({"attention"})
    assert _checkpoint_weight_roles(name) == frozenset({"ffn"})


def test_unclassified_layer_paths_are_rejected(_checkpoint_weight_roles) -> None:
    with pytest.raises(
        RuntimeError,
        match="unclassified DeepSeek-V4.1 checkpoint weight",
    ):
        _checkpoint_weight_roles("model.layers.0.unexpected.weight")


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_role_filter_preserves_original_loader_names(weight_policy, role):
    weights = [
        ("model.image_end", "excluded"),
        ("model.mtp.layers.0.ffn.gate.weight", "excluded"),
        ("model.embed_tokens.weight", "embedding"),
        ("model.layers.0.attn.q_a_proj.weight", "attention"),
        ("layers.0.ffn.experts.0.gate_proj.weight", "expert"),
    ]
    expected = [weights[2], weights[3] if role == "attention" else weights[4]]
    # A one-pass checkpoint stream must retain original names and values.
    assert (
        list(weight_policy["_iter_role_weights"](iter(weights), role=role)) == expected
    )


def test_async_ffn_rejects_execution_outside_layered_runner(weight_policy):
    layer = SimpleNamespace(mlp=SimpleNamespace(), compute_gate_on_attention=True)
    with pytest.raises(
        RuntimeError, match="Async CAM FFN requires the layered executor"
    ):
        weight_policy["compute_ffn_output"](layer, None)
