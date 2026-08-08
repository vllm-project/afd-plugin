from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.models import qwen3_5 as native  # noqa: E402

from afd_plugin.model_executor.models.qwen3_5 import (  # noqa: E402
    AFDQwen3_5MoeForConditionalGeneration,
    _checkpoint_weight_roles,
)

ATTENTION = frozenset(("attention",))
FFN = frozenset(("ffn",))
NONE = frozenset()


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        ("model.language_model.model.embed_tokens.weight", ATTENTION),
        ("model.language_model.model.norm.weight", ATTENTION),
        ("model.language_model.lm_head.weight", ATTENTION),
        (
            "model.language_model.model.layers.0.linear_attn.in_proj_q.weight",
            ATTENTION,
        ),
        (
            "model.language_model.model.layers.3.self_attn.q_proj.weight",
            ATTENTION,
        ),
        ("model.language_model.model.layers.0.input_layernorm.weight", ATTENTION),
        ("model.language_model.model.layers.0.mlp.gate.weight", FFN),
        (
            "model.language_model.model.layers.0.mlp.experts.0.down_proj.weight",
            FFN,
        ),
        (
            "model.language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            FFN,
        ),
        (
            "model.language_model.model.layers.0.mlp.shared_expert_gate.weight",
            FFN,
        ),
        ("model.visual.patch_embed.proj.weight", NONE),
        ("mtp.layers.0.mlp.experts.down_proj", NONE),
    ],
)
def test_qwen_checkpoint_weight_roles(name, roles):
    assert _checkpoint_weight_roles(name) == roles


def test_qwen_ffn_gate_owns_router_checkpoint_weight():
    name = "model.language_model.model.layers.0.mlp.gate.weight"

    assert _checkpoint_weight_roles(name) == FFN


def test_qwen_load_weights_filters_one_shot_iterator(monkeypatch):
    names = [
        "model.language_model.model.embed_tokens.weight",
        "model.language_model.model.layers.0.mlp.gate.weight",
        "model.language_model.model.layers.0.mlp.experts.0.down_proj.weight",
        "model.language_model.model.layers.0.mlp.shared_expert_gate.weight",
        "model.visual.patch_embed.proj.weight",
    ]
    iterations = 0

    def one_shot():
        nonlocal iterations
        iterations += 1
        if iterations > 1:
            raise AssertionError("checkpoint iterator consumed more than once")
        yield from ((name, torch.tensor([index])) for index, name in enumerate(names))

    seen = []
    expected = {"native.loaded"}

    def native_load_weights(self, weights):
        seen.extend(name for name, _ in weights)
        return expected

    monkeypatch.setattr(
        native.Qwen3_5MoeForConditionalGeneration,
        "load_weights",
        native_load_weights,
    )
    model = object.__new__(AFDQwen3_5MoeForConditionalGeneration)
    object.__setattr__(model, "afd_role", "ffn")
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(role="ffn", compute_gate_on_attention=False),
    )

    result = model.load_weights(one_shot())

    assert result is expected
    assert seen == names[1:4]
    assert iterations == 1


def test_qwen_ffn_gate_loads_router_with_experts(monkeypatch):
    names = [
        "model.language_model.model.layers.0.mlp.gate.weight",
        "model.language_model.model.layers.0.mlp.experts.0.down_proj.weight",
        "model.language_model.model.layers.0.mlp.shared_expert_gate.weight",
    ]
    seen = []

    def native_load_weights(self, weights):
        seen.extend(name for name, _ in weights)
        return set()

    monkeypatch.setattr(
        native.Qwen3_5MoeForConditionalGeneration,
        "load_weights",
        native_load_weights,
    )
    model = object.__new__(AFDQwen3_5MoeForConditionalGeneration)
    object.__setattr__(model, "afd_role", "ffn")
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(role="ffn", compute_gate_on_attention=False),
    )

    model.load_weights(
        (name, torch.tensor([index])) for index, name in enumerate(names)
    )

    assert seen == names
