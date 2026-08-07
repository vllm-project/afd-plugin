from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.models.deepseek_v4.nvidia import model as native  # noqa: E402

from afd_plugin.model_executor.models.deepseek_v4 import (  # noqa: E402
    AFDDeepseekV4ForCausalLM,
    _checkpoint_weight_roles,
)


class _OneShotWeights:
    def __init__(self, names: list[str]) -> None:
        self.items = [(name, torch.tensor([idx])) for idx, name in enumerate(names)]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.items)


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.ffn.gate.weight",
        "layers.1.ffn.experts.0.w1.weight",
        "model.layers.2.ffn.shared_experts.w2.weight",
    ],
)
def test_v4_raw_checkpoint_ffn_paths_are_ffn_owned(name):
    assert _checkpoint_weight_roles(name) == frozenset(("ffn",))


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.attn.fused_wqa_wkv.weight",
        "layers.0.hc_attn_fn",
        "layers.0.hc_ffn_fn",
        "layers.0.attn_norm.weight",
        "layers.0.ffn_norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
    ],
)
def test_v4_raw_checkpoint_attention_paths_are_attention_owned(name):
    assert _checkpoint_weight_roles(name) == frozenset(("attention",))


@pytest.mark.parametrize(
    "name",
    [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "model.embed_tokens.weight",
    ],
)
def test_v4_raw_checkpoint_public_paths_are_shared(name):
    assert _checkpoint_weight_roles(name) == frozenset(("attention", "ffn"))


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        (
            "attention",
            [
                "layers.0.attn.fused_wqa_wkv.weight",
                "layers.0.hc_ffn_fn",
                "model.hc_head_fn",
                "embed.weight",
            ],
        ),
        (
            "ffn",
            [
                "layers.0.ffn.gate.weight",
                "embed.weight",
            ],
        ),
    ],
)
def test_v4_load_weights_filters_raw_checkpoint_names_once(
    monkeypatch,
    role,
    expected_names,
):
    names = [
        "layers.0.attn.fused_wqa_wkv.weight",
        "layers.0.hc_ffn_fn",
        "layers.0.ffn.gate.weight",
        "model.hc_head_fn",
        "embed.weight",
    ]
    weights = _OneShotWeights(names)
    seen = []
    native_result = {"native.loaded"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(name for name, _ in filtered_weights)
        return native_result

    monkeypatch.setattr(
        native.DeepseekV4ForCausalLM,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(AFDDeepseekV4ForCausalLM)
    object.__setattr__(model, "afd_role", role)

    result = model.load_weights(weights)

    assert result is native_result
    assert seen == expected_names
    assert weights.iterations == 1
