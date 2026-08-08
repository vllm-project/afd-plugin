# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402

ATTENTION_ROLE = frozenset(("attention",))
FFN_ROLE = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))
WEIGHT_ROLE_CASES = (
    ("embedding", "model.embed_tokens.weight", BOTH_ROLES),
    ("final-norm", "model.norm.weight", BOTH_ROLES),
    ("lm-head", "lm_head.weight", BOTH_ROLES),
    (
        "attention-q-projection",
        "model.layers.0.self_attn.q_proj.weight",
        ATTENTION_ROLE,
    ),
    (
        "attention-k-scale",
        "model.layers.0.self_attn.attn.k_scale",
        ATTENTION_ROLE,
    ),
    (
        "input-layernorm",
        "model.layers.0.input_layernorm.weight",
        ATTENTION_ROLE,
    ),
    (
        "post-attention-layernorm",
        "model.layers.0.post_attention_layernorm.weight",
        ATTENTION_ROLE,
    ),
    ("moe-gate", "model.layers.0.mlp.gate.weight", FFN_ROLE),
    (
        "moe-expert-projection",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        FFN_ROLE,
    ),
    (
        "moe-expert-scale",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        FFN_ROLE,
    ),
    (
        "shared-expert-projection",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        FFN_ROLE,
    ),
    (
        "shared-expert-gate",
        "model.layers.0.mlp.shared_expert_gate.weight",
        FFN_ROLE,
    ),
    (
        "dense-gate-projection",
        "model.layers.3.mlp.gate_proj.weight",
        FFN_ROLE,
    ),
)


class _OneShotWeights:
    def __init__(self, names: list[str]) -> None:
        self.items = [(name, torch.tensor([index])) for index, name in enumerate(names)]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.items)


@pytest.mark.parametrize(
    ("checkpoint_name", "roles"),
    [case[1:] for case in WEIGHT_ROLE_CASES],
    ids=[case[0] for case in WEIGHT_ROLE_CASES],
)
def test_weight_role_policy(
    checkpoint_name: str,
    roles: frozenset[str],
) -> None:
    assert adapter._checkpoint_weight_roles(checkpoint_name) == roles


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        (
            "attention",
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.input_layernorm.weight",
                "lm_head.weight",
            ],
        ),
        (
            "ffn",
            [
                "model.embed_tokens.weight",
                "model.layers.0.mlp.gate.weight",
                "model.layers.0.mlp.experts.0.down_proj.weight",
                "lm_head.weight",
            ],
        ),
    ],
)
def test_load_weights_filters_once_and_delegates_to_native_loader(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_names: list[str],
) -> None:
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "lm_head.weight",
    ]
    weights = _OneShotWeights(names)
    seen: list[tuple[str, torch.Tensor]] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(filtered_weights)
        return native_result

    monkeypatch.setattr(
        adapter.native.Qwen3MoeForCausalLM,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(adapter.AFDQwen3MoeForCausalLM)
    object.__setattr__(model, "afd_role", role)

    result = model.load_weights(weights)

    assert result is native_result
    assert [name for name, _tensor in seen] == expected_names
    assert [tensor for _name, tensor in seen] == [
        weights.items[names.index(name)][1] for name in expected_names
    ]
    assert weights.iterations == 1


def test_native_loader_and_mapping_remain_native_owned() -> None:
    assert "forward" not in adapter.AFDQwen3MoeModel.__dict__
    assert "load_weights" not in adapter.AFDQwen3MoeModel.__dict__
    assert (
        adapter.AFDQwen3MoeModel.hf_to_vllm_mapper
        is adapter.native.Qwen3MoeModel.hf_to_vllm_mapper
    )
    assert (
        adapter.AFDQwen3MoeForCausalLM.hf_to_vllm_mapper
        is adapter.native.Qwen3MoeForCausalLM.hf_to_vllm_mapper
    )


def test_model_reports_config_driven_moe_layer_indices() -> None:
    model = object.__new__(adapter.AFDQwen3MoeModel)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            decoder_sparse_step=2,
            mlp_only_layers=[3],
            num_experts=8,
            num_hidden_layers=6,
        ),
    )

    assert model.get_experts_layer_indices() == (1, 5)
