# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

pytest.importorskip("vllm")

from vllm.model_executor.models import deepseek_v2 as native  # noqa: E402

from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
    _checkpoint_weight_roles,
)
from tests.unit.fakes.weights import OneShotWeights as _OneShotWeights  # noqa: E402

ATTENTION_ROLES = frozenset(("attention",))
FFN_ROLES = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))
WEIGHT_ROLE_GROUPS = (
    (
        BOTH_ROLES,
        BOTH_ROLES,
        (
            ("embedding", "model.embed_tokens.weight"),
            ("decoder-norm", "model.layers.0.input_layernorm.weight"),
        ),
    ),
    (
        ATTENTION_ROLES,
        ATTENTION_ROLES,
        (("mha-q-projection", "model.layers.0.self_attn.q_proj.weight"),),
    ),
    (
        FFN_ROLES,
        ATTENTION_ROLES,
        (("dense-gate-projection", "model.layers.0.mlp.gate_proj.weight"),),
    ),
    (FFN_ROLES, BOTH_ROLES, (("moe-gate", "model.layers.3.mlp.gate.weight"),)),
    (
        FFN_ROLES,
        FFN_ROLES,
        (
            ("moe-expert-projection", "model.layers.3.mlp.experts.0.gate_proj.weight"),
            (
                "shared-expert-projection",
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
            ),
        ),
    ),
)
WEIGHT_ROLE_CASES = tuple(
    (case_id, checkpoint_name, standard_roles, attention_gate_roles)
    for standard_roles, attention_gate_roles, cases in WEIGHT_ROLE_GROUPS
    for case_id, checkpoint_name in cases
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        first_k_dense_replace=3,
        moe_layer_freq=1,
        model_type="deepseek",
        n_routed_experts=64,
        n_shared_experts=1,
        num_nextn_predict_layers=0,
    )


@pytest.mark.parametrize(
    ("checkpoint_name", "standard_roles", "attention_gate_roles"),
    [case[1:] for case in WEIGHT_ROLE_CASES],
    ids=[case[0] for case in WEIGHT_ROLE_CASES],
)
def test_weight_role_policy(
    checkpoint_name: str,
    standard_roles: frozenset[str],
    attention_gate_roles: frozenset[str],
) -> None:
    assert (
        _checkpoint_weight_roles(
            checkpoint_name,
            _config(),
            compute_gate_on_attention=False,
        )
        == standard_roles
    )
    assert (
        _checkpoint_weight_roles(
            checkpoint_name,
            _config(),
            compute_gate_on_attention=True,
        )
        == attention_gate_roles
    )


def test_load_weights_passes_one_shot_generator_to_native_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
    ]
    weights = _OneShotWeights(names)
    seen: list[str] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(name for name, _ in filtered_weights)
        return native_result

    monkeypatch.setattr(
        native.DeepseekV2ForCausalLM,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    object.__setattr__(model, "afd_role", "attention")
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(
            compute_gate_on_attention=False, connector="CAMP2pAFDConnector"
        ),
    )
    object.__setattr__(model, "config", _config())

    result = model.load_weights(weights)

    assert result is native_result
    assert seen == names[:2]
    assert weights.iterations == 1


def test_async_shared_weights_belong_to_attention():
    name = "model.layers.3.mlp.shared_experts.gate_proj.weight"
    assert (
        _checkpoint_weight_roles(
            name,
            _config(),
            compute_gate_on_attention=True,
            attention_shared_experts=True,
        )
        == ATTENTION_ROLES
    )
    assert (
        _checkpoint_weight_roles(
            name,
            _config(),
            compute_gate_on_attention=True,
        )
        == FFN_ROLES
    )


M2N_CHECKPOINT_NAMES = (
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.3.mlp.gate.weight",
    "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "model.layers.3.mlp.experts.0.down_proj.weight",
)


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("attention", M2N_CHECKPOINT_NAMES[:3]),
        ("ffn", M2N_CHECKPOINT_NAMES[1:2] + M2N_CHECKPOINT_NAMES[3:]),
    ],
)
def test_deepep_m2n_loads_shared_experts_on_attention(
    monkeypatch: pytest.MonkeyPatch, role: str, expected: tuple[str, ...]
) -> None:
    """Under DeepEP M2N, Attention loads the dense MLP, the gate and the
    shared experts; the FFN loads the gate and the routed experts only.

    Why: the DeepEP M2N FFN runs routed experts only, and Attention adds the
    shared experts' output to the combined routed output.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        native.DeepseekV2ForCausalLM,
        "load_weights",
        lambda self, weights: seen.extend(name for name, _ in weights),
    )
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    object.__setattr__(model, "afd_role", role)
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(
            compute_gate_on_attention=True, connector="DeepEPM2NAFDConnector"
        ),
    )
    object.__setattr__(model, "config", _config())

    model.load_weights(_OneShotWeights(list(M2N_CHECKPOINT_NAMES)))

    assert tuple(seen) == expected
