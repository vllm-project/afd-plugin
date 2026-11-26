# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

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
from tests.unit.fakes.weights import OneShotWeights  # noqa: E402

ATTENTION = frozenset(("attention",))
FFN = frozenset(("ffn",))
BOTH = ATTENTION | FFN
NONE: frozenset[str] = frozenset()


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


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        ("model.language_model.model.layers.0.mlp.gate.weight", BOTH),
        (
            "model.language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            BOTH,
        ),
        ("model.language_model.model.layers.0.mlp.shared_expert_gate.weight", BOTH),
        (
            "model.language_model.model.layers.0.mlp.experts.0.down_proj.weight",
            FFN,
        ),
    ],
)
def test_qwen_checkpoint_weight_roles_when_attention_routes(name, roles):
    """With DeepEP M2N, both roles load the gate and the gated shared expert,
    and only the FFN loads the routed experts.

    Why: Attention routes and runs the shared expert with them, and the FFN's
    native block still holds them, which vLLM's strict loader requires.
    """
    assert _checkpoint_weight_roles(name, routes_on_attention=True) == roles


NAMES = (
    "model.language_model.model.embed_tokens.weight",
    "model.language_model.model.layers.0.mlp.gate.weight",
    "model.language_model.model.layers.0.mlp.experts.0.down_proj.weight",
    "model.language_model.model.layers.0.mlp.shared_expert_gate.weight",
    "model.visual.patch_embed.proj.weight",
)


@pytest.mark.parametrize(
    ("role", "connector", "loaded"),
    [
        ("ffn", "P2pNcclAFDConnector", NAMES[1:4]),
        ("attention", "DeepEPM2NAFDConnector", (NAMES[0], NAMES[1], NAMES[3])),
    ],
    ids=["p2p-ffn", "deepep-m2n-attention"],
)
def test_qwen_load_weights_filters_one_shot_iterator(
    monkeypatch, role, connector, loaded
):
    names = list(NAMES)
    weights = OneShotWeights(names)

    seen: list[str] = []
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
    object.__setattr__(model, "afd_role", role)
    object.__setattr__(
        model,
        "afd_config",
        SimpleNamespace(
            role=role,
            connector=connector,
            compute_gate_on_attention=connector == "DeepEPM2NAFDConnector",
        ),
    )

    result = model.load_weights(weights)

    assert result is expected
    assert seen == list(loaded)
    assert weights.iterations == 1
