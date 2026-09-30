# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from pathlib import Path
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
    AFDDeepseekV2Model,
    _checkpoint_weight_roles,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ATTENTION_ROLES = frozenset(("attention",))
FFN_ROLES = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))
WEIGHT_ROLE_GROUPS = (
    (
        BOTH_ROLES,
        BOTH_ROLES,
        (
            ("embedding", "model.embed_tokens.weight"),
            ("final-norm", "model.norm.weight"),
            ("lm-head", "lm_head.weight"),
            ("decoder-norm", "model.layers.0.input_layernorm.weight"),
        ),
    ),
    (
        ATTENTION_ROLES,
        ATTENTION_ROLES,
        (
            ("mha-q-projection", "model.layers.0.self_attn.q_proj.weight"),
            ("mha-k-projection", "model.layers.0.self_attn.k_proj.weight"),
            ("mha-v-projection", "model.layers.0.self_attn.v_proj.weight"),
            ("mla-q-a-projection", "model.layers.0.self_attn.q_a_proj.weight"),
            (
                "mla-kv-a-projection",
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            ),
            ("indexer-projection", "model.layers.3.self_attn.indexer.wq_b.weight"),
            ("attention-kv-scale", "model.layers.3.self_attn.attn.k_scale"),
        ),
    ),
    (
        FFN_ROLES,
        ATTENTION_ROLES,
        (
            ("dense-gate-projection", "model.layers.0.mlp.gate_proj.weight"),
            ("dense-up-projection", "model.layers.0.mlp.up_proj.weight"),
            ("dense-down-projection", "model.layers.0.mlp.down_proj.weight"),
        ),
    ),
    (
        FFN_ROLES,
        BOTH_ROLES,
        (
            ("moe-gate", "model.layers.3.mlp.gate.weight"),
            ("moe-gate-bias", "model.layers.3.mlp.gate.e_score_correction_bias"),
        ),
    ),
    (
        FFN_ROLES,
        FFN_ROLES,
        (
            (
                "moe-expert-projection",
                "model.layers.3.mlp.experts.0.gate_proj.weight",
            ),
            (
                "moe-expert-scale",
                "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
            ),
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


@pytest.mark.parametrize("connector", ["P2pNcclAFDConnector", "CAMP2pAFDConnector"])
def test_load_weights_passes_one_shot_generator_to_native_loader(
    monkeypatch: pytest.MonkeyPatch,
    connector: str,
) -> None:
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight_scale_inv",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
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
        SimpleNamespace(compute_gate_on_attention=False, connector=connector),
    )
    object.__setattr__(model, "config", _config())

    result = model.load_weights(weights)

    assert result is native_result
    assert seen == names[:2]
    assert weights.iterations == 1


def test_native_model_loader_packs_mha_qkv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_name = "layers.0.self_attn.qkv_proj.weight"
    loaded_shards: list[tuple[str, torch.Tensor]] = []
    parameter = torch.nn.Parameter(torch.zeros(1))

    def load_qkv(param, loaded_weight, shard_id):
        assert param is parameter
        loaded_shards.append((shard_id, loaded_weight))

    parameter.weight_loader = load_qkv
    monkeypatch.setattr(
        native.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        native,
        "fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(native, "get_pp_missing_layer_names", lambda *args: set())
    monkeypatch.setattr(native, "is_pp_missing_parameter", lambda *args: False)

    model = object.__new__(AFDDeepseekV2Model)
    object.__setattr__(model, "config", _config())
    object.__setattr__(model, "use_mha", True)
    object.__setattr__(model, "num_redundant_experts", 0)
    object.__setattr__(
        model,
        "named_parameters",
        lambda: iter(((target_name, parameter),)),
    )
    qkv_weights = [
        (f"layers.0.self_attn.{projection}_proj.weight", torch.tensor([i]))
        for i, projection in enumerate(("q", "k", "v"))
    ]

    loaded_params = model.load_weights(iter(qkv_weights))

    assert loaded_params == {target_name}
    assert [shard_id for shard_id, _ in loaded_shards] == ["q", "k", "v"]
    assert [weight.item() for _, weight in loaded_shards] == [0, 1, 2]


def test_moe_metadata_and_backend_loading_remain_native_owned() -> None:
    assert "set_moe_parameters" not in AFDDeepseekV2ForCausalLM.__dict__
    source = (
        REPO_ROOT / "afd_plugin" / "model_executor" / "models" / "deepseek_v2.py"
    ).read_text(encoding="utf-8")
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    assert "vllm_ascend" not in module_imports
    assert "vllm_ascend.ascend_config" not in source


@pytest.mark.parametrize("suffix", ["weight", "weight_scale", "input_scale", "bias"])
def test_async_shared_weights_belong_to_attention(suffix):
    name = f"model.layers.3.mlp.shared_experts.gate_proj.{suffix}"
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
