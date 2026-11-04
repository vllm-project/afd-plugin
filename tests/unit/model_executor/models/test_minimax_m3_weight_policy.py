# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models.npu import minimax_m3 as adapter  # noqa: E402
from tests.unit.fakes.weights import OneShotWeights as _OneShotWeights  # noqa: E402

ATTENTION_ROLE = frozenset(("attention",))
FFN_ROLE = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))
NO_ROLES: frozenset[str] = frozenset()
WEIGHT_ROLE_CASES = (
    ("embedding", "model.embed_tokens.weight", BOTH_ROLES),
    ("final-norm", "model.norm.weight", BOTH_ROLES),
    ("lm-head", "lm_head.weight", BOTH_ROLES),
    (
        "language-model-prefixed-attention",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        ATTENTION_ROLE,
    ),
    (
        "attention-qkv-projection",
        "model.layers.0.self_attn.qkv_proj.weight",
        ATTENTION_ROLE,
    ),
    (
        "attention-index-key-projection",
        "model.layers.0.self_attn.index_k_proj.weight",
        ATTENTION_ROLE,
    ),
    (
        "attention-kv-scale",
        "model.layers.0.self_attn.k_scale",
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
    ("moe-gate", "model.layers.0.block_sparse_moe.gate.weight", FFN_ROLE),
    (
        "moe-routed-expert",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        FFN_ROLE,
    ),
    (
        "moe-routed-expert-scale",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight_scale_inv",
        FFN_ROLE,
    ),
    (
        "moe-routing-bias",
        "model.layers.0.block_sparse_moe.e_score_correction_bias",
        FFN_ROLE,
    ),
    (
        "moe-shared-expert",
        "model.layers.0.block_sparse_moe.shared_experts.gate_up_proj.weight",
        FFN_ROLE,
    ),
    ("dense-mlp", "model.layers.3.mlp.gate_up_proj.weight", FFN_ROLE),
    (
        "vision-tower",
        "model.vision_tower.blocks.0.attn.proj.weight",
        NO_ROLES,
    ),
    (
        "vision-projector",
        "model.multi_modal_projector.linear_1.weight",
        NO_ROLES,
    ),
    ("vision-patch-merge", "model.patch_merge_mlp.fc1.weight", NO_ROLES),
    (
        "mtp-layer",
        "model.mtp.layers.0.self_attn.q_proj.weight",
        NO_ROLES,
    ),
)


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


def test_unclassified_decoder_weight_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match="unclassified MiniMax-M3 checkpoint weight",
    ):
        adapter._checkpoint_weight_roles("model.layers.0.unknown_stage.weight")


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
                "model.layers.0.block_sparse_moe.gate.weight",
                "model.layers.0.block_sparse_moe.experts.0.w1.weight",
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
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "lm_head.weight",
    ]
    weights = _OneShotWeights(names)
    seen: list[tuple[str, Tensor]] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(filtered_weights)
        return native_result

    monkeypatch.setattr(
        adapter.native.MiniMaxM3SparseForCausalLM,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(adapter.AFDMiniMaxM3SparseForCausalLM)
    object.__setattr__(model, "afd_role", role)

    result = model.load_weights(weights)

    assert result is native_result
    assert [name for name, _tensor in seen] == expected_names
    assert [tensor for _name, tensor in seen] == [
        weights.items[names.index(name)][1] for name in expected_names
    ]
    assert weights.iterations == 1


def test_native_loader_and_mapping_remain_native_owned() -> None:
    assert "forward" not in adapter.AFDMiniMaxM3Model.__dict__
    assert "load_weights" not in adapter.AFDMiniMaxM3Model.__dict__
    assert (
        adapter.AFDMiniMaxM3SparseForCausalLM.hf_to_vllm_mapper
        is adapter.native.MiniMaxM3SparseForCausalLM.hf_to_vllm_mapper
    )


def test_model_reports_config_driven_experts_layer_indices() -> None:
    model = object.__new__(adapter.AFDMiniMaxM3Model)
    layers = []
    for layer_idx, is_layer_sparse in enumerate((False, True, False, True)):
        layer = object.__new__(adapter.AFDMiniMaxM3DecoderLayer)
        object.__setattr__(layer, "layer_idx", layer_idx)
        object.__setattr__(layer, "is_layer_sparse", is_layer_sparse)
        layers.append(layer)
    object.__setattr__(model, "layers", layers)

    assert model.get_experts_layer_indices() == (1, 3)


def _supported_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
        ),
        speculative_config=None,
        lora_config=None,
    )


@pytest.mark.parametrize(
    ("vllm_config", "afd_config", "message"),
    [
        (
            _supported_vllm_config(),
            AFDConfig(compute_gate_on_attention=True),
            "requires compute_gate_on_attention=false",
        ),
        (
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=True,
                    enable_eplb=False,
                    pipeline_parallel_size=1,
                ),
                speculative_config=None,
                lora_config=None,
            ),
            AFDConfig(),
            "does not support sequence-parallel MoE",
        ),
        (
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=False,
                    enable_eplb=True,
                    pipeline_parallel_size=1,
                ),
                speculative_config=None,
                lora_config=None,
            ),
            AFDConfig(),
            "does not support EPLB",
        ),
        (
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=False,
                    enable_eplb=False,
                    pipeline_parallel_size=2,
                ),
                speculative_config=None,
                lora_config=None,
            ),
            AFDConfig(),
            "does not support pipeline parallelism",
        ),
        (
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=False,
                    enable_eplb=False,
                    pipeline_parallel_size=1,
                ),
                speculative_config=SimpleNamespace(method="eagle3"),
                lora_config=None,
            ),
            AFDConfig(),
            "does not support speculative decoding",
        ),
        (
            SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=False,
                    enable_eplb=False,
                    pipeline_parallel_size=1,
                ),
                speculative_config=None,
                lora_config=SimpleNamespace(),
            ),
            AFDConfig(),
            "does not support LoRA",
        ),
    ],
)
def test_unsupported_configurations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
    afd_config: AFDConfig,
    message: str,
) -> None:
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, afd_config)


def test_non_npu_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )

    with pytest.raises(RuntimeError, match="supports Ascend NPU only"):
        adapter._validate_supported_config(_supported_vllm_config(), AFDConfig())
