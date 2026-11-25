# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the shared experts-only FFN shell (``AFDFFNExpertsModel``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import (  # noqa: E402
    filter_safetensors_files_by_weight_name,
)
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.ffn.experts_shell import (  # noqa: E402
    AFDFFNExpertsModel,
    FFNExpertsSpec,
)

MIXED_NUM_LAYERS = 3
DENSE_LAYER = 0
MOE_LAYERS = (1, 2)
OUT_OF_RANGE_LAYER = MIXED_NUM_LAYERS
WEIGHT_SHAPE = (2, 3)
LAYER_1_FILL = 1.5
LAYER_2_FILL = 2.5


class _FakeRoutedExperts(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.w13_weight = nn.Parameter(torch.zeros(WEIGHT_SHAPE))


class _FakeSparseBlock(nn.Module):
    def __init__(self, layer_idx: int, prefix: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.experts = nn.Module()
        self.experts.routed_experts = _FakeRoutedExperts(layer_idx)


def _vllm_config(
    *,
    num_layers: int = MIXED_NUM_LAYERS,
    moe_layers: tuple[int, ...] = MOE_LAYERS,
) -> Any:
    text_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        moe_layers=frozenset(moe_layers),
    )
    return SimpleNamespace(
        additional_config={
            "afd": {
                "role": "ffn",
                "connector": DEEPEP_M2N_CONNECTOR,
            }
        },
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        quant_config=None,
    )


def _is_moe_layer(config: Any, layer_idx: int) -> bool:
    return layer_idx in config.moe_layers


def _is_routed_weight(name: str) -> bool:
    return ".experts." in name


def _make_spec(
    *,
    ffn_submodule: str = "mlp",
    built_calls: list[tuple[int, str]] | None = None,
    built_blocks: dict[int, _FakeSparseBlock] | None = None,
    is_routed_expert_weight: Callable[[str], bool] = _is_routed_weight,
    routed_experts: Callable[[nn.Module], Any] | None = None,
    hf_to_vllm_mapper: WeightsMapper | None = None,
    validate: Callable[[Any, AFDConfig], None] | None = None,
) -> FFNExpertsSpec[Any]:
    def build_sparse_block(
        _vllm_config: Any,
        layer_idx: int,
        block_prefix: str,
    ) -> nn.Module:
        if built_calls is not None:
            built_calls.append((layer_idx, block_prefix))
        block = _FakeSparseBlock(layer_idx, block_prefix)
        if built_blocks is not None:
            built_blocks[layer_idx] = block
        return block

    kwargs: dict[str, Any] = {
        "ffn_submodule": ffn_submodule,
        "is_moe_layer": _is_moe_layer,
        "build_sparse_block": build_sparse_block,
        "is_routed_expert_weight": is_routed_expert_weight,
        "hf_to_vllm_mapper": hf_to_vllm_mapper,
        "validate": validate,
    }
    if routed_experts is not None:
        kwargs["routed_experts"] = routed_experts
    return FFNExpertsSpec(**kwargs)


def test_mixed_schedule_builds_only_moe_blocks_and_exposes_routed_experts() -> None:
    built_calls: list[tuple[int, str]] = []
    built_blocks: dict[int, _FakeSparseBlock] = {}

    class _MixedExpertsModel(AFDFFNExpertsModel):
        experts_spec = _make_spec(built_calls=built_calls, built_blocks=built_blocks)

    model = _MixedExpertsModel(vllm_config=_vllm_config())

    assert built_calls == [
        (1, "model.layers.1.mlp"),
        (2, "model.layers.2.mlp"),
    ]
    assert len(model.model.layers) == MIXED_NUM_LAYERS
    assert list(model.model.layers[DENSE_LAYER].parameters()) == []
    assert list(model.model.layers[DENSE_LAYER].children()) == []
    assert set(dict(model.named_parameters())) == {
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        "model.layers.2.mlp.experts.routed_experts.w13_weight",
    }
    assert model.get_experts_layer_indices() == MOE_LAYERS
    assert model.get_routed_experts(1) is built_blocks[1].experts.routed_experts
    assert model.get_routed_experts(2) is built_blocks[2].experts.routed_experts
    for non_moe_layer in (DENSE_LAYER, -1, OUT_OF_RANGE_LAYER):
        with pytest.raises(ValueError, match="not an MoE layer"):
            model.get_routed_experts(non_moe_layer)


def test_custom_submodule_prefix_and_routed_experts_extractor() -> None:
    built_calls: list[tuple[int, str]] = []
    extracted = {1: object(), 2: object()}

    spec = _make_spec(
        ffn_submodule="ffn",
        built_calls=built_calls,
        routed_experts=lambda block: extracted[block.layer_idx],
    )
    model = AFDFFNExpertsModel(
        vllm_config=_vllm_config(),
        prefix="draft",
        spec=spec,
    )

    assert built_calls == [
        (1, "draft.model.layers.1.ffn"),
        (2, "draft.model.layers.2.ffn"),
    ]
    assert set(dict(model.named_parameters())) == {
        "model.layers.1.ffn.experts.routed_experts.w13_weight",
        "model.layers.2.ffn.experts.routed_experts.w13_weight",
    }
    assert model.get_routed_experts(1) is extracted[1]
    assert model.get_routed_experts(2) is extracted[2]


def test_missing_spec_raises_type_error() -> None:
    with pytest.raises(TypeError, match="FFNExpertsSpec"):
        AFDFFNExpertsModel(vllm_config=_vllm_config())


def test_validate_hook_runs_before_building_any_block() -> None:
    built_calls: list[tuple[int, str]] = []
    validated_roles: list[str] = []

    def fail_validate(_vllm_config: Any, afd_config: AFDConfig) -> None:
        validated_roles.append(afd_config.role)
        raise ValueError("unsupported FFN config")

    spec = _make_spec(built_calls=built_calls, validate=fail_validate)

    with pytest.raises(ValueError, match="unsupported FFN config"):
        AFDFFNExpertsModel(vllm_config=_vllm_config(), spec=spec)

    assert validated_roles == ["ffn"]
    assert built_calls == []


def test_empty_moe_schedule_fails_fast_with_runtime_error() -> None:
    built_calls: list[tuple[int, str]] = []
    spec = _make_spec(built_calls=built_calls)

    with pytest.raises(RuntimeError, match="no MoE layers"):
        AFDFFNExpertsModel(vllm_config=_vllm_config(moe_layers=()), spec=spec)

    assert built_calls == []


@pytest.mark.parametrize(
    ("weight_name", "expected_unused"),
    [
        ("model.embed_tokens.weight", True),
        ("model.norm.weight", True),
        ("lm_head.weight", True),
        ("model.layers.1.self_attn.q_proj.weight", True),
        ("model.layers.1.input_layernorm.weight", True),
        ("model.layers.1.post_attention_layernorm.weight", True),
        ("model.layers.0.mlp.gate_proj.weight", True),
        ("model.layers.0.mlp.experts.routed_experts.w13_weight", True),
        (
            f"model.layers.{OUT_OF_RANGE_LAYER}.mlp.experts.routed_experts.w13_weight",
            True,
        ),
        ("model.layers.1.mlp.shared_experts.gate_proj.weight", True),
        ("model.layers.1.mlp.gate.weight", True),
        ("model.layers.1.mlp.experts.routed_experts.w13_weight", False),
        ("model.layers.2.mlp.experts.routed_experts.w13_weight", False),
        ("model.language_model.layers.1.mlp.experts.routed_experts.w13_weight", False),
    ],
)
def test_is_unused_checkpoint_weight_filters_to_moe_routed_weights(
    weight_name: str,
    expected_unused: bool,
) -> None:
    model = AFDFFNExpertsModel(vllm_config=_vllm_config(), spec=_make_spec())

    assert model.is_unused_checkpoint_weight(weight_name) is expected_unused


def test_load_weights_loads_only_kept_weights_and_skips_unused() -> None:
    model = AFDFFNExpertsModel(vllm_config=_vllm_config(), spec=_make_spec())
    layer_1_tensor = torch.full(WEIGHT_SHAPE, LAYER_1_FILL)
    layer_2_tensor = torch.full(WEIGHT_SHAPE, LAYER_2_FILL)
    unused_tensor = torch.ones(WEIGHT_SHAPE)
    checkpoint_weights = iter(
        [
            ("model.embed_tokens.weight", unused_tensor),
            ("lm_head.weight", unused_tensor),
            ("model.layers.0.mlp.gate_proj.weight", unused_tensor),
            ("model.layers.1.self_attn.q_proj.weight", unused_tensor),
            ("model.layers.1.mlp.shared_experts.gate_proj.weight", unused_tensor),
            (
                f"model.layers.{OUT_OF_RANGE_LAYER}.mlp.experts.routed_experts.w13_weight",
                unused_tensor,
            ),
            ("model.layers.1.mlp.experts.routed_experts.w13_weight", layer_1_tensor),
            ("model.layers.2.mlp.experts.routed_experts.w13_weight", layer_2_tensor),
        ]
    )

    loaded = model.load_weights(checkpoint_weights)

    assert loaded == {
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        "model.layers.2.mlp.experts.routed_experts.w13_weight",
    }
    assert torch.equal(model.get_routed_experts(1).w13_weight, layer_1_tensor)
    assert torch.equal(model.get_routed_experts(2).w13_weight, layer_2_tensor)


def test_load_weights_applies_hf_to_vllm_mapper() -> None:
    spec = _make_spec(
        is_routed_expert_weight=lambda name: (
            name.startswith("model.language_model.") and ".experts." in name
        ),
        hf_to_vllm_mapper=WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "model."}
        ),
    )
    model = AFDFFNExpertsModel(vllm_config=_vllm_config(), spec=spec)
    layer_1_tensor = torch.full(WEIGHT_SHAPE, LAYER_1_FILL)
    layer_2_tensor = torch.full(WEIGHT_SHAPE, LAYER_2_FILL)
    unused_tensor = torch.ones(WEIGHT_SHAPE)

    loaded = model.load_weights(
        [
            (
                "model.visual.layers.1.mlp.experts.routed_experts.w13_weight",
                unused_tensor,
            ),
            ("model.language_model.embed_tokens.weight", unused_tensor),
            (
                "model.language_model.layers.1.mlp.experts.routed_experts.w13_weight",
                layer_1_tensor,
            ),
            (
                "model.language_model.layers.2.mlp.experts.routed_experts.w13_weight",
                layer_2_tensor,
            ),
        ]
    )

    assert loaded == {
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        "model.layers.2.mlp.experts.routed_experts.w13_weight",
    }
    assert torch.equal(model.get_routed_experts(1).w13_weight, layer_1_tensor)
    assert torch.equal(model.get_routed_experts(2).w13_weight, layer_2_tensor)


def test_filter_safetensors_files_skips_non_routed_shards(tmp_path: Path) -> None:
    model = AFDFFNExpertsModel(vllm_config=_vllm_config(), spec=_make_spec())
    assert model.fall_back_to_pt_during_load is False

    unused_shard = tmp_path / "model-00001-of-00002.safetensors"
    routed_shard = tmp_path / "model-00002-of-00002.safetensors"
    out_of_range_weight = (
        f"model.layers.{OUT_OF_RANGE_LAYER}.mlp.experts.routed_experts.w13_weight"
    )
    save_file(
        {
            name: torch.zeros(WEIGHT_SHAPE)
            for name in (
                "model.embed_tokens.weight",
                "model.layers.0.mlp.experts.routed_experts.w13_weight",
                "model.layers.1.self_attn.q_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                out_of_range_weight,
            )
        },
        str(unused_shard),
    )
    save_file(
        {
            name: torch.zeros(WEIGHT_SHAPE)
            for name in (
                "model.layers.1.self_attn.v_proj.weight",
                "model.layers.1.mlp.experts.routed_experts.w13_weight",
            )
        },
        str(routed_shard),
    )

    kept = filter_safetensors_files_by_weight_name(
        [str(unused_shard), str(routed_shard)],
        model.is_unused_checkpoint_weight,
    )

    assert kept == [str(routed_shard)]
