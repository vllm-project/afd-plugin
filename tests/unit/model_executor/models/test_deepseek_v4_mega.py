# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-owned admission and native weight extraction for Mega."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu import deepseek_v4_mega as adapter  # noqa: E402


@pytest.fixture
def flash_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                hidden_size=4096,
                moe_intermediate_size=2048,
                n_routed_experts=256,
                num_experts_per_tok=6,
                expert_dtype="fp4",
                swiglu_limit=10.0,
            )
        ),
        additional_config={},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "other"),
        ("hidden_size", 8192),
        ("expert_dtype", "int4"),
    ],
)
def test_flash_admission_rejects_unvalidated_models(flash_config, field, value):
    setattr(flash_config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash"):
        adapter.flash_mega_model_spec(flash_config)


def test_flash_admission_exports_explicit_geometry_and_activation(flash_config):
    spec = adapter.flash_mega_model_spec(flash_config)
    assert (spec.hidden_size, spec.intermediate_size, spec.num_experts, spec.top_k) == (
        4096,
        2048,
        256,
        6,
    )
    assert spec.activation_parameters == (10.0, 1.702, 1.0)


def test_flash_admission_rejects_native_fused_weight_layout(flash_config):
    flash_config.additional_config["enable_fused_mc2"] = 1
    with pytest.raises(ValueError, match="enable_fused_mc2=0"):
        adapter.flash_mega_model_spec(flash_config)


def test_extraction_preserves_native_axes_and_reuses_converted_storage(monkeypatch):
    from vllm_ascend.quantization.quant_type import QuantType

    original = torch.arange(24, dtype=torch.uint8).reshape(1, 4, 6)
    owner = SimpleNamespace(
        dynamic_eplb=False,
        w13_weight=torch.nn.Parameter(original.clone(), requires_grad=False),
        w2_weight=torch.nn.Parameter(original.clone(), requires_grad=False),
        w13_weight_scale=torch.zeros(1, 2, 6, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(1, 2, 6, dtype=torch.uint8),
    )
    layer = SimpleNamespace(
        layer_idx=7,
        mlp=SimpleNamespace(
            ep_rank=1,
            physical_expert_start=1,
            n_local_physical_experts=1,
            experts=SimpleNamespace(
                quant_type=QuantType.W4A8MXFP, routed_experts=owner
            ),
        ),
    )

    # Isolate device format conversion; the adapter still handles native axes
    # and replaces the model's original parameter storage itself.
    def convert(w1, w2, s1, s2, **dimensions):
        return [w1.contiguous() + 1], [w2.contiguous() + 2], [s1], [s2]

    monkeypatch.setattr(adapter, "convert_expert_weights", convert)
    weights = adapter.extract_mega_expert_weights([layer], 1, 8, 3)
    assert set(weights) == {7}
    assert torch.equal(weights[7][0][0], original.transpose(1, 2) + 1)
    assert torch.equal(owner.w13_weight, original + 1)
    assert torch.equal(owner.w2_weight, original + 2)
    assert (
        owner.w13_weight.untyped_storage().data_ptr()
        == weights[7][0][0].untyped_storage().data_ptr()
    )
    assert torch.equal(weights[7][2][0], owner.w13_weight_scale.transpose(1, 2))
