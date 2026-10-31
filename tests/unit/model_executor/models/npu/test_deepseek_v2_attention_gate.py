# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU post-quant scale ownership and CAM MLP consumption contracts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.unit.model_executor.models.attention_gate_test_utils import (
    load_attention_gate_moe_ffn,
)

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

_MODEL_ROOT = Path(__file__).resolve().parents[5] / "afd_plugin/model_executor/models"
_HELPER_MODULE = "afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate"


class _RoutedExperts(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.w13_weight_scale = nn.Parameter(
            torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=dtype),
            requires_grad=False,
        )
        self.w13_weight = torch.ones((2, 2, 2), dtype=torch.int8)
        self.w2_weight = torch.ones((2, 1, 2), dtype=torch.int8)
        self.w2_weight_scale = torch.ones((2, 2), dtype=dtype)
        self.dynamic_eplb = False
        self.activation = "silu"
        self.quant_method = SimpleNamespace(quant_method=SimpleNamespace())

    def get_eplb_parameter(self, name):
        # Exact native accessor contract: absent derived state raises directly.
        return getattr(self, name)


def _make_layer(owner, quant_type):
    layer = nn.Module()
    layer.is_moe_layer = True
    layer.mlp = nn.Module()
    layer.mlp.routed_scaling_factor = 1.0
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.quant_type = quant_type
    layer.mlp.experts.routed_experts = owner
    layer.mlp.experts.shared_experts = None
    layer.mlp.experts.moe_config = SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    return layer


def _cpu_mlp(*, mlp_compute_input, quant_method):
    """Consume the actual helper's routed payload with ordinary CPU matmuls."""
    payload = mlp_compute_input
    w1, w2 = payload.weights.w1[0], payload.weights.w2[0]
    w1_scale, w2_scale = payload.weights.w1_scale[0], payload.weights.w2_scale[0]
    outputs = []
    start = 0
    for expert, end in enumerate(payload.group_list.tolist()):
        states = payload.hidden_states[start:end].float()
        gate_up = (states @ w1[expert].float()) * w1_scale[expert]
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up
        outputs.append((hidden @ w2[expert].float()) * w2_scale[expert])
        start = end
    return torch.cat(outputs).to(torch.bfloat16), None


class _FFNModel(nn.Module):
    def __init__(self, layer, compute_ffn):
        super().__init__()
        self.layers = nn.ModuleList([layer])
        self._compute_ffn = compute_ffn

    def compute_ffn_output(self, states, layer_idx, **kwargs):
        return self._compute_ffn(self.layers[layer_idx], hidden_states=states, **kwargs)


class _NativeCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = vllm_config.model


def _compile_nodes(path, nodes, namespace):
    # Execute production functions against CPU collaborators, without importing
    # vLLM/Ascend or asserting on production source text.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def runtime(monkeypatch):
    compute_ffn, quant_type = load_attention_gate_moe_ffn(monkeypatch, _cpu_mlp)
    helper_path = _MODEL_ROOT / "npu/deepseek_v2_attention_gate.py"
    prepare = next(
        node
        for node in ast.parse(helper_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_attention_gate_w8a8_scales"
    )
    namespace = {"torch": torch, "cast": cast}
    _compile_nodes(helper_path, [prepare], namespace)
    helper = SimpleNamespace(
        prepare_attention_gate_w8a8_scales=namespace[
            "prepare_attention_gate_w8a8_scales"
        ],
    )
    monkeypatch.setitem(sys.modules, _HELPER_MODULE, helper)

    platform = SimpleNamespace(device_type="npu")
    wrappers = {}
    for family, filename, class_name in (
        ("v2", "deepseek_v2.py", "AFDDeepseekV2ForCausalLM"),
        ("v4", "npu/deepseek_v4.py", "AFDDeepseekV4ForCausalLM"),
    ):
        path = _MODEL_ROOT / filename
        classes = []
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == class_name:
                node.body = [
                    member
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                    and member.name
                    in (
                        "__init__",
                        "process_weights_after_loading",
                        "compute_ffn_output",
                    )
                ]
                classes.append(node)
            elif family == "v2" and node.name in (
                "AFDDeepseekForCausalLM",
                "AFDDeepseekV3ForCausalLM",
                "AFDGlmMoeDsaForCausalLM",
            ):
                classes.append(node)
        wrapper_namespace = {
            "native": SimpleNamespace(
                current_platform=platform,
                DeepseekV2ForCausalLM=_NativeCausalLM,
                AscendDeepseekV4ForCausalLM=_NativeCausalLM,
            ),
            "parse_afd_config": lambda config, **kwargs: config.afd,
        }
        _compile_nodes(path, classes, wrapper_namespace)
        wrappers[family] = cast(type[_NativeCausalLM], wrapper_namespace[class_name])
        if family == "v2":
            for alias in (
                "AFDDeepseekForCausalLM",
                "AFDDeepseekV3ForCausalLM",
                "AFDGlmMoeDsaForCausalLM",
            ):
                wrappers[alias] = cast(type[_NativeCausalLM], wrapper_namespace[alias])

    def make_model(family, *, dtype=torch.bfloat16, role="ffn", gate_on_a=True):
        owner = _RoutedExperts(dtype)
        layer = _make_layer(owner, quant_type.W8A8)
        config = SimpleNamespace(
            afd=SimpleNamespace(role=role, compute_gate_on_attention=gate_on_a),
            model=_FFNModel(layer, compute_ffn),
        )
        return wrappers[family](vllm_config=config), owner

    return SimpleNamespace(
        make_model=make_model, quant_type=quant_type, platform=platform
    )


def _ffn_output(model):
    return model.compute_ffn_output(
        torch.ones((2, 2), dtype=torch.bfloat16),
        0,
        group_list=torch.tensor([1, 2]),
        dynamic_scales=None,
        expand_x_shared=None,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=0,
    ).routed_output


def _expected_output(scales):
    # Both test weights are one, so each routed expert receives [2, 2].
    values = torch.nn.functional.silu(scales[:, 0].float() * 2)
    values = values * scales[:, 1].float() * 2
    return values[:, None].expand(-1, 2).to(torch.bfloat16)


@pytest.mark.parametrize("family", ["v2", "v4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_post_quant_scales_feed_cam_mlp_once(runtime, family, dtype):
    model, owner = runtime.make_model(family, dtype=dtype)
    canonical = owner.w13_weight_scale
    expected = canonical.detach().float().clone()

    model.process_weights_after_loading()
    cached = owner.w13_weight_scale_fp32
    model.process_weights_after_loading()

    assert owner.w13_weight_scale is canonical
    assert owner.w13_weight_scale.dtype == dtype
    assert owner.w13_weight_scale_fp32 is cached
    assert cached.dtype == torch.float32
    torch.testing.assert_close(cached, expected)
    torch.testing.assert_close(_ffn_output(model), _expected_output(expected))


@pytest.mark.parametrize("family", ["v2", "v4"])
def test_existing_post_quant_scale_is_preserved_and_consumed(runtime, family):
    model, owner = runtime.make_model(family)
    existing = torch.tensor([[2.0, 0.5], [1.0, 1.5]], dtype=torch.float32)
    owner.w13_weight_scale_fp32 = existing
    del owner.w13_weight_scale

    model.process_weights_after_loading()

    assert owner.w13_weight_scale_fp32 is existing
    torch.testing.assert_close(_ffn_output(model), _expected_output(existing))


@pytest.mark.parametrize("family", ["v2", "v4"])
def test_quant_refinalize_rebuilds_scale_from_new_canonical_values(runtime, family):
    model, owner = runtime.make_model(family)
    model.process_weights_after_loading()
    previous = owner.w13_weight_scale_fp32
    # Native quant finalize derives scales again and drops this cache for MegaMoE.
    owner.w13_weight_scale.data.add_(1)
    del owner.w13_weight_scale_fp32

    model.process_weights_after_loading()

    assert owner.w13_weight_scale_fp32 is not previous
    torch.testing.assert_close(
        _ffn_output(model), _expected_output(owner.w13_weight_scale)
    )


@pytest.mark.parametrize("family", ["v2", "v4"])
@pytest.mark.parametrize(
    ("role", "gate_on_a", "device"),
    [("attention", True, "npu"), ("ffn", False, "npu"), ("ffn", True, "cuda")],
)
def test_disabled_role_or_backend_does_not_import_npu_helper(
    monkeypatch, runtime, family, role, gate_on_a, device
):
    model, owner = runtime.make_model(family, role=role, gate_on_a=gate_on_a)
    runtime.platform.device_type = device
    # No preparation symbol exists: reaching the lazy NPU import would fail.
    monkeypatch.setitem(sys.modules, _HELPER_MODULE, SimpleNamespace())

    model.process_weights_after_loading()

    with pytest.raises(AttributeError, match="w13_weight_scale_fp32"):
        _ = owner.w13_weight_scale_fp32


@pytest.mark.parametrize("family", ["v2", "v4"])
@pytest.mark.parametrize("kind", ["dense", "none", "w4a8", "expert_list"])
def test_only_stacked_w8a8_owners_receive_a_scale(runtime, family, kind):
    model, owner = runtime.make_model(family)
    layer = model.model.layers[0]
    if kind == "dense":
        layer.is_moe_layer = False
        del layer.mlp
    elif kind == "expert_list":
        owner.dynamic_eplb = True
    else:
        layer.mlp.experts.quant_type = (
            runtime.quant_type.NONE if kind == "none" else runtime.quant_type.W4A8
        )
    del owner.w13_weight_scale

    model.process_weights_after_loading()

    with pytest.raises(AttributeError, match="w13_weight_scale_fp32"):
        _ = owner.w13_weight_scale_fp32


@pytest.mark.parametrize("family", ["v2", "v4"])
def test_present_invalid_cache_is_not_silently_repaired(runtime, family):
    model, owner = runtime.make_model(family)
    owner.w13_weight_scale_fp32 = None

    model.process_weights_after_loading()

    assert owner.w13_weight_scale_fp32 is None


@pytest.mark.parametrize("family", ["v2", "v4"])
def test_missing_canonical_scale_keeps_the_native_attribute_error(runtime, family):
    model, owner = runtime.make_model(family)
    del owner.w13_weight_scale

    with pytest.raises(AttributeError, match="w13_weight_scale"):
        model.process_weights_after_loading()


@pytest.mark.parametrize(
    "family",
    ["AFDDeepseekForCausalLM", "AFDDeepseekV3ForCausalLM", "AFDGlmMoeDsaForCausalLM"],
)
def test_v2_family_models_inherit_the_post_quant_lifetime(runtime, family):
    model, owner = runtime.make_model(family)

    model.process_weights_after_loading()

    torch.testing.assert_close(
        _ffn_output(model), _expected_output(owner.w13_weight_scale)
    )
