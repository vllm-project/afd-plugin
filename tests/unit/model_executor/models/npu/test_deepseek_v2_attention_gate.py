# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU post-quant scale ownership and CAM MLP consumption contracts."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.unit.fakes.ascend_moe import (
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

    def make_model(family, *, dtype=torch.bfloat16):
        owner = _RoutedExperts(dtype)
        layer = _make_layer(owner, quant_type.W8A8)
        config = SimpleNamespace(
            afd=SimpleNamespace(role="ffn", compute_gate_on_attention=True),
            model=_FFNModel(layer, compute_ffn),
        )
        return wrappers[family](vllm_config=config), owner

    return SimpleNamespace(make_model=make_model)


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


@pytest.mark.parametrize(
    "quant_name,custom_activation,routed_scale_applied_in_topk,input_dtype",
    [
        ("NONE", False, False, torch.bfloat16),
        ("NONE", True, False, torch.bfloat16),
        ("W8A8", True, True, torch.bfloat16),
        ("W8A8", False, False, torch.float16),
        ("NONE", True, True, torch.float16),
    ],
)
def test_attention_gate_passes_activation_and_scaling_contract(
    monkeypatch,
    quant_name,
    custom_activation,
    routed_scale_applied_in_topk,
    input_dtype,
):  # long-test: external native activation ABI and routed/shared scaling agree
    captured = []

    def apply_mlp(*, mlp_compute_input, quant_method):
        # This is a deterministic CPU reference for the MoEMlpComputeInput
        # contract, not an implementation or validation of an Ascend kernel.
        del quant_method
        captured.append(mlp_compute_input)
        inputs = mlp_compute_input.hidden_states.to(torch.float32)
        gate = inputs[:, :2]
        up = inputs[:, 2:]
        if mlp_compute_input.swiglu_limit > 0:
            limit = mlp_compute_input.swiglu_limit
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        # This CPU reference mirrors the native SILU clamp path only; the other
        # activation fields are checked as forwarded inputs, not interpreted here.
        routed = torch.nn.functional.silu(gate) * up
        return routed, None

    compute_ffn, quant_type = load_attention_gate_moe_ffn(monkeypatch, apply_mlp)
    requested_quant_type = getattr(quant_type, quant_name)

    owner = torch.nn.Module()
    owner.get_eplb_parameter = lambda name: getattr(owner, name)
    owner.w13_weight = torch.ones((2, 4, 4))
    owner.w2_weight = torch.ones((2, 4, 4))
    owner.w13_weight_scale_fp32 = torch.ones((2, 4))
    owner.w2_weight_scale = torch.ones((2, 4))
    owner.quant_method = SimpleNamespace(
        quant_method=SimpleNamespace(is_per_channel_weight=False)
    )
    owner.dynamic_eplb = False
    owner.activation = "silu"

    activation_values: dict[str, float | None]
    if custom_activation:
        activation_values = {
            "swiglu_limit": 2.0,
            "swiglu_alpha": 1.5,
            "swiglu_beta": 0.25,
            "activation_situ_beta": 0.1,
            "activation_situ_linear_beta": 0.2,
        }
    else:
        activation_values = {
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        }
    moe_config = SimpleNamespace(has_bias=False, **activation_values)
    experts = SimpleNamespace(
        quant_type=requested_quant_type,
        routed_experts=owner,
        shared_experts=SimpleNamespace(
            _layer=lambda inputs: torch.full_like(inputs, 7.0)
        ),
        moe_config=moe_config,
    )
    scaling_factor = 2.5
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=experts, routed_scaling_factor=scaling_factor)
    )
    hidden_states = torch.full((1, 4), 3.0, dtype=input_dtype)
    shared_states = torch.ones((1, 4))
    result = compute_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.tensor([1, 1]),
        dynamic_scales=None,
        expand_x_shared=shared_states,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=0,
        routed_scale_applied_in_topk=routed_scale_applied_in_topk,
    )

    activation = captured[0]
    expected_parameters = (
        (2.0, 1.5, 0.25, 0.1, 0.2) if custom_activation else (0.0, 1.0, 0.0, None, None)
    )
    assert (
        activation.swiglu_limit,
        activation.swiglu_alpha,
        activation.swiglu_beta,
        activation.activation_situ_beta,
        activation.activation_situ_linear_beta,
    ) == expected_parameters

    limit = 2.0 if custom_activation else 0.0
    expected_value = 3.0 if limit == 0 else limit
    expected_value = expected_value / (1.0 + math.exp(-expected_value)) * expected_value
    expected_routed = torch.full((1, 2), expected_value)
    if not routed_scale_applied_in_topk and input_dtype != torch.float16:
        expected_routed *= scaling_factor
    torch.testing.assert_close(result.routed_output, expected_routed)
    expected_shared_value = 7.0
    if not routed_scale_applied_in_topk and input_dtype == torch.float16:
        expected_shared_value /= scaling_factor
    torch.testing.assert_close(
        result.shared_output, torch.full((1, 4), expected_shared_value)
    )


@pytest.fixture
def w4a8_owner(monkeypatch, dynamic_eplb, per_channel, with_bias):
    owner = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        owner.register_parameter(
            name,
            torch.nn.Parameter(
                torch.ones((2, 4, 4), dtype=torch.int32), requires_grad=False
            ),
        )
    for name in ("w13_weight_scale", "w2_weight_scale"):
        owner.register_parameter(
            name, torch.nn.Parameter(torch.ones((2, 4)), requires_grad=False)
        )
    for name in ("w13_scale_bias", "w2_scale_bias"):
        owner.register_parameter(
            name,
            torch.nn.Parameter(torch.ones((2, 4)), requires_grad=False)
            if with_bias
            else None,
        )
    owner.w13_weight_list = [owner.w13_weight]
    owner.w2_weight_list = [owner.w2_weight]
    owner.w13_weight_scale_list = [owner.w13_weight_scale]
    owner.w2_weight_scale_list = [owner.w2_weight_scale]
    owner.w13_scale_bias_list = [owner.w13_scale_bias] if with_bias else None
    owner.w2_scale_bias_list = [owner.w2_scale_bias] if with_bias else None
    owner.quant_method = SimpleNamespace(
        quant_method=SimpleNamespace(is_per_channel_weight=per_channel)
    )
    owner.dynamic_eplb = dynamic_eplb
    owner.activation = "silu"
    return monkeypatch, owner


@pytest.mark.parametrize(
    "dynamic_eplb,per_channel,with_bias,rows",
    [
        (False, False, False, 2),
        (False, True, True, 2),
        (True, True, True, 2),
        (True, False, False, 0),
    ],
)
def test_w4a8_cam_mlp_contract(
    w4a8_owner, dynamic_eplb, per_channel, with_bias, rows
):  # long-test: native W4A8 payload includes weight views, quantization and activation
    monkeypatch, owner = w4a8_owner
    calls: list[SimpleNamespace] = []

    def apply_mlp(*, mlp_compute_input, quant_method):
        assert quant_method is owner.quant_method.quant_method
        calls.append(mlp_compute_input)
        return torch.ones((rows, 4), dtype=torch.bfloat16), None

    compute_ffn, quant_type = load_attention_gate_moe_ffn(monkeypatch, apply_mlp)
    # The native MoE config owns the activation contract.
    owner.swiglu_limit = None
    experts = SimpleNamespace(
        quant_type=quant_type.W4A8,
        routed_experts=owner,
        shared_experts=None,
        moe_config=SimpleNamespace(
            swiglu_limit=10.0,
            swiglu_alpha=1.7,
            swiglu_beta=0.5,
            activation_situ_beta=0.8,
            activation_situ_linear_beta=0.3,
        ),
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=experts, routed_scaling_factor=1.0)
    )
    hidden_states = torch.ones((rows, 4), dtype=torch.int8)
    scales = torch.ones(rows)
    output = compute_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.tensor([rows, rows]),
        dynamic_scales=scales,
        expand_x_shared=None,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=0,
    )
    assert output.routed_output.shape == (rows, 4)
    assert output.routed_output.dtype == torch.bfloat16
    if not rows:
        return
    contract = calls[0]
    assert contract.hidden_states is hidden_states
    assert contract.layer is owner
    assert contract.dynamic_scale is scales
    assert contract.quant.quant_type == quant_type.W4A8
    assert contract.quant.is_per_channel_weight == per_channel
    assert contract.swiglu_limit == 10.0
    assert contract.swiglu_alpha == 1.7
    assert contract.swiglu_beta == 0.5
    assert contract.activation_situ_beta == 0.8
    assert contract.activation_situ_linear_beta == 0.3
    assert contract.weights.w1[0].data_ptr() == owner.w13_weight.data_ptr()
    assert contract.weights.w2[0].data_ptr() == owner.w2_weight.data_ptr()
    assert contract.weights.w1[0].dtype == torch.int32
    assert contract.weights.w1_scale[0] is owner.w13_weight_scale
    assert contract.weights.w2_scale[0] is owner.w2_weight_scale
    assert (contract.weights.w1_scale_bias is not None) == with_bias
    assert (contract.weights.w2_scale_bias is not None) == with_bias
    assert contract.fusion is False


@pytest.fixture
def empty_rank_ffn(monkeypatch):
    def routed_mlp(*, mlp_compute_input, quant_method):
        states = mlp_compute_input.hidden_states
        assert states.shape[0] > 0, "native MLP requires nonempty input"
        return torch.ones_like(states, dtype=torch.bfloat16), None

    def shared_mlp(owner, states, scales, *, swiglu_limit, output_dtype):
        assert states.shape[0] > 0, "shared MLP requires nonempty input"
        return torch.zeros_like(states, dtype=output_dtype)

    compute, quant = load_attention_gate_moe_ffn(
        monkeypatch, routed_mlp, compute_shared=shared_mlp
    )
    owner = _RoutedExperts(torch.bfloat16)
    owner.w13_weight_scale_fp32 = owner.w13_weight_scale.float()
    layer = _make_layer(owner, quant.W8A8)
    layer.mlp.routed_scaling_factor = 2.0
    layer.mlp.experts.shared_experts = SimpleNamespace(_layer=object())
    return compute, layer


@pytest.mark.parametrize(
    "routed_tokens,shared_tokens", [(2, 2), (2, 0), (0, 2), (0, 0)]
)
def test_deepseek_afd_ffn_skips_empty_rank_local_moe_work(
    empty_rank_ffn, routed_tokens, shared_tokens
):
    compute, layer = empty_rank_ffn
    states = torch.zeros((routed_tokens, 2), dtype=torch.int8)
    shared_states = torch.zeros((shared_tokens, 2), dtype=torch.int8)
    output = compute(
        layer,
        hidden_states=states,
        group_list=torch.tensor([routed_tokens, 0]),
        dynamic_scales=torch.ones(routed_tokens),
        expand_x_shared=shared_states,
        dynamic_scales_shared=torch.ones(shared_tokens),
        topk_scales=None,
        group_list_type=1,
    )
    assert output.routed_output.shape == states.shape
    assert output.routed_output.dtype == torch.bfloat16
    if routed_tokens:
        torch.testing.assert_close(
            output.routed_output,
            torch.full((routed_tokens, 2), 2.0, dtype=torch.bfloat16),
        )
    if shared_tokens:
        torch.testing.assert_close(
            output.shared_output, torch.zeros((shared_tokens, 2), dtype=torch.bfloat16)
        )
    else:
        assert output.shared_output is None
