# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contracts for device-controlled layered quantized execution."""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.model_executor.npu.async_cam_layered import (  # noqa: E402
    AsyncCAMLayeredExecutor,
    LayeredFFNWeights,
)


def weights(layer_idx=0, per_channel=True, quantization="W4A8"):
    experts, hidden, intermediate = 2, 32, 16
    if quantization == "W8A8":
        return LayeredFFNWeights(
            layer_idx=layer_idx,
            w13=torch.empty(experts, hidden, 2 * intermediate, dtype=torch.int8),
            w2=torch.empty(experts, intermediate, hidden, dtype=torch.int8),
            w13_scale=torch.ones(experts, 2 * intermediate),
            w2_scale=torch.ones(experts, hidden, dtype=torch.bfloat16),
            w13_bias=None,
            w2_bias=None,
            per_channel=True,
            swiglu_limit=0.0,
            routed_scaling_factor=2.0,
            quantization="W8A8",
        )
    return LayeredFFNWeights(
        layer_idx=layer_idx,
        w13=torch.empty(experts, hidden, intermediate // 4, dtype=torch.int32),
        w2=torch.empty(experts, intermediate, hidden // 8, dtype=torch.int32),
        w13_scale=torch.empty(
            (experts, 2 * intermediate)
            if per_channel
            else (experts, 2, 2 * intermediate),
            dtype=torch.int64,
        ),
        w2_scale=torch.empty(
            experts, 1 if per_channel else 2, hidden, dtype=torch.int64
        ),
        w13_bias=torch.ones(experts, 2 * intermediate),
        w2_bias=torch.ones(experts, hidden),
        per_channel=per_channel,
        swiglu_limit=0.0,
        routed_scaling_factor=2.0,
    )


@pytest.fixture
def operators(monkeypatch):
    calls = []

    def w13(*args, **kwargs):
        calls.append(("w13", args, kwargs))
        return torch.ones(args[0].shape[0], 16, dtype=torch.int8), torch.ones(
            args[0].shape[0]
        )

    def w2(*args, **kwargs):
        calls.append(("w2", args, kwargs))
        return [torch.ones(args[0][0].shape[0], 32, dtype=kwargs["output_dtype"])]

    monkeypatch.setattr(
        torch.ops.afd_ascend, "gmm_swiglu_quant_v2_layered", w13, raising=False
    )
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "grouped_matmul_layered",
        w2,
        raising=False,
    )
    return calls


@pytest.mark.parametrize(
    "quantization,per_channel", [("W4A8", True), ("W4A8", False), ("W8A8", True)]
)
@pytest.mark.parametrize("swiglu_limit", [0.0, 10.0])
@pytest.mark.parametrize("layer_ids", [(0, 1), (2, 5)])
@pytest.mark.parametrize("via_graph", [False, True])
def test_device_layer_selection_capacity_and_scaling(
    monkeypatch,
    operators,
    quantization,
    per_channel,
    swiglu_limit,
    layer_ids,
    via_graph,
):
    layers = [
        replace(weights(idx, per_channel, quantization), swiglu_limit=swiglu_limit)
        for idx in reversed(layer_ids)
    ]
    executor = AsyncCAMLayeredExecutor(layers)
    hidden = torch.ones(8, 32, dtype=torch.int8)
    scales = torch.ones(8)
    counts = torch.tensor([0, 0], dtype=torch.int64)
    metadata = torch.tensor([100, 11, layer_ids[-1], 0, 77], dtype=torch.int64)
    original = metadata.clone()

    def reject_host_read(*args, **kwargs):
        raise AssertionError("host metadata read in the work-item path")

    with monkeypatch.context() as hot:
        for method in ("cpu", "item", "tolist", "__int__", "__bool__"):
            hot.setattr(torch.Tensor, method, reject_host_read)
        if via_graph:
            output = executor(hidden, scales, counts, metadata, retained_tensors=[])
        else:
            output = executor(hidden, scales, counts, metadata)
    torch.testing.assert_close(metadata, original)
    assert output.shape == (8, 32)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.full_like(output, 2))
    first, second = operators
    assert first[1][0] is hidden
    assert first[1][4] is scales
    assert first[1][5] is second[1][5] is counts
    assert first[1][6] is second[1][4]
    assert first[1][6].tolist() == [1]
    assert first[2]["dequant_mode"] == (0 if per_channel else 1)
    assert first[2]["swiglu_limit"] == (swiglu_limit if quantization == "W4A8" else 0.0)
    assert first[2]["group_list_type"] == second[2]["group_list_type"] == 1
    assert first[1][1][0] is layers[-1].w13
    if quantization == "W8A8":
        assert first[1][3] == []
        biases = second[1][2]
        assert all(
            t.dtype == torch.int32 and torch.count_nonzero(t) == 0 for t in biases
        )
        assert biases[0] is biases[1]
    if layer_ids == (0, 1):
        assert first[1][6].storage_offset() == 0
        assert first[1][6].untyped_storage().nbytes() == first[1][6].element_size()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"w13_bias": torch.empty(2, 32, dtype=torch.int64)}, "compensation"),
        ({"w13_scale": torch.empty(2, 32)}, "encoded INT64"),
        ({"w2_scale": torch.empty(2, 32, dtype=torch.int64)}, "w2_scale"),
    ],
)
def test_reject_incompatible_parameters_before_receive(operators, change, message):
    with pytest.raises(ValueError, match=message):
        AsyncCAMLayeredExecutor([replace(weights(), **change)])
    assert operators == []


@pytest.mark.parametrize(
    "change, message",
    [
        ({"w2": torch.empty(2, 16, 32, dtype=torch.int32)}, "INT8"),
        ({"w13_scale": torch.empty(2, 32, dtype=torch.bfloat16)}, "FP32"),
        ({"w2_scale": torch.empty(2, 1, 32)}, "w2_scale"),
        (
            {
                "w2_scale": torch.empty(2, 32, dtype=torch.float32),
                "output_dtype": torch.float16,
            },
            "BF16 W2",
        ),
        ({"w2_bias": torch.empty(2, 32)}, "compensation"),
        ({"per_channel": False}, "per-channel"),
    ],
)
def test_w8a8_rejects_incompatible_layout_before_receive(operators, change, message):
    with pytest.raises(ValueError, match=message):
        AsyncCAMLayeredExecutor([replace(weights(quantization="W8A8"), **change)])
    assert operators == []


def test_graph_signature_distinguishes_quantization(operators):
    w4 = AsyncCAMLayeredExecutor([weights()])
    w8 = AsyncCAMLayeredExecutor([weights(quantization="W8A8")])
    assert w4.graph_signature(32) != w8.graph_signature(32)
    with pytest.raises(ValueError, match="homogeneous"):
        AsyncCAMLayeredExecutor([weights(0), weights(1, quantization="W8A8")])


def test_repeated_interleaved_layer_ids(operators):
    executor = AsyncCAMLayeredExecutor([weights(2), weights(5)])
    for idx in (5, 2, 5, 5):
        executor(
            torch.empty(8, 32, dtype=torch.int8),
            torch.ones(8),
            torch.tensor([2, 1]),
            torch.tensor([99, 0, idx, 3]),
        )
    assert [call[1][6].item() for call in operators[::2]] == [1, 0, 1, 1]


def runner_method(name, namespace):
    tree = ast.parse(
        Path("afd_plugin/v1/worker/npu/async_ffn_model_runner.py").read_text()
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncNPUFFNModelRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<runner-contract>", "exec"),
        namespace,
    )
    return namespace[name]


def test_runner_preserves_raw_context_and_chunk_capacity():
    payloads = [
        SimpleNamespace(
            hidden_states=torch.empty(8, 32),
            context=SimpleNamespace(
                states=SimpleNamespace(
                    dynamic_scales=torch.ones(8),
                    group_list=torch.tensor([0, 0]),
                    token_nums_rankid_layeridx=torch.tensor([99, 1, 5, 0]),
                )
            ),
        )
        for _ in range(2)
    ]
    received = iter(payloads)
    sent = []
    computed = []
    connector = SimpleNamespace(
        recv_attn_output=lambda: next(received),
        send_ffn_output=lambda result, ctx: sent.append((result, ctx)),
    )

    def compute(*args):
        computed.append(args)
        return torch.empty(8, 32, dtype=torch.bfloat16)

    runner = SimpleNamespace(
        model=object(),
        connector=connector,
        _two_stream=None,
        _layered_executor=compute,
        layer_ids=(2, 5),
    )
    method = runner_method(
        "_ffn_forward_connector_driven",
        {
            "cast": cast,
            "CAMAsyncAFDConnector": SimpleNamespace,
            "AFDAsyncTransferState": SimpleNamespace,
        },
    )
    method(runner)
    assert [ctx for _, ctx in sent] == [p.context for p in payloads]
    for args, payload in zip(computed, payloads, strict=True):
        assert args[0] is payload.hidden_states
        assert args[3] is payload.context.states.token_nums_rankid_layeridx


def test_runner_switch_off_does_not_touch_model_or_ops():
    runner = SimpleNamespace(model=object(), _layered_gmm_requested=False)
    method = runner_method(
        "_initialize_layered_executor",
        {"logger": SimpleNamespace(info=lambda *args: None)},
    )
    method(runner)


@pytest.mark.parametrize(
    ("role", "connector", "is_dsv4", "message"),
    [
        ("attention", "CAMAsyncAFDConnector", True, "FFN role"),
        ("ffn", "CAMP2pAFDConnector", True, "async CAM"),
        ("ffn", "CAMAsyncAFDConnector", False, "only DeepSeek V4"),
    ],
)
def test_layered_switch_rejects_other_roles_connectors_and_models(
    role, connector, is_dsv4, message
):
    source = Path("afd_plugin/compat/npu/feature_validation.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fail_if_unsupported_npu_afd_features"
    )
    namespace = {
        "AFD_ASYNC_CONNECTOR": "CAMAsyncAFDConnector",
        "async_cam_layered_gmm_enabled": lambda: True,
        "_is_dsv4_target": lambda config: is_dsv4,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<feature-validation>", "exec"),
        namespace,
    )
    afd_config = SimpleNamespace(role=role, connector=connector)
    with pytest.raises(RuntimeError, match=message):
        cast(Callable[..., None], namespace["fail_if_unsupported_npu_afd_features"])(
            SimpleNamespace(), afd_config=afd_config
        )


@pytest.mark.parametrize(
    "eligible,draft_ids,layer_ids,error,enabled",
    [
        (False, None, (0,), None, False),
        (True, None, (0,), None, True),
        (True, None, (2,), "remote MoE layer IDs", False),
        (True, (43, 44, 45), (0, 43, 44, 45), None, True),
        (True, (43,), (0, 43, 44, 45), "remote MoE layer IDs", False),
        (True, (), (0, 43, 44, 45), None, False),
    ],
)
@pytest.mark.parametrize("quantization", ["W4A8", "W8A8"])
def test_runner_requires_complete_layered_target_and_draft_coverage(
    operators, eligible, draft_ids, layer_ids, error, enabled, quantization
):
    class Connector:
        dynamic_quant = 1
        layered_gmm_enabled = False

    layer = weights(quantization=quantization)
    runner = SimpleNamespace(
        _layered_gmm_requested=True,
        connector=Connector(),
        afd_config=SimpleNamespace(compute_gate_on_attention=True),
        model=SimpleNamespace(
            get_async_cam_layered_weights=lambda: (
                ([layer], "") if eligible else ([], "mixed quantization")
            )
        ),
        draft_model=SimpleNamespace(
            get_async_cam_layered_weights=lambda: (
                [weights(idx, quantization=quantization) for idx in draft_ids],
                "draft quantization is unsupported" if not draft_ids else "",
            )
        )
        if draft_ids is not None
        else None,
        layer_ids=layer_ids,
        _layered_executor=None,
    )

    method = runner_method(
        "_initialize_layered_executor",
        {
            "CAMAsyncAFDConnector": Connector,
            "AscendDeviceType": SimpleNamespace(A3="A3"),
            "get_ascend_device_type": lambda: "A3",
            "ensure_cam_async_ops_available": lambda: None,
            "AsyncCAMLayeredExecutor": AsyncCAMLayeredExecutor,
            "logger": SimpleNamespace(info=lambda *args: None),
        },
    )
    if error:
        with pytest.raises(ValueError, match=error):
            method(runner)
    else:
        method(runner)
    assert runner.connector.layered_gmm_enabled == enabled


def test_layered_debug_logging_does_not_read_metadata(monkeypatch):
    source = Path("afd_plugin/connectors/npu/async_cam.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_log_cam_op_values"
    )
    messages = []
    namespace = {
        "os": os,
        "Tensor": torch.Tensor,
        "_CAM_OP_IO_LOG_ENV": "AFD_CAM_OP_IO_LOG",
        "_CAM_LOG_SKIPPED_ARGS": frozenset(),
        "logger": SimpleNamespace(warning=lambda *args: messages.append(args)),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<cam-log>", "exec"),
        namespace,
    )
    monkeypatch.setenv("AFD_CAM_OP_IO_LOG", "1")

    def reject(*args, **kwargs):
        raise AssertionError("metadata D2H")

    monkeypatch.setattr(torch.Tensor, "cpu", reject)
    namespace["_log_cam_op_values"](
        "combine",
        "inputs",
        metadata_values=False,
        token_nums_rankid_layeridx=torch.tensor([10, 1, 3]),
    )
    assert "first5" not in messages[0][-1]
    assert "shape=(3,)" in messages[0][-1]


def model_weights_method(monkeypatch):
    source = Path("afd_plugin/model_executor/models/npu/deepseek_v4.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_extract_async_cam_layered_weights"
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.activation",
        SimpleNamespace(MoEActivation=SimpleNamespace(SILU="silu")),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        SimpleNamespace(QuantType=SimpleNamespace(W4A8="w4a8", W8A8="w8a8")),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {"cast": cast, "torch": torch}
    exec(
        compile(ast.fix_missing_locations(module), "<model-weights>", "exec"), namespace
    )
    return namespace["_extract_async_cam_layered_weights"]


def model_layer(idx, quantization="W4A8"):
    spec = weights(idx, quantization=quantization)
    parameters = dict(
        w13_weight=spec.w13,
        w2_weight=spec.w2,
        w13_weight_scale=spec.w13_scale,
        w2_weight_scale=spec.w2_scale,
        w13_scale_bias=spec.w13_bias,
        w2_scale_bias=spec.w2_bias,
    )
    if quantization == "W8A8":
        parameters.pop("w13_scale_bias")
        parameters.pop("w2_scale_bias")
    owner = SimpleNamespace(
        **parameters,
        _parameters=parameters,
        dynamic_eplb=False,
        quant_method=SimpleNamespace(
            quant_method=SimpleNamespace(is_per_channel_weight=True)
        ),
        w13_weight_scale_fp32=spec.w13_scale,
    )
    experts = SimpleNamespace(
        quant_type=quantization.lower(),
        activation="silu",
        _shared_experts=None,
        routed_experts=owner,
    )
    return SimpleNamespace(
        layer_idx=idx,
        is_moe_layer=True,
        mlp=SimpleNamespace(
            experts=experts, swiglu_limit=0.0, routed_scaling_factor=2.5
        ),
    )


@pytest.mark.parametrize("quantization", ["W4A8", "W8A8"])
def test_model_extraction_preserves_weights_and_topk_scaling(monkeypatch, quantization):
    method = model_weights_method(monkeypatch)
    layer = model_layer(3, quantization)
    specs, reason = method([layer])
    assert reason == ""
    assert specs[0].layer_idx == 3
    assert specs[0].w13 is layer.mlp.experts.routed_experts.w13_weight
    assert specs[0].routed_scaling_factor == 1.0
    assert specs[0].quantization == quantization
    assert specs[0].output_dtype == torch.bfloat16
    if quantization == "W8A8":
        owner = layer.mlp.experts.routed_experts
        assert specs[0].w13_scale is owner.w13_weight_scale_fp32
        assert specs[0].w2_scale is owner.w2_weight_scale
        assert specs[0].w13_bias is specs[0].w2_bias is None


@pytest.mark.parametrize("mixed", [False, True])
def test_model_extraction_skips_other_quantization(monkeypatch, mixed):
    method = model_weights_method(monkeypatch)
    layers = [model_layer(2), model_layer(5)]
    layers[0].mlp.experts.quant_type = "w8a8" if mixed else "bf16"
    if not mixed:
        layers[1].mlp.experts.quant_type = "bf16"
    specs, reason = method(layers)
    assert specs == []
    assert "homogeneous W4A8 or W8A8" in reason


def test_model_extraction_rejects_missing_compensation(monkeypatch):
    method = model_weights_method(monkeypatch)
    layer = model_layer(5)
    del layer.mlp.experts.routed_experts._parameters["w13_scale_bias"]
    with pytest.raises(ValueError, match="layer 5: missing loaded w13_scale_bias"):
        method([layer])


def test_model_extraction_skips_heterogeneous_semantics(monkeypatch):
    method = model_weights_method(monkeypatch)
    layers = [model_layer(2), model_layer(5)]
    owner = layers[1].mlp.experts.routed_experts
    owner.quant_method.quant_method.is_per_channel_weight = False
    specs, reason = method(layers)
    assert specs == []
    assert "heterogeneous" in reason
