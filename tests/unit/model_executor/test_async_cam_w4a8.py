# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contracts for device-controlled layered W4A8 execution."""

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

from afd_plugin.model_executor.npu.async_cam_w4a8 import (  # noqa: E402
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)


def weights(layer_idx=0, per_channel=True):
    experts, hidden, intermediate = 2, 32, 16
    return W4A8LayerWeights(
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


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("layer_ids", [(0, 1), (2, 5)])
def test_device_layer_selection_capacity_and_scaling(
    monkeypatch, operators, per_channel, layer_ids
):
    layers = [weights(idx, per_channel) for idx in reversed(layer_ids)]
    executor = AsyncCAMW4A8Executor(layers)
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
    assert first[2]["group_list_type"] == second[2]["group_list_type"] == 1
    assert first[1][1][0] is layers[-1].w13
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
        AsyncCAMW4A8Executor([replace(weights(), **change)])
    assert operators == []


def test_nonzero_swiglu_limit_starts_without_passing_a_clamp(operators):
    executor = AsyncCAMW4A8Executor([replace(weights(), swiglu_limit=10.0)])
    executor(
        torch.ones(1, 32, dtype=torch.int8),
        torch.ones(1),
        torch.ones(2, dtype=torch.int64),
        torch.tensor([2, 0, 0, 1]),
    )
    assert len(operators) == 2
    assert "clamp_value" not in operators[0][2]


def test_repeated_interleaved_layer_ids(operators):
    executor = AsyncCAMW4A8Executor([weights(2), weights(5)])
    for idx in (5, 2, 5, 5):
        executor(
            torch.empty(8, 32, dtype=torch.int8),
            torch.ones(8),
            torch.tensor([2, 1]),
            torch.tensor([99, 0, idx, 3]),
        )
    assert [call[1][6].item() for call in operators[::2]] == [1, 0, 1, 1]


def runner_method(name, namespace):
    tree = ast.parse(Path("afd_plugin/v1/worker/npu/ffn_model_runner.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDNPUFFNModelRunner"
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

    runner = SimpleNamespace(connector=connector, _layered_executor=compute)
    method = runner_method(
        "_ffn_forward_connector_driven",
        {
            "cast": cast,
            "CAMAsyncAFDConnector": SimpleNamespace,
            "AFDAsyncTransferState": SimpleNamespace,
            "_ffn_layer_indices": lambda self: [2, 5],
        },
    )
    method(runner)
    assert [ctx for _, ctx in sent] == [p.context for p in payloads]
    for args, payload in zip(computed, payloads, strict=True):
        assert args[0] is payload.hidden_states
        assert args[3] is payload.context.states.token_nums_rankid_layeridx


def test_runner_switch_off_does_not_touch_model_or_ops():
    runner = SimpleNamespace(_layered_gmm_requested=False)
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


@pytest.mark.parametrize("eligible", [False, True])
def test_runner_initializes_ops_before_executor(eligible):
    events: list[str] = []

    class Connector:
        dynamic_quant = 1

    layer = weights()
    runner = SimpleNamespace(
        _layered_gmm_requested=True,
        connector=Connector(),
        afd_config=SimpleNamespace(compute_gate_on_attention=True),
        model=SimpleNamespace(
            get_async_cam_w4a8_layers=lambda: (
                ([layer], "") if eligible else ([], "mixed quantization")
            )
        ),
        use_aclgraph=False,
        vllm_config=SimpleNamespace(use_v2_model_runner=False),
        _layered_executor=None,
    )

    def build(layers):
        assert events == ["ops"]
        events.append("executor")
        return SimpleNamespace(layer_id_to_slot=None)

    method = runner_method(
        "_initialize_layered_executor",
        {
            "CAMAsyncAFDConnector": Connector,
            "AscendDeviceType": SimpleNamespace(A3="A3"),
            "get_ascend_device_type": lambda: "A3",
            "_ffn_layer_indices": lambda runner: [0],
            "ensure_cam_async_ops_available": lambda: events.append("ops"),
            "AsyncCAMW4A8Executor": build,
            "logger": SimpleNamespace(info=lambda *args: None),
        },
    )
    method(runner)
    assert events == (["ops", "executor"] if eligible else [])
    assert (runner._layered_executor is not None) == eligible


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
        and node.name == "_extract_async_cam_w4a8_layers"
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.activation",
        SimpleNamespace(MoEActivation=SimpleNamespace(SILU="silu")),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        SimpleNamespace(QuantType=SimpleNamespace(W4A8="w4a8")),
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
    namespace: dict[str, object] = {}
    exec(
        compile(ast.fix_missing_locations(module), "<model-weights>", "exec"), namespace
    )
    return namespace["_extract_async_cam_w4a8_layers"]


def model_layer(idx):
    spec = weights(idx)
    parameters = dict(
        w13_weight=spec.w13,
        w2_weight=spec.w2,
        w13_weight_scale=spec.w13_scale,
        w2_weight_scale=spec.w2_scale,
        w13_scale_bias=spec.w13_bias,
        w2_scale_bias=spec.w2_bias,
    )
    owner = SimpleNamespace(
        **parameters,
        _parameters=parameters,
        quant_method=SimpleNamespace(
            quant_method=SimpleNamespace(is_per_channel_weight=True)
        ),
    )
    experts = SimpleNamespace(
        quant_type="w4a8",
        dynamic_eplb=False,
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


def test_model_extraction_preserves_weights_and_topk_scaling(monkeypatch):
    method = model_weights_method(monkeypatch)
    layer = model_layer(3)
    specs, reason = method([layer])
    assert reason == ""
    assert specs[0].layer_idx == 3
    assert specs[0].w13 is layer.mlp.experts.routed_experts.w13_weight
    assert specs[0].routed_scaling_factor == 1.0


@pytest.mark.parametrize("mixed", [False, True])
def test_model_extraction_skips_other_quantization(monkeypatch, mixed):
    method = model_weights_method(monkeypatch)
    layers = [model_layer(2), model_layer(5)]
    layers[0].mlp.experts.quant_type = "w8a8"
    if not mixed:
        layers[1].mlp.experts.quant_type = "w8a8"
    specs, reason = method(layers)
    assert specs == []
    assert "non-W4A8 or mixed" in reason


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
