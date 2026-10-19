# SPDX-License-Identifier: Apache-2.0
"""Exercise Window data and resource contracts without importing NPU extensions."""

from __future__ import annotations

import ast
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_functions(relative_path, names, namespace):
    # Compile the actual functions, following test_window_u2_padding.py, so
    # CPU tests can exercise their contracts without an Ascend installation.
    path = Path(__file__).resolve().parents[4] / relative_path
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *functions], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("hidden_size,record_bytes", [(7168, 7680), (33, 512)])
def test_mxfp8_window_records_include_aligned_scale_storage(hidden_size, record_bytes):
    functions = _load_functions(
        "afd_plugin/connectors/npu/window.py",
        {"_align_up", "_window_sizes"},
        {"_COMM_CONTEXT_WINDOW_ALIGNMENT": 2 * 1024 * 1024},
    )
    _, _, a2f, f2a = functions["_window_sizes"](
        attention_size=2,
        micro_batch_num=2,
        micro_batch_size=8,
        selected_expert_num=16,
        hidden_size=hidden_size,
        quant_mode=4,
    )
    assert a2f == record_bytes
    assert f2a == hidden_size * 2


@pytest.mark.parametrize("graph_capturing", [False, True])
def test_operator_core_limits_are_restored_on_failure(graph_capturing):
    limits = {"cube_core_num": 64, "vector_core_num": 128}
    scope = Mock()
    scope.__enter__ = Mock()
    scope.__exit__ = Mock(return_value=False)
    npu = SimpleNamespace(
        get_stream_limit=Mock(return_value=limits),
        set_stream_limit=Mock(),
        npugraph_ex=SimpleNamespace(
            scope=SimpleNamespace(limit_core_num=Mock(return_value=scope))
        ),
    )
    functions = _load_functions(
        "afd_plugin/v1/worker/npu/ffn_model_runner.py",
        {"_limit_window_ffn_operator_cores"},
        {"contextmanager": contextmanager, "torch": SimpleNamespace(npu=npu)},
    )
    with (
        pytest.raises(RuntimeError, match="operator failed"),
        functions["_limit_window_ffn_operator_cores"](
            "stream", graph_capturing=graph_capturing, aic_num=32, aiv_num=16
        ),
    ):
        raise RuntimeError("operator failed")
    if graph_capturing:
        npu.set_stream_limit.assert_not_called()
        scope.__exit__.assert_called_once()
    else:
        assert npu.set_stream_limit.call_args_list == [
            (("stream",), {"cube_num": 32, "vector_num": 16}),
            (("stream",), {"cube_num": 64, "vector_num": 128}),
        ]


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_explicit_profiler_requests_are_idempotent_and_restartable(role):
    profiler = SimpleNamespace()
    create = Mock(return_value=profiler)
    stop = Mock()
    functions = _load_functions(
        f"afd_plugin/v1/worker/npu/{role}_model_runner.py",
        {"start_afd_profiler", "stop_afd_profiler"},
        {"create_afd_npu_profiler": create, "stop_afd_npu_profiler": stop},
    )
    runner_type = type(
        "ProfilerRunner",
        (),
        {name: functions[name] for name in ("start_afd_profiler", "stop_afd_profiler")},
    )
    runner = runner_type()
    runner.prof = None
    runner._profiler_rank = 2
    runner.start_afd_profiler()
    runner.start_afd_profiler()
    create.assert_called_once_with(role, role_rank=2)
    runner.stop_afd_profiler()
    stop.assert_called_once_with(profiler)
    assert runner.prof is None
    runner.start_afd_profiler()
    assert create.call_count == 2


@pytest.mark.parametrize("multistream", [False, True])
def test_window_graph_retains_every_transaction_and_replays_without_rebatching(
    multistream,
):
    graph = SimpleNamespace(replay=Mock())
    current_stream = SimpleNamespace(wait_stream=Mock())
    capture_flags = Mock()
    functions = _load_functions(
        "afd_plugin/v1/worker/npu/ffn_model_runner.py",
        {"_run_window_ffn_transaction", "_run_window_ffn_graph"},
        {
            "torch": SimpleNamespace(
                npu=SimpleNamespace(
                    NPUGraph=lambda: graph,
                    current_stream=lambda: current_stream,
                    graph=lambda *_args, **_kwargs: nullcontext(),
                )
            ),
            "graph_capture": lambda **_kwargs: nullcontext(),
            "set_cudagraph_capturing_enabled": capture_flags,
            "logger": Mock(),
            "_WINDOW_FFN_PIPELINE_DEPTH": 3,
        },
    )
    runner_type = type(
        "GraphRunner",
        (),
        {
            name: functions[name]
            for name in ("_run_window_ffn_transaction", "_run_window_ffn_graph")
        },
    )
    runner = runner_type()
    payload = SimpleNamespace(context="context")
    runner.connector = SimpleNamespace(
        recv_attn_output=Mock(return_value=payload), send_ffn_output=Mock()
    )
    runner._compute_window_async_batch = Mock(return_value="output")
    runner._run_window_ffn_multistream_transaction = Mock(
        side_effect=lambda **_kwargs: (payload, torch.empty(1))
    )
    runner.window_ffn_multistream_enabled = multistream
    runner.ffn_send_stream = "send"
    runner._window_ffn_acl_graph = None
    runner._window_ffn_graph_warmed_up = False
    runner.device = "npu"
    runner.graph_pool = None
    runner.max_num_tokens = 8
    runner._run_window_ffn_graph()  # Warmup.
    assert runner._window_ffn_acl_graph is None
    runner._run_window_ffn_graph()  # Capture.
    retained = runner._window_ffn_acl_graph["retained"]
    assert len(retained) == (3 if multistream else 1)
    if multistream:
        assert len({id(output) for _, output in retained}) == 3
    runner._run_window_ffn_graph()  # Replay uses the retained buffers.
    graph.replay.assert_called_once_with()
    assert runner._run_window_ffn_multistream_transaction.call_count == (
        6 if multistream else 0
    )
    assert runner.connector.recv_attn_output.call_count == (0 if multistream else 2)
    assert capture_flags.call_args_list == [((True,),), ((False,),)]


@pytest.mark.parametrize("routed", [False, True])
def test_mxfp8_ffn_consumes_received_scales_and_type2_groups(monkeypatch, routed):
    operator = SimpleNamespace(
        npu_dynamic_quant=Mock(side_effect=AssertionError("unexpected quantization")),
        maybe_normalize_mxfp_scale_layout=Mock(side_effect=lambda scale: scale),
        npu_grouped_matmul_gmm2=Mock(
            side_effect=[torch.zeros((2, 4)), torch.ones((2, 32))]
        ),
    )
    for name, attributes in {
        "torch_npu": {"float4_e2m1fn_x2": "fp4"},
        "vllm_ascend.device.device_op": {"DeviceOperator": operator},
        "vllm_ascend.device.mxfp_compat": {"FLOAT8_E8M0FNU_DTYPE": "e8m0"},
        "vllm_ascend.quantization.quant_type": {
            "QuantType": SimpleNamespace(W4A8MXFP="w4a8", MXFP8="mxfp8")
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    activation = Mock(return_value=(torch.zeros((2, 2)), torch.ones((2, 1)), None))
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_swiglu_group_quant", activation, raising=False
    )
    functions = _load_functions(
        "afd_plugin/model_executor/models/npu/deepseek_v4_window_ffn.py",
        {"compute_window_global_mxfp_ffn"},
        {"torch": torch},
    )
    groups = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    scales = torch.ones((2, 1), dtype=torch.uint8)
    weights = SimpleNamespace(
        w1=torch.empty((2, 32, 4)),
        w1_scale=None,
        w2=None,
        w2_scale=None,
        is_routed=routed,
        swiglu_limit=7.0,
        routed_scaling_factor=2.0,
    )
    output = functions["compute_window_global_mxfp_ffn"](
        hidden_states=torch.zeros((2, 32), dtype=torch.float8_e4m3fn),
        compact_group_list=groups,
        dynamic_scale=scales,
        weights=weights,
        output_dtype=torch.bfloat16,
    )
    operator.npu_dynamic_quant.assert_not_called()
    assert operator.npu_grouped_matmul_gmm2.call_count == 2
    for call in operator.npu_grouped_matmul_gmm2.call_args_list:
        assert call.kwargs["group_list"] is groups
        assert call.kwargs["group_list_type"] == 2
    first = operator.npu_grouped_matmul_gmm2.call_args_list[0].kwargs
    assert first["per_token_scale"] is scales
    assert activation.call_args.kwargs["clamp_value"] == 7.0
    assert torch.equal(output, torch.full((2, 32), 2.0 if routed else 1.0))
