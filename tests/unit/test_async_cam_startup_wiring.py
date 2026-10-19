# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real startup call sites with host stubs; not device execution coverage."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from afd_plugin.async_cam_startup import AsyncCAMGraphStartup
from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_npu_afd_async_features,
)
from afd_plugin.config import parse_afd_config


@pytest.fixture
def modules(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[2] / path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    stub("torch", inference_mode=lambda: lambda function: function)
    for name in (
        "torch.distributed",
        "torch.distributed.distributed_c10d",
        "torch.distributed.rendezvous",
        "vllm.logger",
        "vllm.distributed",
        "vllm.utils.torch_utils",
        "vllm_ascend.utils",
        "afd_plugin.compat.npu.ops",
        "afd_plugin.compat.npu.forward_context",
        "afd_plugin.compat.npu.profiler",
        "afd_plugin.model_executor.npu.async_cam_w4a8",
        "afd_plugin.v1.worker.ffn_metadata",
    ):
        stub(name)
    stub(
        "afd_plugin.connectors.base", AFDConnectorBase=object, ConnectorExtraInfo=object
    )
    stub("afd_plugin.connectors.metadata", AFDTransferState=object)
    group = load(
        "host_afd_process_group", "afd_plugin/distributed/afd_process_group.py"
    )
    stub("afd_plugin.distributed", init_afd_process_group=group.init_afd_process_group)
    connector = load(
        "afd_plugin.connectors.npu.async_cam", "afd_plugin/connectors/npu/async_cam.py"
    )
    runtime = load(
        "host_async_ffn_runner", "afd_plugin/v1/worker/npu/async_ffn_model_runner.py"
    )
    return group, connector, runtime


@pytest.mark.parametrize(
    "invalid,dbo_flag",
    [
        (False, None),
        (False, "enable_dbo"),
        (False, "use_ubatching"),
        (False, "async_moe_ubatching"),
        (True, None),
    ],
)
def test_static_validation_precedes_hccl_creation(
    modules, monkeypatch, invalid, dbo_flag
):
    group, connector_module, _ = modules
    order = []
    store = MagicMock()
    monkeypatch.setattr(group, "rendezvous", lambda *a, **k: iter([(store, 0, 2)]))
    monkeypatch.setattr(group, "PrefixStore", lambda *a: store)
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "test")
    pg = MagicMock()
    pg._get_backend.return_value = backend

    def create(*args, **kwargs):
        order.append("hccl")
        return pg, None

    monkeypatch.setattr(group, "_new_process_group_helper", create)
    connector = connector_module.CAMAsyncAFDConnector.__new__(
        connector_module.CAMAsyncAFDConnector
    )
    connector.__dict__.update(
        _initialized=False,
        afd_config=SimpleNamespace(host="localhost", port=1),
        topology=SimpleNamespace(world_size=2),
        world_rank=0,
        attn_size=1,
        ffn_size=1,
        hidden_size=32,
        topk=1,
        num_routed_experts=1,
        activation_dtype="bf16",
        dynamic_quant=1,
        tp_size=1,
        max_num_batched_tokens=8,
        hccl_buffer_size_mb=256,
        local_rank=0,
        extra_info=SimpleNamespace(async_moe_ubatching=False),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=not invalid,
            parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
        ),
        graph_startup=AsyncCAMGraphStartup(0, 1, 2),
    )
    if dbo_flag == "async_moe_ubatching":
        connector.extra_info.async_moe_ubatching = True
    elif dbo_flag is not None:
        setattr(connector.vllm_config.parallel_config, dbo_flag, True)

    def negotiate(store, signature):
        order.append("negotiate")
        connector.graph_startup.mode = "full"

    monkeypatch.setattr(connector.graph_startup, "initialize", negotiate)
    monkeypatch.setattr(
        connector.graph_startup, "phase", lambda stage: order.append(stage)
    )
    monkeypatch.setattr(
        connector.graph_startup, "fail", lambda exc: order.append("failure")
    )
    monkeypatch.setattr(
        connector,
        "_run_attention_graph_startup",
        lambda: order.append("attention-warmup"),
    )
    if invalid:
        with pytest.raises(ValueError, match="MRV2"):
            connector.init_afd_connector()
        assert order == ["negotiate", "failure"]
    else:
        connector.init_afd_connector()
        assert order == ["negotiate", "validated", "hccl", "attention-warmup"]


@pytest.mark.parametrize(
    "role,native_flag,num_ubatches,error",
    [
        ("ffn", "enable_dbo", 2, "requires the Attention role"),
        ("ffn", "use_ubatching", 2, "requires the Attention role"),
        ("attention", "enable_dbo", 3, "requires exactly two ubatches"),
    ],
)
def test_graph_dbo_keeps_existing_role_and_count_validation(
    modules, role, native_flag, num_ubatches, error
):
    _, connector_module, _ = modules
    config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": role,
                "connector": "CAMAsyncAFDConnector",
                "async_dp": True,
            }
        },
        model_config=SimpleNamespace(enforce_eager=role == "attention"),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL")),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            use_ubatching=False,
            num_ubatches=num_ubatches,
            tensor_parallel_size=1,
        ),
    )
    setattr(config.parallel_config, native_flag, True)
    with pytest.raises(RuntimeError, match=error):
        _fail_if_unsupported_npu_afd_async_features(
            config, parse_afd_config(config), connector_module.AFDAsyncExtraInfo()
        )


def test_ffn_announces_service_only_on_receiver_execution(modules):
    _, _, runtime = modules
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    calls: list[str | tuple[str, str | int]] = []
    runner.__dict__.update(
        prof=None,
        _graph=object(),
        _service_announced=False,
        layer_ids=(2, 5),
        connector=SimpleNamespace(
            graph_startup=SimpleNamespace(service_ready=lambda: calls.append("ready"))
        ),
        _submit_graph=lambda count: calls.append(("submit", count)),
        _complete_graph=lambda: True,
    )
    assert calls == []
    assert runner.execute_connector_driven_step()
    assert runner.execute_connector_driven_step()
    assert calls == ["ready", ("submit", 2), ("submit", 2)]


@pytest.mark.parametrize("capture_fails", [False, True])
def test_ffn_warmup_capture_order_and_failure_propagation(modules, capture_fails):
    _, _, runtime = modules
    calls: list[str | tuple[str, str | int]] = []
    startup = AsyncCAMGraphStartup(2, 2, 4)
    startup.phase = lambda stage: calls.append(stage)
    startup.fail = lambda exc: calls.append(("failure", str(exc)))

    def initialize_connector():
        calls.append("connector")
        assert startup.plan.signature == "weights"
        assert startup.plan.layer_ids == (2, 5)
        startup.mode = "full"

    def capture():
        calls.append("capture")
        if capture_fails:
            raise RuntimeError("capture failed")

    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.__dict__.update(
        _failure=None,
        requested_mode="full",
        device="npu:0",
        model=SimpleNamespace(get_experts_layer_indices=lambda: (5, 2)),
        _initialize_layered_executor=lambda: None,
        _layered_executor=SimpleNamespace(graph_signature=lambda capacity: "weights"),
        connector=SimpleNamespace(
            is_initialized=False,
            graph_startup=startup,
            init_afd_connector=initialize_connector,
            attn_size=2,
            tp_size=1,
            graph_startup_chunks=lambda: 2,
        ),
        _ffn_forward_connector_driven=lambda **kw: calls.append("transaction"),
        _wait_graph_device=lambda event: calls.append("device-done"),
        _initialize_graph=lambda stream: calls.append("graph-resources"),
        _capture_graph=capture,
    )
    if capture_fails:
        with pytest.raises(RuntimeError, match="capture failed"):
            runner.initialize_afd_connector()
    else:
        runner.initialize_afd_connector()
    assert calls == [
        "connector",
        *(["transaction"] * 4),
        "device-done",
        "warmup",
        "graph-resources",
        "capture",
        ("failure", "capture failed") if capture_fails else "captured",
    ]
    # Receiver readiness is intentionally not published by initialization.
    assert "service" not in calls


def test_attention_keeps_paired_warmup_and_waits_for_receiver(modules):
    _, connector_module, _ = modules
    calls: list[str | tuple[str, str | int]] = []
    connector = connector_module.CAMAsyncAFDConnector.__new__(
        connector_module.CAMAsyncAFDConnector
    )
    connector.__dict__.update(
        local_rank=0,
        hidden_size=32,
        activation_dtype="bf16",
        topk=1,
        graph_startup=SimpleNamespace(
            warmup_layer_id=2,
            phase=lambda stage: calls.append(stage),
            wait_service_ready=lambda: calls.append("service"),
            fail=lambda exc: calls.append("failure"),
        ),
        send_attn_output=lambda *a, **k: calls.append("send"),
        recv_ffn_output=lambda *a, **k: calls.append("receive"),
        wait_graph_device=lambda event: calls.append("device-done"),
    )
    connector._run_attention_graph_startup()
    assert calls == ["send", "receive", "device-done", "warmup", "captured", "service"]
