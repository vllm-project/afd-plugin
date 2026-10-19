# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real startup call sites with host stubs; not device execution coverage."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from afd_plugin.async_cam_startup import AsyncCAMGraphStartup
from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_npu_afd_async_features,
)
from afd_plugin.config import parse_afd_config


def load_host_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        monkeypatch.setitem(sys.modules, name, module)
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
    group = load_host_module(
        monkeypatch,
        "host_afd_process_group",
        "afd_plugin/distributed/afd_process_group.py",
    )
    stub("afd_plugin.distributed", init_afd_process_group=group.init_afd_process_group)
    connector = load_host_module(
        monkeypatch,
        "afd_plugin.connectors.npu.async_cam",
        "afd_plugin/connectors/npu/async_cam.py",
    )
    runtime = load_host_module(
        monkeypatch,
        "afd_plugin.v1.worker.npu.async_ffn_model_runner",
        "afd_plugin/v1/worker/npu/async_ffn_model_runner.py",
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


@pytest.fixture
def ffn_owners(modules, monkeypatch):
    _, _, async_runtime = modules
    for name in (
        "vllm.compilation.monitor",
        "vllm.config",
        "vllm.forward_context",
        "vllm.platforms",
        "afd_plugin.v1.worker.attention_metadata",
        "afd_plugin.v1.worker.cuda_graph",
        "vllm.distributed.parallel_state",
        "vllm.v1.worker.worker_base",
        "vllm.v1.worker.workspace",
        "vllm_ascend.ascend_config",
        "vllm_ascend.cpu_binding",
        "vllm_ascend.worker.worker",
        "afd_plugin.model_executor.models.model_utils",
    ):
        module = ModuleType(name)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        if name == "vllm_ascend.worker.worker":
            module.__dict__["NPUWorker"] = object
        monkeypatch.setitem(sys.modules, name, module)
    sync_runtime = load_host_module(
        monkeypatch,
        "afd_plugin.v1.worker.npu.ffn_model_runner",
        "afd_plugin/v1/worker/npu/ffn_model_runner.py",
    )
    worker = load_host_module(
        monkeypatch, "host_ffn_worker", "afd_plugin/v1/worker/npu/ffn_worker.py"
    )
    return sync_runtime, async_runtime, worker


@pytest.mark.parametrize("async_connector", [False, True])
def test_worker_selects_owner_and_passes_device_rank(
    ffn_owners, monkeypatch, async_connector
):
    _, _, worker = ffn_owners
    config, device = object(), SimpleNamespace(type="npu", index=3)
    connector, selected = object(), object()
    order = []
    monkeypatch.setattr(
        worker,
        "validate_npu_model_runner_v2_config",
        lambda *a, **k: order.append("validated"),
    )
    monkeypatch.setattr(
        worker,
        "parse_afd_config",
        lambda *a, **k: SimpleNamespace(
            connector="CAMAsyncAFDConnector"
            if async_connector
            else "CAMP2pAFDConnector"
        ),
    )
    monkeypatch.setattr(worker, "get_world_group", lambda: SimpleNamespace(rank=7))

    def create_connector(rank, local_rank, *args):
        assert (rank, local_rank) == (7, 3)
        order.append("connector")
        return connector

    monkeypatch.setattr(
        worker.AFDConnectorFactory, "create_connector", create_connector
    )

    def sync(cfg, dev):
        assert cfg is config and dev is device and not async_connector
        order.append("sync")
        return selected

    def asynchronous(cfg, dev, conn):
        assert cfg is config and dev is device and conn is connector and async_connector
        order.append("async")
        return selected

    monkeypatch.setattr(worker, "AFDNPUFFNModelRunner", sync)
    monkeypatch.setattr(worker, "AsyncNPUFFNModelRunner", asynchronous)
    assert worker.create_npu_ffn_model_runner(config, device) is selected
    assert order == (
        ["validated", "connector", "async"]
        if async_connector
        else ["validated", "sync"]
    )


def test_worker_rejects_v1_before_constructing_either_owner(ffn_owners, monkeypatch):
    _, _, worker = ffn_owners

    def reject(*args, **kwargs):
        raise ValueError("requires MRV2")

    monkeypatch.setattr(worker, "validate_npu_model_runner_v2_config", reject)

    def unexpected(*args, **kwargs):
        pytest.fail("rejected configuration must not create an owner or communicator")

    monkeypatch.setattr(worker, "parse_afd_config", unexpected)
    monkeypatch.setattr(worker.AFDConnectorFactory, "create_connector", unexpected)
    monkeypatch.setattr(worker, "AFDNPUFFNModelRunner", unexpected)
    monkeypatch.setattr(worker, "AsyncNPUFFNModelRunner", unexpected)
    with pytest.raises(ValueError, match="MRV2"):
        worker.create_npu_ffn_model_runner(object(), SimpleNamespace(type="npu"))


@pytest.mark.parametrize("async_connector", [False, True])
@pytest.mark.parametrize("speculative_tokens", [None, 3])
def test_separate_owners_construct_and_load_without_cross_graph_pool(
    ffn_owners, monkeypatch, async_connector, speculative_tokens
):
    sync_runtime, async_runtime, _ = ffn_owners
    runtime = async_runtime if async_connector else sync_runtime
    calls: list[str | tuple] = []
    for name in (
        "validate_npu_model_runner_v2_config",
        "fail_if_unsupported_npu_afd_features",
    ):
        monkeypatch.setattr(runtime, name, lambda *a, **k: calls.append("validated"))
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    connector, device, model, pool = (
        object(),
        SimpleNamespace(type="npu"),
        object(),
        object(),
    )
    config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "ffn",
                "connector": "CAMAsyncAFDConnector"
                if async_connector
                else "CAMP2pAFDConnector",
            }
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=43), enforce_eager=False
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=6),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL")),
        load_config=object(),
        speculative_config=None
        if speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=speculative_tokens),
    )

    def global_pool():
        assert not async_connector, "Async must not acquire the synchronous graph pool"
        calls.append("sync-pool")
        return pool

    monkeypatch.setattr(
        sync_runtime.current_platform, "get_global_graph_pool", global_pool
    )
    owner = (
        runtime.AsyncNPUFFNModelRunner
        if async_connector
        else runtime.AFDNPUFFNModelRunner
    )
    runner = owner(config, device, connector)
    assert calls == (
        ["validated", "validated"]
        if async_connector
        else ["validated", "validated", "sync-pool"]
    )
    assert runner.model is None
    if async_connector:
        assert runner._graph is None and runner._graph_capture_count == 0
    else:
        assert runner._acl_graphs == {} and runner.graph_pool is pool

    @contextmanager
    def memory_profiler():
        yield SimpleNamespace(consumed_memory=1234)

    def load_model(**kwargs):
        assert kwargs == {"vllm_config": config, "model_config": config.model_config}
        calls.append("load")
        return model

    for name, attrs in (
        (
            "vllm.model_executor.model_loader",
            {"get_model_loader": lambda cfg: SimpleNamespace(load_model=load_model)},
        ),
        ("vllm.utils.mem_utils", {"DeviceMemoryProfiler": memory_profiler}),
        (
            "vllm_ascend.ascend_forward_context",
            {
                "set_mc2_tokens_capacity": lambda *args: calls.append(
                    ("capacity", args)
                ),
                "set_mc2_mask": lambda *args: calls.append(("mask", args)),
            },
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    runner.load_model()
    assert calls[-3:] == [
        ("capacity", (config, 6, 1 + (speculative_tokens or 0))),
        ("mask", (config, device)),
        "load",
    ]
    assert runner.get_model() is model and runner.model_memory_usage == 1234
    with pytest.raises(RuntimeError, match="already loaded"):
        runner.load_model()
