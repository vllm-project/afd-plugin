# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real startup call sites with host stubs; not device execution coverage."""

import json
import sys
import threading
import weakref
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
    AsyncCAMGraphStartup,
    FFNGraphPlan,
    resolve_ffn_mode,
)
from afd_plugin.validation import validate_npu_model_runner_v2_config
from tests.unit.fakes.async_cam_startup import MemoryStore, startup_world
from tests.unit.fakes.async_ffn import (
    load_async_modules,
    load_host_module,
)
from tests.unit.fakes.graph import Graph
from tests.unit.fakes.vllm_config import npu_dspark_config, npu_vllm_config


@pytest.fixture
def modules(monkeypatch):
    return load_async_modules(monkeypatch)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("model_type", ["deepseek_v2", "deepseek_v3", "deepseek_v32"])
def test_async_startup_rejects_models_without_layered_weights(role, model_type):
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        extra_config={"dynamicQuant": 1},
    )
    config.model_config.hf_text_config.model_type = model_type
    with pytest.raises(RuntimeError, match="requires DeepSeek V4"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("dynamic_quant", [0, 1])
def test_async_startup_requires_dynamic_quantization_for_v4(
    modules, monkeypatch, role, dynamic_quant
):
    _, connector_module, _ = modules
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        extra_config={"dynamicQuant": dynamic_quant},
    )
    config.model_config.hf_text_config.model_type = "deepseek_v4"
    factory = ModuleType("afd_plugin.connectors.factory")
    factory.__dict__["AFDConnectorFactory"] = SimpleNamespace(
        parse_connector_extra_info=lambda _, source: (
            connector_module.AFDAsyncExtraInfo.from_mapping(
                source.additional_config["afd"]["connector_extra_config"]
            )
        )
    )
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    if dynamic_quant:
        fail_if_unsupported_npu_afd_features(config)
    else:
        with pytest.raises(RuntimeError, match="dynamicQuant=1"):
            fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("model_type", ["deepseek_v2", "deepseek_v3"])
def test_sync_camp2p_does_not_require_layered_async_weights(
    modules, monkeypatch, model_type
):
    class CAMP2PExtraInfo:
        def validate_supported(self):
            pass

    connector = ModuleType("afd_plugin.connectors.npu.camp2p")
    connector.__dict__["CAMP2PExtraInfo"] = CAMP2PExtraInfo
    factory = ModuleType("afd_plugin.connectors.factory")
    factory.__dict__["AFDConnectorFactory"] = SimpleNamespace(
        parse_connector_extra_info=lambda *_: CAMP2PExtraInfo()
    )
    monkeypatch.setitem(sys.modules, connector.__name__, connector)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    config = npu_vllm_config(connector="CAMP2pAFDConnector", use_mla=True)
    config.model_config.hf_text_config.model_type = model_type

    fail_if_unsupported_npu_afd_features(config)


@pytest.fixture
def layered_startup_runner(modules, monkeypatch, mode, invalid):
    _, _, runtime = modules

    def executor(layers):
        signature = repr(tuple(layer.layer_idx for layer in layers))
        return SimpleNamespace(
            layer_id_to_slot=None, graph_signature=lambda capacity: signature
        )

    monkeypatch.setattr(runtime, "AsyncCAMLayeredExecutor", executor)
    monkeypatch.setattr(
        runtime,
        "expert_weight_signature",
        lambda layers, capacity: repr(tuple(layer.layer_idx for layer in layers)),
    )
    target = (
        []
        if invalid == "target"
        else [
            SimpleNamespace(
                layer_idx=2, quantization="W4A8", per_channel=True, swiglu_limit=0
            )
        ]
    )
    draft = [] if invalid == "draft" else [SimpleNamespace(layer_idx=5)]
    if invalid == "layers":
        draft = [SimpleNamespace(layer_idx=6)]
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.model = SimpleNamespace(
        get_experts_layer_indices=lambda: (2,),
        get_quantized_expert_weights=lambda: (target, "unsupported target"),
    )
    runner.draft_model = SimpleNamespace(
        get_experts_layer_indices=lambda: (5,),
        get_quantized_expert_weights=lambda: (draft, "unsupported draft"),
    )
    connector = SimpleNamespace(
        is_initialized=False,
        dynamic_quant=1,
        extra_info=SimpleNamespace(ffn_mega_moe=False),
        graph_startup=AsyncCAMGraphStartup(1, 1, 2),
    )
    connector.init_afd_connector = lambda: resolve_ffn_mode(
        [connector.graph_startup.plan]
    )
    runner.connector = connector
    runner._failure = runner._layered_executor = runner._mega_executor = None
    runner.requested_mode = mode
    return runner, connector


@pytest.mark.parametrize(
    "mode,invalid,message",
    [
        (mode, invalid, message)
        for mode in ("eager", "full")
        for invalid, message in (
            ("target", "target weights"),
            ("draft", "draft weights"),
            ("layers", "layer IDs"),
        )
    ]
    + [("eager", None, None)],
)
def test_ffn_startup_validates_target_and_draft_weights(
    layered_startup_runner, mode, invalid, message
):
    runner, connector = layered_startup_runner
    if message:
        with pytest.raises(ValueError, match=message):
            runner.initialize_afd_connector()
    else:
        runner.initialize_afd_connector()
        assert runner.effective_mode == "eager"
        assert connector.graph_startup.plan.signature == "(2, 5)"


@pytest.mark.parametrize("mode", ["eager", "full"])
def test_mega_executor_failure_is_published_during_startup(modules, monkeypatch, mode):
    _, _, runtime = modules
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.model = SimpleNamespace(
        get_experts_layer_indices=lambda: (2,),
        get_quantized_expert_weights=lambda: ([SimpleNamespace(layer_idx=2)], ""),
    )
    runner.draft_model = None
    runner._failure = runner._layered_executor = runner._mega_executor = None
    runner.requested_mode = mode
    connector = SimpleNamespace(
        is_initialized=False,
        extra_info=SimpleNamespace(ffn_mega_moe=True),
        dynamic_quant=1,
        topology=None,
        max_num_batched_tokens=8,
        topk=2,
        tp_size=1,
        graph_startup=AsyncCAMGraphStartup(1, 1, 2),
    )
    connector.init_afd_connector = lambda: resolve_ffn_mode(
        [connector.graph_startup.plan]
    )
    runner.connector = connector

    def fail_executor(*_args, **_kwargs):
        raise ValueError("unsupported Mega weights")

    monkeypatch.setattr(runtime, "AsyncCAMMegaExecutor", fail_executor)
    with pytest.raises(ValueError, match="unsupported Mega weights"):
        runner.initialize_afd_connector()
    assert connector.graph_startup.plan.executor_error == "unsupported Mega weights"


@pytest.fixture
def independent_mega_runner(modules, monkeypatch):
    _, _, runtime = modules
    peers = startup_world(attn=1, ffn=1)
    model = SimpleNamespace(
        get_experts_layer_indices=lambda: (2,),
        get_quantized_expert_weights=lambda: ([SimpleNamespace(layer_idx=2)], ""),
    )
    connector = SimpleNamespace(
        is_initialized=False,
        dynamic_quant=1,
        extra_info=SimpleNamespace(ffn_mega_moe=True),
        graph_startup=peers[1],
        topology=None,
        max_num_batched_tokens=8,
        topk=2,
        tp_size=1,
        comm_args=object(),
        group_name="ready",
    )

    def initialize():
        connector.graph_startup.mode = resolve_ffn_mode([connector.graph_startup.plan])
        connector.is_initialized = True

    connector.init_afd_connector = initialize
    fail_binding = [False]
    mega = MagicMock()

    def bind(*args):
        if not connector.is_initialized:
            raise RuntimeError("communication was bound before initialization")
        if fail_binding[0]:
            raise RuntimeError("communication binding failed")

    mega.bind_communication = bind

    def forbidden_layered(*args):
        raise AssertionError("Mega must not construct Layered")

    monkeypatch.setattr(runtime, "AsyncCAMLayeredExecutor", forbidden_layered)
    monkeypatch.setattr(runtime, "AsyncCAMMegaExecutor", lambda *args, **kw: mega)
    monkeypatch.setattr(
        runtime, "expert_weight_signature", lambda layers, capacity: "weights"
    )
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.__dict__.update(
        model=model,
        draft_model=None,
        connector=connector,
        requested_mode="eager",
        _failure=None,
        _layered_executor=None,
        _mega_executor=None,
        _first_step=False,
        _graph=None,
        _is_shutdown=False,
        prof=None,
        session=sys.modules["afd_plugin.connectors.base"].FFNExecutionSession(),
    )
    return SimpleNamespace(
        runner=runner, connector=connector, peer=peers[0], fail_binding=fail_binding
    )


def test_mega_serves_without_constructing_layered(independent_mega_runner):
    h = independent_mega_runner
    h.runner.initialize_afd_connector()
    assert h.connector.graph_startup.plan.signature == "native_mc2:weights"
    assert h.runner.execute_connector_driven_step()
    h.peer.check_failure()


@pytest.mark.parametrize("mode", ["eager", "full"])
def test_mega_binding_failure_reaches_peer_and_retains_resources(
    independent_mega_runner, mode
):
    h = independent_mega_runner
    h.runner.requested_mode = mode
    h.fail_binding[0] = True
    with pytest.raises(RuntimeError, match="communication binding failed"):
        h.runner.initialize_afd_connector()
    with pytest.raises(RuntimeError, match="communication binding failed"):
        h.peer.check_failure()
    with pytest.raises(RuntimeError, match="startup retains resources"):
        h.runner.shutdown()
    assert h.connector.is_initialized


@pytest.mark.parametrize(
    "field",
    [
        "ffn_two_stream",
        "ffn_compute_aic_num",
        "ffn_compute_aiv_num",
        "ffn_combine_aiv_num",
    ],
)
def test_removed_two_stream_options_are_rejected(modules, field):
    _, connector, _ = modules
    with pytest.raises(ValueError, match="unknown AFD async connector_extra_config"):
        connector.AFDAsyncExtraInfo.from_mapping({field: True})


@pytest.mark.parametrize("mode", ["eager", "full"])
@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("tokens,tp", [(65535, 1), (65536, 1), (65536, 2), (131072, 2)])
def test_full_row_bound_precedes_hccl_creation(
    modules, monkeypatch, mode, role, tokens, tp
):  # long-test: negotiated mode gates the same pre-HCCL boundary on both roles
    group, connector_module, _ = modules
    order = []
    store = MagicMock()
    rank = 0 if role == "attention" else tp
    monkeypatch.setattr(
        group, "rendezvous", lambda *a, **k: iter([(store, rank, tp + 1)])
    )
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
        extra_info=connector_module.AFDAsyncExtraInfo(),
        afd_config=SimpleNamespace(host="localhost", port=1),
        topology=SimpleNamespace(world_size=tp + 1),
        world_rank=rank,
        attn_size=tp,
        ffn_size=1,
        hidden_size=32,
        topk=1,
        num_routed_experts=1,
        activation_dtype="bf16",
        dynamic_quant=1,
        tp_size=tp,
        max_num_batched_tokens=tokens,
        hccl_buffer_size_mb=256,
        local_rank=0,
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
        ),
        graph_startup=AsyncCAMGraphStartup(rank, tp, tp + 1),
    )
    connector._session = connector_module.PushCAMSession(connector.activation_dtype)

    def negotiate(store, signature):
        order.append("negotiate")
        connector.graph_startup.mode = mode

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
    if mode == "full" and tokens // tp > 65535:
        with pytest.raises(ValueError, match="uint16"):
            connector.init_afd_connector()
        assert order == ["negotiate", "failure"]
    else:
        connector.init_afd_connector()
        expected = ["negotiate"]
        if mode == "full":
            expected.append("validated")
        expected.append("hccl")
        if mode == "full" and role == "attention":
            expected.append("attention-warmup")
        assert order == expected


def test_executor_error_reaches_store_without_retrying_transport_loader(
    modules, monkeypatch
):  # long-test: real Store callback must publish the construction error before HCCL
    group, _, _ = modules

    def unavailable_transport_ops():
        raise RuntimeError("transport loader retried")

    monkeypatch.setattr(
        sys.modules["afd_plugin.compat.npu.ops"],
        "ensure_cam_async_ops_available",
        unavailable_transport_ops,
    )
    module = load_host_module(monkeypatch, "afd_plugin.connectors.npu.async_cam")
    config = npu_vllm_config(
        role="ffn",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        extra_config={"dynamicQuant": 1},
    )
    config.model_config.dtype = "bf16"
    connector = module.CAMAsyncAFDConnector(1, 0, config, parse_afd_config(config), 0)
    connector.graph_startup.plan = FFNGraphPlan(
        executor_error="unsupported compute operator"
    )
    store = MemoryStore()
    nonce = "000000000000001"
    store.set("ffn_graph/nonce", nonce)
    store.set(
        f"ffn_graph/{nonce}/plan/0",
        json.dumps(
            {
                "transport": "wire",
                "shared_errors": True,
                "ffn": None,
            }
        ),
    )
    monkeypatch.setattr(group, "rendezvous", lambda *a, **k: iter([(store, 1, 2)]))
    monkeypatch.setattr(group, "PrefixStore", lambda *args: store)

    def unexpected_hccl(*args, **kwargs):
        pytest.fail("invalid executor must fail before HCCL construction")

    monkeypatch.setattr(group, "_new_process_group_helper", unexpected_hccl)
    with pytest.raises(ValueError, match="unsupported compute operator"):
        connector.init_afd_connector()
    assert (
        "unsupported compute operator" in store.get(f"ffn_graph/{nonce}/error").decode()
    )


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
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        num_ubatches=num_ubatches,
    )
    config.model_config.enforce_eager = role == "attention"
    setattr(config.parallel_config, native_flag, True)
    with pytest.raises(RuntimeError, match=error):
        validate_npu_model_runner_v2_config(
            config, expected_role=role, device_type="npu"
        )


def test_ffn_announces_service_only_on_receiver_execution(modules):
    _, connector_module, runtime = modules
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    calls: list[str | tuple[str, str | int]] = []
    runner.__dict__.update(
        prof=None,
        _graph=object(),
        _service_announced=False,
        _first_step=True,
        _graph_stream=MagicMock(),
        device="npu:0",
        session=sys.modules["tests.unit.fakes.connector"].FakeFFNSession(),
        layer_ids=(2, 5),
        connector=SimpleNamespace(
            extra_info=connector_module.AFDAsyncExtraInfo(),
            graph_startup=SimpleNamespace(service_ready=lambda: calls.append("ready")),
        ),
        _submit_graph=lambda count: calls.append(("submit", count)),
        _complete_graph=lambda: True,
    )
    assert calls == []
    assert runner.execute_connector_driven_step()
    assert runner.execute_connector_driven_step()
    assert calls == ["ready", ("submit", 2), ("submit", 2)]


@pytest.mark.parametrize("session_serves", [False, True])
def test_ffn_step_lets_session_decide_service_after_shutdown(modules, session_serves):
    _, connector_module, runtime = modules
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    submitted: list[int] = []
    serving = [True] if session_serves else []
    runner.__dict__.update(
        prof=None,
        _graph=object(),
        _service_announced=True,
        _first_step=False,
        session=sys.modules["tests.unit.fakes.connector"].FakeFFNSession(
            serving=serving
        ),
        layer_ids=(2, 5),
        connector=SimpleNamespace(extra_info=connector_module.AFDAsyncExtraInfo()),
        _submit_graph=submitted.append,
        _complete_graph=lambda: True,
    )
    # Push stops on the shutdown request; Pull keeps serving until A is quiet.
    assert runner.execute_connector_driven_step(True) is session_serves
    assert submitted == ([2] if session_serves else [])


@pytest.mark.parametrize("capture_fails", [False, True])
def test_ffn_warmup_capture_order_and_failure_propagation(
    modules, monkeypatch, capture_fails
):  # long-test: warmup/capture/readiness are one ordered startup protocol
    _, connector_module, runtime = modules
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
        draft_model=None,
        _initialize_ffn_executor=lambda: "weights",
        _layered_executor=None,
        _mega_executor=None,
        connector=SimpleNamespace(
            is_initialized=False,
            graph_startup=startup,
            extra_info=connector_module.AFDAsyncExtraInfo(),
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
    ):
        module = ModuleType(name)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        if name == "vllm_ascend.worker.worker":
            module.__dict__["NPUWorker"] = object
        monkeypatch.setitem(sys.modules, name, module)
    load_host_module(monkeypatch, "afd_plugin.model_executor.models.model_utils")
    load_host_module(monkeypatch, "afd_plugin.connectors.factory")
    sync_runtime = load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.ffn_model_runner"
    )
    worker = load_host_module(monkeypatch, "afd_plugin.v1.worker.npu.ffn_worker")
    return sync_runtime, async_runtime, worker


@pytest.fixture(params=["attention", "ffn"])
def npu_worker(request, ffn_owners, monkeypatch):
    _, _, ffn_module = ffn_owners
    role = request.param
    events = []
    if role == "attention":
        runner_module = ModuleType("afd_plugin.v1.worker.npu.attention_model_runner_v2")
        runner_module.__dict__["AFDNPUAttentionModelRunnerV2"] = lambda *args: (
            events.append("runner")
        )
        monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
        module = load_host_module(
            monkeypatch, "afd_plugin.v1.worker.npu.attention_worker"
        )
        worker_class = module.AFDNPUAttentionWorker
    else:
        module = ffn_module
        worker_class = module.AFDNPUFFNWorker
        monkeypatch.setattr(
            module, "create_npu_ffn_model_runner", lambda *args: events.append("runner")
        )
        module.get_ascend_device_type.return_value = module.AscendDeviceType.A3
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        num_ffn_ranks=2,
        data_parallel_size=2 if role == "ffn" else 1,
        extra_config={"dynamicQuant": 1},
        use_ubatching=role == "attention",
        num_ubatches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=2,
        worker_cls=(
            "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
            if role == "attention"
            else "afd_plugin.v1.worker.npu.AFDNPUFFNWorker"
        ),
        all2all_backend="native-default",
    )
    config.model_config.hf_config.model_type = "deepseek_v4"
    config.model_config.hf_config.architectures = ["DeepseekV4ForCausalLM"]
    worker = worker_class.__new__(worker_class)
    worker.vllm_config = config
    worker.model_config = config.model_config
    harness = SimpleNamespace(
        worker=worker, module=module, events=events, fail_ops=False
    )

    def initialize_device():
        events.append("device")
        return SimpleNamespace(type="npu", index=0)

    def load_ops():
        events.append("ops")
        if harness.fail_ops:
            raise RuntimeError("transport operators unavailable")

    worker._init_device = initialize_device
    monkeypatch.setattr(module, "ensure_cam_async_ops_available", load_ops)
    monkeypatch.setattr(
        module, "init_workspace_manager", lambda *args: events.append("workspace")
    )
    return harness


@pytest.mark.parametrize(
    "invalid,error",
    [
        ("v1", "requires ModelRunnerV2"),
        ("pp", "does not support PP or CP"),
        ("world", "ranks must match DP"),
        ("ep", "static expert parallelism"),
        ("quant", "dynamicQuant=1"),
        ("experts", "experts divisible by FFN ranks"),
    ],
)
def test_worker_rejects_deployment_before_device_initialization(
    npu_worker, invalid, error
):
    config = npu_worker.worker.vllm_config
    if invalid == "v1":
        config.use_v2_model_runner = False
    elif invalid == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif invalid == "world":
        config.parallel_config.data_parallel_size += 1
    elif invalid == "ep":
        config.parallel_config.enable_expert_parallel = False
    elif invalid == "quant":
        config.additional_config["afd"]["connector_extra_config"]["dynamicQuant"] = 2
    else:
        config.model_config.hf_config.n_routed_experts = 7
    with pytest.raises((RuntimeError, ValueError), match=error):
        npu_worker.worker.init_device()
    assert npu_worker.events == []


def test_worker_prepares_role_config_and_dbo_before_native_device(npu_worker):
    worker = npu_worker.worker
    worker.init_device()
    assert npu_worker.events == ["device", "ops", "workspace", "runner"]
    assert worker.model_config is worker.vllm_config.model_config
    assert worker.model_config.hf_config.architectures == ["AFDDeepseekV4ForCausalLM"]
    parallel = worker.vllm_config.parallel_config
    assert parallel.enable_dbo == (worker.afd_expected_role == "attention")
    assert parallel.all2all_backend == "flashinfer_all2allv"


@pytest.mark.parametrize("npu_worker", ["attention"], indirect=True)
def test_dbo_normalization_rechecks_threshold_before_device(npu_worker):
    npu_worker.worker.vllm_config.parallel_config.dbo_decode_token_threshold = 1
    with pytest.raises(ValueError, match="at least the number of microbatches"):
        npu_worker.worker.init_device()
    assert npu_worker.events == []


def test_dspark_transport_dimensions_fail_before_device(npu_worker):
    config = npu_worker.worker.vllm_config
    config.parallel_config.use_ubatching = False
    config.speculative_config = npu_dspark_config().speculative_config
    config.speculative_config.draft_model_config.hf_text_config.hidden_size = 64
    config.scheduler_config.max_num_seqs = 2
    with pytest.raises(ValueError, match="transport hidden_size mismatch"):
        npu_worker.worker.init_device()
    assert npu_worker.events == []


def test_missing_transport_operators_fail_before_workspace_or_runner(npu_worker):
    npu_worker.fail_ops = True
    with pytest.raises(RuntimeError, match="transport operators unavailable"):
        npu_worker.worker.init_device()
    assert npu_worker.events == ["device", "ops"]


@pytest.mark.parametrize("npu_worker", ["ffn"], indirect=True)
def test_ffn_device_capability_fails_before_workspace_or_runner(npu_worker):
    npu_worker.module.get_ascend_device_type.return_value = object()
    with pytest.raises(ValueError, match="requires Ascend 910C/A3"):
        npu_worker.worker.init_device()
    assert npu_worker.events == ["device"]


@pytest.fixture
def receiver_graph_worker(ffn_owners, monkeypatch):
    _, runtime, module = ffn_owners
    # Admission has its own tests; this host loader replaces connector bases.
    config = npu_vllm_config(
        role="ffn",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        enable_expert_parallel=True,
        enable_elastic_ep=False,
    )
    state = SimpleNamespace(
        token_nums_rankid_layeridx=object(),
        group_list=object(),
        dynamic_scales=object(),
    )
    connector = SimpleNamespace(
        is_initialized=True,
        graph_startup=SimpleNamespace(service_ready=lambda: None),
        ffn_session=sys.modules["afd_plugin.connectors.base"].FFNExecutionSession,
        close=lambda: None,
        recv_attn_output=lambda: SimpleNamespace(
            hidden_states=object(), context=SimpleNamespace(states=state)
        ),
        send_ffn_output=lambda *args: None,
    )
    runner = runtime.AsyncNPUFFNModelRunner(
        config, SimpleNamespace(type="npu"), connector
    )
    runner.effective_mode = "full"
    monkeypatch.setattr(runtime.torch.npu, "NPUGraph", Graph)
    monkeypatch.setattr(
        runtime.torch.npu, "graph", lambda *args, **kwargs: nullcontext()
    )
    runner._layered_executor = lambda hidden, *args, **kwargs: hidden
    runner._initialize_graph(runtime.torch.npu.Stream(device=runner.device))
    runner._capture_graph()
    shutdown = ModuleType("vllm.v1.worker.gpu.shutdown")
    shutdown.__dict__["free_before_shutdown"] = lambda config: None
    monkeypatch.setitem(sys.modules, shutdown.__name__, shutdown)
    worker = module.AFDNPUFFNWorker.__new__(module.AFDNPUFFNWorker)
    worker.model_runner = runner
    worker.device = runner.device
    worker._ffn_thread = None
    worker._ffn_loop_error = None
    worker._cpu_binding_attempted = True
    return worker, weakref.ref(runner._graph)


def test_receiver_exit_prevents_further_graph_submission(receiver_graph_worker):
    worker, _ = receiver_graph_worker
    worker.model_runner.layer_ids = (2,)
    worker.model_runner.mark_receiver_exited()
    with pytest.raises(RuntimeError, match="not ready for replay"):
        worker.model_runner.execute_connector_driven_step()


@pytest.mark.parametrize("failure", [None, "serve", "drain", "finish", "pending"])
def test_full_receiver_exit_releases_graph_only_after_success(
    receiver_graph_worker, failure
):
    worker, graph = receiver_graph_worker
    runner = worker.model_runner
    runner.session = sys.modules["tests.unit.fakes.connector"].FakeFFNSession(
        serving=(False,),
        requires_drain=True,
        drain_error=RuntimeError("drain failed") if failure == "drain" else None,
        finish_error=RuntimeError("finish failed") if failure == "finish" else None,
    )

    def step(shutdown_requested):
        if failure == "serve":
            raise RuntimeError("partial device completion")
        if failure == "pending":
            runner._submit_graph()
        return False

    runner.execute_connector_driven_step = step
    worker.start_ffn_server_loop()
    thread: threading.Thread = worker._ffn_thread
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert graph() is not None
    if failure:
        with pytest.raises(RuntimeError, match="FFN worker loop failed"):
            worker.stop_ffn_server_loop()
        assert graph() is not None
        with pytest.raises(RuntimeError, match="FFN worker loop failed"):
            worker.raise_ffn_loop_error_if_any()
    else:
        worker.stop_ffn_server_loop()
        worker.raise_ffn_loop_error_if_any()
        assert graph() is None


@pytest.mark.parametrize("state", ["live", "failed", "drained"])
@pytest.mark.parametrize("requires_drain", [False, True])
def test_worker_retains_resources_until_clean_exit(ffn_owners, state, requires_drain):
    _, runtime, module = ffn_owners
    calls = []
    timeouts = []
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.effective_mode = "full"
    runner.session = SimpleNamespace(requires_drain=requires_drain)
    runner.connector = SimpleNamespace(close=lambda: calls.append("connector-close"))
    runner.shutdown = lambda: calls.append("release")
    worker = module.AFDNPUFFNWorker.__new__(module.AFDNPUFFNWorker)
    worker.model_runner = runner
    worker._ffn_shutdown_event = SimpleNamespace(set=lambda: calls.append("stop"))

    def join(**kwargs):
        calls.append("join")
        timeouts.append(kwargs["timeout"])

    worker._ffn_thread = SimpleNamespace(
        join=join,
        is_alive=lambda: state == "live",
    )
    worker._ffn_loop_error = (
        RuntimeError("native failure") if state == "failed" else None
    )
    if state == "drained":
        worker.stop_ffn_server_loop()
        assert calls == ["stop", "join", "release"]
    else:
        with pytest.raises(RuntimeError, match="retain resources|loop failed"):
            worker.stop_ffn_server_loop()
        assert calls == ["stop", "join"]
        if state == "failed":
            for _ in range(2):
                with pytest.raises(RuntimeError, match="loop failed"):
                    worker.raise_ffn_loop_error_if_any()
            with pytest.raises(RuntimeError):
                worker.start_ffn_server_loop()
    assert timeouts == [60 if requires_drain else 5]


@pytest.mark.parametrize(
    "config,error",
    [
        (AFDConfig(role="invalid"), "AFD role"),
        (AFDConfig(num_ffn_ranks=0), "num_ffn_ranks must be positive"),
        (AFDConfig(port=0), "AFD port"),
    ],
)
def test_connector_factory_validates_supplied_config_before_construction(
    modules, monkeypatch, config, error
):
    factory = load_host_module(monkeypatch, "afd_plugin.connectors.factory")
    with pytest.raises(ValueError, match=error):
        factory.AFDConnectorFactory.create_connector(0, 0, object(), config)


@pytest.mark.parametrize("async_connector", [False, True])
def test_worker_selects_runner_for_connector(ffn_owners, monkeypatch, async_connector):
    _, _, worker = ffn_owners
    config, device, calls = object(), SimpleNamespace(type="npu", index=3), MagicMock()
    parsed = SimpleNamespace(
        connector="CAMAsyncAFDConnector" if async_connector else "CAMP2pAFDConnector"
    )
    monkeypatch.setattr(worker, "parse_afd_config", lambda *a, **k: parsed)
    monkeypatch.setattr(worker, "get_world_group", lambda: SimpleNamespace(rank=7))
    monkeypatch.setattr(worker.AFDConnectorFactory, "create_connector", calls.connector)
    monkeypatch.setattr(worker, "AFDNPUFFNModelRunner", calls.sync)
    monkeypatch.setattr(worker, "AsyncNPUFFNModelRunner", calls.asynchronous)
    selected = calls.asynchronous if async_connector else calls.sync
    assert worker.create_npu_ffn_model_runner(config, device) is selected.return_value


@pytest.mark.parametrize("async_connector", [False, True])
@pytest.mark.parametrize("speculative_tokens", [None, 3])
def test_ffn_runner_loads_model_and_rejects_second_load(
    ffn_owners, monkeypatch, async_connector, speculative_tokens
):  # long-test: loader setup
    sync_runtime, async_runtime, _ = ffn_owners
    runtime = async_runtime if async_connector else sync_runtime
    calls = MagicMock()
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    connector = SimpleNamespace(
        ffn_session=sys.modules["afd_plugin.connectors.base"].FFNExecutionSession
    )
    device, model = SimpleNamespace(type="npu"), object()
    connector_name = "CAMAsyncAFDConnector" if async_connector else "CAMP2pAFDConnector"
    config = npu_vllm_config(role="ffn", connector=connector_name)
    config.model_config.enforce_eager = False
    config.model_config.hf_config = SimpleNamespace(num_hidden_layers=43)
    config.scheduler_config.max_num_seqs = 6
    config.load_config = object()
    if speculative_tokens is not None:
        config.speculative_config = SimpleNamespace(
            method="mtp", num_speculative_tokens=speculative_tokens
        )
    monkeypatch.setattr(
        sync_runtime.current_platform, "get_global_graph_pool", calls.pool
    )
    owner = (
        runtime.AsyncNPUFFNModelRunner
        if async_connector
        else runtime.AFDNPUFFNModelRunner
    )
    runner = owner(config, device, connector)

    def get_loader(load_config):
        assert load_config is config.load_config
        assert calls.mock_calls[-2:] == [
            call.capacity(config, 6, 1 + (speculative_tokens or 0)),
            call.mask(config, device),
        ]
        return SimpleNamespace(load_model=calls.load)

    calls.load.return_value = model
    profiler = MagicMock()
    profiler.__enter__.return_value.consumed_memory = 1234
    for name, attrs in (
        ("vllm.model_executor.model_loader", {"get_model_loader": get_loader}),
        ("vllm.utils.mem_utils", {"DeviceMemoryProfiler": lambda: profiler}),
        (
            "vllm_ascend.ascend_forward_context",
            {"set_mc2_tokens_capacity": calls.capacity, "set_mc2_mask": calls.mask},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    runner.load_model()
    calls.load.assert_called_once_with(
        vllm_config=config, model_config=config.model_config
    )
    assert runner.get_model() is model and runner.model_memory_usage == 1234
    with pytest.raises(RuntimeError, match="already loaded"):
        runner.load_model()


@pytest.fixture
def connector_world(monkeypatch):
    _, module, runtime = load_async_modules(monkeypatch)
    owners = startup_world()
    connectors = []
    for rank, startup in enumerate(owners):
        role = "attention" if rank < 2 else "ffn"
        config = npu_vllm_config(
            role=role,
            connector="CAMAsyncAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=2,
            extra_config={"ffn_mega_moe": True, "dynamicQuant": 1},
        )
        config.model_config.hf_config = SimpleNamespace(
            hidden_size=32, num_experts_per_tok=2, n_routed_experts=4
        )
        config.model_config.dtype = "bfloat16"
        connector = module.CAMAsyncAFDConnector(
            rank, 0, config, parse_afd_config(config), rank % 2
        )
        connector.graph_startup = startup
        connector.ffn_session().startup = startup
        connector._initialized = True
        connector.ffn_session().comm_args = object()
        connectors.append(connector)
    return SimpleNamespace(connectors=connectors, runtime=runtime)


def test_default_session_obeys_shutdown_without_drain(connector_world):
    session = sys.modules["afd_plugin.connectors.base"].FFNExecutionSession()
    assert session.keep_serving(False)
    assert not session.keep_serving(True)
    assert not session.requires_drain
    session.before_first_step()
    session.after_step()
    session.drain(lambda: pytest.fail("Push must not execute a drain transaction"))
    session.finish_receiver()
    session.finish_receiver()
    session.validate_shutdown()


@pytest.mark.parametrize("failure", ["unfinished", "active", "status"])
def test_close_preserves_receiver_communication_on_unsafe_session(
    connector_world, monkeypatch, failure
):
    connector = connector_world.connectors[2]
    resource = connector.comm_args
    connector.cam_pg = group = object()
    session = connector.ffn_session()
    if failure != "unfinished":
        monkeypatch.setattr(
            connector.ffn_session(), "pull_has_active_source", lambda: False
        )
        session.drain(lambda: pytest.fail("idle source must not launch"))
        for attention in connector_world.connectors[:2]:
            attention.graph_startup._publish("pull_quiet")
        session.finish_receiver()
    if failure == "active":
        monkeypatch.setattr(
            connector.ffn_session(), "pull_has_active_source", lambda: True
        )
    elif failure == "status":

        def fail_status():
            raise RuntimeError("sticky status -5")

        monkeypatch.setattr(connector.ffn_session(), "check_pull_status", fail_status)
    expected = "sticky status -5" if failure == "status" else "has not drained"
    with pytest.raises(RuntimeError, match=expected):
        connector.close()
    assert connector.is_initialized and connector.comm_args is resource
    assert connector.cam_pg is group


@pytest.mark.parametrize("failure", ["body", "status"])
def test_attention_step_recovers_after_error(connector_world, monkeypatch, failure):
    connector = connector_world.connectors[0]
    error: list[RuntimeError | None] = [None]

    def check_status():
        if error[0] is not None:
            raise error[0]

    monkeypatch.setattr(connector.ffn_session(), "check_pull_status", check_status)
    expected = ValueError if failure == "body" else RuntimeError
    with pytest.raises(expected, match="failed"), connector.attention_step():
        if failure == "body":
            raise ValueError("model failed")
        error[0] = RuntimeError("status failed")
    error[0] = RuntimeError("status failed after recovery")
    with (
        pytest.raises(RuntimeError, match="after recovery"),
        connector.attention_step(),
    ):
        pass
    error[0] = None
    with connector.attention_step():
        pass


def test_nested_attention_step_defers_status_until_outer_boundary(
    connector_world, monkeypatch
):
    connector = connector_world.connectors[0]

    def check_status():
        raise RuntimeError("sticky status")

    monkeypatch.setattr(connector.ffn_session(), "check_pull_status", check_status)
    with pytest.raises(RuntimeError, match="sticky status"), connector.attention_step():
        with connector.attention_step():
            pass
        # Reaching this line proves the nested scope did not check sticky status.
        completed_inner_scope = True
    assert completed_inner_scope


@pytest.mark.parametrize("fails", [False, True])
def test_runner_drain_returns_completed_response_or_session_error(
    connector_world, fails
):
    runtime = connector_world.runtime
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.model = object()
    runner.connector = connector_world.connectors[2]
    runner.session = sys.modules["tests.unit.fakes.connector"].FakeFFNSession(
        transactions=1,
        drain_error=RuntimeError("source did not drain") if fails else None,
    )
    runner._graph = None
    runner.layer_ids = (2, 5)
    responses = []
    runner.connector.send_ffn_output = lambda output, context: responses.append(output)
    # The host executor emits a peer response; it does not simulate device compute.
    runner._mega_executor = lambda: runner.connector.send_ffn_output("response", None)
    if fails:
        with pytest.raises(RuntimeError, match="source did not drain"):
            runner.drain()
    else:
        runner.drain()
    assert responses == ["response"]


@pytest.mark.parametrize("failure", ["unfinished", "active", "status"])
def test_runner_shutdown_retains_model_and_communication_on_unsafe_session(
    connector_world, monkeypatch, failure
):
    runtime = connector_world.runtime
    connector = connector_world.connectors[2]
    runner = runtime.AsyncNPUFFNModelRunner(
        connector.vllm_config, SimpleNamespace(type="npu"), connector
    )
    runner.model = model = object()
    runner.mark_receiver_exited()
    resource = connector.comm_args
    if failure != "unfinished":
        monkeypatch.setattr(
            connector.ffn_session(), "pull_has_active_source", lambda: False
        )
        runner.session.drain(lambda: pytest.fail("idle source must not launch"))
        for attention in connector_world.connectors[:2]:
            attention.graph_startup._publish("pull_quiet")
        runner.session.finish_receiver()
    if failure == "active":
        monkeypatch.setattr(
            connector.ffn_session(), "pull_has_active_source", lambda: True
        )
    elif failure == "status":

        def fail_status():
            raise RuntimeError("status failed; retain resources")

        monkeypatch.setattr(connector.ffn_session(), "check_pull_status", fail_status)
    with pytest.raises(RuntimeError, match="retain resources"):
        runner.shutdown()
    assert runner.get_model() is model
    assert connector.is_initialized and connector.comm_args is resource
