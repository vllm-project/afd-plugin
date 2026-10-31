# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real startup call sites with host stubs; not device execution coverage."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from afd_plugin.async_cam_startup import AsyncCAMGraphStartup
from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_npu_afd_async_features,
)
from afd_plugin.config import parse_afd_config
from tests.unit.fakes.async_ffn import (
    load_async_modules,
    load_host_module,
)
from tests.unit.fakes.vllm_config import npu_vllm_config


@pytest.fixture
def modules(monkeypatch):
    return load_async_modules(monkeypatch)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_two_stream_rejects_invalid_core_settings(modules, value):
    _, connector, _ = modules
    with pytest.raises((TypeError, ValueError), match="ffn_compute_aic_num"):
        connector.AFDAsyncExtraInfo.from_mapping({"ffn_compute_aic_num": value})


def test_two_stream_config_round_trip_and_default_off(modules):
    _, connector, _ = modules
    default = connector.AFDAsyncExtraInfo.from_mapping({})
    assert not default.ffn_two_stream
    assert "ffn_two_stream" not in default.to_mapping()
    enabled = connector.AFDAsyncExtraInfo.from_mapping({"ffn_two_stream": True})
    assert enabled.ffn_compute_aic_num == 12
    assert enabled.ffn_compute_aiv_num == enabled.ffn_combine_aiv_num == 24
    assert connector.AFDAsyncExtraInfo.from_mapping(enabled.to_mapping()) == enabled
    disabled = connector.AFDAsyncExtraInfo.from_mapping({"ffn_compute_aic_num": 8})
    assert connector.AFDAsyncExtraInfo.from_mapping(disabled.to_mapping()) == disabled


@pytest.mark.parametrize(
    "feature", [None, "enable_dbo", "use_sequence_parallel_moe", "flashcomm1"]
)
def test_two_stream_attention_preflight_rejects_unqualified_modes(
    modules, monkeypatch, feature
):
    _, module, _ = modules
    connector = module.CAMAsyncAFDConnector.__new__(module.CAMAsyncAFDConnector)
    connector.vllm_config = npu_vllm_config(
        enable_dbo=feature == "enable_dbo",
        num_ubatches=2,
        use_sequence_parallel_moe=feature == "use_sequence_parallel_moe",
    )
    monkeypatch.setenv(
        "VLLM_ASCEND_ENABLE_FLASHCOMM1", "1" if feature == "flashcomm1" else "0"
    )
    if feature in (None, "enable_dbo"):
        connector._validate_two_stream_attention()
    else:
        with pytest.raises(ValueError, match="two-stream"):
            connector._validate_two_stream_attention()


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "value,enabled",
    [
        (None, False),
        ("0", False),
        ("", False),
        ("false", False),
        (" False ", False),
        ("1", True),
        ("true", True),
        (" TRUE ", True),
        ("yes", False),
        ("2", False),
    ],
)
def test_two_stream_flashcomm1_matches_ascend_boolean_parsing(
    modules, monkeypatch, role, value, enabled
):
    _, module, _ = modules
    if value is None:
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", value)
    monkeypatch.setenv("AFD_ASYNC_CAM_LAYERED_GMM", "1")
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_mla=True,
    )
    if role == "ffn":

        def validate():
            _fail_if_unsupported_npu_afd_async_features(
                config,
                parse_afd_config(config),
                module.AFDAsyncExtraInfo(ffn_two_stream=True, dynamic_quant=1),
            )

    else:
        connector = module.CAMAsyncAFDConnector.__new__(module.CAMAsyncAFDConnector)
        connector.vllm_config = config
        validate = connector._validate_two_stream_attention
    if enabled:
        with pytest.raises((ValueError, RuntimeError), match="two-stream"):
            validate()
    else:
        validate()


@pytest.mark.parametrize(
    "invalid,dbo_flag",
    [
        (False, None),
        (False, "enable_dbo"),
        (False, "use_ubatching"),
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
        extra_info=connector_module.AFDAsyncExtraInfo(),
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
        vllm_config=SimpleNamespace(
            use_v2_model_runner=not invalid,
            parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
        ),
        graph_startup=AsyncCAMGraphStartup(0, 1, 2),
    )
    if dbo_flag is not None:
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
    config = npu_vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        num_ubatches=num_ubatches,
    )
    config.model_config.enforce_eager = role == "attention"
    setattr(config.parallel_config, native_flag, True)
    with pytest.raises(RuntimeError, match=error):
        _fail_if_unsupported_npu_afd_async_features(
            config, parse_afd_config(config), connector_module.AFDAsyncExtraInfo()
        )


def test_ffn_announces_service_only_on_receiver_execution(modules):
    _, connector_module, runtime = modules
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    calls: list[str | tuple[str, str | int]] = []
    runner.__dict__.update(
        prof=None,
        _graph=object(),
        _service_announced=False,
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


@pytest.mark.parametrize("capture_fails", [False, True])
@pytest.mark.parametrize("two_stream", [False, True])
def test_ffn_warmup_capture_order_and_failure_propagation(
    modules, monkeypatch, capture_fails, two_stream
):
    _, connector_module, runtime = modules
    calls: list[str | tuple[str, str | int]] = []
    startup = AsyncCAMGraphStartup(2, 2, 4)
    startup.phase = lambda stage: calls.append(stage)
    startup.fail = lambda exc: calls.append(("failure", str(exc)))

    def initialize_connector():
        calls.append("connector")
        assert startup.plan.signature == "weights"
        assert startup.plan.layer_ids == (2, 5)
        assert not startup.plan.stream_error
        assert bool(startup.plan.stream_signature) == two_stream
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
        _mega_executor=None,
        connector=SimpleNamespace(
            is_initialized=False,
            graph_startup=startup,
            extra_info=connector_module.AFDAsyncExtraInfo(ffn_two_stream=two_stream),
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
        *([] if two_stream else ["graph-resources"]),
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
        monkeypatch, "afd_plugin.v1.worker.npu.ffn_model_runner"
    )
    worker = load_host_module(monkeypatch, "afd_plugin.v1.worker.npu.ffn_worker")
    return sync_runtime, async_runtime, worker


@pytest.mark.parametrize("state", ["live", "failed", "drained"])
@pytest.mark.parametrize("two_stream", [False, True])
def test_worker_retains_resources_until_clean_exit(ffn_owners, state, two_stream):
    _, runtime, module = ffn_owners
    calls = []
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.effective_mode = "eager" if two_stream else "full"
    runner._two_stream = object() if two_stream else None
    runner.connector = SimpleNamespace(close=lambda: calls.append("connector-close"))
    runner.shutdown = lambda: calls.append("release")
    worker = module.AFDNPUFFNWorker.__new__(module.AFDNPUFFNWorker)
    worker.model_runner = runner
    worker._ffn_shutdown_event = SimpleNamespace(set=lambda: calls.append("stop"))
    worker._ffn_thread = SimpleNamespace(
        join=lambda **kw: calls.append("join"), is_alive=lambda: state == "live"
    )
    worker._ffn_loop_error = (
        RuntimeError("native failure") if state == "failed" else None
    )
    if state == "drained":
        worker.stop_ffn_server_loop()
        assert calls == ["stop", "join", "release"] and worker._ffn_thread is None
    else:
        with pytest.raises(RuntimeError, match="retain resources|loop failed"):
            worker.stop_ffn_server_loop()
        assert calls == ["stop", "join"] and worker._ffn_thread is not None
        if state == "failed":
            for _ in range(2):
                with pytest.raises(RuntimeError, match="loop failed"):
                    worker.raise_ffn_loop_error_if_any()
            with pytest.raises(RuntimeError):
                worker.start_ffn_server_loop()


@pytest.mark.parametrize("async_connector", [False, True])
def test_worker_selects_owner_and_passes_device_rank(
    ffn_owners, monkeypatch, async_connector
):
    _, _, worker = ffn_owners
    config, device, calls = object(), SimpleNamespace(type="npu", index=3), MagicMock()
    parsed = SimpleNamespace(
        connector="CAMAsyncAFDConnector" if async_connector else "CAMP2pAFDConnector"
    )
    monkeypatch.setattr(worker, "parse_afd_config", lambda *a, **k: parsed)
    monkeypatch.setattr(worker, "get_world_group", lambda: SimpleNamespace(rank=7))
    monkeypatch.setattr(worker, "validate_npu_model_runner_v2_config", calls.validate)
    monkeypatch.setattr(worker.AFDConnectorFactory, "create_connector", calls.connector)
    monkeypatch.setattr(worker, "AFDNPUFFNModelRunner", calls.sync)
    monkeypatch.setattr(worker, "AsyncNPUFFNModelRunner", calls.asynchronous)
    selected = calls.asynchronous if async_connector else calls.sync
    assert worker.create_npu_ffn_model_runner(config, device) is selected.return_value
    expected = [call.validate(config, expected_role="ffn", device_type="npu")]
    expected += (
        [
            call.connector(7, 3, config, parsed),
            call.asynchronous(config, device, calls.connector.return_value),
        ]
        if async_connector
        else [call.sync(config, device)]
    )
    assert calls.mock_calls == expected


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
    calls = MagicMock()
    monkeypatch.setattr(runtime, "validate_npu_model_runner_v2_config", calls.validate)
    monkeypatch.setattr(runtime, "fail_if_unsupported_npu_afd_features", calls.validate)
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    connector, device, model = object(), SimpleNamespace(type="npu"), object()
    connector_name = "CAMAsyncAFDConnector" if async_connector else "CAMP2pAFDConnector"
    config = npu_vllm_config(role="ffn", connector=connector_name)
    config.model_config.enforce_eager = False
    config.model_config.hf_config = SimpleNamespace(num_hidden_layers=43)
    config.scheduler_config.max_num_seqs = 6
    config.load_config = object()
    if speculative_tokens is not None:
        config.speculative_config = SimpleNamespace(
            num_speculative_tokens=speculative_tokens
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
    assert calls.validate.call_count == 2
    assert calls.pool.call_count == (0 if async_connector else 1)
    assert runner.model is None
    if async_connector:
        assert runner._graph is None and runner._graph_capture_count == 0
    else:
        assert runner._acl_graphs == {} and runner.graph_pool is calls.pool.return_value

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
