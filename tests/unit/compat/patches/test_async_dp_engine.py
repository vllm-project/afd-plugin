# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import importlib
import logging
import sys
import types
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


def _config(
    *,
    connector: str = "CAMAsyncAFDConnector",
    role: str = "attention",
    async_dp: bool = True,
    is_moe: bool = True,
    data_parallel_size: int = 2,
    device_type: str = "npu",
):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": connector,
                "role": role,
                "async": async_dp,
            },
        },
        model_config=SimpleNamespace(is_moe=is_moe, multimodal_config=None),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size,
            data_parallel_rank=0,
            data_parallel_rank_local=None,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_rpc_port=0,
            local_engines_only=False,
            data_parallel_backend="mp",
            enable_elastic_ep=False,
            _coord_store_port=None,
            numa_bind=False,
        ),
        kv_transfer_config=None,
        needs_dp_coordinator=True,
        cache_config=SimpleNamespace(),
        device_config=SimpleNamespace(device_type=device_type),
        logging_config=None,
        watchdog_config=SimpleNamespace(),
    )


def _install_fake_vllm_engine(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.34.0"
    vllm_v1_module = types.ModuleType("vllm.v1")
    engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    utils_module = types.ModuleType("vllm.v1.engine.utils")
    client_module = types.ModuleType("vllm.v1.engine.core_client")
    serve_module = types.ModuleType("vllm.entrypoints.cli.serve")

    engine_module.EngineCoreRequestType = SimpleNamespace(ADD="ADD", WAKEUP="WAKEUP")

    class EngineCoreProc:
        def __init__(self, *_args, engine_index=0, **_kwargs):
            self.kind = "engine"
            self.engine_index = engine_index
            self.input_queue = SimpleNamespace(put_nowait=lambda _item: None)
            self.shutdown_state = None
            self.shutdown_called = False
            core_module.last_engine = self

        @staticmethod
        def run_engine_core(*_args, **_kwargs):
            return None

        def run_busy_loop(self):
            return None

        def shutdown(self):
            self.shutdown_called = True

        def _send_engine_dead(self):
            self.sent_engine_dead = True

    class DPEngineCoreProc(EngineCoreProc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.kind = "dp"

    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.logger = logging.getLogger("fake-async-dp-engine")
    core_module.last_engine = None
    core_module.maybe_register_config_serialize_by_value = lambda: None
    core_module.set_process_title = lambda _title: None
    core_module.maybe_init_worker_tracer = lambda *_args: None
    core_module.decorate_logs = lambda: None
    core_module.start_watch_dog = lambda *_args: None
    core_module.watchdog_failures = []
    core_module.get_watch_dog = lambda: SimpleNamespace(
        handle_failure=core_module.watchdog_failures.append,
    )
    core_module.EngineShutdownState = SimpleNamespace(
        REQUESTED="REQUESTED",
        SHUTTING_DOWN="SHUTTING_DOWN",
    )
    core_module.signal = SimpleNamespace(
        SIGTERM="SIGTERM",
        SIGINT="SIGINT",
        SIG_DFL="SIG_DFL",
        signal=lambda *_args: None,
    )

    class SignalCallback:
        def __init__(self, callback):
            self.callback = callback
            self.stopped = False

        def trigger(self):
            self.callback()

        def stop(self):
            self.stopped = True

    core_module.SignalCallback = SignalCallback

    @dataclass
    class CoreEngineLaunch:
        engine_manager: object | None
        coordinator: object | None
        addresses: object
        tensor_queue: object | None
        watched_frontend_processes: tuple = field(default=())

    class DPCoordinator:
        def __init__(
            self, parallel_config, enable_wave_coordination=True, logging_config=None
        ):
            self.parallel_config = parallel_config
            self.enable_wave_coordination = enable_wave_coordination
            self.logging_config = logging_config
            self.proc = SimpleNamespace(pid=123)

        def get_engine_socket_addresses(self):
            return "coord-input", "coord-output"

        def get_stats_publish_address(self):
            return "stats-pub"

    class CoreEngine:
        def __init__(self, index, local):
            self.index = index
            self.local = local

    class CoreEngineProcManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def sentinels(self):
            return []

    class CoreEngineActorManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @contextmanager
    def zmq_socket_ctx(*_args, **_kwargs):
        yield SimpleNamespace()

    def wait_for_engine_startup(*_args, **_kwargs):
        return None

    @contextmanager
    def launch_core_engines(
        vllm_config,
        _executor_class,
        _log_stats,
        _addresses,
    ):
        yield utils_module.CoreEngineLaunch(
            None,
            utils_module.DPCoordinator(
                vllm_config.parallel_config,
                enable_wave_coordination=True,
            ),
            _addresses,
            None,
        )

    utils_module.CoreEngineLaunch = CoreEngineLaunch
    utils_module.DPCoordinator = DPCoordinator
    utils_module.CoreEngine = CoreEngine
    utils_module.CoreEngineProcManager = CoreEngineProcManager
    utils_module.CoreEngineActorManager = CoreEngineActorManager
    utils_module.get_engine_client_zmq_addr = lambda *_args: "handshake"
    utils_module.get_open_zmq_ipc_path = lambda: "ipc"
    utils_module.zmq_socket_ctx = zmq_socket_ctx
    utils_module.zmq = SimpleNamespace(ROUTER="ROUTER")
    utils_module.wait_for_engine_startup = wait_for_engine_startup
    utils_module.logger = logging.getLogger("fake-async-dp-utils")
    utils_module.launch_core_engines = launch_core_engines
    client_module.launch_core_engines = launch_core_engines
    serve_module.launch_core_engines = launch_core_engines

    class DPAsyncMPClient:
        async def add_request_async(self, request):
            return None

    client_module.DPAsyncMPClient = DPAsyncMPClient

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", client_module)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.serve", serve_module)
    distributed_module = types.ModuleType("vllm.distributed")
    distributed_utils = types.ModuleType("vllm.distributed.utils")
    distributed_utils.create_tcp_store = lambda *_args, **_kwargs: SimpleNamespace(
        port=41234
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed.utils", distributed_utils)
    return core_module, utils_module, client_module


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm_engine(monkeypatch)
    return _import_patch_module()


def _import_patch_module():
    module_name = "afd_plugin.compat.patches.async_dp_engine"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_async_dp_attention_uses_regular_engine_core(monkeypatch):
    _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]

    core_module.EngineCoreProc.run_engine_core(
        vllm_config=_config(),
        dp_rank=1,
    )
    engine = core_module.last_engine

    assert engine.kind == "engine"
    assert engine.engine_index == 1
    assert engine.shutdown_called is True


def test_async_dp_engine_patch_preserves_non_async_moe_dp(monkeypatch):
    _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]

    core_module.EngineCoreProc.run_engine_core(
        vllm_config=_config(connector="CAMP2pAFDConnector", async_dp=False),
        dp_rank=1,
    )
    engine = core_module.last_engine

    assert engine.kind == "dp"


@pytest.mark.parametrize("async_dp", [False, True])
@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_engine_honors_shutdown_received_during_construction(
    monkeypatch, async_dp, device_type
):
    patch = _load_patch_module(monkeypatch)
    core = sys.modules["vllm.v1.engine.core"]
    handlers: dict[int, Callable[[int, None], None] | int] = {}
    signals = SimpleNamespace(
        SIGTERM=15,
        SIGINT=2,
        SIG_DFL=0,
        Signals=lambda _number: SimpleNamespace(name="SIGTERM"),
        signal=lambda number, handler: handlers.__setitem__(number, handler),
    )
    base = patch.EngineCoreProc if async_dp else core.DPEngineCoreProc

    def interrupted_engine(*args, **kwargs):
        handler = handlers[signals.SIGTERM]
        assert callable(handler)
        handler(signals.SIGTERM, None)
        return base(*args, **kwargs)

    monkeypatch.setattr(core, "signal", signals)
    if async_dp:
        monkeypatch.setattr(patch, "EngineCoreProc", interrupted_engine)
    else:
        monkeypatch.setattr(core, "DPEngineCoreProc", interrupted_engine)
    patch.run_engine_core(
        vllm_config=_config(async_dp=async_dp, device_type=device_type)
    )

    assert core.last_engine.shutdown_state == core.EngineShutdownState.REQUESTED
    assert core.last_engine.shutdown_called is True
    assert core.watchdog_failures == ["SIGTERM"]
    assert handlers[signals.SIGTERM] == signals.SIG_DFL
    assert handlers[signals.SIGINT] == signals.SIG_DFL


def test_async_dp_engine_patch_preserves_moe_dp_without_afd_config(monkeypatch):
    _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]
    config = _config()
    config.additional_config = {}

    core_module.EngineCoreProc.run_engine_core(
        vllm_config=config,
        dp_rank=1,
    )
    engine = core_module.last_engine

    assert engine.kind == "dp"


def test_async_dp_engine_patch_rebinds_after_backend_override(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]

    core_module.EngineCoreProc.run_engine_core = staticmethod(lambda **_kwargs: None)

    assert patch_module.apply_async_dp_engine_patch() is True
    core_module.EngineCoreProc.run_engine_core(vllm_config=_config(), dp_rank=1)
    assert core_module.last_engine.kind == "engine"
    assert core_module.last_engine.engine_index == 1
    assert core_module.last_engine.shutdown_called is True


@pytest.mark.parametrize(
    "entrypoint",
    ["vllm.v1.engine.core_client", "vllm.entrypoints.cli.serve"],
)
def test_async_dp_coordinator_disables_wave_coordination(monkeypatch, entrypoint):
    _load_patch_module(monkeypatch)
    entrypoint_module = sys.modules[entrypoint]

    addresses = SimpleNamespace()
    with entrypoint_module.launch_core_engines(
        _config(),
        object,
        False,
        addresses,
    ) as launch:
        assert launch.coordinator.enable_wave_coordination is False
        assert launch.addresses is addresses


def test_non_afd_launch_keeps_upstream_wave_coordination(monkeypatch):
    _load_patch_module(monkeypatch)
    utils_module = sys.modules["vllm.v1.engine.utils"]

    addresses = SimpleNamespace()
    with utils_module.launch_core_engines(
        _config(async_dp=False),
        object,
        False,
        addresses,
    ) as launch:
        assert launch.coordinator.enable_wave_coordination is True


@pytest.mark.parametrize(
    "async_dp, role, wave_coordination",
    [(False, "attention", True), (True, "attention", False), (True, "ffn", True)],
)
def test_npu_launch_uses_target_coordinator_contract(
    monkeypatch, async_dp, role, wave_coordination
):
    patch = _load_patch_module(monkeypatch)
    config = _config(async_dp=async_dp, role=role)
    addresses = SimpleNamespace()
    with patch.launch_core_engines(config, object, False, addresses) as launch:
        assert launch.coordinator.enable_wave_coordination is wave_coordination
        assert launch.addresses.coordinator_input == "coord-input"
        assert launch.addresses.coordinator_output == "coord-output"
        assert launch.addresses.frontend_stats_publish_address == "stats-pub"
        assert config.parallel_config._coord_store_port == 41234


@pytest.mark.parametrize(
    "elastic_ep, needs_coordinator, extends_store_lifetime",
    [
        (False, True, False),
        (False, False, True),
        (True, True, True),
        (True, False, True),
    ],
)
def test_dp_launch_preserves_coordination_store_lifetime(
    monkeypatch, elastic_ep, needs_coordinator, extends_store_lifetime
):
    patch = _load_patch_module(monkeypatch)
    references = []

    class Store:
        port = 41234

    def create_store(*_args, **_kwargs):
        store = Store()
        references.append(weakref.ref(store))
        return store

    monkeypatch.setattr(
        sys.modules["vllm.distributed.utils"], "create_tcp_store", create_store
    )
    config = _config(async_dp=False)
    config.needs_dp_coordinator = needs_coordinator
    config.parallel_config.enable_elastic_ep = elastic_ep
    with patch.launch_core_engines(config, object, False, SimpleNamespace()) as launch:
        assert config.parallel_config._coord_store_port == Store.port
        assert references[0]() is not None
    assert (references[0]() is not None) is extends_store_lifetime
    del launch
    assert references[0]() is None


@pytest.mark.parametrize("version", ["0.32.0", "0.29.0"])
def test_non_target_vllm_release_keeps_native_engine_bindings(monkeypatch, version):
    core_module, utils_module, client_module = _install_fake_vllm_engine(monkeypatch)
    sys.modules["vllm"].__version__ = version
    native_run_engine_core = core_module.EngineCoreProc.run_engine_core
    native_launch = utils_module.launch_core_engines
    native_add_request = client_module.DPAsyncMPClient.add_request_async

    patch_module = _import_patch_module()

    assert patch_module.apply_async_dp_engine_patch() is False
    assert core_module.EngineCoreProc.run_engine_core is native_run_engine_core
    assert utils_module.launch_core_engines is native_launch
    assert client_module.DPAsyncMPClient.add_request_async is native_add_request


def test_target_vllm_keeps_native_dp_request_admission(monkeypatch):
    _, _, client_module = _install_fake_vllm_engine(monkeypatch)
    native_add_request = client_module.DPAsyncMPClient.add_request_async

    patch = _import_patch_module()

    assert patch.apply_async_dp_engine_patch() is True
    assert client_module.DPAsyncMPClient.add_request_async is native_add_request
