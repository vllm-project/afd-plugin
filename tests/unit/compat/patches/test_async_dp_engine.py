# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import asyncio
import copy
import importlib
import logging
import multiprocessing
import sys
import time
import types
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _config(
    *,
    connector: str = "CAMAsyncAFDConnector",
    role: str = "attention",
    async_dp: bool = True,
    is_moe: bool = True,
    data_parallel_size: int = 2,
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
            numa_bind=False,
        ),
        kv_transfer_config=None,
        needs_dp_coordinator=True,
        cache_config=SimpleNamespace(),
    )


def _install_fake_vllm_engine(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.28.0"
    vllm_v1_module = types.ModuleType("vllm.v1")
    engine_module = types.ModuleType("vllm.v1.engine")
    coordinator_module = types.ModuleType("vllm.v1.engine.coordinator")
    core_module = types.ModuleType("vllm.v1.engine.core")
    utils_module = types.ModuleType("vllm.v1.engine.utils")
    client_module = types.ModuleType("vllm.v1.engine.core_client")

    engine_module.EngineCoreRequestType = SimpleNamespace(ADD="ADD", WAKEUP="WAKEUP")

    class EngineCoreProc:
        def __init__(self, *_args, engine_index=0, **_kwargs):
            self.kind = "engine"
            self.engine_index = engine_index
            self.input_queue = SimpleNamespace(put_nowait=lambda _item: None)
            self.shutdown_state = None
            self.shutdown_called = False
            core_module.last_engine = self

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
    core_module.EngineShutdownState = SimpleNamespace(REQUESTED="REQUESTED")
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

    class EngineCoreOutputs:
        def __init__(
            self,
            *,
            scheduler_stats=None,
            engine_index=0,
            outputs=None,
            utility_output=None,
            wave_complete=None,
            start_wave=None,
        ):
            self.scheduler_stats = scheduler_stats
            self.engine_index = engine_index
            self.outputs = outputs or []
            self.utility_output = utility_output
            self.wave_complete = wave_complete
            self.start_wave = start_wave

    core_module.EngineCoreOutputs = EngineCoreOutputs

    class EngineState:
        def __init__(self):
            self.request_counts = [0, 0]

    class DPCoordinatorProc:
        def __init__(
            self,
            engine_count,
            min_stats_update_interval_ms=100,
            enable_wave_coordination=True,
        ):
            self.ctx = object()
            self.engines = [EngineState() for _ in range(engine_count)]
            self.stats_update_interval_ms = min_stats_update_interval_ms
            self.enable_wave_coordination = enable_wave_coordination

        def _get_engine_counts(self, do_copy=False):
            if do_copy:
                return [copy.copy(engine.request_counts) for engine in self.engines]
            return [engine.request_counts for engine in self.engines]

        def _send_start_wave(self, _socket, _wave, _exclude_engine_index):
            return None

    class DPCoordinator:
        def __init__(self, parallel_config, enable_wave_coordination=True):
            self.parallel_config = parallel_config
            self.enable_wave_coordination = enable_wave_coordination
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
        _num_api_servers=1,
    ):
        yield utils_module.DPCoordinator(
            vllm_config.parallel_config,
            enable_wave_coordination=True,
        )

    msgpack_codec = SimpleNamespace(
        encode=lambda value: value,
        decode=lambda value: value,
    )
    coordinator_module.copy = copy
    coordinator_module.time = time
    coordinator_module.msgspec = SimpleNamespace(msgpack=msgpack_codec)
    coordinator_module.zmq = SimpleNamespace(
        XPUB="XPUB",
        PULL="PULL",
        POLLIN="POLLIN",
        LAST_ENDPOINT="LAST_ENDPOINT",
    )
    coordinator_module.logger = logging.getLogger("fake-async-dp-coordinator")
    coordinator_module.make_zmq_socket = None
    coordinator_module.MsgpackDecoder = None
    coordinator_module.EngineState = EngineState
    coordinator_module.DPCoordinatorProc = DPCoordinatorProc

    utils_module.DPCoordinator = DPCoordinator
    utils_module.CoreEngine = CoreEngine
    utils_module.CoreEngineProcManager = CoreEngineProcManager
    utils_module.CoreEngineActorManager = CoreEngineActorManager
    utils_module.get_engine_client_zmq_addr = lambda *_args: "handshake"
    utils_module.get_open_port = lambda: 12345
    utils_module.get_open_zmq_ipc_path = lambda: "ipc"
    utils_module.zmq_socket_ctx = zmq_socket_ctx
    utils_module.zmq = SimpleNamespace(ROUTER="ROUTER")
    utils_module.wait_for_engine_startup = wait_for_engine_startup
    utils_module.logger = logging.getLogger("fake-async-dp-utils")
    utils_module.launch_core_engines = launch_core_engines
    client_module.launch_core_engines = launch_core_engines

    class DPAsyncMPClient:
        async def add_request_async(self, request):
            self._ensure_stats_update_task()
            request.current_wave = self.current_wave
            request.client_index = self.client_index
            chosen_engine = self.get_core_engine_for_request(request)
            to_await = self._send_input("ADD", request, chosen_engine)
            if not self.engines_running:
                await self.first_req_send_socket.send(("FIRST_REQ", chosen_engine))
            await to_await
            self._ensure_output_queue_task()

    client_module.msgspec = SimpleNamespace(msgpack=msgpack_codec)
    client_module.DPAsyncMPClient = DPAsyncMPClient

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.coordinator",
        coordinator_module,
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", client_module)
    return core_module, utils_module, client_module


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm_engine(monkeypatch)
    module_name = "afd_plugin.compat.patches.async_dp_engine"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _inspect_spawned_coordinator_binding(result_queue):
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_fake_vllm_engine(monkeypatch)
        module_name = "afd_plugin.compat.patches.async_dp_engine"
        sys.modules.pop(module_name, None)
        patch_module = importlib.import_module(module_name)
        coordinator = sys.modules["vllm.v1.engine.coordinator"].DPCoordinatorProc
        result_queue.put(
            (
                coordinator.run_coordinator is patch_module.run_coordinator,
                coordinator.process_input_socket is patch_module.process_input_socket,
                coordinator.run_coordinator.__module__,
            )
        )
    finally:
        monkeypatch.undo()


def test_spawn_import_installs_plugin_owned_coordinator_target():
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_inspect_spawned_coordinator_binding,
        args=(result_queue,),
    )
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 0
    assert result_queue.get(timeout=5) == (
        True,
        True,
        "afd_plugin.compat.patches.async_dp_engine",
    )
    result_queue.close()


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


def test_async_dp_engine_patch_reload_is_idempotent(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]
    importlib.reload(patch_module)

    core_module.EngineCoreProc.run_engine_core(
        vllm_config=_config(connector="CAMP2pAFDConnector", async_dp=False),
        dp_rank=1,
    )
    engine = core_module.last_engine

    assert engine.kind == "dp"


def test_async_dp_engine_patch_rebinds_after_backend_override(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]

    core_module.EngineCoreProc.run_engine_core = staticmethod(lambda **_kwargs: None)

    assert patch_module.apply_async_dp_engine_patch() is True
    assert core_module.EngineCoreProc.run_engine_core is patch_module.run_engine_core


def test_async_dp_coordinator_disables_wave_coordination(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    coordinator_module = sys.modules["vllm.v1.engine.coordinator"]
    utils_module = sys.modules["vllm.v1.engine.utils"]
    client_module = sys.modules["vllm.v1.engine.core_client"]

    addresses = SimpleNamespace()
    with utils_module.launch_core_engines(
        _config(),
        object,
        False,
        addresses,
    ) as launch_result:
        _, coordinator, yielded_addresses, _ = launch_result
        assert coordinator.enable_wave_coordination is False
        assert yielded_addresses is addresses

    assert client_module.launch_core_engines is utils_module.launch_core_engines
    assert (
        coordinator_module.DPCoordinatorProc.process_input_socket
        is patch_module.process_input_socket
    )
    assert (
        coordinator_module.DPCoordinatorProc.run_coordinator
        is patch_module.run_coordinator
    )
    assert patch_module.run_coordinator.__module__ == (
        "afd_plugin.compat.patches.async_dp_engine"
    )


def test_newer_dev_vllm_keeps_native_coordinator(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    vllm_module = sys.modules["vllm"]
    coordinator_module = sys.modules["vllm.v1.engine.coordinator"]

    def native_run_coordinator(*_args, **_kwargs):
        return None

    def native_process_input_socket(*_args, **_kwargs):
        return None

    coordinator_module.DPCoordinatorProc.run_coordinator = staticmethod(
        native_run_coordinator
    )
    coordinator_module.DPCoordinatorProc.process_input_socket = (
        native_process_input_socket
    )
    vllm_module.__version__ = "0.27.0.dev1"

    assert patch_module.apply_async_dp_engine_patch() is True
    assert (
        coordinator_module.DPCoordinatorProc.run_coordinator is native_run_coordinator
    )
    assert (
        coordinator_module.DPCoordinatorProc.process_input_socket
        is native_process_input_socket
    )


@pytest.mark.parametrize(
    ("enable_wave_coordination", "expected_timeouts", "expects_warning"),
    (
        (False, [4000, 0, 0], False),
        (True, [4000, 50, 50], True),
    ),
)
def test_dp_coordinator_separates_independent_and_lockstep_stats(
    monkeypatch,
    caplog,
    enable_wave_coordination,
    expected_timeouts,
    expects_warning,
):
    _load_patch_module(monkeypatch)
    coordinator_module = sys.modules["vllm.v1.engine.coordinator"]
    core_module = sys.modules["vllm.v1.engine.core"]

    class CoordinatorStoppedError(Exception):
        pass

    class Socket:
        def __init__(self, received=(), *, stop_on_send=False):
            self.received = deque(received)
            self.sent = []
            self.stop_on_send = stop_on_send

        def recv(self):
            return self.received.popleft()

        def send(self, message):
            self.sent.append(message)
            if self.stop_on_send:
                raise CoordinatorStoppedError

    first_stats = SimpleNamespace(
        step_counter=10,
        current_wave=4,
        num_waiting_reqs=3,
        num_running_reqs=1,
    )
    stale_global_step_stats = SimpleNamespace(
        step_counter=1,
        current_wave=0,
        num_waiting_reqs=4,
        num_running_reqs=2,
    )
    output_front = Socket(stop_on_send=True)
    output_back = Socket(
        (
            core_module.EngineCoreOutputs(
                scheduler_stats=first_stats,
                engine_index=0,
            ),
            core_module.EngineCoreOutputs(
                scheduler_stats=stale_global_step_stats,
                engine_index=1,
            ),
        )
    )
    publish_back = Socket((b"\x01", b"\x01"))
    sockets = {
        "front": output_front,
        "outputs": output_back,
        "engines": publish_back,
    }

    @contextmanager
    def make_zmq_socket(*, path, **_kwargs):
        yield sockets[path]

    poll_timeouts = []
    event_batches = deque(
        (
            [(output_back, "POLLIN")],
            [(output_back, "POLLIN")],
            [],
        )
    )

    class Poller:
        def register(self, _socket, _event):
            return None

        def poll(self, timeout):
            poll_timeouts.append(timeout)
            return event_batches.popleft()

    class Decoder:
        def __init__(self, _output_type):
            pass

        def decode(self, value):
            return value

    coordinator_module.make_zmq_socket = make_zmq_socket
    coordinator_module.MsgpackDecoder = Decoder
    coordinator_module.zmq.Poller = Poller
    coordinator_module.time = SimpleNamespace(time=lambda: 1.0)

    coordinator = coordinator_module.DPCoordinatorProc(
        engine_count=2,
        enable_wave_coordination=enable_wave_coordination,
    )
    with (
        caplog.at_level(logging.WARNING, logger="fake-async-dp-coordinator"),
        pytest.raises(CoordinatorStoppedError),
    ):
        coordinator.process_input_socket("front", "outputs", "engines")

    assert output_front.sent == [([[3, 1], [4, 2]], 0, False)]
    assert poll_timeouts == expected_timeouts
    assert ("out-of-order update" in caplog.text) is expects_warning


def test_async_dp_client_skips_first_req(monkeypatch):
    _load_patch_module(monkeypatch)
    client_module = sys.modules["vllm.v1.engine.core_client"]

    class Sender:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(message)

    client = client_module.DPAsyncMPClient()
    client.vllm_config = _config()
    client.current_wave = 7
    client.client_index = 3
    client.engines_running = False
    client.first_req_send_socket = Sender()
    client.stats_ready = False
    client.output_ready = False
    client.sent_inputs = []
    client._ensure_stats_update_task = lambda: setattr(client, "stats_ready", True)
    client._ensure_output_queue_task = lambda: setattr(client, "output_ready", True)
    client.get_core_engine_for_request = lambda request: 1

    async def send_input(request_type, request, engine):
        client.sent_inputs.append((request_type, request, engine))

    client._send_input = send_input
    request = SimpleNamespace()

    asyncio.run(client.add_request_async(request))

    assert request.current_wave == 7
    assert request.client_index == 3
    assert client.first_req_send_socket.messages == []
    assert client.stats_ready is True
    assert client.output_ready is True


def test_non_afd_client_preserves_first_req_wakeup(monkeypatch):
    _load_patch_module(monkeypatch)
    client_module = sys.modules["vllm.v1.engine.core_client"]

    class Sender:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(message)

    client = client_module.DPAsyncMPClient()
    client.vllm_config = _config()
    client.vllm_config.additional_config = {}
    client.current_wave = 2
    client.client_index = 0
    client.engines_running = False
    client.first_req_send_socket = Sender()
    client._ensure_stats_update_task = lambda: None
    client._ensure_output_queue_task = lambda: None
    client.get_core_engine_for_request = lambda _request: 1

    async def send_input(_request_type, _request, _engine):
        return None

    client._send_input = send_input
    asyncio.run(client.add_request_async(SimpleNamespace()))

    assert client.first_req_send_socket.messages == [("FIRST_REQ", 1)]
