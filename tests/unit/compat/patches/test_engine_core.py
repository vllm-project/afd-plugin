# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import importlib
import logging
import queue
import sys
import types
from enum import IntEnum
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.fakes.kv_cache import KVCacheExecutor


class _EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1


def _install_fake_vllm_core(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str = "cuda",
):
    # Reload a fresh patch module against the fake, leaving any existing real
    # module and its already-imported references untouched. Undo restores both
    # the sys.modules entry and the package attribute, including when absent.
    patch_name = "afd_plugin.compat.patches.engine_core"
    patch_package = importlib.import_module("afd_plugin.compat.patches")
    monkeypatch.setitem(sys.modules, patch_name, None)
    monkeypatch.delitem(sys.modules, patch_name)
    monkeypatch.setattr(patch_package, "engine_core", None, raising=False)

    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.34.0"
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type=device_type)
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    plugins_module = types.ModuleType("vllm.plugins")

    def load_general_plugins():
        return None

    plugins_module.load_general_plugins = load_general_plugins

    class EngineCoreRequestType:
        UTILITY = "utility"

    class UtilityResult:
        def __init__(self, result):
            self.result = result

    class UtilityOutput:
        def __init__(self, call_id):
            self.call_id = call_id
            self.failure_message = None
            self.result = None

    class EngineCoreOutputs:
        def __init__(self, *, utility_output):
            self.utility_output = utility_output

    class EngineCore:
        def __init__(
            self,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback=None,
            include_finished_set=False,
        ):
            self.vllm_config = vllm_config
            self.original_init_called = True
            self.model_executor = executor_class(vllm_config)
            self.structured_output_manager = object()
            core_module.measure_encoder_output_widths(vllm_config, self.collective_rpc)
            self._initialize_kv_caches(vllm_config)
            self.prefix_hash_config = (
                SimpleNamespace(
                    hash_algo=vllm_config.cache_config.prefix_caching_hash_algo,
                    num_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
                )
                if vllm_config.cache_config.enable_prefix_caching
                else None
            )

        def shutdown(self):
            self.original_shutdown_called = True

        def _initialize_kv_caches(self, vllm_config):
            del vllm_config
            self.original_initialize_kv_caches_called = True

        def step(self):
            return None

        def step_with_batch_queue(self):
            return None

        def get_supported_tasks(self):
            supported_tasks = self.model_executor.supported_tasks
            self._log_pooler_config(supported_tasks)
            return supported_tasks

        def _log_pooler_config(self, supported_tasks):
            del supported_tasks
            if self._pooler_config_logged:
                return

            model_config = self.vllm_config.model_config
            pooler_config = model_config.pooler_config
            if (
                self.vllm_config.parallel_config.data_parallel_rank_local
                or model_config.runner_type != "pooling"
                or pooler_config is None
            ):
                return

        def reset_mm_cache(self):
            if self.scheduler.has_unfinished_requests():
                return
            if self.mm_receiver_cache is not None:
                self.mm_receiver_cache.clear_cache()
            self.model_executor.reset_mm_cache()

        def _handle_client_request(self, request_type, request):
            if request_type != EngineCoreRequestType.UTILITY:
                raise AssertionError(f"unexpected request type: {request_type}")

            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                result = getattr(self, method_name)(*args)
                output.result = UtilityResult(result)
            except Exception as exc:
                output.failure_message = f"Call to {method_name} method failed: {exc}"
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )

        def collective_rpc(self, _method, *args, **kwargs):
            return []

        def _initialize_effective_attention_block_size(self):
            self.vllm_config.cache_config.effective_attention_block_size = None

    class EngineCoreProc(EngineCore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_queue: queue.Queue[Any] = queue.Queue()
            self.output_queue: queue.Queue[Any] = queue.Queue()

        def _make_ready_response(self):
            return SimpleNamespace(prefix_hash_config=self.prefix_hash_config)

        def run_busy_loop(self):
            self.original_run_busy_loop_called = True

    class DPEngineCoreProc(EngineCoreProc):
        pass

    def instrument(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def fault_tolerant_wrapper(busy_loop_func):
        # Mirrors the upstream wrapper contract that matters here: SystemExit
        # propagates, and exceptions re-raise while fault tolerance is off.
        def run_with_fault_tolerance(self):
            while True:
                try:
                    busy_loop_func(self)
                except SystemExit:
                    raise
                except Exception:
                    # Fault tolerance stays disabled in the fake: re-raise.
                    if not getattr(self, "enable_fault_tolerance", False):
                        raise
                    raise AssertionError(
                        "fault tolerance unsupported in fake"
                    ) from None

        return run_with_fault_tolerance

    def get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory, global_kv_cache_groups
    ):
        del vllm_config, available_gpu_memory, global_kv_cache_groups
        return kv_cache_specs

    def generate_scheduler_kv_cache_config(kv_cache_configs):
        del kv_cache_configs
        return SimpleNamespace(num_blocks=0, kv_cache_groups=[])

    def update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config):
        vllm_config.cache_config.kv_cache_capacity_updated = True
        del scheduler_kv_cache_config

    def resolve_kv_cache_layout(vllm_config, supported_layouts, specs):
        del vllm_config, specs
        return SimpleNamespace(name=supported_layouts[0])

    def register_all_kvcache_specs(_vllm_config):
        return None

    def resolve_kv_cache_block_sizes(_kv_cache_config, _vllm_config):
        return 16, 16

    core_module.EngineCore = EngineCore
    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.EngineCoreRequestType = EngineCoreRequestType
    core_module.EngineShutdownState = _EngineShutdownState
    core_module.VLLM_VERSION = vllm_module.__version__
    core_module.logger = logging.getLogger("fake-vllm-core")
    core_module.logger.info_once = lambda *args, **kwargs: None
    core_module.envs = SimpleNamespace(VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=False)
    core_module.measure_encoder_output_widths = lambda _config, _rpc: None
    core_module.instrument = instrument
    core_module.fault_tolerant_wrapper = fault_tolerant_wrapper
    core_module.get_watch_dog = lambda: SimpleNamespace(feed=lambda: None)
    core_module.get_global_kv_cache_groups = lambda *_args: []
    core_module.get_kv_cache_configs = get_kv_cache_configs
    core_module.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
    core_module.update_kv_cache_capacity = update_kv_cache_capacity
    core_module.resolve_kv_cache_layout = resolve_kv_cache_layout
    core_module.register_all_kvcache_specs = register_all_kvcache_specs
    core_module.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
    core_module.resolve_cache_hit_alignment_tokens = lambda *_args: 32

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", vllm_engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins_module)
    return core_module


def _load_patch_module() -> types.ModuleType:
    module_name = "afd_plugin.compat.patches.engine_core"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _config(role: str | None, *, async_dp: bool = False, device_type: str = "cuda"):
    class Scheduler:
        connector = None
        ec_connector = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.request_counts = (0, 0)

        def get_kv_connector(self):
            return None

        def get_request_counts(self):
            return self.request_counts

        def get_kv_cache_usage(self):
            return 0.0

        def shutdown(self):
            self.shutdown_called = True

    scheduler_config = SimpleNamespace(
        async_scheduling=False,
        enable_chunked_prefill=False,
        get_scheduler_cls=lambda: Scheduler,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        enable_prefix_caching=False,
        enable_extensible_kv_cache=False,
        prefix_caching_hash_algo="builtin",
        num_gpu_blocks=None,
        kv_cache_layout="layered",
    )
    parallel_config = SimpleNamespace(
        data_parallel_rank_local=0,
        cp_kv_cache_interleave_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    model_config = SimpleNamespace(
        max_model_len=8,
        runner_type="generate",
        is_diffusion=False,
        pooler_config=None,
    )

    def validate_block_size():
        cache_config.validated = True

    afd_config: dict[str, object] = {"role": role}
    if async_dp:
        afd_config.update(
            {
                "async": True,
                "connector": "CAMAsyncAFDConnector",
            }
        )

    return SimpleNamespace(
        additional_config={} if role is None else {"afd": afd_config},
        parallel_config=parallel_config,
        device_config=SimpleNamespace(device_type=device_type),
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        model_config=model_config,
        speculative_config=None,
        ec_transfer_config=None,
        kv_transfer_config=None,
        adjust_dcp_kv_cache_interleave_size=lambda _groups: None,
        is_mm_encoder_only=False,
        max_concurrent_batches=1,
        compilation_config=SimpleNamespace(
            compilation_time=0.0,
            encoder_compilation_time=0.0,
        ),
        validate_block_size=validate_block_size,
    )


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_engine_core_patch_skips_kv_scheduler_init_for_ffn(monkeypatch, device_type):
    core_module = _install_fake_vllm_core(monkeypatch, device_type=device_type)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    class Executor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.calls = []

        def register_failure_callback(self, callback):
            self.callback = callback

        def collective_rpc(self, method):
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    from afd_plugin.compat.patches.engine_core import _AFDFFNNoopScheduler

    config = _config("ffn", device_type=device_type)
    engine = core_module.EngineCore(config, Executor, log_stats=True)

    assert engine.afd_config.role == "ffn"
    assert isinstance(engine.scheduler, _AFDFFNNoopScheduler)
    assert engine.structured_output_manager is None
    assert isinstance(engine.model_executor, Executor)
    assert config.cache_config.num_gpu_blocks == 0
    assert config.cache_config.num_cpu_blocks == 0


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_late_loaded_ffn_returns_empty_kv_without_executor(monkeypatch, device_type):
    core = _install_fake_vllm_core(monkeypatch, device_type=device_type)
    _load_patch_module()
    config = _config("ffn", device_type=device_type)
    config.cache_config.enable_prefix_caching = True
    config.scheduler_config.enable_chunked_prefill = True

    result = core.EngineCore._initialize_kv_caches(SimpleNamespace(), config)

    assert result.kv_cache_groups == []
    assert config.cache_config.num_gpu_blocks == 0
    assert config.cache_config.num_cpu_blocks == 0
    assert config.cache_config.enable_prefix_caching is False
    assert config.scheduler_config.enable_chunked_prefill is False
    scheduler = config.scheduler_config.get_scheduler_cls()()
    assert scheduler.get_kv_connector() is None
    assert scheduler.get_ec_connector() is None


def test_ffn_noop_scheduler_exposes_native_ready_protocol(monkeypatch):
    """Both plugin-loading orders must survive the native ready handshake.

    When the patches are installed before FFN EngineCore construction (fork
    inheritance, in-process engines), the FFN divert at the top of
    ``EngineCore.__init__`` runs and ``_make_ready_response`` later calls
    ``scheduler.get_kv_event_publisher_config()`` unconditionally. The
    early-loaded path therefore needs the same noop scheduler the
    late-loaded path already gets.
    """
    _install_fake_vllm_core(monkeypatch)
    _load_patch_module()
    from afd_plugin.compat.patches.engine_core import _AFDFFNNoopScheduler

    noop = _AFDFFNNoopScheduler()
    assert noop.connector is None
    assert noop.ec_connector is None
    assert noop.get_kv_connector() is None
    assert noop.get_ec_connector() is None
    assert noop.get_kv_event_publisher_config() is None
    assert noop.has_requests() is False
    assert noop.has_unfinished_requests() is False


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_attention_uses_native_constructor_with_kv_layout_and_warmup(
    monkeypatch, device_type
):
    core = _install_fake_vllm_core(monkeypatch, device_type=device_type)
    _load_patch_module()
    config = _config("attention", device_type=device_type)
    config.parallel_config.cp_kv_cache_interleave_size = 64

    engine = core.EngineCore(config, KVCacheExecutor, log_stats=False)

    assert engine.original_init_called
    assert engine.model_executor.warmup_called
    assert engine.model_executor.kv_cache_layout == "layered"
    assert engine.model_executor.cp_kv_cache_interleave_size == 64


@pytest.mark.parametrize("role", [None, "attention"])
def test_v034_non_ffn_ready_response_keeps_native_prefix_hash_config(monkeypatch, role):
    core = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()
    # This test exercises native initialization and the ready payload. KV
    # allocation is covered separately and does not participate in the check.
    monkeypatch.setattr(
        core.EngineCore,
        "_initialize_kv_caches",
        lambda self, config: SimpleNamespace(kv_cache_groups=[]),
    )
    config = _config(role)
    config.cache_config.enable_prefix_caching = True
    config.num_hash_lookahead_tokens = 3

    engine = core.EngineCoreProc(config, KVCacheExecutor, log_stats=False)
    assert engine._make_ready_response().prefix_hash_config == SimpleNamespace(
        hash_algo="builtin", num_lookahead_tokens=3
    )


@pytest.mark.parametrize("engine_type", ["EngineCoreProc", "DPEngineCoreProc"])
def test_v034_non_ffn_keeps_native_busy_loop_exit(monkeypatch, engine_type):
    core = _install_fake_vllm_core(monkeypatch)

    def exit_native_loop(self):
        raise SystemExit(engine_type)

    engine_class = getattr(core, engine_type)
    monkeypatch.setattr(engine_class, "run_busy_loop", exit_native_loop)
    engine = engine_class(_config(None), KVCacheExecutor, log_stats=False)
    _load_patch_module()
    engine._handle_shutdown = lambda: False

    with pytest.raises(SystemExit, match=engine_type):
        engine.run_busy_loop()


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("role", [None, "attention"])
def test_native_kv_config_preserves_worker_and_scheduler_alignment(
    monkeypatch, device_type, role
):
    core = _install_fake_vllm_core(monkeypatch, device_type=device_type)
    _load_patch_module()
    config = _config(role, device_type=device_type)
    worker_config = SimpleNamespace(kv_cache_groups=[], num_blocks=0)
    monkeypatch.setattr(core, "get_kv_cache_configs", lambda *_args: [worker_config])
    executor = KVCacheExecutor(config)
    executor.kv_cache_specs = [{}]

    result = core.EngineCore._initialize_kv_caches(
        SimpleNamespace(model_executor=executor), config
    )

    assert result.hash_block_size == 16
    assert result.cache_hit_alignment_tokens == 32
    assert executor.kv_cache_configs[0].hash_block_size == 16
    assert executor.kv_cache_configs[0].cache_hit_alignment_tokens == 32
    assert executor.kv_cache_configs[0].kv_cache_layout == "layered"


@pytest.mark.parametrize("role", [None, "attention"])
def test_npu_measures_encoder_widths_before_kv_allocation(monkeypatch, role):
    core = _install_fake_vllm_core(monkeypatch, device_type="npu")
    _load_patch_module()
    config = _config(role, device_type="npu")
    config.ec_transfer_config = SimpleNamespace(
        is_ec_producer=True, mm_encoder_output_widths={}
    )
    widths = {"image": 1024}

    def measure_encoder_output_widths(config, _rpc):
        config.ec_transfer_config.mm_encoder_output_widths = widths

    def allocate_kv_cache(*_args):
        # The native EC producer must dummy-encode before KV memory allocation.
        assert config.ec_transfer_config.mm_encoder_output_widths == widths
        return []

    monkeypatch.setattr(
        core, "measure_encoder_output_widths", measure_encoder_output_widths
    )
    monkeypatch.setattr(core, "get_kv_cache_configs", allocate_kv_cache)

    engine = core.EngineCore(config, KVCacheExecutor, log_stats=False)

    assert config.ec_transfer_config.mm_encoder_output_widths == widths
    assert engine.model_executor.warmup_called


@pytest.mark.parametrize("engine_type", ["EngineCoreProc", "DPEngineCoreProc"])
def test_engine_core_patch_runs_and_stops_ffn_loop(monkeypatch, engine_type):
    core_module = _install_fake_vllm_core(monkeypatch, device_type="npu")
    _load_patch_module()

    class Executor:
        def __init__(self, vllm_config):
            self.calls = []

        def collective_rpc(self, method):
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    engine_class = getattr(core_module, engine_type)
    engine = engine_class(_config("ffn", device_type="npu"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING
    watchdog_feeds = []
    monkeypatch.setattr(
        core_module,
        "get_watch_dog",
        lambda: SimpleNamespace(feed=lambda: watchdog_feeds.append(True)),
    )

    from afd_plugin.compat.patches import engine_core as engine_core_patch

    def request_shutdown(_seconds):
        engine.shutdown_state = _EngineShutdownState.REQUESTED

    monkeypatch.setattr(engine_core_patch.time, "sleep", request_shutdown)

    with pytest.raises(SystemExit):
        engine.run_busy_loop()

    assert watchdog_feeds == [True]
    assert engine.model_executor.calls == [
        "start_ffn_server_loop",
        "raise_ffn_loop_error_if_any",
        "stop_ffn_server_loop",
    ]


def test_ffn_connector_loop_replies_to_startup_utility_requests(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()

    class Executor:
        supported_tasks = ()

        def __init__(self, vllm_config):
            del vllm_config

        def collective_rpc(self, method):
            del method

        def reset_mm_cache(self):
            return None

        def shutdown(self):
            return None

    engine = core_module.EngineCoreProc(_config("ffn"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING
    engine.input_queue.put_nowait(
        (
            core_module.EngineCoreRequestType.UTILITY,
            (0, 41, "reset_mm_cache", ()),
        )
    )
    engine.input_queue.put_nowait(
        (
            core_module.EngineCoreRequestType.UTILITY,
            (0, 42, "get_supported_tasks", ()),
        )
    )

    from afd_plugin.compat.patches import engine_core as engine_core_patch

    def request_shutdown(_seconds):
        engine.shutdown_state = _EngineShutdownState.REQUESTED

    monkeypatch.setattr(engine_core_patch.time, "sleep", request_shutdown)

    with pytest.raises(SystemExit):
        engine.run_busy_loop()

    replies = {}
    while not engine.output_queue.empty():
        _client_idx, outputs = engine.output_queue.get_nowait()
        replies[outputs.utility_output.call_id] = outputs.utility_output

    assert replies[41].failure_message is None
    assert replies[41].result.result is None
    assert replies[42].failure_message is None
    assert replies[42].result.result == ()


def test_engine_core_ffn_readiness_log_follows_start_rpc(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    events = []

    class Executor:
        def __init__(self, vllm_config):
            self.calls = []

        def collective_rpc(self, method):
            events.append(f"rpc:{method}")

        def shutdown(self):
            self.calls.append("shutdown")

    engine = core_module.EngineCoreProc(_config("ffn"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING

    from afd_plugin.compat.patches import engine_core as engine_core_patch

    def request_shutdown(_seconds):
        events.append("sleep")
        engine.shutdown_state = _EngineShutdownState.REQUESTED

    monkeypatch.setattr(engine_core_patch.time, "sleep", request_shutdown)

    original_info = core_module.logger.info

    def recording_info(message, *args, **kwargs):
        events.append("log:started")
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(core_module.logger, "info", recording_info)

    with pytest.raises(SystemExit):
        engine.run_busy_loop()

    # The readiness log must not precede the collective start RPC: it is only
    # emitted after every FFN worker entered its connector loop.
    assert events.index("rpc:start_ffn_server_loop") < events.index("log:started")
    assert events.index("log:started") < events.index("rpc:raise_ffn_loop_error_if_any")


def test_engine_core_ffn_start_rpc_failure_emits_no_readiness_log(monkeypatch, caplog):
    core_module = _install_fake_vllm_core(monkeypatch)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    class Executor:
        def __init__(self, vllm_config):
            self.calls = []

        def collective_rpc(self, method):
            if method == "start_ffn_server_loop":
                raise RuntimeError("start failed")
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    engine = core_module.EngineCoreProc(_config("ffn"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING

    with (
        caplog.at_level(logging.INFO, logger="fake-vllm-core"),
        pytest.raises(RuntimeError, match="start failed"),
    ):
        engine.run_busy_loop()

    assert "AFD FFN EngineCore started; workers run connector loop." not in caplog.text
