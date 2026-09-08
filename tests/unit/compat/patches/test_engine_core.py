# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import importlib
import logging
import sys
import types
from enum import IntEnum
from types import SimpleNamespace

import pytest


class _EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1


def _install_fake_vllm_core(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    plugins_module = types.ModuleType("vllm.plugins")

    def load_general_plugins():
        return None

    plugins_module.load_general_plugins = load_general_plugins

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

        def shutdown(self):
            self.original_shutdown_called = True

        def _initialize_kv_caches(self, vllm_config):
            del vllm_config
            self.original_initialize_kv_caches_called = True

        def step(self):
            return None

        def step_with_batch_queue(self):
            return None

    class EngineCoreProc(EngineCore):
        def run_busy_loop(self):
            self.original_run_busy_loop_called = True

    class DPEngineCoreProc(EngineCoreProc):
        pass

    class _StructuredOutputManager:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def clear_backend(self):
            self.backend_cleared = True

    class _MMRegistry:
        def engine_receiver_cache_from_config(self, vllm_config):
            return ("mm-cache", vllm_config)

    class _SchedulerStats:
        def __init__(self, num_running_reqs=0, num_waiting_reqs=0):
            self.num_running_reqs = num_running_reqs
            self.num_waiting_reqs = num_waiting_reqs

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory):
        del vllm_config, available_gpu_memory
        return kv_cache_specs

    def generate_scheduler_kv_cache_config(kv_cache_configs):
        del kv_cache_configs
        return SimpleNamespace(num_blocks=0, kv_cache_groups=[])

    def get_hash_fn_by_name(name):
        return name

    def init_none_hash(_hash_fn):
        return None

    def get_request_block_hasher(block_size, hash_fn):
        return block_size, hash_fn

    def register_all_kvcache_specs(_vllm_config):
        return None

    def resolve_kv_cache_block_sizes(_kv_cache_config, _vllm_config):
        return 16, 16

    core_module.EngineCore = EngineCore
    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.EngineShutdownState = _EngineShutdownState
    core_module.VLLM_VERSION = "0.28.0"
    core_module.logger = logging.getLogger("fake-vllm-core")
    core_module.logger.info_once = lambda *args, **kwargs: None
    core_module.envs = SimpleNamespace(VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=False)
    core_module.StructuredOutputManager = _StructuredOutputManager
    core_module.MULTIMODAL_REGISTRY = _MMRegistry()
    core_module.SchedulerStats = _SchedulerStats
    core_module.get_kv_cache_configs = get_kv_cache_configs
    core_module.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
    core_module.get_hash_fn_by_name = get_hash_fn_by_name
    core_module.init_none_hash = init_none_hash
    core_module.get_request_block_hasher = get_request_block_hasher
    core_module.register_all_kvcache_specs = register_all_kvcache_specs
    core_module.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
    core_module.freeze_gc_heap = lambda: None
    core_module.maybe_attach_gc_debug_callback = lambda: None
    core_module.enable_envs_cache = lambda: None
    core_module.EngineCoreOutputs = lambda **kwargs: SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
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


def _config(role: str, *, async_dp: bool = False):
    class Scheduler:
        connector = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.request_counts = (0, 0)

        def get_kv_connector(self):
            return None

        def get_request_counts(self):
            return self.request_counts

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
        prefix_caching_hash_algo="builtin",
        num_gpu_blocks=None,
    )
    parallel_config = SimpleNamespace(
        data_parallel_rank_local=0,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    model_config = SimpleNamespace(
        max_model_len=8,
        runner_type="generate",
        is_diffusion=False,
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
        additional_config={"afd": afd_config},
        parallel_config=parallel_config,
        device_config=SimpleNamespace(device_type="cuda"),
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        model_config=model_config,
        speculative_config=None,
        ec_transfer_config=None,
        max_concurrent_batches=1,
        compilation_config=SimpleNamespace(
            compilation_time=0.0,
            encoder_compilation_time=0.0,
        ),
        validate_block_size=validate_block_size,
    )


def test_engine_core_patch_skips_kv_scheduler_init_for_ffn(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
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

    engine = core_module.EngineCore(_config("ffn"), Executor, log_stats=True)

    assert not hasattr(engine, "original_init_called")
    assert engine.afd_config.role == "ffn"
    assert engine.scheduler is None
    assert engine.structured_output_manager is None
    assert isinstance(engine.model_executor, Executor)


def test_engine_core_patch_leaves_cuda_non_ffn_path_untouched(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()
    monkeypatch.setitem(sys.modules, "vllm_ascend", None)

    class Executor:
        max_concurrent_batches = 1

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def get_kv_cache_specs(self):
            return []

        def initialize_from_config(self, kv_cache_configs):
            self.kv_cache_configs = kv_cache_configs

        def shutdown(self):
            self.shutdown_called = True

    config = _config("attention")
    config.device_config = SimpleNamespace(device_type="cuda")
    engine = core_module.EngineCore(config, Executor, log_stats=False)

    assert not hasattr(engine, "original_init_called")
    assert isinstance(engine.model_executor, Executor)
    assert engine.scheduler is not None
    assert engine.available_gpu_memory_for_kv_cache == -1


def test_engine_core_patch_imports_ascend_kv_cache_patch_on_npu(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()
    imported = []
    real_import = __import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_ascend.patch.platform.patch_kv_cache_utils":
            imported.append(name)
            return types.ModuleType(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    class Executor:
        max_concurrent_batches = 1

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def get_kv_cache_specs(self):
            return []

        def initialize_from_config(self, kv_cache_configs):
            self.kv_cache_configs = kv_cache_configs

        def shutdown(self):
            self.shutdown_called = True

    config = _config("attention")
    config.device_config = SimpleNamespace(device_type="npu")
    engine = core_module.EngineCore(config, Executor, log_stats=False)

    assert imported == ["vllm_ascend.patch.platform.patch_kv_cache_utils"]
    assert engine.scheduler is not None


def test_async_attention_publishes_request_count_lifecycle_without_dp_waves(
    monkeypatch,
    caplog,
):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()

    class Executor:
        max_concurrent_batches = 1

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def get_kv_cache_specs(self):
            return []

        def initialize_from_config(self, kv_cache_configs):
            self.kv_cache_configs = kv_cache_configs

        def shutdown(self):
            return None

    engine = core_module.EngineCoreProc(
        _config("attention", async_dp=True),
        Executor,
        log_stats=False,
    )
    engine.engine_index = 2
    engine.publish_dp_lb_stats = True
    published = []
    engine.output_queue = SimpleNamespace(
        put_nowait=lambda output: published.append(output)
    )

    loop_states = iter((True, True, False))
    count_states = iter(((0, 1), (1, 0), (1, 0), (0, 0)))
    engine._handle_shutdown = lambda: next(loop_states)
    engine._process_input_queue = lambda: setattr(
        engine.scheduler,
        "request_counts",
        next(count_states),
    )
    engine._process_engine_step = lambda: setattr(
        engine.scheduler,
        "request_counts",
        next(count_states),
    )

    def forbidden_sync_path(*_args, **_kwargs):
        pytest.fail("async Attention must not enter DP-wave synchronization")

    engine._has_global_unfinished_reqs = forbidden_sync_path
    engine.execute_dummy_batch = forbidden_sync_path

    with (
        caplog.at_level(logging.DEBUG, logger="fake-vllm-core"),
        pytest.raises(SystemExit),
    ):
        engine.run_busy_loop()

    published_counts = [
        (
            output.scheduler_stats.num_waiting_reqs,
            output.scheduler_stats.num_running_reqs,
        )
        for destination, output in published
        if destination == -1
    ]
    assert published_counts == [(1, 0), (0, 1), (0, 0)]
    assert "AFD async-DP engine 2 published request counts" in caplog.text


def test_engine_core_patch_runs_and_stops_ffn_loop(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()

    class Executor:
        def __init__(self, vllm_config):
            self.calls = []

        def collective_rpc(self, method):
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    engine = core_module.EngineCoreProc(_config("ffn"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING

    from afd_plugin.compat.patches import engine_core as engine_core_patch

    def request_shutdown(_seconds):
        engine.shutdown_state = _EngineShutdownState.REQUESTED

    monkeypatch.setattr(engine_core_patch.time, "sleep", request_shutdown)

    with pytest.raises(SystemExit):
        engine.run_busy_loop()

    assert engine.model_executor.calls == [
        "start_ffn_server_loop",
        "raise_ffn_loop_error_if_any",
        "stop_ffn_server_loop",
    ]


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
