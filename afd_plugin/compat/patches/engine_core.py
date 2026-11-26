# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""EngineCore compatibility patches for AFD runtime modes.

The AFD FFN side runs as a connector daemon, not as a normal request-scheduling
EngineCore. After constructing the model executor, FFN EngineCore
initialization returns before KV cache and scheduler setup. This keeps FFN
startup out of HybridKVCacheCoordinator.

AFD async-DP Attention uses the regular ``EngineCoreProc`` to avoid vLLM's
synchronous DP-wave loop; see the async-DP engine patch for the constructor
selection. The native busy loop publishes DP load-balancer request counts.
"""

from __future__ import annotations

import gc
import queue
import time
from collections.abc import Callable
from contextlib import suppress
from functools import wraps
from typing import TYPE_CHECKING, Any

import vllm.v1.engine.core as core_module

from afd_plugin.compat.vllm import get_target_vllm_version
from afd_plugin.compat.vllm import (
    is_target_vllm_compatible as _is_target_vllm_compatible,
)
from afd_plugin.config import AFDConfig, parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor
    from vllm.v1.kv_cache_interface import KVCacheConfig


# Upstream source: vLLM v0.34.0 (62d4bc3c7e33b1b6c906e233e6af2d3506c482ca),
# vllm/v1/engine/core.py::EngineCore.shutdown.
# Patch reason: AFD FFN daemon engines skip scheduler/KV setup, so upstream
# shutdown can touch attributes that were intentionally not initialized.
# Patch functionality: stops the connector worker loop and shuts down the model
# executor for AFD FFN engines while preserving upstream shutdown for non-AFD
# engines.
# Signature: matches upstream; no added parameters.
def shutdown(self):
    # ### PATCH START: AFD FFN EngineCore shutdown
    # Stop the connector-driven worker loop before shutting down the executor;
    # scheduler/KV state may not exist for FFN daemon engines.
    if _is_afd_ffn_engine(self):
        _stop_ffn_worker_loop(self)
        model_executor = getattr(self, "model_executor", None)
        if model_executor is not None:
            model_executor.shutdown()
        with suppress(Exception):
            gc.unfreeze()
        core_module.cleanup_dist_env_and_memory()
        return
    # ### PATCH END: AFD FFN EngineCore shutdown

    core_module.logger.debug_once("[shutdown] EngineCore: tearing down local resources")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()

    # Undo the gc.freeze() from __init__ so that the objects allocated
    # during engine startup (model weights, KV caches, etc.) become
    # visible to the garbage collector again. Without this, deleting
    # the engine in-process (e.g. unit tests) leaks GPU memory.
    gc.unfreeze()
    # Tear down distributed state initialized in this EngineCore process
    # before it exits and release cached memory.
    core_module.cleanup_dist_env_and_memory()
    core_module.logger.debug_once(
        "[shutdown] EngineCore: local resource teardown complete"
    )


# Upstream source: vLLM v0.34.0 (62d4bc3c7e33b1b6c906e233e6af2d3506c482ca),
# vllm/v1/engine/core.py::EngineCore._initialize_kv_caches.
# Patch reason: late-loaded AFD FFN engines must bypass KV initialization.
# Patch functionality: keeps the 0.34 flow intact -- the capability probe and
# possible disable, the global KV groups and DCP interleave adjustment, the
# two-argument layout publish, capacity bookkeeping skipped while extensible,
# the warmup return value, and the finalize/second-max_model_len/extend steps --
# and short-circuits AFD FFN engines before any of it.
# Signature: matches upstream; no added parameters.
# Removal/upstream plan: delete once vLLM exposes a pre-KV-init hook per engine.
@core_module.instrument(span_name="Prepare model")
def _initialize_kv_caches_v034(self, vllm_config: VllmConfig) -> KVCacheConfig:
    # ### PATCH START: AFD FFN late-loaded KV cache bypass
    # FFN daemon engines do not schedule requests or own KV cache blocks, but
    # late-loaded paths still need a minimal KV cache config-shaped result.
    if _is_afd_ffn_config(vllm_config):
        _prepare_late_loaded_ffn_engine_core(self, vllm_config)
        return _AFDFFNKVCacheConfig()
    # ### PATCH END: AFD FFN late-loaded KV cache bypass

    start = time.time()

    # register all kvcache specs in enginecore process.
    core_module.register_all_kvcache_specs(vllm_config)

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # Some layers (e.g. Prefix LM attention) run non-causally and tag their
    # KV cache spec with ``non_causal=True``. The specs are collected here in
    # the engine-core process (the same process that builds the scheduler),
    # so this is the multiproc-safe place to translate that layer-level
    # signal into a scheduling policy: chunked prefill and prefix caching
    # both assume causal attention and would corrupt non-causal prefill.
    if any(
        getattr(spec, "non_causal", False)
        for worker_specs in kv_cache_specs
        for spec in worker_specs.values()
    ):
        if vllm_config.scheduler_config.enable_chunked_prefill:
            core_module.logger.info(
                "Disabling chunked prefill: model has non-causal attention layers."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
        if vllm_config.cache_config.enable_prefix_caching:
            core_module.logger.info(
                "Disabling prefix caching: model has non-causal attention layers."
            )
            vllm_config.cache_config.enable_prefix_caching = False

    # Resolve the KV cache layout before memory profiling: workers that
    # capture full cudagraphs initialize a minimal KV cache during it.
    # Attention-free models resolve the default so layout reads never precede
    # resolution.
    if vllm_config.cache_config.enable_extensible_kv_cache:
        # Disable it, everywhere, where it cannot be used.
        executor = self.model_executor
        reason = executor.extensible_kv_cache_unsupported_reason(kv_cache_specs)
        if reason is not None:
            executor.disable_extensible_kv_cache(reason)
    layout = core_module.resolve_kv_cache_layout(
        vllm_config,
        self.model_executor.get_supported_kv_cache_layouts(),
        [s for specs in kv_cache_specs for s in specs.values()],
    )

    global_kv_cache_groups = core_module.get_global_kv_cache_groups(
        vllm_config, kv_cache_specs
    )
    vllm_config.adjust_dcp_kv_cache_interleave_size(global_kv_cache_groups)
    self.model_executor.set_kv_cache_layout(
        layout.name, vllm_config.parallel_config.cp_kv_cache_interleave_size
    )

    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
    if has_kv_cache:
        if core_module.envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # NOTE(yongji): should already be set
            # during _eep_scale_up_before_kv_init
            assert self.available_gpu_memory_for_kv_cache > 0
            available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                kv_cache_specs
            )
        else:
            # Profiles the peak memory usage of the model to determine how
            # much memory can be allocated for kv cache.
            available_gpu_memory = self.model_executor.determine_available_memory()
            self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    else:
        # Attention free models don't need memory for kv cache
        available_gpu_memory = [0] * len(kv_cache_specs)

    assert len(kv_cache_specs) == len(available_gpu_memory)

    # Track max_model_len before KV cache config to detect auto-fit changes
    max_model_len_before = vllm_config.model_config.max_model_len

    kv_cache_configs = core_module.get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory, global_kv_cache_groups
    )
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_cache_layout = vllm_config.cache_config.kv_cache_layout

    # If auto-fit reduced max_model_len, sync the new value to workers.
    # This is needed because workers were spawned before memory profiling
    # and have the original (larger) max_model_len cached.
    max_model_len_after = vllm_config.model_config.max_model_len
    if max_model_len_after != max_model_len_before:
        self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

    scheduler_kv_cache_config = core_module.generate_scheduler_kv_cache_config(
        kv_cache_configs
    )
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
    if kv_cache_groups:
        # Exclude groups that opt out of prefix caching (e.g. a circular
        # buffer, a 1-block/req scratch buffer with a small ring block):
        # their small block_size would otherwise drag the global block_size
        # below the real allocator block size and desync it from mamba.
        participating = [
            g.kv_cache_spec.block_size
            for g in kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ]
        vllm_config.cache_config.block_size = min(
            participating
            if participating
            else [g.kv_cache_spec.block_size for g in kv_cache_groups]
        )
    # With an extensible KV cache, `num_blocks` is the reserved capacity
    # until the post-warmup measurement below.
    extensible = vllm_config.cache_config.enable_extensible_kv_cache
    if kv_cache_groups and not extensible:
        core_module.update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)

    vllm_config.validate_block_size()

    scheduler_block_size, hash_block_size = core_module.resolve_kv_cache_block_sizes(
        scheduler_kv_cache_config, vllm_config
    )
    cache_hit_alignment_tokens = core_module.resolve_cache_hit_alignment_tokens(
        scheduler_kv_cache_config,
        vllm_config,
        scheduler_block_size,
        hash_block_size,
    )
    # The scheduler's config is a deep copy taken before these are resolved,
    # so set them there too: the KV connector's scheduler side reads the same
    # fields as its workers.
    for kv_cache_config in (*kv_cache_configs, scheduler_kv_cache_config):
        kv_cache_config.hash_block_size = hash_block_size
        kv_cache_config.cache_hit_alignment_tokens = cache_hit_alignment_tokens

    self.model_executor.initialize_from_config(kv_cache_configs)
    compilation_times = []
    if not core_module.envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        compilation_times = self.model_executor.compile_or_warm_up_model()
    if extensible:
        num_blocks = core_module.finalize_extensible_kv_cache(
            vllm_config,
            kv_cache_configs,
            scheduler_kv_cache_config,
            compilation_times,
        )
        max_model_len = vllm_config.model_config.max_model_len
        if max_model_len != max_model_len_after:
            self.collective_rpc("update_max_model_len", args=(max_model_len,))
        self.model_executor.extend_kv_cache(num_blocks)
    # Under the extensible KV cache, workers create the connector in
    # `extend_kv_cache`, so ask for its block count only afterwards.
    kv_transfer_config = vllm_config.kv_transfer_config
    if (
        kv_transfer_config is not None
        and kv_transfer_config.has_connector("SimpleCPUOffloadConnector")
        and vllm_config.cache_config.enable_prefix_caching
    ):
        aligned_num_cpu_blocks = self.model_executor.get_simple_cpu_offload_num_blocks()
        assert aligned_num_cpu_blocks is not None, (
            "SimpleCPUOffloadConnector workers must report their block count"
        )
        vllm_config.cache_config.simple_cpu_offload_num_blocks = aligned_num_cpu_blocks

    elapsed = time.time() - start
    compile_time = vllm_config.compilation_config.compilation_time
    encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
    if encoder_compile_time > 0:
        core_module.logger.info_once(
            "init engine (profile, create kv cache, warmup model) took "
            "%.2f s (compilation: %.2f s — language_model: %.2f s, "
            "encoder: %.2f s)",
            elapsed,
            compile_time + encoder_compile_time,
            compile_time,
            encoder_compile_time,
        )
    elif compile_time > 0:
        core_module.logger.info_once(
            "init engine (profile, create kv cache, warmup model) took "
            "%.2f s (compilation: %.2f s)",
            elapsed,
            compile_time,
        )
    else:
        core_module.logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f s",
            elapsed,
        )
    return scheduler_kv_cache_config


class _AFDFFNKVCacheConfig:
    kv_cache_groups: list[Any] = []


class _AFDFFNNoopScheduler:
    # A cold FFN engine loads the patches mid-native-init and then runs on
    # with this noop, so it must mirror every scheduler member upstream
    # touches after KV-cache setup.
    connector = None
    ec_connector = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_kv_connector(self) -> None:
        return None

    def get_kv_event_publisher_config(self) -> None:
        return None

    def get_ec_connector(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def has_requests(self) -> bool:
        return False

    def has_unfinished_requests(self) -> bool:
        return False

    def get_num_unfinished_requests(self) -> int:
        return 0

    def finish_requests(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _initialize_ffn_engine_core(
    self,
    core_module: Any,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None,
) -> None:
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    except Exception:
        core_module.logger.debug(
            "AFD FFN EngineCore could not reload vLLM plugins",
            exc_info=True,
        )

    afd_config = _get_afd_config(vllm_config)
    self.vllm_config = vllm_config
    self.afd_config = afd_config
    self.log_stats = log_stats

    with suppress(Exception):
        vllm_config.afd_config = afd_config

    parallel_config = getattr(vllm_config, "parallel_config", None)
    local_dp_rank = getattr(parallel_config, "data_parallel_rank_local", 0)
    if not local_dp_rank:
        version = getattr(core_module, "VLLM_VERSION", "unknown")
        core_module.logger.info(
            "Initializing an AFD FFN V1 engine (v%s) with config: %s",
            version,
            vllm_config,
        )

    self.model_executor = executor_class(vllm_config)
    if executor_fail_callback is not None:
        self.model_executor.register_failure_callback(executor_fail_callback)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0

    # These attributes let common shutdown/debug utility paths tolerate the
    # intentionally skipped KV/scheduler initialization. The noop scheduler
    # (not None) is required: the native ready handshake calls
    # scheduler.get_kv_event_publisher_config() unconditionally.
    self.available_gpu_memory_for_kv_cache = -1
    self.structured_output_manager = None
    self.scheduler = _AFDFFNNoopScheduler()
    self.mm_receiver_cache = None
    self.batch_queue_size = 0
    self.batch_queue = None
    self.request_block_hasher = None
    self.aborts_queue = queue.Queue()
    self._idle_state_callbacks = []
    self.use_spec_decode = False
    self.is_pooling_model = False
    # The 0.34 ready handshake reads this when it builds
    # EngineCoreReadyResponse (core.py:2155). The native __init__ sets it to
    # None first (core.py:235); an FFN daemon never publishes prefix hashes,
    # so None is also the correct value here.
    self.prefix_hash_config = None
    # The native startup utility RPC reads this even when the worker reports no
    # supported tasks. Early-loaded FFN engines bypass the standard EngineCore
    # constructor, so initialize it explicitly for the frontend handshake.
    self._pooler_config_logged = False


def _prepare_late_loaded_ffn_engine_core(
    self,
    vllm_config: VllmConfig,
) -> None:
    afd_config = _get_afd_config(vllm_config)
    self.afd_config = afd_config
    with suppress(Exception):
        vllm_config.afd_config = afd_config

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0
        with suppress(Exception):
            cache_config.enable_prefix_caching = False

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is not None:
        with suppress(Exception):
            scheduler_config.enable_chunked_prefill = False
        with suppress(Exception):
            scheduler_config.get_scheduler_cls = lambda: _AFDFFNNoopScheduler


# FFN replaces the normal busy loop, so service the frontend startup RPCs from
# the connector loop. Otherwise the frontend waits for get_supported_tasks and
# never starts its HTTP health endpoint.
def _drain_frontend_requests(engine_core: Any) -> None:
    while True:
        try:
            req = engine_core.input_queue.get_nowait()
        except Exception:
            return
        engine_core._handle_client_request(*req)


def _run_ffn_busy_loop(self, core_module: Any) -> None:
    started = False
    try:
        self.model_executor.collective_rpc("start_ffn_server_loop")
        started = True
        # Emit readiness only after the collective start RPC succeeded; the
        # worker connector loop may not be ready to receive before this point.
        core_module.logger.info(
            "AFD FFN EngineCore started; workers run connector loop."
        )
        while _is_running(self, core_module):
            core_module.get_watch_dog().feed()
            _drain_frontend_requests(self)
            self.model_executor.collective_rpc("raise_ffn_loop_error_if_any")
            time.sleep(0.5)
    except KeyboardInterrupt:
        core_module.logger.info(
            "AFD FFN EngineCore shutting down after KeyboardInterrupt"
        )
    except Exception:
        core_module.logger.exception("AFD FFN EngineCore encountered a fatal error")
        raise
    finally:
        if started:
            _stop_ffn_worker_loop(self)

    raise SystemExit


def _stop_ffn_worker_loop(self) -> None:
    model_executor = getattr(self, "model_executor", None)
    if model_executor is None:
        return
    try:
        model_executor.collective_rpc("stop_ffn_server_loop")
    except Exception:
        core_module.logger.debug(
            "AFD FFN worker loop stop failed or was already stopped",
            exc_info=True,
        )


def _is_running(self, core_module: Any) -> bool:
    shutdown_state = getattr(self, "shutdown_state", None)
    engine_shutdown_state = getattr(core_module, "EngineShutdownState", None)
    running_state = getattr(engine_shutdown_state, "RUNNING", None)
    if shutdown_state is None or running_state is None:
        return True
    return shutdown_state == running_state


def _is_afd_ffn_engine(self) -> bool:
    return _is_afd_ffn_config(getattr(self, "vllm_config", None))


def _is_afd_ffn_config(vllm_config: VllmConfig | None) -> bool:
    config = _get_afd_config(vllm_config)
    return config is not None and config.role == "ffn"


def _get_afd_config(vllm_config: VllmConfig | None) -> AFDConfig | None:
    existing = getattr(vllm_config, "afd_config", None)
    if isinstance(existing, AFDConfig):
        return existing
    try:
        return parse_optional_afd_config(vllm_config, validate=False)
    except Exception:
        core_module.logger.debug(
            "Unable to parse AFD config from vLLM config",
            exc_info=True,
        )
        return None


def _with_ffn_engine_init(native_init: Callable[..., None]) -> Callable[..., None]:
    # Upstream source: vLLM v0.34.0 (62d4bc3c7e33b1b6c906e233e6af2d3506c482ca),
    # vllm/v1/engine/core.py::EngineCore.__init__.
    # Patch reason: FFN engines own only the connector worker, not request KV
    # cache and scheduler state.
    # Patch functionality: use the FFN initialization branch for that role.
    # Expansion exception: the native constructor is large and owns prefix
    # hashing and the KV transfer handshake. The FFN decision precedes native
    # initialization, so delegate to vLLM for all other configs.
    # Removal plan: use an upstream connector-daemon engine mode when available.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # ### PATCH START: AFD FFN EngineCore daemon initialization
        if _is_afd_ffn_config(vllm_config):
            _initialize_ffn_engine_core(
                self,
                core_module,
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
            )
            return
        # ### PATCH END: AFD FFN EngineCore daemon initialization
        native_init(
            self,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
            include_finished_set,
        )

    return __init__


def _with_ffn_busy_loop(
    native_busy_loop: Callable[[core_module.EngineCoreProc], None],
) -> Callable[[core_module.EngineCoreProc], None]:
    @core_module.fault_tolerant_wrapper
    def ffn_busy_loop(self):
        _run_ffn_busy_loop(self, core_module)

    # Upstream source: vLLM v0.34.0 (62d4bc3c7e33b1b6c906e233e6af2d3506c482ca),
    # vllm/v1/engine/core.py::EngineCoreProc/DPEngineCoreProc.run_busy_loop.
    # Patch reason: FFN engines use a connector loop in place of the scheduler.
    # Patch functionality: select the fault-tolerant FFN loop for that role.
    # Expansion exception: delegate the long native DP loop and the base loop
    # to their corresponding already-decorated methods. Each path enters
    # fault tolerance once and vLLM retains elastic-EP scheduling.
    # Removal plan: use an upstream connector-daemon engine mode when available.
    # Signature: matches upstream; no added parameters.
    @wraps(native_busy_loop, assigned=("__doc__",))
    def run_busy_loop(self):
        # ### PATCH START: AFD FFN connector busy loop
        if _is_afd_ffn_engine(self):
            return ffn_busy_loop(self)
        # ### PATCH END: AFD FFN connector busy loop
        return native_busy_loop(self)

    return run_busy_loop


if _is_target_vllm_compatible():
    # The installed closures already retain the original methods on reload.
    if core_module.EngineCore.__init__.__module__ != __name__:
        core_module.EngineCore.__init__ = _with_ffn_engine_init(
            core_module.EngineCore.__init__
        )
        core_module.DPEngineCoreProc.run_busy_loop = _with_ffn_busy_loop(
            core_module.DPEngineCoreProc.run_busy_loop
        )
        core_module.EngineCoreProc.run_busy_loop = _with_ffn_busy_loop(
            core_module.EngineCoreProc.run_busy_loop
        )
    core_module.EngineCore._initialize_kv_caches = _initialize_kv_caches_v034
    core_module.EngineCore.shutdown = shutdown
    core_module.logger.debug("AFD EngineCore patch applied")
else:
    core_module.logger.warning(
        "Skipping AFD EngineCore patch: vLLM %s does not match target %s",
        getattr(core_module, "VLLM_VERSION", "unknown"),
        get_target_vllm_version(),
    )


__all__: list[str] = []
