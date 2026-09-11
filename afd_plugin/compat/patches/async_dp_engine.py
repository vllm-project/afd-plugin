# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP engine scheduling.

This module patches:
1. ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``
2. ``vllm.v1.engine.utils.launch_core_engines``
3. ``vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async``

Why:
    vLLM 0.28.0's native MoE DP path uses ``DPEngineCoreProc`` and DP wave
    notifications. AFD async-DP Attention ranks are connector-driven and must
    step independently while keeping the original DP/EP topology for expert
    placement and weight loading.

How:
    AFD async configs are selected by plugin-owned
    ``additional_config["afd"]["async"]``. Attention-side MoE DP engine
    processes instantiate ``EngineCoreProc`` instead of ``DPEngineCoreProc``;
    launch disables wave coordination on the DP coordinator, which vLLM
    0.28.0 also uses to skip lockstep stat aggregation, and client
    ``FIRST_REQ`` wakeups are skipped for AFD async configs.

Future plan:
    Remove the remaining patch when vLLM exposes an external async-DP
    scheduling hook selectable by plugin config. The copied DP-coordinator
    patches that existed for vLLM 0.26.0 were removed because vLLM 0.28.0
    absorbed the non-lockstep stats handling upstream (#49204): the
    coordinator now gates lockstep timeouts and step/wave ordering on
    ``enable_wave_coordination`` itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPAsyncMPClient

from afd_plugin.compat.vllm import (
    is_target_vllm_compatible as _is_target_vllm_compatible,
)
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.utils import CoreEngineLaunch, EngineZmqAddresses
    from vllm.v1.executor import Executor


# Patch reason: vLLM's MoE DP engine process uses DPEngineCoreProc, but AFD
# async Attention ranks are connector-driven and must not run DP wave logic.
# Patch functionality: keep upstream startup flow while selecting EngineCoreProc
# for AFD async Attention configs.
# Signature: matches upstream; no added parameters.
def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Replace MoE DP proc selection for AFD async Attention engines."""

    engine_core_module.maybe_register_config_serialize_by_value()

    engine_core = None
    signal_callback = None
    try:
        vllm_config = kwargs["vllm_config"]
        parallel_config = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        engine_core_module.set_process_title(process_title)
        engine_core_module.maybe_init_worker_tracer(
            "vllm.engine_core",
            "engine_core",
            process_title,
        )
        engine_core_module.decorate_logs()
        if parallel_config.numa_bind:
            engine_core_module.numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
            )
            engine_core_module.logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            # ### PATCH START: AFD async-DP Attention engine selection
            # Async-DP Attention ranks are connector-driven, so use the regular
            # EngineCoreProc instead of DPEngineCoreProc while keeping the
            # original DP rank metadata.
            if _is_afd_async_attention_config(vllm_config):
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            else:
                engine_core = engine_core_module.DPEngineCoreProc(*args, **kwargs)
            # ### PATCH END: AFD async-DP Attention engine selection
        else:
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            parallel_config.reconfigure_for_independent_dp_rank()
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        assert engine_core is not None

        def wakeup_engine():
            # Wakes up idle engine via input_queue when shutdown is requested
            # Not safe in a signal handler - we may interrupt the main thread
            # while it is holding the non-reentrant input_queue.mutex
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        signal_callback = engine_core_module.SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            signal_name = engine_core_module.signal.Signals(signum).name
            engine_core_module.logger.info(
                "[shutdown] EngineCore: trigger received signal=%s",
                signal_name,
            )
            engine_core.shutdown_state = (
                engine_core_module.EngineShutdownState.REQUESTED
            )
            signal_callback.trigger()

        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            signal_handler,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            signal_handler,
        )

        engine_core.run_busy_loop()

    except SystemExit:
        engine_core_module.logger.info_once("[shutdown] EngineCore: exiting busy loop")
        raise
    except Exception as exc:
        if engine_core is None:
            engine_core_module.logger.exception("EngineCore failed to start.")
        else:
            engine_core_module.logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise exc
    finally:
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            engine_core_module.signal.SIG_DFL,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            engine_core_module.signal.SIG_DFL,
        )
        if signal_callback is not None:
            signal_callback.stop()
        if engine_core is not None:
            engine_core.shutdown()


# Patch reason: vLLM enables MoE DP wave coordination when launching DP cores,
# while AFD async-DP only needs coordinator stats.
# Patch functionality: preserve upstream engine launch behavior but disable wave
# coordination for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
@contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
    """Disable coordinator wave mode while launching AFD async-DP engines."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # Create a single tensor IPC queue for sharing multimodal tensors between
    # API servers and engine core. Returns a single queue since we only support
    # DP=1 for this data flow.
    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = engine_utils_module.get_mp_context().Queue()

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = engine_utils_module.DPCoordinator(
            parallel_config,
            # ### PATCH START: AFD async-DP coordinator wave mode
            # Keep DP coordinator stats, but disable wave coordination for
            # connector-driven async-DP. vLLM 0.28.0 keys its lockstep stat
            # aggregation off this flag, so async Attention ranks get
            # independent request-count publication.
            enable_wave_coordination=(
                vllm_config.model_config.is_moe and not is_afd_async_dp(vllm_config)
            ),
            # ### PATCH END: AFD async-DP coordinator wave mode
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        engine_utils_module.logger.info(
            "Started DP Coordinator process (PID: %d)",
            coordinator.proc.pid,
        )
    else:
        coordinator = None

    if parallel_config.data_parallel_backend == "ray":
        engine_utils_module.logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = engine_utils_module.CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_utils_module.CoreEngineLaunch(
            engine_actor_manager, coordinator, addresses, tensor_queue
        )
        return

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=dp_rank, local=True),
        ]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    handshake_address = engine_utils_module.get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        parallel_config.data_parallel_rpc_port,
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = engine_utils_module.get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with engine_utils_module.zmq_socket_ctx(
        local_handshake_address,
        engine_utils_module.zmq.ROUTER,
        bind=True,
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = engine_utils_module.CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        launch = engine_utils_module.CoreEngineLaunch(
            local_engine_manager, coordinator, addresses, tensor_queue
        )
        yield launch
        engine_utils_module.wait_for_engine_startup(
            handshake_socket,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            launch,
        )


# Patch reason: vLLM sends FIRST_REQ wakeups to coordinate DP waves, which AFD
# async-DP engines intentionally do not use.
# Patch functionality: preserve request routing and stats updates while skipping
# FIRST_REQ for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
async def add_request_async(
    self,
    request: EngineCoreRequest,
) -> None:
    """Skip the DP wave ``FIRST_REQ`` notification for AFD async-DP."""

    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    chosen_engine = self.get_core_engine_for_request(request)
    to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
    # ### PATCH START: AFD async-DP request wakeup
    # Async-DP engines step independently, so skip the coordinator FIRST_REQ
    # wakeup while preserving normal routing.
    if not self.engines_running and not is_afd_async_dp(self.vllm_config):
        req_msg = core_client_module.msgspec.msgpack.encode(
            ("FIRST_REQ", chosen_engine),
        )
        await self.first_req_send_socket.send(req_msg)
    # ### PATCH END: AFD async-DP request wakeup

    await to_await

    # The output queue task delivers completed responses and is independent of
    # the DP-wave FIRST_REQ coordination skipped above.
    self._ensure_output_queue_task()


def _is_afd_async_attention_config(vllm_config: VllmConfig) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    return (
        afd_config is not None
        and is_afd_async_dp(vllm_config)
        and afd_config.role == "attention"
    )


def apply_async_dp_engine_patch() -> bool:
    """Install the async-DP patches against the current vLLM bindings.

    vLLM-Ascend installs platform patches after general plugins and also wraps
    ``EngineCoreProc.run_engine_core``. Re-applying this installer after the
    final AFD Attention config is built ensures the process target captured by
    vLLM uses AFD's async-DP entry point. The caller scopes that late install to
    AFD async Attention, so non-AFD and FFN Ascend scheduling remain untouched.
    """

    if not _is_target_vllm_compatible():
        return False

    EngineCoreProc.run_engine_core = staticmethod(run_engine_core)
    engine_utils_module.launch_core_engines = launch_core_engines
    core_client_module.launch_core_engines = launch_core_engines
    DPAsyncMPClient.add_request_async = add_request_async
    engine_core_module.logger.debug("AFD async-DP engine patch applied")
    return True


apply_async_dp_engine_patch()


__all__ = ["apply_async_dp_engine_patch"]
