# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Narrow EEP integration seams, ported from vLLM 0.30.0 (ced6857afa0e)
and re-anchored to main@544f182d9 where marked inline.

No ElasticEPScalingState code is patched. Worker role actions are implemented
by AFDElasticGPUExecutor; role-aware orchestration drives the native
prepare_elastic_ep/commit_elastic_ep client RPCs through ElasticAFDClient.
Upstream targets: an optional EEP worker factory, post-READY initialization
and an explicit EEP-complete utility.
"""

from __future__ import annotations

import asyncio
import time
from multiprocessing import Queue

from vllm.config import VllmConfig
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_scaling_elastic_ep,
    set_scaling_elastic_ep,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import PauseState
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerManager

from afd_plugin.elastic.client import ElasticAFDClient
from afd_plugin.elastic.config import Role, is_elastic_attention

logger = init_logger(__name__)


def afd_eep_complete(self) -> bool:
    """Nonblocking EngineCore utility; a worker RPC cannot observe this state."""
    return self.eep_scaling_state is None


def afd_reset_after_warmup(self) -> None:
    # Re-anchor main@544f182d: post-commit pause/running state legitimately
    # differs per engine (scale-up joiners boot unpaused; the wave sync can
    # leave engines_running True on retained engines). The bare asserts died
    # with an empty-message AssertionError and killed the whole resize; log
    # the unexpected shape loudly and skip the (only cache-hygiene) reset
    # instead, so a genuine state bug stays visible without aborting scale.
    pause_state = self.scheduler.pause_state
    engines_running = self.engines_running
    pending_pause = self.pending_pause
    if pause_state != PauseState.PAUSED_ALL or engines_running or pending_pause:
        logger.warning(
            "[AFD] afd_reset_after_warmup: dp_rank=%s unexpected state "
            "pause=%s engines_running=%s pending_pause=%s; skipping reset",
            self.vllm_config.parallel_config.data_parallel_rank,
            pause_state,
            engines_running,
            pending_pause,
        )
        return
    self._reset_caches()


async def initialize_afd(self) -> bool:
    if not is_elastic_attention(self.vllm_config):
        return False
    assert isinstance(self.engine_core, DPLBAsyncMPClient)
    controller = ElasticAFDClient(self.engine_core)
    self.engine_core._afd_elastic = controller
    try:
        await controller.initialize()
    except BaseException:
        controller.ffn.shutdown()
        raise
    return True


def install() -> None:
    original_init = DPEngineCoreProc.__init__
    original_scale = AsyncLLM.scale_elastic_ep
    original_shutdown = DPLBAsyncMPClient.shutdown

    # Patch reason: native READY precedes the AFD cross-role rendezvous.
    # Change: keep new/initial A paused after normal engine construction.
    # Signature: exact v0.30 DPEngineCoreProc.__init__.
    # Delegation exception: preserve constructor super() and handshake ordering;
    # only the post-construction pause is AFD-owned.
    def init(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        original_init(
            self,
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            tensor_queue,
        )
        # ### PATCH START: pause before the engine enters its busy loop.
        if is_elastic_attention(vllm_config):
            self.scheduler.set_pause_state(PauseState.PAUSED_ALL)
            self.engines_running = False
            self.ignore_start_dp_wave = True
        # ### PATCH END: pause before the engine enters its busy loop.

    # Patch reason: native scale compares only with A size, drains optionally
    # and owns no second role.
    # Change: AFD owns role dispatch; the attention path runs the controller
    # (pause/link teardown -> native prepare_elastic_ep/commit_elastic_ep ->
    # cross-role reconnect) then mirrors upstream's stat-logger and config
    # bookkeeping. Non-AFD deployments keep delegating to the native method.
    # Signature: adds role="attention" to v0.30 AsyncLLM.scale_elastic_ep.
    async def scale(
        self,
        new_data_parallel_size: int,
        drain_timeout: int = 300,
        role: Role = "attention",
    ):
        # ### PATCH START: AFD role-aware drain and scale gate.
        if not is_elastic_attention(self.vllm_config):
            if role != "attention":
                raise ValueError("role=ffn requires elastic AFD")
            return await original_scale(self, new_data_parallel_size, drain_timeout)
        controller = self.engine_core._afd_elastic
        target = controller.topology.resize(role, new_data_parallel_size)
        if get_scaling_elastic_ep():
            raise RuntimeError("An elastic AFD resize is already in progress")
        if target == controller.topology:
            return
        set_scaling_elastic_ep(True)
        try:
            # Coordinator idle stats can lag an accepted request. Include the
            # frontend request registry and in-flight A routing before pausing.
            deadline = time.monotonic() + drain_timeout
            while (
                self.output_processor.has_unfinished_requests()
                or self.engine_core.reqs_in_flight
                or self.engine_core.dp_engines_running()
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out draining elastic AFD requests")
                await asyncio.sleep(0.1)
            old_attention_dp = controller.topology.attention_dp
            await controller.resize(new_data_parallel_size, role)
            if role == "attention":
                # Mirror v0.30 _scale_elastic_ep bookkeeping for the A path.
                if new_data_parallel_size > old_attention_dp and self.log_stats:
                    self.logger_manager = StatLoggerManager(
                        vllm_config=self.vllm_config,
                        engine_idxs=list(range(new_data_parallel_size)),
                        custom_stat_loggers=None,
                    )
                    if hasattr(self, "_logger_ref"):
                        self._logger_ref[0] = self.logger_manager
                    self.logger_manager.log_engine_initialized()
                self.vllm_config.parallel_config.data_parallel_size = (
                    new_data_parallel_size
                )
        finally:
            set_scaling_elastic_ep(False)
        # ### PATCH END: AFD role-aware drain and scale gate.

    # Patch reason: native client teardown owns only A actors.
    # Change: release this client's F actor pool, also on initialization failure.
    # Signature: exact v0.30 MPClient.shutdown; inherited by DPLBAsyncMPClient.
    # Delegation exception: native process/thread/socket cleanup stays upstream.
    def shutdown(self, timeout: float | None = None) -> None:
        # ### PATCH START: close the optional per-client F sidecar.
        controller = vars(self).get("_afd_elastic")
        if controller is not None:
            controller.ffn.shutdown()
        # ### PATCH END: close the optional per-client F sidecar.
        original_shutdown(self, timeout)

    DPEngineCoreProc.__init__ = init
    DPEngineCoreProc.afd_eep_complete = afd_eep_complete
    DPEngineCoreProc.afd_reset_after_warmup = afd_reset_after_warmup
    DPLBAsyncMPClient.shutdown = shutdown
    AsyncLLM.scale_elastic_ep = scale
    AsyncLLM.initialize_afd = initialize_afd


install()
