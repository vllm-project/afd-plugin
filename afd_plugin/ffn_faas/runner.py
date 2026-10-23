# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone FFN worker/runner with bounded native execution and ownership."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.model import CheckpointDescription, FFNModel
from afd_plugin.ffn_faas.protocol import Mode, Result, SessionSpec
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
from afd_plugin.ffn_faas.v4_runtime import V4FFNModel

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.npu import NPUExecutor


@dataclass(frozen=True)
class WorkerConfig:
    device: int
    ffn_rank: int
    host: str
    port: int
    timeout: float = 30.0
    startup_timeout: float = 300.0
    window_mb: int = 64


class FFNModelRunner:
    """Own the model and executor; the executor borrows the model's tensors."""

    def __init__(
        self,
        service: FFNService,
        checkpoint: CheckpointDescription | V4CheckpointDescription,
        spec: SessionSpec,
        config: WorkerConfig,
    ) -> None:
        self.service, self.spec, self.config = service, spec, config
        self.model: FFNModel | V4FFNModel
        if isinstance(checkpoint, V4CheckpointDescription):
            self.model = V4FFNModel(checkpoint, spec, config.ffn_rank)
        else:
            self.model = FFNModel(checkpoint, spec, config.ffn_rank)
        self.executor: NPUExecutor | None = None

    def run(self) -> None:
        # The external launcher creates one process per F rank. No vLLM model
        # runner, scheduler, EngineCore or KV cache is constructed. V4 owns
        # F-local startup groups only while its native loader packs parameters.
        import torch
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import bootstrap_custom_op_env

        from afd_plugin.compat.npu.ops import bootstrap_afd_custom_op_env

        # Native platform setup prepares OPP discovery before first device use.
        # The standalone service bypasses its worker/runner startup.
        bootstrap_custom_op_env()
        bootstrap_afd_custom_op_env()

        from afd_plugin.ffn_faas.npu import NPUExecutor

        c, service, spec = self.config, self.service, self.spec
        torch.npu.set_device(c.device)
        started = time.monotonic()
        failure: BaseException | None = None
        try:
            service.phase(
                ServicePhase.LOADING, parameter_bytes=self.model.parameter_bytes
            )
            self.model.load(c.device, service.loading_progress)
            service.phase(ServicePhase.CONNECTING)
            # Retain the executor before connecting, including partial failures.
            if isinstance(self.model, V4FFNModel):
                from afd_plugin.ffn_faas.native_npu import NativeW4A8Executor

                self.executor = NativeW4A8Executor(
                    spec,
                    rank=spec.attention_ranks + c.ffn_rank,
                    device=c.device,
                    host=c.host,
                    port=c.port,
                    weights=self.model.native_view,
                    timeout=c.timeout,
                    startup_timeout=c.startup_timeout,
                    window_mb=c.window_mb,
                    auto_initialize=False,
                )
            else:
                self.executor = NPUExecutor(
                    spec,
                    rank=spec.attention_ranks + c.ffn_rank,
                    device=c.device,
                    host=c.host,
                    port=c.port,
                    weights=self.model.native_view,
                    timeout=c.timeout,
                    startup_timeout=c.startup_timeout,
                    window_mb=c.window_mb,
                    auto_initialize=False,
                )
            executor = self.executor
            executor.initialize(stop=service.stop_requested)
            try:
                executor.bind_host_threads()
            except Exception as error:
                # CPU placement is optional for correctness, matching the
                # backend's worker policy on constrained/unsupported hosts.
                logging.getLogger(__name__).warning(
                    "FFN CPU binding skipped: %s", error
                )
            store = executor.store
            assert store is not None
            service.phase(ServicePhase.WARMUP)
            # An idle launch proves windows and native descriptors are usable
            # before A sends the paired zero-input expert warmup.
            if executor.retire(executor.submit(Mode.EXECUTE)) != Result.IDLE:
                raise RuntimeError("unexpected work before paired warmup")
            store.set(f"prepared/{c.ffn_rank}", "1")
            warmup_keys = [f"warmup/{rank}" for rank in range(spec.attention_ranks)]
            ready = False
            warmup_deadline = time.monotonic() + c.startup_timeout
            while True:
                if service.stop_requested.is_set():
                    raise InterruptedError("standalone FFN stop aborts the A/F session")
                executor.check_peers()
                # Only the BF16 fused backend executes a bounded batch per
                # launch; the V4 host backend retires one transaction.
                mode = (
                    Mode.SERVE
                    if ready and not isinstance(self.model, V4FFNModel)
                    else Mode.EXECUTE
                )
                completion = executor.submit(mode)
                completed = 0
                try:
                    code = executor.retire(completion)
                finally:
                    # A valid receipt exists only after the stream completed.
                    # Preserve a partial count even if retire raises on a peer
                    # failure; a device/wait failure must not inspect storage.
                    if mode == Mode.SERVE and completion.completed_work is not None:
                        completed = completion.completed_work
                        if completed:
                            service.completed_work(completed)
                if mode == Mode.EXECUTE and code == Result.OK:
                    completed = 1
                    service.completed_work()
                if code == Result.DRAINED:
                    if not ready:
                        raise RuntimeError("session drained before paired warmup")
                    service.phase(ServicePhase.DRAINING)
                    break
                if code == Result.OK and not completed:
                    raise RuntimeError("successful fused execution completed no work")
                if code not in (Result.OK, Result.IDLE):
                    raise RuntimeError(
                        f"unexpected fused execution result: {code.name}"
                    )
                if not ready:
                    if store.check(warmup_keys):
                        store.set(f"ready/{c.ffn_rank}", "1")
                        service.phase(
                            ServicePhase.READY,
                            startup_milliseconds=int(
                                (time.monotonic() - started) * 1000
                            ),
                        )
                        ready = True
                    elif time.monotonic() > warmup_deadline:
                        raise TimeoutError("paired Attention warmup timed out")
        except BaseException as error:
            failure = error
            if self.executor is not None:
                self.executor.publish_failure(error)
        finally:
            try:
                if self.executor is not None:
                    self.executor.close(abort=failure is not None)
                self.model.close()
            except BaseException as cleanup_error:
                service.phase(
                    ServicePhase.FAILED,
                    error=str(cleanup_error),
                    resources_released=False,
                )
                if failure is not None:
                    raise cleanup_error from failure
                raise
            service.phase(
                ServicePhase.STOPPED if failure is None else ServicePhase.FAILED,
                resources_released=True,
                **({"error": str(failure)} if failure is not None else {}),
            )
        if failure is not None:
            raise failure
