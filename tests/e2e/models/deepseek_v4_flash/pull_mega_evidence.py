# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Functional-only Pull observations; device fences exclude performance use."""

from __future__ import annotations

import os
import sys
from functools import wraps
from pathlib import Path
from threading import Lock

import torch
from vllm.distributed import get_tensor_model_parallel_rank

from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_session import PullCAMSession
from afd_plugin.model_executor.npu.async_cam_mega import AsyncCAMMegaExecutor
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker


def install_observers() -> type:
    """Keep counters in each worker, using the harness's evaluation markers."""
    marker = Path(os.environ["AFD_E2E_MRV2_EVIDENCE_START"])
    complete = marker.with_name("evaluation-complete")
    lock = Lock()
    window = ""
    completed = 0
    emitted = False
    combine = CAMAsyncAFDConnector.recv_ffn_output
    native = AsyncCAMMegaExecutor.__call__
    health = AFDNPUFFNWorker.raise_ffn_loop_error_if_any

    def arm() -> str:
        nonlocal window, completed, emitted
        if marker.exists():
            current = marker.read_text()
            with lock:
                if current != window:
                    window, completed, emitted = current, 0, False
        return window

    def observe(connector, active: str, valid: bool) -> None:
        nonlocal completed
        if (
            valid
            and active
            and connector.extra_info.ffn_mega_moe
            and isinstance(connector._session, PullCAMSession)
        ):
            with lock:
                if active == window and not emitted:
                    completed += 1

    def report(worker) -> None:
        nonlocal emitted
        connector = worker.model_runner.connector
        enabled = connector.extra_info.ffn_mega_moe and isinstance(
            connector._session, PullCAMSession
        )
        with lock:
            if emitted:
                return
            emitted = True
            line = (
                "AFD E2E execution: "
                f"window={window} "
                f"dp_rank={worker.vllm_config.parallel_config.data_parallel_rank} "
                f"tp_rank={get_tensor_model_parallel_rank()} "
                f"pull_mega={int(enabled)} native_completed={completed}\n"
            )
        sys.stdout.write(line)
        sys.stdout.flush()

    class Worker:
        def afd_reset_execution_evidence(self, evaluation_window="evaluation") -> None:
            nonlocal window, completed, emitted
            with lock:
                window, completed, emitted = evaluation_window, 0, False

        def afd_log_execution_evidence(self) -> None:
            report(self)

    # Observe plugin-owned methods only; upstream serving code stays untouched.
    @wraps(combine)
    def combined(self, *args, **kwargs):
        active = window
        result = combine(self, *args, **kwargs)
        if active:
            torch.npu.current_stream().synchronize()
            observe(self, active, result.shape[0] > 0)
        return result

    @wraps(native)
    def executed(self):
        active = arm()
        result = native(self)
        if active:
            torch.npu.current_stream().synchronize()
            header = self.batch_info.cpu().tolist()
            observe(
                self.connector,
                active,
                self.op is torch.ops.afd_ascend.afd_mega_moe_pull_native
                and header[0] > 0
                and header[2] >= 0,
            )
        return result

    @wraps(health)
    def checked(self):
        result = health(self)
        if window and complete.exists() and complete.read_text() == window:
            report(self)
        return result

    def observe_shutdown(shutdown):
        @wraps(shutdown)
        def closed(self):
            rank = self.connector.world_rank
            result = shutdown(self)
            if self.connector.is_initialized:
                raise RuntimeError("Pull Mega shutdown retained the transport")
            sys.stdout.write(f"AFD Pull Mega shutdown: rank={rank} complete\n")
            sys.stdout.flush()
            return result

        return closed

    CAMAsyncAFDConnector.recv_ffn_output = combined
    AsyncCAMMegaExecutor.__call__ = executed
    AFDNPUFFNWorker.raise_ffn_loop_error_if_any = checked
    AFDNPUAttentionModelRunnerV2.shutdown = observe_shutdown(
        AFDNPUAttentionModelRunnerV2.shutdown
    )
    AsyncNPUFFNModelRunner.shutdown = observe_shutdown(AsyncNPUFFNModelRunner.shutdown)
    return Worker


Worker = install_observers()
