# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bridge profiling RPCs to each Attention or FFN worker's session.

The opt-in mixin wires the connector/session/backend, forwards endpoint.py RPCs
and enforces native/static profiler exclusivity without eager TorchNPU imports.
Attention status advances native lifecycle; FFN uses its receive-loop waiter.
"""

from __future__ import annotations

import os
import socket
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from afd_plugin.profiling import runtime_profiling_enabled
from afd_plugin.profiling.native import RuntimeProfiler

if TYPE_CHECKING:
    import torch
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.profiling.session import WorkerSession
    from afd_plugin.profiling.tracing import DeviceTracer
    from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner


class CAMProfileWorkerMixin:
    device: torch.device
    model_runner: (
        AFDNPUAttentionModelRunnerV2 | AFDNPUFFNModelRunner | AsyncNPUFFNModelRunner
    )
    profiler: TorchNPUProfilerWrapper | None = None
    cam_profile_session: WorkerSession | None = None

    def initialize_cam_profiling(self) -> None:
        if not runtime_profiling_enabled():
            return
        # Keep TorchNPU profiling resources out of workers with profiling disabled.
        from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
        from afd_plugin.profiling.npu import FFNDeviceWaiter
        from afd_plugin.profiling.session import WorkerSession

        connector = self.model_runner.connector
        if not isinstance(connector, CAMAsyncAFDConnector):
            raise ValueError("runtime CAM profiling requires the async CAM connector")
        self.cam_profile_session = WorkerSession(
            connector.profile_worker_id,
            connector.topology.role,
            Path(os.environ.get("AFD_CAM_PROFILE_DIR", "cam-profiles")),
            None,
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "device": str(self.device),
                "cam_rank": connector.world_rank,
                "attention_ranks": connector.attn_size,
                "ffn_ranks": connector.ffn_size,
                "tp_size": connector.tp_size,
            },
            tracer_factory=partial(self._create_cam_tracer, connector),
            profiler_factory=lambda trace_dir, mode: RuntimeProfiler(
                "attention" if connector.topology.role == "attention" else "ffn",
                trace_dir,
                mode,
                graph=connector.topology.role == "ffn"
                and connector.graph_startup.mode == "full",
            ),
        )
        if connector.topology.role == "ffn":
            self.cam_profile_session.device_waiter = FFNDeviceWaiter(
                self.device, self.cam_profile_session
            )
        connector.profile_session = self.cam_profile_session
        self.cam_profile_session.start(self.device)

    def _create_cam_tracer(self, connector: CAMAsyncAFDConnector) -> DeviceTracer:
        from afd_plugin.profiling.tracing import DeviceTracer

        return DeviceTracer(
            self.device,
            connector.profile_worker_id,
        )

    def _require_cam_profile_session(self) -> WorkerSession:
        if self.cam_profile_session is None:
            raise ValueError("runtime CAM profiling was not enabled at startup")
        return self.cam_profile_session

    def cam_profile_prepare(self, command: dict) -> dict:
        return self._cam_profile_command("prepare", command)

    def cam_profile_commit(self, command: dict) -> dict:
        return self._cam_profile_command("commit", command)

    def cam_profile_stop(self, command: dict) -> dict:
        return self._cam_profile_command("stop", command)

    def _cam_profile_command(
        self, action: Literal["prepare", "commit", "stop"], command: dict
    ) -> dict:
        session = self._require_cam_profile_session()
        try:
            if action == "prepare":
                # _active covers delayed starts and clears on automatic stop.
                # shortcut: pinned vLLM state; use a public active API when available.
                return session.prepare(
                    command,
                    backend_busy=(self.profiler is not None and self.profiler._active)
                    or self.model_runner.prof is not None,
                )
            if action == "commit":
                return session.commit(command)
            return session.stop(command)
        except (ValueError, OSError) as exc:
            result = session.status()
            result["error"] = f"{type(exc).__name__}: {exc}"
            return result

    def cam_profile_status(self, command: dict) -> dict:
        session = self._require_cam_profile_session()
        # Attention RPCs share the model thread; FFN RPCs do not.
        if session.role == "attention":
            session.advance_profiler()
        return session.status(command)

    # Ascend 8d4409d6 NPUWorker.profile also owns a native profiler. Its large
    # backend setup remains upstream; this guard adds only runtime exclusivity.
    # Signature matches upstream, including its omitted return annotation.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ### PATCH START: CAM session exclusivity
        session = self.cam_profile_session
        if (
            is_start
            and session is not None
            and (
                session.state not in {"IDLE", "COMPLETED", "FAILED"}
                or not session.profiler_reusable
            )
        ):
            raise ValueError("a runtime CAM profiling session is active")
        # ### PATCH END: CAM session exclusivity
        # This mixin is composed immediately before the pinned NPUWorker.
        return super().profile(is_start, profile_prefix)  # type: ignore[misc]
