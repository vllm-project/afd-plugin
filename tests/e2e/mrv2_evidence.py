# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Test-only counters for completed GPU/NPU graph and two-stage DBO execution.

After Attention readiness, native RPC resets its counters and a marker arms FFN
on its first evaluation call. Attention reports after evaluation; FFN reports
through its existing health RPC after evaluation. Its daemon does not service
HTTP RPC.
No per-step logging or runner-class identity checks are used.
Pinned vLLM: ced6857afa0ea7b2e3f0846a62e1394e90f15607 (0.30.0).
Remove these wrappers when upstream exposes equivalent live Attention/FFN
replay and microbatch counters.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from afd_plugin.v1.worker.cuda_graph import AFDGraphRunMode, make_ffn_graph_key

if current_platform.device_type == "npu":
    from afd_plugin.v1.worker.npu import ffn_model_runner, ffn_worker

    FFN_RUNNER = ffn_model_runner.AFDNPUFFNModelRunner
    FFN_WORKER = ffn_worker.AFDNPUFFNWorker
else:
    from afd_plugin.v1.worker import ffn_model_runner, ffn_worker

    FFN_RUNNER = ffn_model_runner.GPUFFNModelRunner
    FFN_WORKER = ffn_worker.AFDFFNWorker

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import DPMetadata
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from afd_plugin.connectors import AFDDPMetadata

__all__ = ["Worker"]

DBO_NUM_STAGES = 2
MRV2_EVIDENCE_START_ENV = "AFD_E2E_MRV2_EVIDENCE_START"
ExecutionLayout = tuple[tuple[int, tuple[int, ...]], ...]


@dataclass
class _ExecutionCounts:
    graph_replays: int = 0
    two_stage_steps: int = 0
    two_stage_graph_replays: int = 0
    two_stage_layouts: set[ExecutionLayout] = field(default_factory=set)
    consecutive_graph_layouts: set[ExecutionLayout] = field(default_factory=set)
    previous_replay_layout: ExecutionLayout | None = None

    def record(self, replayed: bool, layout: ExecutionLayout | None) -> None:
        self.graph_replays += int(replayed)
        if layout is not None:
            self.two_stage_steps += 1
            self.two_stage_layouts.add(layout)
            self.two_stage_graph_replays += int(replayed)
            if replayed and layout == self.previous_replay_layout:
                self.consecutive_graph_layouts.add(layout)
        self.previous_replay_layout = layout if replayed else None


def _install_probes() -> type:
    counts = _ExecutionCounts()
    counts_lock = Lock()
    armed = False
    ffn_window_started = False
    start_marker = Path(os.environ[MRV2_EVIDENCE_START_ENV])
    complete_marker = start_marker.with_name("evaluation-complete")
    attention_replayed = False
    ffn_mode: AFDGraphRunMode | None = None
    native_execute = GPUModelRunner.execute_model
    native_replay = ModelCudaGraphManager.run_fullgraph
    ffn_execute = FFN_RUNNER.execute_model
    ffn_check = FFN_WORKER.raise_ffn_loop_error_if_any
    select_mode = ffn_model_runner.graph_run_mode

    def log_execution_evidence(
        vllm_config: VllmConfig, *, only_if_armed: bool = False
    ) -> None:
        nonlocal armed, counts
        with counts_lock:
            if only_if_armed and not armed:
                return
            armed = False
            # All writers use this lock. Detach the completed window so its
            # counters/sets stay immutable while encoding and writing stdout.
            snapshot = counts
            counts = _ExecutionCounts()
        layouts = json.dumps(
            {
                "two_stage": sorted(snapshot.two_stage_layouts),
                "consecutive_graph": sorted(snapshot.consecutive_graph_layouts),
            },
            separators=(",", ":"),
        )
        # Native worker log decoration splits print's body/newline writes.
        # Keep the frame together; stdout also bypasses logging-level filters.
        sys.stdout.write(
            "AFD E2E execution: "
            f"dp_rank={vllm_config.parallel_config.data_parallel_rank} "
            f"tp_rank={get_tensor_model_parallel_rank()} "
            f"graph_replays={snapshot.graph_replays} "
            f"two_stage_steps={snapshot.two_stage_steps} "
            f"two_stage_graph_replays={snapshot.two_stage_graph_replays} "
            f"layouts={layouts}\n",
        )
        sys.stdout.flush()

    class Worker:
        """Native worker extension; counters are private to this worker process."""

        vllm_config: VllmConfig

        def afd_reset_execution_evidence(self) -> None:
            nonlocal counts, armed
            with counts_lock:
                counts = _ExecutionCounts()
                armed = True

        def afd_log_execution_evidence(self) -> None:
            log_execution_evidence(self.vllm_config)

    @wraps(native_replay)
    # Patch reason: configured graph mode does not prove replay occurred.
    # Patch functionality: observe completed native FULL replay unchanged.
    # Signature/source: ModelCudaGraphManager.run_fullgraph, vLLM ced6857,
    # vllm/v1/worker/gpu/cudagraph_utils.py. Delegation is observation-only.
    def replay(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        result = native_replay(self, desc)
        # ### PATCH START: observe completed native replay.
        nonlocal attention_replayed
        attention_replayed = True
        # ### PATCH END: observe completed native replay.
        return result

    @wraps(native_execute)
    # Patch reason: graph/DBO configuration alone can pass accuracy via eager.
    # Patch functionality: count completed live replay and nonempty microbatches.
    # Signature/source: GPUModelRunner.execute_model, vLLM ced6857,
    # vllm/v1/worker/gpu/model_runner.py. Delegation is observation-only.
    def execute(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # ### PATCH START: reset replay observation for this step.
        nonlocal attention_replayed
        attention_replayed = False
        # ### PATCH END: reset replay observation for this step.
        result = native_execute(
            self,
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
            context_len,
            valid_dummy_state_slots,
        )
        # ### PATCH START: count only completed evaluation work.
        with counts_lock:
            metadata = self._afd_pending_metadata
            if armed and not (dummy_run or is_profile) and metadata is not None:
                real_tokens = metadata.tokens_unpadded_lens
                if any(count > 0 for count in real_tokens):
                    layout: ExecutionLayout | None = None
                    if metadata.num_stages == DBO_NUM_STAGES and all(
                        count > 0 for count in real_tokens
                    ):
                        dp_size = (
                            1
                            if self.afd_config.async_dp
                            else self.vllm_config.parallel_config.data_parallel_size
                        )
                        layout = tuple(
                            (stage, (tokens,) * dp_size)
                            for stage, tokens in enumerate(metadata.tokens_lens)
                        )
                    counts.record(attention_replayed, layout)
                else:
                    counts.previous_replay_layout = None
            else:
                counts.previous_replay_layout = None
        # ### PATCH END: count only completed evaluation work.
        return result

    @wraps(select_mode)
    # Patch reason: FFN replay is selected in the connector-driven runtime.
    # Patch functionality: observe the actual selection without recomputing it.
    # Signature/source: afd_plugin.v1.worker.cuda_graph.graph_run_mode.
    # Delegation is observation-only; remove with the FFN execution probe.
    def mode(
        *,
        is_warmup: bool,
        is_graph_capturing: bool,
        is_graph_replaying: bool,
        graph_enabled: bool,
        graph_exists: bool,
    ) -> AFDGraphRunMode:
        # ### PATCH START: observe native FFN mode selection.
        nonlocal ffn_mode
        ffn_mode = select_mode(
            is_warmup=is_warmup,
            is_graph_capturing=is_graph_capturing,
            is_graph_replaying=is_graph_replaying,
            graph_enabled=graph_enabled,
            graph_exists=graph_exists,
        )
        return ffn_mode
        # ### PATCH END: observe native FFN mode selection.

    def arm_ffn() -> None:
        # ### PATCH START: arm FFN once, before its first evaluation call.
        nonlocal counts, armed, ffn_window_started
        if not ffn_window_started and not armed and start_marker.exists():
            with counts_lock:
                counts = _ExecutionCounts()
                armed = ffn_window_started = True
        # ### PATCH END: arm FFN once, before its first evaluation call.

    def record_ffn(dp_metadata_list, is_warmup, is_graph_capturing) -> None:
        # ### PATCH START: count completed live FFN work.
        with counts_lock:
            if armed and not (is_warmup or is_graph_capturing):
                assert dp_metadata_list is not None and ffn_mode is not None
                replayed = ffn_mode is AFDGraphRunMode.REPLAY
                layout = (
                    make_ffn_graph_key(dp_metadata_list)
                    if len(dp_metadata_list) == DBO_NUM_STAGES
                    else None
                )
                counts.record(replayed, layout)
            else:
                counts.previous_replay_layout = None
        # ### PATCH END: count completed live FFN work.

    @wraps(ffn_execute)
    # Patch reason: Attention replay does not prove matching FFN replay ran.
    # Patch functionality: count FFN work only after the real call returns.
    # Signature/source: afd_plugin.v1.worker.ffn_model_runner.execute_model.
    # Delegation is observation-only; remove when FFN exposes equivalent counts.
    def execute_ffn(
        self,
        scheduler_output: SchedulerOutput | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
    ) -> None:
        arm_ffn()
        ffn_execute(
            self,
            scheduler_output,
            intermediate_tensors,
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
            is_graph_replaying=is_graph_replaying,
        )
        record_ffn(dp_metadata_list, is_warmup, is_graph_capturing)

    @wraps(ffn_execute)
    # Patch reason: sync NPU FFN has a profile argument in its concrete API.
    # Patch functionality: count completed connector-driven work in the same
    # evaluation window as Attention. Signature matches AFDNPUFFNModelRunner
    # execute_model. Observation only; remove with the test-only GPU probe.
    def execute_npu_ffn(
        self,
        scheduler_output: SchedulerOutput | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_profile: bool = False,
        is_graph_replaying: bool = False,
    ) -> None:
        # ### PATCH START: observe NPU completion without adding device fences.
        arm_ffn()
        ffn_execute(
            self,
            scheduler_output,
            intermediate_tensors,
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
            is_profile=is_profile,
            is_graph_replaying=is_graph_replaying,
        )
        record_ffn(dp_metadata_list, is_warmup or is_profile, is_graph_capturing)
        # ### PATCH END: observe NPU completion without adding device fences.

    @wraps(ffn_check)
    # Patch reason: the FFN daemon does not service HTTP collective RPC.
    # Patch functionality: report completed work through its existing health RPC.
    # Signature/source: AFDFFNWorker.raise_ffn_loop_error_if_any, afd_plugin/v1/
    # worker/ffn_worker.py. Delegation is observation-only, before GPU teardown.
    # Remove with the FFN probe when equivalent native counters are available.
    def check_ffn(self) -> None:
        nonlocal armed
        try:
            ffn_check(self)
        except BaseException:
            # ### PATCH START: preserve failure without later success evidence.
            with counts_lock:
                armed = False
            # ### PATCH END: preserve failure without later success evidence.
            raise
        # ### PATCH START: freeze once after the completed evaluation window.
        if armed and complete_marker.exists():
            log_execution_evidence(self.vllm_config, only_if_armed=True)
        # ### PATCH END: freeze once after the completed evaluation window.

    GPUModelRunner.execute_model = execute
    ModelCudaGraphManager.run_fullgraph = replay
    ffn_model_runner.graph_run_mode = mode
    FFN_RUNNER.execute_model = (
        execute_npu_ffn if current_platform.device_type == "npu" else execute_ffn
    )
    FFN_WORKER.raise_ffn_loop_error_if_any = check_ffn
    return Worker


Worker = _install_probes()
