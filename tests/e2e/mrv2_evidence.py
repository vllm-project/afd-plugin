# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Test-only counters for completed GPU/NPU graph and two-stage DBO execution.

Native RPC resets each named window; a marker arms FFN for that same window.
Attention reports after evaluation; FFN reports through its existing health
RPC. Its daemon does not service HTTP RPC. CUDA V4 also records a bounded set
of successful live transactions, pairing native A input IDs with the actual
first local Hash-layer F input IDs. Capture retains references without D2H.
CUDA target: 5df63a7203cdb9328d824a27b682e77c7b9928be (0.32.0).
Ascend retains the existing 0.30 call without CUDA's randomize_inputs argument.
Remove these wrappers when upstream exposes equivalent live Attention/FFN
replay and microbatch counters.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

from afd_plugin.v1.worker.cuda_graph import AFDGraphRunMode, make_ffn_graph_key
from tests.e2e.models.deepseek_v4_flash.gpu_config import DSV4_GPU_HASH_WINDOWS

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
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import BatchReqState
    from vllm.v1.worker.gpu.ubatch_utils import UBatchState
    from vllm.v1.worker.ubatch_utils import UBatchSlices

    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata
    from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector

__all__ = ["Worker"]

DBO_NUM_STAGES = 2
MRV2_EVIDENCE_START_ENV = "AFD_E2E_MRV2_EVIDENCE_START"
HASH_INPUT_IDS_ENV = "AFD_E2E_HASH_INPUT_IDS"
HASH_INPUT_MAX_TRANSACTIONS = 16
HASH_INPUT_MAX_STAGE_TOKENS = 8
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


@dataclass
class _HashInputSample:
    control: int
    stage: int
    peer: int
    bucket: int
    real_tokens: int
    input_ids: list[int]
    replayed: bool
    completed: bool = True


@dataclass
class _HashInputProbe:
    """Bounded values from native A inputs and the actual F Hash-layer input."""

    graph_required: bool
    control: int = 0
    selected_transactions: int = 0
    selected: bool = False
    payload: AFDControlPayload | None = None
    input_batch: InputBatch | None = None
    slices: UBatchSlices | None = None
    eager_refs: dict[int, torch.Tensor] = field(default_factory=dict)
    graph_refs: dict[ExecutionLayout, dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    samples: list[_HashInputSample] = field(default_factory=list)

    def reset(self) -> None:
        self.control = self.selected_transactions = 0
        self.selected = False
        self.payload = None
        self.input_batch = None
        self.slices = None
        self.eager_refs.clear()
        self.samples.clear()
        # Capture references belong to the live graph and survive windows.

    def observe_control(
        self, payload: AFDControlPayload, armed: bool, window: str
    ) -> None:
        self.payload = payload
        self.selected = False
        if (
            not armed
            or window not in DSV4_GPU_HASH_WINDOWS
            or payload.is_warmup
            or payload.is_graph_capturing
            or payload.is_profile
        ):
            return
        self.control += 1
        small = all(
            0 < int(count) <= HASH_INPUT_MAX_STAGE_TOKENS
            for metadata in payload.dp_metadata_list.values()
            for count in metadata.num_tokens_across_dp_cpu
        )
        if (
            small
            and (not self.graph_required or payload.is_graph_replaying)
            and self.selected_transactions < HASH_INPUT_MAX_TRANSACTIONS
        ):
            self.selected_transactions += 1
            self.selected = True

    def retain_ffn_input(self, stage: int, input_ids: torch.Tensor) -> None:
        assert self.payload is not None
        if self.payload.is_graph_capturing or self.payload.is_warmup:
            key = make_ffn_graph_key(self.payload.dp_metadata_list)
            self.graph_refs.setdefault(key, {})[stage] = input_ids
        else:
            self.eager_refs[stage] = input_ids

    def sample_attention(
        self, connector: P2pNcclAFDConnector, replayed: bool
    ) -> list[_HashInputSample]:
        if not self.selected:
            return []
        assert self.payload is not None
        samples = []
        batch = self.input_batch
        # Exactly one bounded D2H per selected A transaction, after execute.
        ids = (
            [] if batch is None else batch.input_ids[: batch.num_tokens].cpu().tolist()
        )
        for stage, metadata in sorted(self.payload.dp_metadata_list.items()):
            start = 0 if self.slices is None else self.slices[stage].token_slice.start
            bucket = int(metadata.num_tokens_across_dp_cpu[connector.mapping.role_rank])
            real = max(0, min(start + bucket, len(ids)) - start)
            samples.append(
                _HashInputSample(
                    self.control,
                    stage,
                    connector.mapping.role_rank,
                    bucket,
                    real,
                    ids[start : start + real],
                    replayed,
                )
            )
        return samples

    def sample_ffn(
        self, connector: P2pNcclAFDConnector, replayed: bool
    ) -> list[_HashInputSample]:
        if not self.selected:
            return []
        assert self.payload is not None
        samples = []
        refs = (
            self.graph_refs[make_ffn_graph_key(self.payload.dp_metadata_list)]
            if replayed
            else self.eager_refs
        )
        peers = tuple(
            rank - connector.ffn_size for rank in connector.mapping.subgroup_ranks[1:]
        )
        for stage, metadata in sorted(self.payload.dp_metadata_list.items()):
            # Capture only retains this exact compute_ffn_output Tensor. Read
            # its real values after the selected live eager/replay returns.
            ids = refs[stage].cpu().tolist()
            start = 0
            for peer in peers:
                bucket = int(metadata.num_tokens_across_dp_cpu[peer])
                samples.append(
                    _HashInputSample(
                        self.control,
                        stage,
                        peer,
                        bucket,
                        -1,
                        ids[start : start + bucket],
                        replayed,
                    )
                )
                start += bucket
        return samples


def _install_probes() -> type:
    counts = _ExecutionCounts()
    counts_lock = Lock()
    armed = False
    ffn_in_flight = False
    window = ""
    start_marker = Path(os.environ[MRV2_EVIDENCE_START_ENV])
    complete_marker = start_marker.with_name("evaluation-complete")
    attention_replayed = False
    ffn_mode: AFDGraphRunMode | None = None
    native_execute = GPUModelRunner.execute_model
    native_replay = ModelCudaGraphManager.run_fullgraph
    native_prepare = UBatchRunner.prepare
    native_prepare_inputs = GPUModelRunner.prepare_inputs
    native_baseline = os.environ.get("AFD_E2E_NATIVE_BASELINE") == "1"
    native_layout: ExecutionLayout | None = None
    ffn_execute = FFN_RUNNER.execute_model
    ffn_check = FFN_WORKER.raise_ffn_loop_error_if_any
    select_mode = ffn_model_runner.graph_run_mode
    hash_probe = (
        _HashInputProbe(os.environ.get("AFD_E2E_HASH_REQUIRE_GRAPH") == "1")
        if os.environ.get(HASH_INPUT_IDS_ENV) == "1"
        else None
    )

    def log_execution_evidence(
        vllm_config: VllmConfig, *, only_if_armed: bool = False
    ) -> None:
        nonlocal armed, counts
        with counts_lock:
            # A returned native call may still be waiting to submit its
            # counters/samples. Retry on the next existing health RPC; never
            # wait for execution or D2H while holding this lock.
            if ffn_in_flight or (only_if_armed and not armed):
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
                "hash_ids": []
                if hash_probe is None
                else [asdict(sample) for sample in hash_probe.samples],
            },
            separators=(",", ":"),
        )
        # Native worker log decoration splits print's body/newline writes.
        # Keep the frame together; stdout also bypasses logging-level filters.
        sys.stdout.write(
            "AFD E2E execution: "
            f"window={window} "
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

        def afd_reset_execution_evidence(
            self, evaluation_window: str = "evaluation"
        ) -> None:
            nonlocal counts, armed, window
            with counts_lock:
                counts = _ExecutionCounts()
                window = evaluation_window
                armed = True
                if hash_probe is not None:
                    hash_probe.reset()

        def afd_log_execution_evidence(self) -> None:
            log_execution_evidence(self.vllm_config)

    @wraps(native_prepare)
    # Patch reason: native DBO has no AFD metadata to prove real child work.
    # Patch functionality: observe native slices without device synchronization;
    # count only after their parent execute_model completes successfully.
    # Signature/source: UBatchRunner.prepare, vLLM 5df63a7,
    # vllm/v1/worker/gpu/ubatch_utils.py. Native-baseline observation only;
    # remove when upstream exposes completed live microbatch counters.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        result = native_prepare(
            self, input_batch, block_tables, slot_mappings, cg_mode, for_capture
        )
        # ### PATCH START: retain nonempty native request-stage coordinates.
        nonlocal native_layout
        if hash_probe is not None:
            hash_probe.slices = result.slices
        if (
            not for_capture
            and len(result.slices) == DBO_NUM_STAGES
            and all(
                min(stage.token_slice.stop, input_batch.num_tokens)
                > stage.token_slice.start
                for stage in result.slices
            )
        ):
            dp_size = self.parallel_config.data_parallel_size
            native_layout = tuple(
                (index, (stage.num_tokens,) * dp_size)
                for index, stage in enumerate(result.slices)
            )
        # ### PATCH END: retain nonempty native request-stage coordinates.
        return result

    @wraps(native_prepare_inputs)
    # Patch reason: transported IDs must match native request inputs, including
    # FULL replay. Observe the actual InputBatch before it enters the model.
    # Signature/source: GPUModelRunner.prepare_inputs at vLLM 5df63a7.
    # Observation-only delegation; remove with the Hash input acceptance probe.
    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
        num_active_loras: int,
    ) -> InputBatch:
        result = native_prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        )
        # ### PATCH START: retain real native input IDs without capture-time D2H.
        assert hash_probe is not None
        hash_probe.input_batch = result
        # ### PATCH END: retain real native input IDs without capture-time D2H.
        return result

    @wraps(native_replay)
    # Patch reason: configured graph mode does not prove replay occurred.
    # Patch functionality: observe completed native FULL replay unchanged.
    # Signature/source: ModelCudaGraphManager.run_fullgraph, CUDA 5df63a7
    # and Ascend's ced6857 baseline,
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
    # Signature/source: GPUModelRunner.execute_model, CUDA vLLM 5df63a7,
    # vllm/v1/worker/gpu/model_runner.py. Delegation is observation-only.
    # Ascend retains its 0.30 native call; randomize_inputs is CUDA-only.
    def execute(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # ### PATCH START: reset replay observation for this step.
        nonlocal attention_replayed, native_layout
        attention_replayed = False
        native_layout = None
        if hash_probe is not None:
            hash_probe.selected = False
            hash_probe.input_batch = None
            hash_probe.slices = None
        # ### PATCH END: reset replay observation for this step.
        execute_args = (
            self,
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
            context_len,
            valid_dummy_state_slots,
        )
        result = (
            native_execute(*execute_args)
            if current_platform.device_type == "npu"
            else native_execute(*execute_args, randomize_inputs=randomize_inputs)
        )
        # D2H is bounded and runs outside the publication lock.
        hash_samples = (
            hash_probe.sample_attention(self.connector, attention_replayed)
            if hash_probe is not None and armed
            else []
        )
        # ### PATCH START: count only completed evaluation work.
        with counts_lock:
            if hash_probe is not None and armed:
                hash_probe.samples.extend(hash_samples)
            if native_baseline:
                if (
                    armed
                    and not (dummy_run or is_profile)
                    and (scheduler_output.total_num_scheduled_tokens > 0)
                ):
                    counts.record(attention_replayed, native_layout)
                else:
                    counts.previous_replay_layout = None
                return result
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
        # ### PATCH START: arm once for each explicitly named evaluation window.
        nonlocal counts, armed, window
        next_window = start_marker.read_text() if start_marker.exists() else ""
        if next_window and next_window != window:
            with counts_lock:
                counts = _ExecutionCounts()
                window = next_window
                armed = True
                if hash_probe is not None:
                    hash_probe.reset()
        # ### PATCH END: arm once for each explicitly named evaluation window.

    def record_ffn(
        dp_metadata_list,
        is_warmup,
        is_graph_capturing,
        connector: P2pNcclAFDConnector | None = None,
    ) -> None:
        # ### PATCH START: count completed live FFN work.
        live = not (is_warmup or is_graph_capturing)
        replayed = ffn_mode is AFDGraphRunMode.REPLAY
        hash_samples = []
        if hash_probe is not None and armed and live:
            assert connector is not None
            # ffn_in_flight remains set while sampling outside the lock.
            hash_samples = hash_probe.sample_ffn(connector, replayed)
        with counts_lock:
            if armed and live:
                assert dp_metadata_list is not None and ffn_mode is not None
                layout = (
                    make_ffn_graph_key(dp_metadata_list)
                    if len(dp_metadata_list) == DBO_NUM_STAGES
                    else None
                )
                counts.record(replayed, layout)
                if hash_probe is not None:
                    hash_probe.samples.extend(hash_samples)
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
        nonlocal ffn_in_flight, armed
        arm_ffn()
        with counts_lock:
            ffn_in_flight = True
        try:
            if hash_probe is not None:
                hash_probe.eager_refs.clear()
            ffn_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
                is_graph_replaying=is_graph_replaying,
            )
            record_ffn(dp_metadata_list, is_warmup, is_graph_capturing, self.connector)
        except BaseException:
            with counts_lock:
                armed = False
            raise
        finally:
            with counts_lock:
                ffn_in_flight = False

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
        nonlocal ffn_in_flight, armed
        arm_ffn()
        with counts_lock:
            ffn_in_flight = True
        try:
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
        except BaseException:
            with counts_lock:
                armed = False
            raise
        finally:
            with counts_lock:
                ffn_in_flight = False
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
        if armed and complete_marker.exists() and complete_marker.read_text() == window:
            log_execution_evidence(self.vllm_config, only_if_armed=True)
        # ### PATCH END: freeze once after the completed evaluation window.

    GPUModelRunner.execute_model = execute
    ModelCudaGraphManager.run_fullgraph = replay
    if native_baseline or hash_probe is not None:
        UBatchRunner.prepare = prepare
    if not native_baseline:
        ffn_model_runner.graph_run_mode = mode
        FFN_RUNNER.execute_model = (
            execute_npu_ffn if current_platform.device_type == "npu" else execute_ffn
        )
        FFN_WORKER.raise_ffn_loop_error_if_any = check_ffn
    if hash_probe is not None:
        # Lazy imports keep CUDA-only model/connector probes out of Ascend
        # workers and out of native controls.
        from afd_plugin.connectors.gpu.p2p import P2pNcclAFDControlPlane
        from afd_plugin.model_executor.models.deepseek_v4 import AFDDeepseekV4Model

        send_control = P2pNcclAFDControlPlane.send_dp_metadata_list
        recv_control = P2pNcclAFDControlPlane.recv_dp_metadata_list
        compute_hash_ffn = AFDDeepseekV4Model.compute_ffn_output

        @wraps(send_control)
        # Patch reason: A/F samples need the existing control transaction order.
        # Signature/source: P2pNcclAFDControlPlane.send_dp_metadata_list.
        # Observation only; no production payload or wire changes.
        def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
            # ### PATCH START: number live controls within the armed window.
            hash_probe.observe_control(payload, armed, window)
            # ### PATCH END: number live controls within the armed window.
            send_control(self, payload)

        @wraps(recv_control)
        # Patch reason: the connector daemon cannot receive HTTP reset RPCs.
        # Signature/source: P2pNcclAFDControlPlane.recv_dp_metadata_list.
        # Observation only; pair the received control in its original order.
        def recv_dp_metadata_list(self) -> AFDControlPayload:
            payload = recv_control(self)
            # ### PATCH START: pair the live receive with its named window.
            arm_ffn()
            hash_probe.observe_control(payload, armed, window)
            # ### PATCH END: pair the live receive with its named window.
            return payload

        @wraps(compute_hash_ffn)
        # Patch reason: receive-buffer shape cannot prove the Hash MoE input.
        # Signature/source: AFDDeepseekV4Model.compute_ffn_output.
        # Retain the exact Tensor passed to the first local Hash FFN; capture
        # performs no D2H. Remove with the bounded live Hash input probe.
        def compute_ffn_output(
            self,
            hidden_states: torch.Tensor,
            layer_idx: int,
            *,
            input_ids: torch.Tensor | None,
        ) -> torch.Tensor:
            result = compute_hash_ffn(
                self, hidden_states, layer_idx, input_ids=input_ids
            )
            # ### PATCH START: retain the actual first local Hash-layer input.
            if (
                layer_idx == self.start_layer
                and layer_idx < self.config.num_hash_layers
            ):
                assert input_ids is not None
                stage = (
                    get_forward_context().additional_kwargs["afd_metadata"].stage_idx
                )
                hash_probe.retain_ffn_input(stage, input_ids)
            # ### PATCH END: retain the actual first local Hash-layer input.
            return result

        GPUModelRunner.prepare_inputs = prepare_inputs
        P2pNcclAFDControlPlane.send_dp_metadata_list = send_dp_metadata_list
        P2pNcclAFDControlPlane.recv_dp_metadata_list = recv_dp_metadata_list
        AFDDeepseekV4Model.compute_ffn_output = compute_ffn_output
    return Worker


Worker = _install_probes()
