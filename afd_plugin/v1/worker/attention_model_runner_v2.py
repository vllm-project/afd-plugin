# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for AFD GPU ModelRunnerV2 execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker import utils as v2_worker_utils
from vllm.v1.worker.gpu import cudagraph_utils as v2_cudagraph_utils
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, UBatchState

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDForwardContextMetadata,
)
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.v1.worker.attention_metadata import (
    AFDMetadataProviderMixin,
    _resolve_world_ranks,
)
from afd_plugin.validation import validate_gpu_model_runner_v2_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

_AFD_FULLGRAPH_HOOK_MARKER = "_afd_fullgraph_replay_hook_active"


class _AFDCaptureEventTracker:
    """Consume each FULL descriptor once for warmup, then once for capture."""

    def __init__(
        self,
        capture_descs: list[v2_cudagraph_utils.BatchExecutionDescriptor],
    ) -> None:
        self._expected_events = [
            (desc, is_warmup) for desc in capture_descs for is_warmup in (True, False)
        ]
        self._event_index = 0

    def consume(
        self,
        num_reqs: int,
        num_tokens: int,
        num_ubatches: int = 1,
    ) -> tuple[v2_cudagraph_utils.BatchExecutionDescriptor, bool]:
        if self._event_index >= len(self._expected_events):
            raise RuntimeError(
                "AFD ModelRunnerV2 CUDA Graph observed extra capture input preparation",
            )
        desc, is_warmup = self._expected_events[self._event_index]
        self._event_index += 1
        if (
            num_reqs != desc.num_reqs
            or num_tokens != desc.num_tokens
            or num_ubatches != desc.num_ubatches
        ):
            raise RuntimeError(
                "AFD ModelRunnerV2 CUDA Graph capture descriptor/order drift: "
                f"expected ({desc.num_reqs}, {desc.num_tokens}, {desc.num_ubatches}), "
                f"got ({num_reqs}, {num_tokens}, {num_ubatches})",
            )
        return desc, is_warmup

    def assert_complete(self) -> None:
        if self._event_index != len(self._expected_events):
            raise RuntimeError(
                "AFD ModelRunnerV2 CUDA Graph capture input-preparation "
                "call count drift: "
                f"expected {len(self._expected_events)}, got {self._event_index}",
            )


@contextmanager
def _use_afd_ubatch_preparation(
    runner: AFDAttentionModelRunnerV2,
    event_tracker: _AFDCaptureEventTracker | None = None,
) -> Iterator[None]:
    """Publish one transaction from native prepared state before stage execution."""
    ubatch_runner = runner.ubatch_runner
    if ubatch_runner is None:
        yield
        return
    state = vars(ubatch_runner)
    had_override = "prepare" in state
    previous_override = state.get("prepare")
    original_prepare = ubatch_runner.prepare

    # Patch reason: native MRV2 creates stage contexts through an imported
    # factory alias, bypassing the ordinary AFD context provider. Its graph
    # branch also bypasses prepare_inputs_to_capture.
    # Patch functionality: decorate native stage state and publish one complete
    # A/F payload before threads start (or before FULL replay).
    # Signature: matches UBatchRunner.prepare at vLLM ced6857.
    # Delegation exception: keep native slicing/attention preparation in the
    # bound method. Remove this seam when upstream offers a prepared-state
    # callback; no runner class or module-global function is replaced here.
    def prepare(
        self: UBatchRunner,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: bind AFD to native stage state outside graph execution.
        prepared = original_prepare(
            input_batch, block_tables, slot_mappings, cg_mode, for_capture
        )
        if event_tracker is not None:
            _, is_warmup = event_tracker.consume(
                input_batch.num_reqs,
                input_batch.num_tokens_after_padding,
                len(prepared.slices),
            )
            runner._is_warmup = is_warmup
            runner._afd_is_graph_capturing = not is_warmup
            runner._afd_is_graph_replaying = False
        else:
            runner._afd_is_graph_replaying = cg_mode == CUDAGraphMode.FULL
        runner.install_mrv2_ubatch_metadata(
            prepared.slices, prepared.forward_contexts, input_batch.num_tokens
        )
        runner.send_dp_metadata(None, prepared.slices)
        return prepared
        # ### PATCH END: bind AFD to native stage state outside graph execution.

    ubatch_runner.prepare = MethodType(prepare, ubatch_runner)
    try:
        yield
    finally:
        if had_override:
            ubatch_runner.prepare = previous_override
        else:
            del ubatch_runner.prepare


@contextmanager
def _use_afd_capture_input_preparation(
    runner: AFDAttentionModelRunnerV2,
    event_tracker: _AFDCaptureEventTracker,
) -> Iterator[None]:
    """Scope AFD capture events to the native input-preparation symbol."""

    original_prepare = v2_cudagraph_utils.prepare_inputs_to_capture

    # Patch reason: native ModelCudaGraphManager.capture calls this exact
    # module symbol once before each warmup/formal-capture forward.
    # Patch functionality: preserve native preparation and publish the
    # matching AFD event before the forward starts.
    # Signature: matches vLLM v0.30.0 prepare_inputs_to_capture exactly,
    # including the 0.30.0 trailing pcp_manager parameter.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # prepare_inputs_to_capture; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    def prepare_inputs_to_capture(
        num_reqs: int,
        num_tokens: int,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[v2_worker_utils.AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        full_cudagraph: bool,
        max_query_len: int | None = None,
        pcp_manager: PCPManager | None = None,
    ) -> v2_cudagraph_utils.AttentionState:
        # ### PATCH START: stage one exact AFD capture event.
        attention_state = original_prepare(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            model_state=model_state,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            full_cudagraph=full_cudagraph,
            max_query_len=max_query_len,
            pcp_manager=pcp_manager,
        )
        desc, is_warmup = event_tracker.consume(num_reqs, num_tokens)
        runner._is_warmup = is_warmup
        runner._afd_is_graph_capturing = not is_warmup
        runner._afd_pending_metadata = runner.build_afd_metadata(
            None,
            int(desc.num_tokens),
        )
        runner.send_dp_metadata(
            runner.build_capture_dp_metadata(int(desc.num_tokens)),
            None,
        )
        runner._afd_suppress_metadata_send = True
        return attention_state
        # ### PATCH END: stage one exact AFD capture event.

    v2_cudagraph_utils.prepare_inputs_to_capture = prepare_inputs_to_capture
    try:
        yield
    finally:
        v2_cudagraph_utils.prepare_inputs_to_capture = original_prepare


@contextmanager
def _use_afd_fullgraph_replay_hook(
    runner: AFDAttentionModelRunnerV2,
    real_tokens: int,
) -> Iterator[None]:
    """Scope AFD FULL-replay control to one CUDA graph manager instance."""

    manager = runner.cudagraph_manager
    if manager is None:
        raise RuntimeError(
            "AFD FULL graph replay hook requires an initialized graph manager",
        )
    manager_state = manager.__dict__
    if _AFD_FULLGRAPH_HOOK_MARKER in manager_state:
        raise RuntimeError("AFD FULL graph replay hook is already active")

    had_instance_override = "run_fullgraph" in manager_state
    previous_instance_override = manager_state.get("run_fullgraph")
    original_run_fullgraph = manager.run_fullgraph
    manager_state[_AFD_FULLGRAPH_HOOK_MARKER] = True

    # Patch reason: native FULL replay bypasses ForwardContext creation, so the
    # execute-scoped AFD provider cannot publish runtime control.
    # Patch functionality: wrap only this manager instance and publish one
    # ordinary padded control payload immediately before each native replay.
    # Signature: matches vLLM v0.30.0 ModelCudaGraphManager.run_fullgraph exactly.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # ModelCudaGraphManager.run_fullgraph; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native replay remains wholly in the saved bound
    # method; this scope owns only the AFD pre-replay control seam.
    # Removal/upstream plan: delete this hook when vLLM exposes a per-manager
    # callback immediately before FULL graph replay.
    def run_fullgraph(
        self: v2_cudagraph_utils.ModelCudaGraphManager,
        desc: v2_cudagraph_utils.BatchExecutionDescriptor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        # ### PATCH START: publish one AFD pre-replay payload.
        if desc.num_ubatches > 1:
            # Native execute_model already prepared live stage inputs and our
            # prepare seam sent their complete control payload.
            assert runner._afd_pending_metadata is not None
            assert runner._afd_pending_metadata.num_stages == desc.num_ubatches
            return original_run_fullgraph(desc)
        previous_is_graph_replaying = getattr(
            runner,
            "_afd_is_graph_replaying",
            False,
        )
        try:
            padded_tokens = int(desc.num_tokens)
            metadata = runner.build_afd_metadata(None, real_tokens)
            metadata.tokens_lens = [padded_tokens]
            runner._afd_pending_metadata = metadata
            runner._afd_suppress_metadata_send = True
            runner._is_warmup = False
            runner._afd_is_graph_capturing = False
            runner._afd_is_graph_replaying = True
            runner.send_dp_metadata(
                runner.build_capture_dp_metadata(padded_tokens),
                None,
            )
            result = original_run_fullgraph(desc)
        finally:
            runner._afd_is_graph_replaying = previous_is_graph_replaying
        # ### PATCH END: publish one AFD pre-replay payload.
        return result

    try:
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
        yield
    finally:
        if had_instance_override:
            manager.run_fullgraph = previous_instance_override
        else:
            del manager.run_fullgraph
        del manager_state[_AFD_FULLGRAPH_HOOK_MARKER]


@contextmanager
def _use_afd_execution_context(
    runner: AFDAttentionModelRunnerV2,
    real_tokens: int,
) -> Iterator[None]:
    """Scope AFD replay, metadata-provider, and runner sidecar state."""

    use_fullgraph_replay_hook = (
        runner.vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
        and runner.cudagraph_manager is not None
    )
    previous_metadata = runner._afd_pending_metadata
    previous_suppress_send = runner._afd_suppress_metadata_send
    previous_is_warmup = runner._is_warmup
    previous_is_graph_capturing = runner._afd_is_graph_capturing
    previous_is_graph_replaying = getattr(runner, "_afd_is_graph_replaying", False)
    runner._afd_is_graph_replaying = False

    replay_scope = (
        _use_afd_fullgraph_replay_hook(runner, real_tokens)
        if use_fullgraph_replay_hook
        else nullcontext()
    )
    try:
        with (
            replay_scope,
            _use_afd_ubatch_preparation(runner),
            use_afd_metadata_provider(
                runner.install_afd_metadata_on_forward_context,
            ),
        ):
            yield
    finally:
        runner._afd_pending_metadata = previous_metadata
        runner._afd_suppress_metadata_send = previous_suppress_send
        runner._is_warmup = previous_is_warmup
        runner._afd_is_graph_capturing = previous_is_graph_capturing
        runner._afd_is_graph_replaying = previous_is_graph_replaying


class AFDAttentionModelRunnerV2(AFDMetadataProviderMixin, GPUModelRunnerV2):
    """Thin AFD seam over native vLLM 0.30.0 GPU ModelRunnerV2.

    Native V2 retains request state, input preparation, Attention/KV handling,
    sampling, and output ownership. The inherited AFD metadata methods are
    pure connector/context plumbing and are reused without porting V1's
    execution, dummy, or graph lifecycle methods.
    """

    afd_expected_role = "attention"

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        validate_gpu_model_runner_v2_config(
            vllm_config,
            expected_role="attention",
            device_type=device.type,
        )
        super().__init__(vllm_config, device)
        connector: AFDConnectorBase | None = None
        try:
            self.afd_config = self.parse_config(self.vllm_config)
            rank, local_rank = _resolve_world_ranks()
            connector = AFDConnectorFactory.create_connector(
                rank,
                local_rank,
                self.vllm_config,
                self.afd_config,
            )
            self.connector = connector
            # The connector rendezvous is deferred to the end of ``load_model()``
            # so Attention and FFN weight loading overlap, matching the V1
            # Attention runner lifecycle.
            if connector.control_plane is None:
                raise RuntimeError(
                    "AFD ModelRunnerV2 requires a control-plane-driven connector",
                )
            self._is_warmup = False
            self._afd_is_graph_capturing = False
            self._afd_is_graph_replaying = False
            self._afd_pending_metadata: AFDForwardContextMetadata | None = None
            self._afd_suppress_metadata_send = False
            self._afd_transaction_counter = 0
            self.prof = create_afd_gpu_profiler("attention")
        except BaseException:
            try:
                if connector is not None:
                    connector.close()
            finally:
                super().shutdown()
            raise

    def _afd_num_tokens_for_context(self, forward_context: ForwardContext) -> int:
        # vLLM 0.30.0's token chain is
        # scheduler_output.total_num_scheduled_tokens ->
        # dispatch_cg_and_sync_dp(..., need_eager=is_profile or skip_compiled)
        # -> ordinary ModelCudaGraphManager.dispatch() or the forced-eager
        # descriptor -> BatchExecutionDescriptor.num_tokens ->
        # prepare_inputs(...).num_tokens_after_padding. For eager execution,
        # the ordinary no-graph fallback and profile/dummy forced-eager branch
        # both preserve the native batch-descriptor token count.
        batch_descriptor = forward_context.batch_descriptor
        if batch_descriptor is None:
            raise RuntimeError(
                "AFD ModelRunnerV2 requires a native eager BatchDescriptor",
            )
        return int(batch_descriptor.num_tokens)

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: native V2 load_model has no AFD connector lifecycle.
    # Patch functionality: initialize the AFD connector after native weight
    # loading so Attention and FFN model loading overlap across roles.
    # Signature: matches vLLM v0.30.0 GPUModelRunnerV2.load_model exactly,
    # including the upstream *args/**kwargs pass-through.
    def load_model(
        self,
        load_dummy_weights: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        if not self.connector.is_initialized:
            self.connector.init_afd_connector()

    # Patch reason: vLLM 0.30.0 prepares FULL graph inputs before each warmup
    # and formal capture forward, outside torch.cuda.graph, but does not expose
    # that lifecycle to AFD's control plane.
    # Patch functionality: temporarily wrap the exact upstream input-preparation
    # symbol so each native FULL descriptor publishes one warmup and one capture
    # payload before its forward, while the provider installs the pending AFD
    # sidecar without sending control from inside torch.cuda.graph. The pairing
    # is preserved under profile_only memory profiling: native capture() still
    # prepares inputs exactly twice per captured descriptor (warmup + capture);
    # profiling only truncates the descriptor list to
    # manager._max_full_descs_to_capture, which this wrapper mirrors.
    # Signature: matches vLLM v0.30.0 GPUModelRunner.capture_model exactly
    # (profile_only keyword-only parameter added in 0.30.0) and is forwarded
    # unchanged to super().
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.capture_model; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native capture remains wholly in super(); this
    # wrapper owns only the temporary AFD lifecycle seam.
    # Removal/upstream plan: delete this wrapper when vLLM exposes graph
    # warmup/capture metadata hooks around prepare_inputs_to_capture.
    def capture_model(self, *, profile_only: bool = False) -> int:
        # ### PATCH START: publish AFD FULL graph warmup/capture control.
        manager = self.cudagraph_manager
        capture_descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])
        max_full_descs = manager._max_full_descs_to_capture
        if max_full_descs is not None:
            # Mirror the native profile-only truncation in
            # ModelCudaGraphManager.capture so the event pairing stays exact.
            capture_descs = capture_descs[:max_full_descs]
        if not capture_descs:
            raise RuntimeError(
                "AFD ModelRunnerV2 CUDA Graph expected at least one FULL "
                "capture descriptor",
            )
        for desc in capture_descs:
            if desc.cg_mode != CUDAGraphMode.FULL:
                raise RuntimeError(
                    "AFD ModelRunnerV2 CUDA Graph expected only FULL capture "
                    "descriptors",
                )
        event_tracker = _AFDCaptureEventTracker(capture_descs)
        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_warmup = self._is_warmup
        previous_is_graph_capturing = self._afd_is_graph_capturing
        previous_is_graph_replaying = self._afd_is_graph_replaying

        try:
            with (
                _use_afd_capture_input_preparation(self, event_tracker),
                _use_afd_ubatch_preparation(self, event_tracker),
            ):
                with use_afd_metadata_provider(
                    self.install_afd_metadata_on_forward_context,
                ):
                    result = super().capture_model(profile_only=profile_only)
                event_tracker.assert_complete()
            return result
        finally:
            self._afd_pending_metadata = previous_metadata
            self._afd_suppress_metadata_send = previous_suppress_send
            self._is_warmup = previous_is_warmup
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_is_graph_replaying = previous_is_graph_replaying
        # ### PATCH END: publish AFD FULL graph warmup/capture control.

    # Patch reason: native V2 creates ForwardContext inside execute_model, so
    # AFD must install its sidecar at that exact context-construction seam.
    # Patch functionality: delegate all request/input/Attention/KV/sampling/
    # output work to native V2 while temporarily installing AFD metadata.
    # Signature: matches vLLM v0.30.0 GPUModelRunnerV2.execute_model exactly,
    # including the 0.30.0 trailing context_len and valid_dummy_state_slots
    # parameters, which are forwarded by keyword.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.execute_model; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: the upstream method is intentionally not copied;
    # only this narrow provider/profiler wrapper is AFD-specific.
    # Removal/upstream plan: delete this wrapper when vLLM exposes a plugin
    # ForwardContext/set_forward_context sidecar/provider hook.
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # ### PATCH START: scope AFD metadata provider/replay and profiler step.
        step_afd_gpu_profiler(self.prof)
        with _use_afd_execution_context(
            self,
            int(scheduler_output.total_num_scheduled_tokens),
        ):
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                context_len=context_len,
                valid_dummy_state_slots=valid_dummy_state_slots,
            )
        # ### PATCH END: scope AFD metadata provider/replay and profiler step.

    # Patch reason: native V2 shutdown does not know about AFD's profiler,
    # connector, or pending metadata sidecar.
    # Patch functionality: preserve delegated native cleanup and guarantee all
    # AFD cleanup layers run when any earlier layer raises.
    # Signature: matches vLLM v0.30.0 GPUModelRunnerV2.shutdown exactly.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.shutdown; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native resource release remains in super().shutdown;
    # this override contains only AFD-specific finalizers.
    # Removal/upstream plan: delete this wrapper if AFD-owned lifecycle moves
    # to a shared composition/factory owner; it never copies native cleanup.
    def shutdown(self) -> None:
        # ### PATCH START: guarantee profiler/native/connector cleanup.
        try:
            stop_afd_gpu_profiler(self.prof)
        finally:
            try:
                super().shutdown()
            finally:
                self._afd_pending_metadata = None
                self.connector.close()
        # ### PATCH END: guarantee profiler/native/connector cleanup.


__all__ = ["AFDAttentionModelRunnerV2"]
