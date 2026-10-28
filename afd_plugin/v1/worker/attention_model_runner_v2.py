# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for AFD GPU ModelRunnerV2 execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker import utils as v2_worker_utils
from vllm.v1.worker.gpu import cudagraph_utils as v2_cudagraph_utils
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker.utils import AttentionGroup

from afd_plugin.compat.patches.gpu_mrv2_ubatching import (
    use_gpu_mrv2_dbo_dispatch,
    use_gpu_mrv2_ubatch_factory,
)
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
# V4 CuTe compressor positions/slot pointers require eight int64 elements.
_V4_INPUT_ALIGNMENT_ELEMENTS = 8


class _AFDV4UBatchInputs:
    """Stable aligned prefixes shared by every layer in each V4 stage."""

    def __init__(
        self,
        input_buffers: InputBuffers,
        native_slot_mappings: torch.Tensor,
        num_ubatches: int,
    ) -> None:
        capacity = input_buffers.max_num_tokens
        num_kv_groups = native_slot_mappings.shape[0]
        pitch = (
            (capacity + _V4_INPUT_ALIGNMENT_ELEMENTS - 1)
            // _V4_INPUT_ALIGNMENT_ELEMENTS
            * _V4_INPUT_ALIGNMENT_ELEMENTS
        )
        # Native dummy and live inputs share these buffers, and stage 0 starts
        # at token 0. Choose slot borrowing once before graph capture.
        self.positions = (input_buffers.positions,) + tuple(
            input_buffers.positions.new_empty((capacity,))
            for _ in range(1, num_ubatches)
        )
        borrow_first_slots = (
            num_kv_groups <= 1
            or native_slot_mappings.stride(0) % _V4_INPUT_ALIGNMENT_ELEMENTS == 0
        )
        self.slot_mappings = tuple(
            None
            if stage == 0 and borrow_first_slots
            else input_buffers.positions.new_empty((num_kv_groups, pitch))
            for stage in range(num_ubatches)
        )


@contextmanager
def _use_v4_ubatch_inputs(
    runner: AFDAttentionModelRunnerV2,
) -> Iterator[_AFDV4UBatchInputs | None]:
    """Bind stable V4 stage inputs while native preparation/capture owns work."""
    if (
        runner.vllm_config.model_config.hf_config.model_type != "deepseek_v4"
        or runner.device.type != "cuda"
    ):
        yield None
        return
    ubatch_runner = runner.ubatch_runner
    assert ubatch_runner is not None
    if runner._afd_v4_ubatch_inputs is None:
        runner._afd_v4_ubatch_inputs = _AFDV4UBatchInputs(
            runner.input_buffers,
            runner.block_tables.slot_mappings,
            ubatch_runner.num_ubatches,
        )
    inputs = runner._afd_v4_ubatch_inputs
    model_state = runner.model_state
    state = vars(model_state)
    had_prepare = "prepare_attn" in state
    previous_prepare = state.get("prepare_attn")
    # V4 uses the pinned DefaultModelState, whose concrete signature includes
    # the optional model-specific metadata argument absent on its base class.
    original_prepare = cast(DefaultModelState, model_state).prepare_attn
    ubatch_state = vars(ubatch_runner)
    had_begin = "begin_capturable_run" in ubatch_state
    previous_begin = ubatch_state.get("begin_capturable_run")
    original_begin = ubatch_runner.begin_capturable_run

    # Patch reason: native ubatch views can start at a non-64-byte boundary.
    # Patch functionality: retain native stage-zero inputs where their fixed
    # layout is aligned; stage other inputs before native metadata builders.
    # Stable addresses are refreshed before FULL replay as well as eager/capture.
    # Signature: matches DefaultModelState.prepare_attn at vLLM 5df63a7,
    # vllm/v1/worker/gpu/model_states/default.py. Delegation retains native
    # metadata construction; remove when native stage buffers meet V4's ABI.
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        # ### PATCH START: stable aligned V4 stage metadata inputs.
        if ubatch_idx:
            positions = inputs.positions[ubatch_idx][: input_batch.positions.shape[0]]
            positions.copy_(input_batch.positions)
            input_batch = replace(input_batch, positions=positions)
        slots = inputs.slot_mappings[ubatch_idx]
        if slots is None:
            slots = slot_mappings
        else:
            slots = slots[:, : slot_mappings.shape[1]]
            slots.copy_(slot_mappings)
        return original_prepare(
            input_batch,
            cudagraph_mode,
            block_tables,
            slots,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        # ### PATCH END: stable aligned V4 stage metadata inputs.

    # Patch reason: native model-input slicing recreates the unaligned view.
    # Patch functionality: bind the model call to the same persistent positions
    # prepared for this stage; native threads, streams and finish stay intact.
    # Signature: matches UBatchRunner.begin_capturable_run at vLLM 5df63a7,
    # vllm/v1/worker/gpu/ubatch_utils.py. The callable accepts the native open
    # model-input dictionary. Remove with the aligned V4 preparation adapter.
    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        # ### PATCH START: bind V4 model positions without changing scheduling.
        def aligned_model(**model_kwargs: Any) -> Any:
            stage = dbo_current_ubatch_id()
            model_kwargs["positions"] = inputs.positions[stage][
                : ubatch_state.slices[stage].num_tokens
            ]
            return model(**model_kwargs)

        return original_begin(
            aligned_model, model_inputs, ubatch_state, for_capture=for_capture
        )
        # ### PATCH END: bind V4 model positions without changing scheduling.

    model_state.prepare_attn = MethodType(prepare_attn, model_state)
    ubatch_runner.begin_capturable_run = MethodType(begin_capturable_run, ubatch_runner)
    try:
        yield inputs
    finally:
        if had_begin:
            ubatch_runner.begin_capturable_run = previous_begin
        else:
            del ubatch_runner.begin_capturable_run
        if had_prepare:
            model_state.prepare_attn = previous_prepare
        else:
            del model_state.prepare_attn


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
    capture_prepared: Callable[
        [v2_cudagraph_utils.BatchExecutionDescriptor, UBatchState, bool], None
    ]
    | None = None,
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
    # A/F payload before threads start (or before FULL replay). V4 indexer
    # schedules use this ubatch plan's compute SM budget during preparation,
    # while retaining the native physical-capacity buffer and planner refill.
    # Restore builder budgets before returning, including preparation failures.
    # Remove this V4 lease when native UBatchRunner.prepare propagates its SM
    # budget to the per-stage indexer metadata builders.
    # Signature: matches UBatchRunner.prepare in
    # vllm/v1/worker/gpu/ubatch_utils.py at vLLM
    # 5df63a7203cdb9328d824a27b682e77c7b9928be. Indexer producer:
    # vllm/v1/attention/backends/mla/indexer.py at the same pinned revision.
    # Delegation exception: keep native slicing/attention preparation in the
    # bound method. Remove this seam when upstream offers a prepared-state
    # callback; no runner class or module-global function is replaced here.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: bind AFD to native stage state outside graph execution.
        # ### PATCH START: generate V4 index schedules for the ubatch SM budget.
        indexer_budgets: list[tuple[DeepseekV32IndexerMetadataBuilder, int]] = []
        try:
            if stage_inputs is not None:
                for groups in self.attn_groups:
                    for group in groups:
                        for stage in range(self.num_ubatches):
                            builder = group.get_metadata_builder(stage)
                            if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                                indexer_budgets.append((builder, builder.num_sms))
                                builder.num_sms = self.sm_control.compute_sms
            prepared = original_prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
        finally:
            # Reverse order also restores correctly if groups share a builder.
            for builder, num_sms in reversed(indexer_budgets):
                builder.num_sms = num_sms
        # ### PATCH END: generate V4 index schedules for the ubatch SM budget.
        # FULL replay uses the context captured by the graph, not this fresh one.
        if stage_inputs is not None and (cg_mode != CUDAGraphMode.FULL or for_capture):
            for stage, context in enumerate(prepared.forward_contexts):
                slots = stage_inputs.slot_mappings[stage]
                if slots is not None:
                    context.slot_mapping = build_slot_mappings_by_layer(
                        slots[:, : prepared.slices[stage].num_tokens],
                        runner.kv_cache_config,
                    )
        if event_tracker is not None:
            desc, is_warmup = event_tracker.consume(
                input_batch.num_reqs,
                input_batch.num_tokens_after_padding,
                len(prepared.slices),
            )
            runner._is_warmup = is_warmup
            runner._afd_is_graph_capturing = not is_warmup
            runner._afd_is_graph_replaying = False
            if capture_prepared is not None:
                capture_prepared(desc, prepared, is_warmup)
        else:
            runner._afd_is_graph_replaying = cg_mode == CUDAGraphMode.FULL
        runner.install_mrv2_ubatch_metadata(
            prepared.slices, prepared.forward_contexts, input_batch.num_tokens
        )
        if runner.connector.control_plane is not None:
            runner.send_dp_metadata(None, prepared.slices)
        return prepared
        # ### PATCH END: bind AFD to native stage state outside graph execution.

    with _use_v4_ubatch_inputs(runner) as stage_inputs:
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
    # Signature: matches vLLM v0.32.0 prepare_inputs_to_capture exactly,
    # including the trailing pcp_manager parameter.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # prepare_inputs_to_capture; commit
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
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
        runner._afd_is_graph_replaying = False
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
    # Signature: matches vLLM v0.32.0 ModelCudaGraphManager.run_fullgraph exactly.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # ModelCudaGraphManager.run_fullgraph; commit
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation exception: native replay remains wholly in the saved bound
    # method; this scope owns only the AFD pre-replay control seam.
    # Removal/upstream plan: delete this hook when vLLM exposes a per-manager
    # callback immediately before FULL graph replay.
    def run_fullgraph(
        self,
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
            use_gpu_mrv2_dbo_dispatch(),
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
    """Thin AFD seam over native vLLM 0.32.0 GPU ModelRunnerV2.

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
        self._afd_v4_ubatch_inputs: _AFDV4UBatchInputs | None = None
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
        # vLLM 0.32.0's token chain is
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

    # Native ActiveKVConnector creates a context without attention metadata or
    # a batch descriptor for KV-only work, including idle steps and async loads
    # after eager forward. It must not start or repeat an AFD model transaction.
    # Keep this exception in GPU V2; native model/ubatch contexts still use the
    # shared metadata provider, and NPU context handling is unchanged.
    def install_afd_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        if (
            forward_context.batch_descriptor is None
            and forward_context.attn_metadata is None
        ):
            return
        super().install_afd_metadata_on_forward_context(forward_context)

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: native V2 load_model has no AFD connector lifecycle.
    # Patch functionality: initialize the AFD connector after native weight
    # loading so Attention and FFN model loading overlap across roles.
    # Signature: matches vLLM v0.32.0 GPUModelRunnerV2.load_model exactly.
    # Upstream: vllm/v1/worker/gpu/model_runner.py, load_model;
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation exception: native loading and model injection remain in super;
    # only the AFD connector handshake follows native loading.
    # Removal plan: remove when native loading supports connector lifecycle hooks.
    def load_model(
        self,
        load_dummy_weights: bool = False,
        *args,
        model: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().load_model(load_dummy_weights, *args, model=model, **kwargs)
        # ### PATCH START: connect after native loading.
        if not self.connector.is_initialized:
            self.connector.init_afd_connector()
        # ### PATCH END: connect after native loading.

    # Patch reason: native skip-attention profiling visits only workspace slot0;
    # later V4 UBatch prefill must not grow scratch after graph capture locks it.
    # Patch functionality: reserve each real V4 FlashMLA owner's native bound
    # before the unchanged profile so resident scratch overlaps its normal peak.
    # Signature/decorator: GPUModelRunner.profile_run, vllm/v1/worker/gpu/
    # model_runner.py, 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation keeps the full native activation/sampler profile exactly once.
    # Remove when native profiling reserves scratch for all execution owners.
    @torch.inference_mode()
    def profile_run(self, randomize_inputs: bool = False) -> None:
        # ### PATCH START: pre-reserve only CUDA V4 UBatch attention scratch.
        if (
            self.device.type == "cuda"
            and self.vllm_config.model_config.hf_config.model_type == "deepseek_v4"
            and self.parallel_config.use_ubatching
        ):
            # GPU-only backend import must not extend the shared NPU boundary.
            from afd_plugin.model_executor.models.deepseek_v4 import (
                _reserve_flashmla_ubatch_workspace,
            )

            _reserve_flashmla_ubatch_workspace(self.model, self.vllm_config)
        # ### PATCH END: pre-reserve only CUDA V4 UBatch attention scratch.
        super().profile_run(randomize_inputs=randomize_inputs)

    # Patch reason: native KV initialization excludes DP1 from UBatchRunner
    # construction, although AFD can coordinate its two microbatches locally.
    # Patch functionality: scope only the native factory alias to AFD's DP1
    # adapter; native KV allocation, runner and CUDA graph setup stay upstream.
    # Signature: matches GPUModelRunner.initialize_kv_cache at the pinned commit.
    # Upstream: vllm/v1/worker/gpu/model_runner.py, initialize_kv_cache;
    # 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
    # Delegation exception: the large native KV/graph lifecycle remains in super;
    # only its factory seam is missing. Remove when upstream supports DP1 DBO.
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # ### PATCH START: enable native DP1 UBatchRunner construction.
        with use_gpu_mrv2_ubatch_factory():
            super().initialize_kv_cache(
                kv_cache_config,
                is_profiling=is_profiling,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
        # ### PATCH END: enable native DP1 UBatchRunner construction.

    # Patch reason: vLLM 0.32.0 prepares FULL graph inputs before each warmup
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
    # Signature: matches vLLM v0.32.0 GPUModelRunner.capture_model exactly
    # (including the keyword-only profile_only parameter) and is forwarded
    # unchanged to super().
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.capture_model; commit
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
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
    # Signature: matches vLLM v0.32.0 GPUModelRunnerV2.execute_model exactly,
    # including randomize_inputs, forwarded unchanged to native dummy execution.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.execute_model; commit
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
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
        randomize_inputs: bool = False,
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
                randomize_inputs=randomize_inputs,
            )
        # ### PATCH END: scope AFD metadata provider/replay and profiler step.

    # Patch reason: native V2 shutdown does not know about AFD's profiler,
    # connector, or pending metadata sidecar.
    # Patch functionality: preserve delegated native cleanup and guarantee all
    # AFD cleanup layers run when any earlier layer raises.
    # Signature: matches vLLM v0.32.0 GPUModelRunnerV2.shutdown exactly.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.shutdown; commit
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
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
                # Native graphs and threads no longer reference stage inputs.
                self._afd_v4_ubatch_inputs = None
            finally:
                self._afd_pending_metadata = None
                self.connector.close()
        # ### PATCH END: guarantee profiler/native/connector cleanup.


__all__ = ["AFDAttentionModelRunnerV2"]
