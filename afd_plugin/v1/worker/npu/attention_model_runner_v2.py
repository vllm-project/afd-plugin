# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for AFD ModelRunnerV2 execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker import utils as v2_worker_utils
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu import cudagraph_utils as v2_cudagraph_utils
from vllm.v1.worker.gpu import model_runner as v2_model_runner
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

from afd_plugin.compat.npu import fail_if_unsupported_npu_afd_features
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
)
from afd_plugin.v1.worker.attention_metadata import (
    AFDMetadataProviderMixin,
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.npu.async_cam_metadata_v2 import use_async_cam_stage_metadata
from afd_plugin.validation import validate_npu_model_runner_v2_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

_AFD_FULLGRAPH_HOOK_MARKER = "_afd_fullgraph_replay_hook_active"


@contextmanager
def _use_afd_fullgraph_replay_hook(
    runner: AFDNPUAttentionModelRunnerV2,
    real_tokens: int,
) -> Iterator[None]:
    """Scope AFD full ACL-graph replay control to one graph manager instance."""

    manager = runner.cudagraph_manager
    if manager is None:
        raise RuntimeError(
            "AFD ACL graph replay hook requires an initialized graph manager",
        )
    manager_state = manager.__dict__
    if _AFD_FULLGRAPH_HOOK_MARKER in manager_state:
        raise RuntimeError("AFD ACL graph replay hook is already active")

    had_instance_override = "run_fullgraph" in manager_state
    previous_instance_override = manager_state.get("run_fullgraph")
    original_run_fullgraph = manager.run_fullgraph
    manager_state[_AFD_FULLGRAPH_HOOK_MARKER] = True

    # Patch reason: native FULL replay bypasses ForwardContext creation, so the
    # execute-scoped AFD provider cannot publish runtime control.
    # Patch functionality: wrap only this manager instance and publish one
    # ordinary padded control payload immediately before each native replay.
    # Signature: matches vLLM v0.26.0 ModelCudaGraphManager.run_fullgraph exactly.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # ModelCudaGraphManager.run_fullgraph; commit
    # 568afb3a13806beb53bb2e6bd518269357b237c0.
    # Delegation exception: native replay remains wholly in the saved bound
    # method; this scope owns only the AFD pre-replay control seam.
    # Removal/upstream plan: delete this hook when vLLM exposes a per-manager
    # callback immediately before FULL graph replay.
    def run_fullgraph(
        self: v2_cudagraph_utils.ModelCudaGraphManager,
        desc: v2_cudagraph_utils.BatchExecutionDescriptor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        # ### PATCH START: publish one AFD pre-replay payload.
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
def _use_camp2p_dp_padding(attention_size: int, ffn_size: int) -> Iterator[None]:
    """Pad each CAMP2P receiver group equally before input preparation (FFN TP=1)."""
    original_dispatch = v2_model_runner.dispatch_cg_and_sync_dp

    # Patch reason: native execute calls this module-level function before input
    # preparation; no runner method exposes that boundary.
    # Patch functionality: apply CAMP2P group padding to the agreed eager batch.
    # Signature: matches vLLM v0.30.0 dispatch_cg_and_sync_dp exactly.
    # Upstream source: vllm/v1/worker/gpu/dp_utils.py;
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: preserve native graph selection and DP collectives.
    # Removal plan: replace when native dispatch exposes a padding callback.
    def dispatch_cg_and_sync_dp(
        cudagraph_manager: v2_cudagraph_utils.CudaGraphManager | None,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        dp_size: int,
        dp_rank: int,
        max_query_len: int | None = None,
        need_eager: bool = False,
        num_active_loras: int = 0,
        parallel_config: ParallelConfig | None = None,
        allow_ubatching: bool = False,
        uniform_decode: bool = False,
        dp_sync: DPSyncState | None = None,
    ) -> tuple[v2_cudagraph_utils.BatchExecutionDescriptor, DPSyncState | None]:
        desc, sync = original_dispatch(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len=max_query_len,
            need_eager=need_eager,
            num_active_loras=num_active_loras,
            parallel_config=parallel_config,
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode,
            dp_sync=dp_sync,
        )
        # ### PATCH START: equal CAMP2P strides for eager and graph fallback.
        if desc.cg_mode == CUDAGraphMode.NONE and sync is not None:
            attention_tp = attention_size // dp_size
            # Rank a sends to FFN a % ffn_size. Contiguous Attention TP peers
            # must also pad equally; non-aligned TP blocks connect all groups.
            num_groups = ffn_size // attention_tp if ffn_size % attention_tp == 0 else 1
            padded_counts = (
                sync.num_tokens_across_dp.reshape(-1, num_groups)
                .amax(dim=0)
                .repeat(dp_size // num_groups)
            )
            desc = replace(desc, num_tokens=int(padded_counts[dp_rank].item()))
            sync = replace(sync, num_tokens_across_dp=padded_counts)
        # ### PATCH END: equal CAMP2P strides for eager and graph fallback.
        return desc, sync

    try:
        v2_model_runner.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
        yield
    finally:
        v2_model_runner.dispatch_cg_and_sync_dp = original_dispatch


class AFDNPUAttentionModelRunnerV2(AFDMetadataProviderMixin, NPUModelRunnerV2):
    """Thin AFD seam over native vLLM-Ascend ModelRunnerV2.

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
        validate_npu_model_runner_v2_config(
            vllm_config,
            expected_role="attention",
            device_type=device.type,
        )
        fail_if_unsupported_npu_afd_features(vllm_config)
        super().__init__(vllm_config, device)
        connector: AFDConnectorBase | None = None
        try:
            self.afd_config = self.parse_config(self.vllm_config)
            rank, _ = _resolve_world_ranks()
            local_rank = int(device.index)
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
            if connector.control_plane is None and not self.afd_config.async_dp:
                raise RuntimeError(
                    "AFD ModelRunnerV2 requires a control-plane-driven connector",
                )
            self._is_warmup = False
            self._afd_is_graph_capturing = False
            self._afd_is_profile = False
            self._afd_is_graph_replaying = False
            self._afd_pending_metadata: AFDForwardContextMetadata | None = None
            self._afd_async_moe_ubatch_metadata: AsyncMoeUbatchMetadata | None = None
            self._afd_suppress_metadata_send = False
            self._afd_transaction_counter = 0
            self.prof = create_afd_npu_profiler("attention")
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
        # CAMP2P many-to-one execution pads the descriptor and DP counts before
        # native input preparation, including eager fallback and dummy ranks.
        batch_descriptor = forward_context.batch_descriptor
        if batch_descriptor is None:
            raise RuntimeError(
                "AFD ModelRunnerV2 requires a native eager BatchDescriptor",
            )
        return int(batch_descriptor.num_tokens)

    def install_afd_metadata_on_forward_context(
        self, forward_context: ForwardContext
    ) -> None:
        super().install_afd_metadata_on_forward_context(forward_context)
        if self._afd_async_moe_ubatch_metadata is not None:
            assert forward_context.additional_kwargs is not None
            forward_context.additional_kwargs[ASYNC_MOE_UBATCH_METADATA_KEY] = (
                self._afd_async_moe_ubatch_metadata
            )

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: native V2 load_model has no AFD connector lifecycle.
    # Patch functionality: preserve native model loading, then rendezvous the
    # AFD connector after both worker roles have loaded their weights.
    def load_model(
        self,
        load_dummy_weights: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        if not self.connector.is_initialized:
            self.connector.init_afd_connector()

    # Patch reason: vLLM v0.30.0 prepares FULL graph inputs before each warmup
    # and formal capture forward, outside torch.cuda.graph, but does not expose
    # that lifecycle to AFD's control plane.
    # Patch functionality: temporarily wrap the exact upstream input-preparation
    # symbol so each native FULL descriptor publishes one warmup and one capture
    # payload before its forward, while the provider installs the pending AFD
    # sidecar without sending control from inside torch.cuda.graph.
    # Signature: matches vLLM v0.30.0 GPUModelRunner.capture_model exactly.
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
        # Native graph-memory profiling captures only this descriptor prefix.
        capture_descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])[
            : manager._max_full_descs_to_capture
        ]
        if not capture_descs:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 ACL graph expected at least one FULL "
                "capture descriptor",
            )
        for desc in capture_descs:
            if desc.cg_mode != CUDAGraphMode.FULL:
                raise RuntimeError(
                    "AFD NPU ModelRunnerV2 ACL graph expected only FULL capture "
                    "descriptors",
                )
        expected_events = [
            (desc, is_warmup) for desc in capture_descs for is_warmup in (True, False)
        ]
        event_index = 0
        original_prepare = v2_cudagraph_utils.prepare_inputs_to_capture
        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_warmup = self._is_warmup
        previous_is_graph_capturing = self._afd_is_graph_capturing

        # Patch reason: native ModelCudaGraphManager.capture calls this exact
        # module symbol once before each warmup/formal-capture forward.
        # Patch functionality: preserve native preparation and publish the
        # matching AFD event before the forward starts.
        # Signature: matches vLLM v0.30.0 prepare_inputs_to_capture exactly.
        # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
        # prepare_inputs_to_capture; commit
        # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
        # Delegation exception: preserve native attention preparation; this
        # wrapper only publishes the AFD event before each capture forward.
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
            nonlocal event_index
            # ### PATCH START: stage one exact AFD capture event.
            attention_state = original_prepare(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=full_cudagraph,
                max_query_len=max_query_len,
                pcp_manager=pcp_manager,
            )
            if event_index >= len(expected_events):
                raise RuntimeError(
                    "AFD NPU ModelRunnerV2 ACL graph observed extra capture input "
                    "preparation",
                )
            desc, is_warmup = expected_events[event_index]
            event_index += 1
            if num_reqs != desc.num_reqs or num_tokens != desc.num_tokens:
                raise RuntimeError(
                    "AFD NPU ModelRunnerV2 ACL graph capture descriptor/order drift: "
                    f"expected ({desc.num_reqs}, {desc.num_tokens}), got "
                    f"({num_reqs}, {num_tokens})",
                )
            self._is_warmup = is_warmup
            self._afd_is_graph_capturing = not is_warmup
            self._afd_pending_metadata = self.build_afd_metadata(
                None,
                int(desc.num_tokens),
            )
            self.send_dp_metadata(
                self.build_capture_dp_metadata(int(desc.num_tokens)),
                None,
            )
            self._afd_suppress_metadata_send = True
            return attention_state
            # ### PATCH END: stage one exact AFD capture event.

        v2_cudagraph_utils.prepare_inputs_to_capture = prepare_inputs_to_capture
        try:
            with use_afd_metadata_provider(
                self.install_afd_metadata_on_forward_context,
            ):
                result = super().capture_model(profile_only=profile_only)
            if event_index != len(expected_events):
                raise RuntimeError(
                    "AFD NPU ModelRunnerV2 ACL graph capture input-preparation "
                    "call count drift: "
                    f"expected {len(expected_events)}, got {event_index}",
                )
            return result
        finally:
            v2_cudagraph_utils.prepare_inputs_to_capture = original_prepare
            self._afd_pending_metadata = previous_metadata
            self._afd_suppress_metadata_send = previous_suppress_send
            self._is_warmup = previous_is_warmup
            self._afd_is_graph_capturing = previous_is_graph_capturing
        # ### PATCH END: publish AFD FULL graph warmup/capture control.

    # Patch reason: native V2 creates ForwardContext inside execute_model, so
    # AFD must install its sidecar at that exact context-construction seam.
    # Patch functionality: delegate all request/input/Attention/KV/sampling/
    # output work to native V2 while temporarily installing AFD metadata.
    # Signature: matches vLLM v0.30.0 GPUModelRunner.execute_model, including
    # the dummy-state parameters forwarded through Ascend NPUModelRunnerV2.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.execute_model; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: the upstream method is intentionally not copied;
    # only this narrow provider/profiler wrapper is AFD-specific.
    # Async CAM uses the v0.30 native DP opt-out for requests and profile/dummy
    # execution. The scope leaves TP collectives and the configured topology intact.
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
        step_afd_npu_profiler(self.prof)
        use_fullgraph_replay_hook = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)
            and self.cudagraph_manager is not None
        )
        previous_metadata = self._afd_pending_metadata
        previous_stage_metadata = self._afd_async_moe_ubatch_metadata
        self._afd_async_moe_ubatch_metadata = None
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_warmup = self._is_warmup
        previous_is_graph_capturing = self._afd_is_graph_capturing
        previous_is_graph_replaying = getattr(self, "_afd_is_graph_replaying", False)
        previous_is_profile = self._afd_is_profile
        self._afd_is_profile = bool(is_profile)
        self._afd_is_graph_replaying = False

        replay_scope = (
            _use_afd_fullgraph_replay_hook(
                self,
                int(scheduler_output.total_num_scheduled_tokens),
            )
            if use_fullgraph_replay_hook
            else nullcontext()
        )
        try:
            with (
                skip_dp_coordination() if self.afd_config.async_dp else nullcontext(),
                _use_camp2p_dp_padding(
                    self.afd_config.num_attention_ranks, self.afd_config.num_ffn_ranks
                )
                if not self.afd_config.async_dp
                and self.afd_config.num_attention_ranks > self.afd_config.num_ffn_ranks
                else nullcontext(),
                replay_scope,
                use_async_cam_stage_metadata(self)
                if self.afd_config.async_dp
                and cast(
                    AFDAsyncExtraInfo, self.connector.extra_info
                ).async_moe_ubatching
                else nullcontext(),
                use_afd_metadata_provider(
                    self.install_afd_metadata_on_forward_context,
                ),
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
        finally:
            self._afd_pending_metadata = previous_metadata
            self._afd_async_moe_ubatch_metadata = previous_stage_metadata
            self._afd_suppress_metadata_send = previous_suppress_send
            self._is_warmup = previous_is_warmup
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_is_graph_replaying = previous_is_graph_replaying
            self._afd_is_profile = previous_is_profile
        # ### PATCH END: scope AFD metadata provider/replay and profiler step.

    # Patch reason: native V2 shutdown does not know about AFD's profiler,
    # connector, or pending metadata sidecar.
    # Patch functionality: preserve delegated native cleanup and guarantee all
    # AFD cleanup layers run when any earlier layer raises.
    # Signature: matches vLLM v0.26.0 NPUModelRunnerV2.shutdown exactly.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.shutdown; commit
    # 568afb3a13806beb53bb2e6bd518269357b237c0.
    # Delegation exception: native resource release remains in super().shutdown;
    # this override contains only AFD-specific finalizers.
    # Removal/upstream plan: delete this wrapper if AFD-owned lifecycle moves
    # to a shared composition/factory owner; it never copies native cleanup.
    def shutdown(self) -> None:
        # ### PATCH START: guarantee profiler/native/connector cleanup.
        try:
            stop_afd_npu_profiler(self.prof)
        finally:
            try:
                super().shutdown()
            finally:
                self._afd_pending_metadata = None
                self.connector.close()
        # ### PATCH END: guarantee profiler/native/connector cleanup.


__all__ = ["AFDNPUAttentionModelRunnerV2"]
