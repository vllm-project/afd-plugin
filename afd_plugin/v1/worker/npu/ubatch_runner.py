# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend inputs, contexts and MLA updates for native MRV2 microbatching."""

import threading
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import ubatching
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, UBatchState
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import use_workspace_lane
from vllm_ascend.ascend_forward_context import (
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
)
from vllm_ascend.compilation.acl_graph import (
    GraphParams,
    get_graph_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import (
    ModelAclGraphManager,
    _get_graph_update_backend,
)
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.compat.npu.runtime_config import uses_async_cam_dbo
from afd_plugin.compat.patches.npu.mla_graph import AFD_MLA_GRAPH_PARAMS_KEY
from afd_plugin.compat.patches.npu.mrv2_ubatching import (
    use_async_cam_child_dp_metadata,
    use_npu_input_batch_slicing,
    use_npu_ubatch_contexts,
    use_npu_ubatch_output_merge,
)
from afd_plugin.model_executor.models.npu.deepseek_attention_metadata import (
    materialize_deepseek_attention_metadata_by_layer,
)
from afd_plugin.v1.worker.npu.mla_graph import (
    merge_mla_graph_params,
    new_mla_graph_params,
    override_mla_graph_params,
)


class NPUUBatchRunner(UBatchRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
        *,
        kernel_block_sizes: list[int],
        timeout_s: float,
    ):
        super().__init__(
            vllm_config, device, model_state, attn_groups, kv_cache_config, max_num_reqs
        )
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1, timeout=timeout_s)
        self._prepared_state: UBatchState | None = None
        self._mla_capture_params: dict[
            BatchExecutionDescriptor, tuple[GraphParams, GraphParams]
        ] = {}
        hf_config = vllm_config.model_config.hf_text_config
        self._uses_mla_graph_params = (
            vllm_config.model_config.use_mla
            and not hasattr(hf_config, "index_topk")
            and not hasattr(hf_config, "compress_ratios")
        )
        self._stage_block_tables: list[tuple[torch.Tensor, ...] | None] = [
            None for _ in range(self.num_ubatches)
        ]
        self._stage_rope_buffers: list[dict[str, dict[str, torch.Tensor]]] = [
            {} for _ in range(self.num_ubatches)
        ]
        self.stage_attn_groups: list[list[list[AttentionGroup]]] = []
        with set_current_vllm_config(vllm_config):
            for _ in range(self.num_ubatches):
                groups = [
                    [copy(group) for group in cache_group]
                    for cache_group in attn_groups
                ]
                for group_idx, cache_group in enumerate(groups):
                    for group in cache_group:
                        group.create_metadata_builders(
                            vllm_config,
                            device,
                            kernel_block_size=kernel_block_sizes[group_idx],
                            num_metadata_builders=1,
                        )
                self.stage_attn_groups.append(groups)

    # Ascend 8d4409d6 asserts ubatch_idx == 0 and always selects builder zero.
    # Feed its existing builder function an independent group per native child;
    # no request/KV slicing or MRV1 state is reproduced here. Preserve native
    # FULL padding and Ascend's capture-specific metadata builder contract.
    def _prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any] | None:
        # ### PATCH START: build Ascend metadata from native child coordinates.
        assert cudagraph_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
        if input_batch.num_tokens == 0 and cudagraph_mode == CUDAGraphMode.NONE:
            # Ascend's existing no-metadata path zeros attention without touching
            # KV. The native context still runs padded tensors and A/F exchange.
            return None
        num_reqs = (
            input_batch.num_reqs_after_padding
            if cudagraph_mode == CUDAGraphMode.FULL
            else input_batch.num_reqs
        )
        num_input_tokens = (
            input_batch.num_tokens_after_padding
            if cudagraph_mode == CUDAGraphMode.FULL
            else input_batch.num_tokens
        )
        if self._uses_mla_graph_params:
            # Native dummy and live tables share persistent input storage. The
            # target MLA updater retains the table captured in attn_params, so
            # a fresh clone here would replay the dummy block IDs forever.
            stage_block_tables = block_tables
        else:
            # DSA/SFA builders clear padded rows in place. Isolate that write
            # per child, but keep the same storage from capture through replay.
            # Allocate to native request capacity, not this step's slice size.
            buffers = self._stage_block_tables[ubatch_idx]
            if buffers is None:
                capacity = self.ubatch_seq_lens[ubatch_idx].shape[0]
                buffers = tuple(
                    torch.empty(
                        (capacity, *table.shape[1:]),
                        dtype=table.dtype,
                        device=table.device,
                    )
                    for table in block_tables
                )
                self._stage_block_tables[ubatch_idx] = buffers
            stage_block_tables = tuple(
                buffer[: table.shape[0]].copy_(table)
                for buffer, table in zip(buffers, block_tables, strict=True)
            )
        metadata = build_attn_metadata(
            attn_groups=self.stage_attn_groups[ubatch_idx],
            num_reqs=num_reqs,
            num_actual_reqs=input_batch.num_reqs,
            num_tokens=num_input_tokens,
            num_actual_tokens=input_batch.num_tokens,
            num_input_tokens=num_input_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=int(input_batch.num_scheduled_tokens.max(initial=0)),
            seq_lens=input_batch.seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            max_seq_len=self.model_state.max_model_len,
            block_tables=stage_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            parallel_config=self.parallel_config,
            positions=input_batch.positions,
            attn_state=(
                input_batch.attn_state
                if cudagraph_mode == CUDAGraphMode.FULL
                else build_attn_state(
                    self.vllm_config,
                    input_batch.seq_lens_np,
                    input_batch.num_reqs,
                    input_batch.num_scheduled_tokens,
                    input_batch.num_scheduled_tokens,
                    kv_cache_config,
                )
            ),
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            for_cudagraph_capture=for_capture,
        )
        if cudagraph_mode == CUDAGraphMode.FULL:
            materialize_deepseek_attention_metadata_by_layer(
                metadata,
                input_batch.positions,
                num_input_tokens,
                rope_buffers=self._stage_rope_buffers[ubatch_idx],
                rope_capacity=self.vllm_config.scheduler_config.max_num_batched_tokens,
            )
        else:
            materialize_deepseek_attention_metadata_by_layer(
                metadata, input_batch.positions, num_input_tokens
            )
        return metadata
        # ### PATCH END: build Ascend metadata from native child coordinates.

    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        previous_override = self.model_state.__dict__.get("prepare_attn")
        self.model_state.prepare_attn = self._prepare_attn
        try:
            with use_npu_input_batch_slicing():
                prepared = super().prepare(
                    input_batch, block_tables, slot_mappings, cg_mode, for_capture
                )
            if for_capture and self._uses_mla_graph_params:
                sizes = tuple(stage.num_tokens for stage in prepared.slices)
                assert len(sizes) == 2 and sizes[0] == sizes[1], sizes
                # Native capture orders each ordinary descriptor before its
                # two-stage twin. Reuse its full-size FIA workspace, as the
                # two native child threads share one compute stream.
                graph_params = get_graph_params()
                assert graph_params is not None
                workspace = graph_params.workspaces[sum(sizes)]
                assert workspace is not None
                for context in prepared.forward_contexts:
                    context.additional_kwargs[AFD_MLA_GRAPH_PARAMS_KEY] = (
                        new_mla_graph_params(sizes[0], workspace)
                    )
            self._prepared_state = prepared
            return prepared
        finally:
            if previous_override is None:
                del self.model_state.prepare_attn
            else:
                self.model_state.prepare_attn = previous_override

    def _make_forward_contexts(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata: list[dict[str, Any] | None],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        with (
            use_async_cam_child_dp_metadata()
            if uses_async_cam_dbo(self.vllm_config)
            else nullcontext()
        ):
            contexts = super()._make_forward_contexts(
                ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
            )
        for stage_idx, (context, stage) in enumerate(
            zip(contexts, ubatch_slices, strict=True)
        ):
            context.additional_kwargs = current_platform.set_additional_forward_context(
                context.attn_metadata,
                self.vllm_config,
                context.dp_metadata,
                num_tokens=stage.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ubatch_slices=ubatch_slices,
            )
            context.additional_kwargs.update(
                ubatch_idx=stage_idx, num_ubatches=len(ubatch_slices), dbo_enabled=True
            )
            # CAMP2P's opaque compiled send/receive ops resolve their HCCL
            # group from this context field at execution time. Python connector
            # assignments run during tracing and cannot initialize each native
            # child: without this, stage 1 silently selects stage 0's group.
            context.ubatch_idx = stage_idx
            mask = context.additional_kwargs.get("mc2_mask")
            if mask is not None:
                context.additional_kwargs["mc2_mask"] = mask.clone()
        return contexts

    # Native graph capture calls begin_capturable_run directly, bypassing run.
    # Scope NPU aliases until the native finish/abort joins its own threads.
    # Signature matches UBatchRunner.begin_capturable_run at ced6857a. Native
    # execution is too large to copy: only context and CUDA-only BLAS setup are
    # adapted here. Remove when upstream offers NPU thread initialization hooks.
    def begin_capturable_run(
        self,
        model: Callable,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        # ### PATCH START: NPU scopes also cover parked capture threads.
        is_profile = get_mrv2_in_profile_run()

        def forward(**inputs):
            with (
                set_current_vllm_config(self.vllm_config),
                override_mrv2_in_profile_run(is_profile),
                skip_dp_coordination()
                if uses_async_cam_dbo(self.vllm_config)
                else nullcontext(),
                use_workspace_lane(0),
            ):
                return model(**inputs)

        # Native prepare, threads, handoff and merge remain upstream. Restore the
        # parent context even if a child fails before native begin returns finish.
        stack = ExitStack()
        stack.enter_context(
            override_forward_context(ubatching.forward_context._forward_context)
        )
        stack.enter_context(use_npu_ubatch_contexts())
        stack.enter_context(use_npu_ubatch_output_merge())
        if for_capture:
            # Native capture preallocates CUDA cuBLAS handles in each child.
            # Ascend GEMMs initialize during the native warmup instead. The
            # handle return value is unused; restore the exact alias on finish.
            original_blas_handle = torch.cuda.current_blas_handle
            torch.cuda.current_blas_handle = lambda: 0
            stack.callback(
                setattr, torch.cuda, "current_blas_handle", original_blas_handle
            )
        try:
            native_finish = super().begin_capturable_run(
                forward, model_inputs, ubatch_state, for_capture
            )
        except BaseException:
            stack.close()
            raise

        def finish() -> Any:
            try:
                return native_finish()
            finally:
                stack.close()

        self._pending_finish = finish
        return finish
        # ### PATCH END: NPU scopes also cover parked capture threads.

    def record_capture_graph_params(
        self,
        desc: BatchExecutionDescriptor,
        prepared: UBatchState,
        is_warmup: bool,
    ) -> None:
        if is_warmup or not self._uses_mla_graph_params:
            return
        first, second = prepared.forward_contexts
        self._mla_capture_params[desc] = (
            first.additional_kwargs[AFD_MLA_GRAPH_PARAMS_KEY],
            second.additional_kwargs[AFD_MLA_GRAPH_PARAMS_KEY],
        )

    def run_fullgraph(
        self, manager: ModelAclGraphManager, desc: BatchExecutionDescriptor
    ) -> Any:
        """Replay the native graph and update the two captured MLA stages."""
        if not self._uses_mla_graph_params:
            return ModelAclGraphManager.run_fullgraph(manager, desc)
        prepared = self._prepared_state
        assert prepared is not None and len(prepared.slices) == desc.num_ubatches
        first, second = prepared.forward_contexts
        stage_tokens = prepared.slices[0].num_tokens
        metadata, params = merge_mla_graph_params(
            [first.attn_metadata, second.attn_metadata],
            self._mla_capture_params[desc],
            stage_tokens,
        )
        manager.update_stream.wait_stream(torch.npu.current_stream(self.device))
        # Skip only Ascend's single-batch updater. The native manager still owns
        # graph replay and persistent output buffers for this exact descriptor.
        output = ModelCudaGraphManager.run_fullgraph(manager, desc)
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=stage_tokens,
                num_tokens_across_dp=first.dp_metadata.num_tokens_across_dp_cpu,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
            ),
        ):
            context = get_forward_context()
            with override_mla_graph_params(context, metadata, params):
                update_full_graph_params(
                    _get_graph_update_backend(self.attn_groups),
                    manager.update_stream,
                    context,
                    stage_tokens,
                    self.vllm_config,
                    self.vllm_config.speculative_config,
                )
        return output
