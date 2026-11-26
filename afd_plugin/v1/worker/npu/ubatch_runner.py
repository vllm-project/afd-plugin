# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend inputs, contexts and graph updates for native MRV2 microbatching."""

import threading
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import ubatching
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
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
from vllm_ascend.compilation.updatable_graph import _ACTIVE_GRAPH, UpdatableGraph
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.compat.npu.runtime_config import uses_async_cam_dbo
from afd_plugin.compat.patches.npu.mrv2_ubatching import (
    use_async_cam_child_dp_metadata,
    use_npu_input_batch_slicing,
    use_npu_ubatch_contexts,
    use_npu_ubatch_output_merge,
)
from afd_plugin.v1.worker.npu.attention_metadata import (
    materialize_attention_metadata_by_layer,
)
from afd_plugin.v1.worker.npu.mla_graph import (
    use_ubatch_capture_graph,
    use_ubatch_graph_registration,
    use_ubatch_replay_metadata,
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
        hf_config = vllm_config.model_config.hf_text_config
        self._shares_mla_block_tables = (
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

    # Upstream: vLLM-Ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # worker/v2/attn_utils.py and worker/v2/model_states/default.py::prepare_attn.
    # Native build_attn_metadata always selects builder zero.
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
        if self._shares_mla_block_tables:
            # Native dummy and live tables share persistent input storage. The
            # target MLA task retains the table captured in its kwargs, so
            # the same table must be populated at capture and replay.
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
            full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL,
        )
        if cudagraph_mode == CUDAGraphMode.FULL:
            materialize_attention_metadata_by_layer(
                metadata,
                input_batch.positions,
                num_input_tokens,
                rope_buffers=self._stage_rope_buffers[ubatch_idx],
                rope_capacity=self.vllm_config.scheduler_config.max_num_batched_tokens,
            )
        else:
            materialize_attention_metadata_by_layer(
                metadata, input_batch.positions, num_input_tokens
            )
        return metadata
        # ### PATCH END: build Ascend metadata from native child coordinates.

    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        batch_desc: BatchExecutionDescriptor,
        dp_sync_state: DPSyncState | None = None,
        for_capture: bool = False,
    ) -> UBatchState:
        previous_override = self.model_state.__dict__.get("prepare_attn")
        self.model_state.prepare_attn = self._prepare_attn
        try:
            with use_npu_input_batch_slicing():
                prepared = super().prepare(
                    input_batch,
                    block_tables,
                    slot_mappings,
                    batch_desc,
                    dp_sync_state,
                    for_capture,
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
    # Signature matches vllm/v1/worker/gpu/ubatch_utils.py::
    # UBatchRunner.begin_capturable_run at
    # 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca. Native execution is too large
    # to copy. Scope Ascend 7dcdb690's active graph and CUDA-only BLAS setup
    # around the native parked threads. Remove when upstream offers NPU
    # thread initialization and capture-context hooks.
    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        # ### PATCH START: NPU scopes also cover parked capture threads.
        is_profile = get_mrv2_in_profile_run()
        capture_graph: UpdatableGraph | None = None

        def forward(**inputs):
            with (
                use_ubatch_capture_graph(capture_graph)
                if capture_graph is not None
                else nullcontext(),
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
            nonlocal capture_graph
            try:
                # Native capture parks the threads before capture_begin; its
                # finish callback releases them after the parent binds the graph.
                capture_graph = _ACTIVE_GRAPH.get() if for_capture else None
                with (
                    use_ubatch_graph_registration(capture_graph)
                    if capture_graph is not None
                    else nullcontext()
                ):
                    return native_finish()
            finally:
                capture_graph = None
                stack.close()

        self._pending_finish = finish
        return finish
        # ### PATCH END: NPU scopes also cover parked capture threads.

    def run_fullgraph(
        self, manager: ModelAclGraphManager, desc: BatchExecutionDescriptor
    ) -> Any:
        """Replay the native aggregate graph with the currently prepared stages."""
        prepared = self._prepared_state
        assert prepared is not None and len(prepared.slices) == desc.num_ubatches
        stages = [context.attn_metadata for context in prepared.forward_contexts]
        with use_ubatch_replay_metadata(manager, stages):
            return ModelAclGraphManager.run_fullgraph(manager, desc)
