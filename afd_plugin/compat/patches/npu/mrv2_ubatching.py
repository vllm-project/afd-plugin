# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped Ascend seams for vLLM ced6857a's native UBatchRunner.

Ascend baseline: 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6.
MRV2 owns preparation, threads, handoff and graph lifecycle.
All bindings are scoped to AFD runners and restored on exit.
Remove when upstream supports Ascend child inputs, streams and cancellation.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import ParallelConfig, VllmConfig
from vllm.utils import torch_utils
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import ubatching
from vllm.v1.worker.gpu import model_runner, ubatch_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    check_ubatch_thresholds,
    get_num_ubatches,
)
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.config import is_afd_async_dp, parse_afd_config

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


class NPUUBatchContext(ubatching.UBatchContext):
    """Keep the native registry/handoff, with NPU streams and peer failure wakeup."""

    # Native CUDA API aliases do not update vLLM's cached stream after Ascend
    # replaces torch.cuda.set_stream. Update both views of the stream together.
    def update_stream(self, stream):
        # ### PATCH START: NPU stream and native TLS agree.
        self.current_stream = stream
        torch.npu.set_stream(stream)
        torch_utils._current_stream_tls.value = stream
        # ### PATCH END: NPU stream and native TLS agree.

    def _abort(self):
        self.ready_barrier.abort()
        self.cpu_wait_event.set()
        self.cpu_signal_event.set()

    def _check_cancelled(self):
        if self.ready_barrier.broken:
            raise RuntimeError("NPU DBO peer failed; restart the A/F worker group")

    # Native __enter__ leaves its registry entry behind if barrier/device setup
    # fails. The bounded barrier also catches initialization before __enter__.
    def __enter__(self):
        # ### PATCH START: unwind failed native context entry.
        try:
            torch.npu.set_device(self.compute_stream.device)
            super().__enter__()
            self._check_cancelled()
            return self
        except BaseException:
            self._abort()
            ubatching._THREAD_ID_TO_CONTEXT.pop(threading.get_ident(), None)
            ubatching._CURRENT_CONTEXTS[self.id] = None
            raise
        # ### PATCH END: unwind failed native context entry.

    # Abort the existing handoff on a model or receive-hook failure. Waking a
    # peer does not reclaim a communication window, so this runner cannot retry.
    def __exit__(self, exc_type, exc_val, exc_tb):
        # ### PATCH START: always wake the native peer on failure.
        if exc_type is not None:
            self._abort()
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        except BaseException:
            self._abort()
            raise
        finally:
            self.cpu_signal_event.set()
        # ### PATCH END: always wake the native peer on failure.

    # Check the same barrier before and after native yield; do not create a
    # second cancellation registry or change the normal thread schedule.
    def _cpu_yield(self):
        # ### PATCH START: a dead peer cannot park this thread again.
        self._check_cancelled()
        super()._cpu_yield()
        self._check_cancelled()
        # ### PATCH END: a dead peer cannot park this thread again.


@contextmanager
def use_npu_ubatch_factory(runner: "AFDNPUAttentionModelRunnerV2") -> Iterator[None]:
    """Replace the factory alias actually called by GPUModelRunner initialization."""
    original_factory = model_runner.maybe_build_ubatch_runner
    original_sm_control = ubatch_utils.create_sm_control_context

    # This scope runs only during AFD NPU KV initialization. Keep the native
    # eligibility check, except Async CAM needs no DP handshake at DP=1.
    # Signature matches maybe_build_ubatch_runner at ced6857a.
    def maybe_build_ubatch_runner(
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
    ) -> ubatch_utils.UBatchRunner | None:
        # ### PATCH START: AFD NPU constructor dispatch.
        from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner

        parallel = vllm_config.parallel_config
        if not parallel.use_ubatching or (
            parallel.data_parallel_size <= 1 and not is_afd_async_dp(vllm_config)
        ):
            return None
        return NPUUBatchRunner(
            vllm_config,
            device,
            model_state,
            attn_groups,
            kv_cache_config,
            max_num_reqs,
            kernel_block_sizes=runner.kernel_block_sizes,
            timeout_s=parse_afd_config(vllm_config).afd_process_group_timeout_s,
        )
        # ### PATCH END: AFD NPU constructor dispatch.

    try:
        model_runner.maybe_build_ubatch_runner = maybe_build_ubatch_runner
        ubatch_utils.create_sm_control_context = lambda parallel: nullcontext()
        yield
    finally:
        model_runner.maybe_build_ubatch_runner = original_factory
        ubatch_utils.create_sm_control_context = original_sm_control


@contextmanager
def use_async_cam_dbo_dispatch() -> Iterator[None]:
    """Extend native threshold selection to Async CAM Attention DP=1."""
    original_dispatch = model_runner.dispatch_cg_and_sync_dp

    # Patch reason: native dispatch returns early at DP=1 before DBO selection.
    # Patch functionality: apply native eligibility and thresholds at DP=1;
    # native dispatch continues to own every DP>1 descriptor.
    # Signature matches dispatch_cg_and_sync_dp at ced6857a.
    # Delegation exception: native descriptor/threshold selection stays upstream.
    # Removal plan: remove when native dispatch supports local DP=1 DBO.
    def dispatch_cg_and_sync_dp(
        cudagraph_manager: CudaGraphManager | None,
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
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
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
        # ### PATCH START: native DBO eligibility without a DP handshake.
        if (
            dp_size == 1
            and allow_ubatching
            and parallel_config is not None
            and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)
            and num_tokens >= get_num_ubatches(parallel_config)
        ):
            desc = replace(desc, num_ubatches=get_num_ubatches(parallel_config))
        # ### PATCH END: native DBO eligibility without a DP handshake.
        return desc, sync

    model_runner.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    try:
        yield
    finally:
        model_runner.dispatch_cg_and_sync_dp = original_dispatch


@contextmanager
def use_async_cam_child_dp_metadata() -> Iterator[None]:
    """Build native Async contexts without FFN expert DP coordination metadata."""
    original_metadata = ubatch_utils.DPMetadata

    class AsyncCamDPMetadata:
        # Native ced6857a UBatchRunner calls DPMetadata.make unconditionally.
        # Async FFN owns its own DP layout; DP1 without SP also fails that factory.
        # This factory alias is scoped to native child context construction;
        # the real forward_context.DPMetadata class remains untouched.
        # Signature matches DPMetadata.make except its intentional None result.
        # Remove when native UBatchRunner supports optional child DP metadata.
        @staticmethod
        def make(
            parallel_config: ParallelConfig,
            num_tokens: int,
            num_tokens_across_dp_cpu: torch.Tensor,
        ) -> None:
            # ### PATCH START: Async children do not coordinate FFN DP.
            return None
            # ### PATCH END: Async children do not coordinate FFN DP.

    ubatch_utils.DPMetadata = AsyncCamDPMetadata
    try:
        yield
    finally:
        ubatch_utils.DPMetadata = original_metadata


@contextmanager
def use_npu_input_batch_slicing() -> Iterator[None]:
    original_slice = ubatch_utils._slice_input_batch

    # Native replace() retains the Ascend subtype but leaves seq_lens_np whole.
    # Delegation preserves native query/sequence truncation. Apply exactly its
    # CPU truncation delta to Ascend's exact lengths, including straddling requests.
    # Signature matches _slice_input_batch at ced6857a.
    def _slice_input_batch(
        input_batch: InputBatch,
        ubatch_slice: UBatchSlice,
        query_start_loc_buf: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        dcp_local_seq_lens_buf: torch.Tensor | None = None,
        *,
        dcp_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> InputBatch:
        # ### PATCH START: extend native slicing with Ascend host lengths.
        child = original_slice(
            input_batch,
            ubatch_slice,
            query_start_loc_buf,
            seq_lens_buf,
            dcp_local_seq_lens_buf,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        assert isinstance(input_batch, AscendInputBatch)
        assert isinstance(child, AscendInputBatch)
        request_slice = ubatch_slice.request_slice
        truncation = (
            input_batch.seq_lens_cpu_upper_bound[request_slice]
            - child.seq_lens_cpu_upper_bound
        ).numpy()
        child.seq_lens_np = input_batch.seq_lens_np[request_slice] - truncation
        return child
        # ### PATCH END: extend native slicing with Ascend host lengths.

    ubatch_utils._slice_input_batch = _slice_input_batch
    try:
        yield
    finally:
        ubatch_utils._slice_input_batch = original_slice


@contextmanager
def use_npu_ubatch_contexts() -> Iterator[None]:
    original_context = ubatching.UBatchContext
    original_stream = ubatching.current_stream
    original_runner_stream = ubatch_utils.current_stream
    try:
        ubatching.UBatchContext = NPUUBatchContext
        ubatching.current_stream = torch.npu.current_stream
        ubatch_utils.current_stream = torch.npu.current_stream
        yield
    finally:
        ubatching.UBatchContext = original_context
        ubatching.current_stream = original_stream
        ubatch_utils.current_stream = original_runner_stream


def cat_ubatch_outputs(
    outputs: list[torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Keep Ascend's nested auxiliary-output contract when native threads merge."""
    first = outputs[0]
    if isinstance(first, tuple):
        pairs = [output for output in outputs if isinstance(output, tuple)]
        assert len(pairs) == len(outputs)
        assert all(len(pair[1]) == len(first[1]) for pair in pairs)
        return (
            torch.cat([pair[0] for pair in pairs], dim=0),
            [
                torch.cat([pair[1][index] for pair in pairs], dim=0)
                for index in range(len(first[1]))
            ],
        )
    tensors = [output for output in outputs if isinstance(output, torch.Tensor)]
    assert len(tensors) == len(outputs)
    return torch.cat(tensors, dim=0)


@contextmanager
def use_npu_ubatch_output_merge() -> Iterator[None]:
    """Adapt only Ascend auxiliary-output merging; keep native input slicing."""
    previous_merge = ubatch_utils.merge_ubatch_outputs

    # Patch reason: Ascend auxiliary outputs are (Tensor, list[Tensor]), which
    # native torch.cat(parts) cannot concatenate. Preserve the tensor merge,
    # adapting only that nested contract. Signature matches
    # merge_ubatch_outputs at ced6857. Remove when upstream handles nested aux.
    def merge_ubatch_outputs(outputs: list[Any]) -> Any:
        # ### PATCH START: Ascend nested auxiliary outputs.
        return cat_ubatch_outputs(outputs)
        # ### PATCH END: Ascend nested auxiliary outputs.

    try:
        ubatch_utils.merge_ubatch_outputs = merge_ubatch_outputs
        yield
    finally:
        ubatch_utils.merge_ubatch_outputs = previous_merge
