# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped DBO seams for the pinned native GPU ModelRunnerV2.

The native UBatchRunner owns input slicing, Attention preparation, threads and
CUDA graphs. AFD extends DP1 eligibility, dispatch and child-context metadata.
Bindings are limited to AFD Attention initialization/execution and restored on
exit; worker initialization and dispatch run synchronously in each process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import Any

import torch
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.forward_context import DPMetadata, ForwardContext, create_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState, sync_cudagraph_and_dp_padding
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner, logger
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    get_num_ubatches,
)
from vllm.v1.worker.utils import AttentionGroup


# Patch reason: the native factory excludes DP1, although AFD schedules its two
# stages locally and sends their complete metadata to the FFN control plane.
# Patch functionality: construct the native UBatchRunner at DP1 and bind only its
# DP1 child-context metadata factory.
# Signature: matches maybe_build_ubatch_runner exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, maybe_build_ubatch_runner;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native GPU DBO supports DP1.
def maybe_build_ubatch_runner(
    vllm_config: VllmConfig,
    device: torch.device,
    model_state: ModelState,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_num_reqs: int,
) -> UBatchRunner | None:
    """Build the microbatch runner, or None when DBO is not in use.

    AFD also coordinates the two stages locally when DP is one.
    """
    parallel_config = vllm_config.parallel_config
    # ### PATCH START: permit local DP1 DBO.
    if not parallel_config.use_ubatching:
        return None
    # ### PATCH END: permit local DP1 DBO.

    logger.info_once("Dual batch overlap is enabled.")
    # ### PATCH START: bind native DP1 child metadata on this instance.
    runner = UBatchRunner(
        vllm_config, device, model_state, attn_groups, kv_cache_config, max_num_reqs
    )
    if parallel_config.data_parallel_size == 1:
        runner._make_forward_contexts = MethodType(_make_forward_contexts, runner)
    # ### PATCH END: bind native DP1 child metadata on this instance.
    return runner


# Patch reason: DPMetadata.make asserts DP>1 or SP; AFD DP1 uses neither.
# Patch functionality: build the same native DPMetadata directly for this DP1
# runner instance. Native context fields, token counts and SP methods are kept.
# Signature: matches UBatchRunner._make_forward_contexts at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, UBatchRunner._make_forward_contexts;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native child contexts support DP1 metadata.
def _make_forward_contexts(
    self,
    ubatch_slices: UBatchSlices,
    attn_metadata: list[dict[str, Any]],
    slot_mappings_by_layer: list[dict[str, torch.Tensor]],
    is_padding: list[torch.Tensor | None],
) -> list[ForwardContext]:
    """Build one forward context per microbatch.

    The expert all-to-all runs once per microbatch, so each context's DP
    metadata carries that microbatch's token count. Microbatching pads every
    DP rank alike, so each rank can assume that count for the others too.
    """
    dp_size = self.parallel_config.data_parallel_size
    forward_contexts = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        num_tokens = ubatch_slice.num_tokens
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )
        forward_contexts.append(
            create_forward_context(
                attn_metadata[i],
                self.vllm_config,
                # ### PATCH START: native metadata without the DP>1 assertion.
                dp_metadata=DPMetadata(num_tokens_across_dp),
                # ### PATCH END: native metadata without the DP>1 assertion.
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mappings_by_layer[i],
                is_padding=is_padding[i],
            )
        )
    return forward_contexts


# Patch reason: native DP1 dispatch returns before its DBO threshold decision.
# Patch functionality: apply the native threshold/graph-padding policy locally;
# retain native DP>1 coordination and native descriptor selection.
# Signature: matches dispatch_cg_and_sync_dp exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/dp_utils.py, dispatch_cg_and_sync_dp;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native dispatch selects DP1 microbatch descriptors.
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
    dp_sync_state: DPSyncState | None = None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Pick a cudagraph descriptor for this batch, agreeing it across DP ranks.

    Runs a collective when dp_size > 1 so every rank dispatches to the same
    shape. Pass `dp_sync_state` from a dispatch already made over this same batch (a
    drafter's prefill runs the target's batch shape) to reuse that agreement
    instead, with no collective.

    Args:
        cudagraph_manager: Manager to dispatch against. May be None only when
            `need_eager` is True (profile run).
        num_reqs: Requests in this rank's batch.
        num_tokens: Tokens in this rank's batch, already padded by the caller.
        uniform_token_count: Per-request token count if this rank's batch is a
            uniform decode, else None. `dp_sync_state.uniform_token_count` takes its
            place when a sync is reused, since that one is agreed across ranks.
        dp_size: Data-parallel world size. 1 skips all cross-rank work.
        dp_rank: This rank's index in the DP group.
        max_query_len: Upper bound on per-request query length, for selecting
            varlen decode graphs. None, as for a batch with a prefill, keeps
            the batch out of graphs that constrain it.
        need_eager: Force `CUDAGraphMode.NONE` instead of dispatching.
        num_active_loras: Active LoRA count for this rank. Does not need
            cross-rank agreement; it never changes a bucket's token count.
        parallel_config: Only needed to decide whether to microbatch, so
            callers that never do (`allow_ubatching=False`) can leave it out.
        allow_ubatching: Whether this rank's batch may be split into
            microbatches. Agreed across ranks before it takes effect.
        uniform_decode: Whether this rank's batch is a uniform decode, used
            to pick the microbatching split.
        dp_sync_state: Agreement from a prior dispatch over this same batch, to reuse.
            Must come from a batch with this same padded `num_tokens` and the
            same `uniform_token_count`; `num_reqs` may differ, as neither
            depends on it. Passing a sync from a different batch is a caller
            error and trips an assert.

    Returns:
        (batch_desc, sync), where `sync` is this batch's agreement for a later
        dispatch to reuse. It is None when `dp_size` is 1 or no rank has work.

    """
    reuse_eager = dp_sync_state is not None and dp_sync_state.eager

    if need_eager or reuse_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            dp_sync_state.uniform_token_count if dp_sync_state else uniform_token_count,
            num_active_loras=num_active_loras,
            max_query_len=max_query_len,
        )

    if dp_size == 1:
        # ### PATCH START: native DBO dispatch without a DP handshake.
        if (
            allow_ubatching
            and parallel_config is not None
            and num_tokens >= get_num_ubatches(parallel_config)
            and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)
        ):
            num_ubatches = get_num_ubatches(parallel_config)
            ubatch_desc = None
            if not need_eager and not reuse_eager:
                assert cudagraph_manager is not None
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
                if num_ubatches * num_tokens < ubatch_desc.num_tokens:
                    # Match the native DP>1 guard against an empty trailing stage.
                    ubatch_desc = None
            if ubatch_desc is None:
                ubatch_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
            return ubatch_desc, None
        # ### PATCH END: native DBO dispatch without a DP handshake.
        return batch_desc, None

    if dp_sync_state is not None:
        assert dp_sync_state.num_tokens_across_dp[dp_rank] == num_tokens, (
            "reusing a DP sync taken over a different batch"
        )
        assert (
            dp_sync_state.uniform_token_count is None
            or uniform_token_count == dp_sync_state.uniform_token_count
        ), "reusing a DP sync taken over a different batch"
        if not dp_sync_state.eager and batch_desc.num_tokens != num_tokens:
            # Capture sizes can differ between managers, so this one may
            # pad further. Every rank pads alike, so report what will run.
            dp_sync_state = replace(
                dp_sync_state,
                num_tokens_across_dp=torch.full_like(
                    dp_sync_state.num_tokens_across_dp, batch_desc.num_tokens
                ),
            )
        return batch_desc, dp_sync_state

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
    )


@contextmanager
def use_gpu_mrv2_ubatch_factory() -> Iterator[None]:
    """Scope the DP1 factory to AFD's native KV initialization call."""
    original_factory = model_runner.maybe_build_ubatch_runner
    model_runner.maybe_build_ubatch_runner = maybe_build_ubatch_runner
    try:
        yield
    finally:
        model_runner.maybe_build_ubatch_runner = original_factory


@contextmanager
def use_gpu_mrv2_dbo_dispatch() -> Iterator[None]:
    """Scope DP1 eligibility to one AFD native execute_model call."""
    original_dispatch = model_runner.dispatch_cg_and_sync_dp
    model_runner.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    try:
        yield
    finally:
        model_runner.dispatch_cg_and_sync_dp = original_dispatch
