# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped DBO seams for the pinned native GPU ModelRunnerV2.

The native UBatchRunner owns input slicing, Attention preparation, threads and
CUDA graphs. AFD extends DP1 eligibility, dispatch and child-context metadata,
and keeps FULL microbatched decodes on the captured request split so an
underfilled DP rank replays the graph instead of running eager.
Bindings are limited to AFD Attention initialization/execution and restored on
exit; worker initialization and dispatch run synchronously in each process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata, ForwardContext, create_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    _slice_input_batch,
    create_ubatch_slices,
    logger,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
    get_num_ubatches,
)
from vllm.v1.worker.utils import AttentionGroup

# A FULL microbatched decode graph is captured with one token per request.
DECODE_QUERY_LEN = 1
# Host-side sequence bound for a microbatch holding only padding requests; the
# graph's kernels were captured with max_model_len and ignore it.
PADDING_ONLY_SEQ_LEN_BOUND = 1


def replays_captured_ubatch_split(vllm_config: VllmConfig) -> bool:
    """Whether FULL microbatched decodes keep the captured request split.

    The split keeps every microbatch's captured request and token range, so a
    rank holding at most half of the graph's requests replays its trailing
    microbatch on padding rows only. Limited to one-token decodes over
    attention without inner state or context parallelism; MLA is not yet
    validated with a padding-only microbatch.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    return (
        vllm_config.speculative_config is None
        and not model_config.has_inner_state
        and not model_config.use_mla
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
    )


def needs_regular_graph_fallback(
    cudagraph_manager: CudaGraphManager,
    num_reqs: int,
    min_num_tokens: int,
    max_num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
    num_ubatches: int,
) -> bool:
    """Whether a microbatched graph would replay a shifted request split.

    Native slicing re-derives the split from the real requests, so a graph only
    replays correctly when the smallest rank fills more than its leading
    microbatches. Without the captured split, such a step replays the regular
    graph instead of running eager. Every rank passes the same agreed token
    bounds, so every rank returns the same answer.
    """
    if replays_captured_ubatch_split(cudagraph_manager.vllm_config):
        return False
    ubatch_desc = cudagraph_manager.dispatch(
        num_reqs,
        max_num_tokens,
        uniform_token_count,
        num_active_loras=num_active_loras,
        num_ubatches=num_ubatches,
    )
    return (
        ubatch_desc.cg_mode == CUDAGraphMode.FULL
        and num_ubatches * min_num_tokens <= ubatch_desc.num_tokens
    )


def captured_ubatch_slices(input_batch: InputBatch, num_ubatches: int) -> UBatchSlices:
    """Slices a FULL one-token-per-request graph was captured with.

    Capture splits a dummy batch whose request and token counts are equal at
    `num_tokens // num_ubatches`, the last microbatch taking the remainder.
    Native slicing yields the same slices whenever real requests reach the
    last microbatch.
    """
    num_tokens = input_batch.num_tokens_after_padding
    split_point = num_tokens // num_ubatches
    bounds = [split_point * i for i in range(num_ubatches)] + [num_tokens]
    return [
        UBatchSlice(slice(start, stop), slice(start, stop))
        for start, stop in pairwise(bounds)
    ]


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
    # ### PATCH START: bind the captured-split prepare on this instance.
    if replays_captured_ubatch_split(vllm_config):
        runner.prepare = MethodType(prepare, runner)
    # ### PATCH END: bind the captured-split prepare on this instance.
    return runner


# Patch reason: native prepare re-derives the microbatch split from the real
# requests and clamps an empty trailing microbatch onto the last real request.
# A FULL microbatched graph replays fixed request/token offsets, so a rank with
# at most half of the graph's requests would replay with shifted metadata, and
# native dispatch runs such steps eager instead (`2 * min < N`).
# Patch functionality: slice FULL one-token decodes at the captured split. A
# trailing microbatch without real requests keeps its padded shape, with a
# one-token query bound and a nonzero host-side sequence bound.
# Signature: matches UBatchRunner.prepare exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/ubatch_utils.py, UBatchRunner.prepare;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native FULL microbatching keeps the captured split.
def prepare(
    self,
    input_batch: InputBatch,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    for_capture: bool = False,
) -> UBatchState:
    """Split the batch into the microbatches the step will run on.

    Attention metadata is built per microbatch and carried in the forward
    contexts the threads install, not in the caller's.

    FULL uses padded sizes; `for_capture` refreshes metadata from dummy
    block tables.
    """
    # ### PATCH START: slice FULL one-token decodes at the captured split.
    keeps_captured_split = (
        cg_mode == CUDAGraphMode.FULL
        and input_batch.num_tokens == input_batch.num_reqs * DECODE_QUERY_LEN
        and input_batch.num_tokens_after_padding
        == input_batch.num_reqs_after_padding * DECODE_QUERY_LEN
    )
    if keeps_captured_split:
        # A microbatch without real requests has no query length to derive.
        input_batch = replace(input_batch, max_query_len=DECODE_QUERY_LEN)
        ubatch_slices = captured_ubatch_slices(input_batch, self.num_ubatches)
    else:
        ubatch_slices = create_ubatch_slices(input_batch, self.num_ubatches)
    # ### PATCH END: slice FULL one-token decodes at the captured split.

    attn_metadata = []
    slot_mappings_by_layer = []
    is_padding = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        ubatch = _slice_input_batch(
            input_batch,
            ubatch_slice,
            self.ubatch_query_start_loc[i],
            self.ubatch_seq_lens[i],
            self.ubatch_dcp_local_seq_lens[i],
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        # ### PATCH START: keep a padding-only microbatch's sequence bound nonzero.
        if keeps_captured_split and ubatch.num_reqs == 0:
            ubatch.seq_lens_cpu_upper_bound[0] = PADDING_ONLY_SEQ_LEN_BOUND
        # ### PATCH END: keep a padding-only microbatch's sequence bound nonzero.
        ubatch_slot_mappings = slot_mappings[:, ubatch_slice.token_slice]
        ubatch_block_tables = tuple(
            block_table[ubatch_slice.request_slice] for block_table in block_tables
        )
        attn_metadata.append(
            self.model_state.prepare_attn(
                ubatch,
                cg_mode,
                ubatch_block_tables,
                ubatch_slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
                ubatch_idx=i,
                for_capture=for_capture,
            )
        )
        slot_mappings_by_layer.append(
            build_slot_mappings_by_layer(ubatch_slot_mappings, self.kv_cache_config)
        )
        is_padding.append(ubatch.is_padding)

    return UBatchState(
        slices=ubatch_slices,
        forward_contexts=self._make_forward_contexts(
            ubatch_slices, attn_metadata, slot_mappings_by_layer, is_padding
        ),
    )


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


# Patch reason: native DP>1 microbatching runs eager whenever the smallest rank
# holds at most half of the graph (`2 * min < N`), because native slicing would
# replay a shifted request split.
# Patch functionality: keep the microbatched FULL graph when AFD's prepare
# replays the captured split; otherwise replay the regular graph, never eager.
# Every other branch is the native coordination unchanged.
# Signature: matches sync_cudagraph_and_dp_padding exactly at the pinned commit.
# Upstream: vllm/v1/worker/gpu/dp_utils.py, sync_cudagraph_and_dp_padding;
# 5df63a7203cdb9328d824a27b682e77c7b9928be (vLLM 0.32.0).
# Removal plan: remove when native FULL microbatching keeps the captured split.
def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    parallel_config: ParallelConfig | None = None,
    allow_ubatching: bool = False,
    uniform_decode: bool = False,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Coordinates the batch descriptor and DP padding across all ranks.

    `parallel_config` is only needed to decide whether to microbatch, so callers
    that never do (`allow_ubatching=False`) can leave it out.

    Returns (synced_batch_desc, sync). `sync` is None when no rank has work.
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(6, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    tensor[4][dp_rank] = int(allow_ubatching)
    tensor[5][dp_rank] = num_reqs
    if should_skip_dp_coordination():
        tensor[:] = tensor[:, dp_rank, None].clone()
    else:
        dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]
    allow_ubatching_across_dp = tensor[4]
    num_reqs_across_dp = tensor[5]

    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    synced_uniform_token_count: int | None = int(uniform_token_counts_across_dp[0])
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    if torch.all(allow_ubatching_across_dp == 1):
        # This rank voted too, so its caller passed a config.
        assert parallel_config is not None
        # A uniform decode only if every rank runs the same uniform query
        # length, so every rank reaches the same answer here.
        uniform_decode_across_dp = uniform_decode and torch.all(
            uniform_token_counts_across_dp == (uniform_token_count or 0)
        )
        # ### PATCH START: regular graph when the captured split cannot serve.
        if check_ubatch_thresholds(
            parallel_config,
            # Thresholds only grow with the token count, so holding the smallest
            # rank to them holds every rank to them.
            int(num_tokens_across_dp.min()),
            uniform_decode=uniform_decode_across_dp,
        ) and not (
            cudagraph_manager is not None
            and needs_regular_graph_fallback(
                cudagraph_manager,
                num_reqs,
                int(num_tokens_across_dp.min()),
                int(num_tokens_across_dp.max()),
                synced_uniform_token_count,
                num_active_loras,
                get_num_ubatches(parallel_config),
            )
        ):
            # ### PATCH END: regular graph when the captured split cannot serve.
            # Expert all-to-all requires every rank to split and pad equally.
            # Empty microbatches run as dummy batches.
            ubatch_num_tokens = int(num_tokens_across_dp.max())
            num_ubatches = get_num_ubatches(parallel_config)
            ubatch_desc = None
            if cudagraph_manager is not None:
                # Match a FULL microbatched CUDA graph, falling back to an eager
                # descriptor if no graph matches.
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    ubatch_num_tokens,
                    synced_uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
                # ### PATCH START: no eager fallback for an underfilled rank.
                # The captured split replays it; otherwise the threshold check
                # above already routed the step to the regular graph.
                # ### PATCH END: no eager fallback for an underfilled rank.
            if ubatch_desc is None:
                ubatch_desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=ubatch_num_tokens,
                    num_reqs=num_reqs,
                    num_ubatches=num_ubatches,
                )
            # Refresh the token count to include CUDA graph padding.
            ubatch_num_tokens = ubatch_desc.num_tokens
            num_reqs = int(num_reqs_across_dp.max())
            if ubatch_desc.cg_mode == CUDAGraphMode.FULL:
                assert ubatch_desc.num_reqs is not None
                num_reqs = ubatch_desc.num_reqs
            return ubatch_desc, DPSyncState(
                num_tokens_across_dp=torch.full_like(
                    num_tokens_across_dp, ubatch_num_tokens
                ),
                uniform_token_count=synced_uniform_token_count,
                eager=ubatch_desc.cg_mode == CUDAGraphMode.NONE,
                num_reqs=num_reqs,
            )

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=desired_batch_desc.num_active_loras,
            ),
            DPSyncState(
                num_tokens_across_dp=num_tokens_across_dp,
                uniform_token_count=synced_uniform_token_count,
                eager=True,
                num_reqs=int(num_reqs_across_dp.max()),
            ),
        )

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max())

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1)):
        synced_max_query_len = int(max_query_lens_across_dp.max())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, DPSyncState(
        num_tokens_across_dp=num_tokens_across_dp,
        uniform_token_count=synced_uniform_token_count,
        eager=False,
        num_reqs=(
            synced_desc.num_reqs
            if synced_desc.cg_mode == CUDAGraphMode.FULL
            and synced_desc.num_reqs is not None
            else int(num_reqs_across_dp.max())
        ),
    )


# Patch reason: native DP1 dispatch returns before its DBO threshold decision.
# Patch functionality: apply the native threshold/graph-padding policy locally,
# replaying the regular graph when the captured split cannot serve the step;
# route DP>1 coordination through AFD's sync_cudagraph_and_dp_padding.
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
                if needs_regular_graph_fallback(
                    cudagraph_manager,
                    num_reqs,
                    num_tokens,
                    num_tokens,
                    uniform_token_count,
                    num_active_loras,
                    num_ubatches,
                ):
                    return batch_desc, None
                ubatch_desc = cudagraph_manager.dispatch(
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
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
