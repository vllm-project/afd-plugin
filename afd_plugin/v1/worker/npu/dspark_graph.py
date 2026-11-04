# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Install AFD metadata around native draft graph capture and replay."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dflash_cudagraph
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager

from afd_plugin.v1.worker.attention_model_runner_v2 import _AFDCaptureEventTracker

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


@contextmanager
def use_afd_draft_capture(
    runner: AFDNPUAttentionModelRunnerV2,
    manager: DFlashAclGraphManager,
) -> Iterator[None]:
    """Associate every native draft capture forward with its own AFD batch."""
    descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])[
        : manager._max_full_descs_to_capture
    ]
    if not runner.speculative_config.enforce_eager and not descs:
        raise RuntimeError("AFD DSpark draft FULL has no native capture descriptors")
    tracker = _AFDCaptureEventTracker(descs)
    original_prepare = dflash_cudagraph._prepare_dflash_inputs_to_capture

    # Upstream: vLLM 5df63a72, dflash/cudagraph.py; Ascend bd42eea0 delegates
    # draft capture here. Signature matches the native preparation function.
    # Native input/KV preparation remains in the saved function; only AFD
    # metadata is added. Remove when draft capture exposes a plugin callback.
    def prepare_inputs(
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        skip_attn: bool,
        causal: bool | Mapping[int, bool],
    ) -> AttentionState:
        # ### PATCH START: stage descriptor-local draft metadata.
        state = original_prepare(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_groups,
            kv_cache_config,
            max_model_len,
            skip_attn,
            causal,
        )
        desc, is_warmup = tracker.consume(num_reqs, num_tokens)
        runner._afd_pending_metadata = runner.build_afd_metadata(None, desc.num_tokens)
        runner._afd_suppress_metadata_send = True
        runner._is_warmup = is_warmup
        runner._afd_is_graph_capturing = not is_warmup
        runner._afd_is_graph_replaying = False
        return state
        # ### PATCH END: stage descriptor-local draft metadata.

    previous_capturing = runner._afd_is_graph_capturing
    previous_replaying = runner._afd_is_graph_replaying
    dflash_cudagraph._prepare_dflash_inputs_to_capture = prepare_inputs
    try:
        yield
        tracker.assert_complete()
    finally:
        dflash_cudagraph._prepare_dflash_inputs_to_capture = original_prepare
        runner._afd_is_graph_capturing = previous_capturing
        runner._afd_is_graph_replaying = previous_replaying


@contextmanager
def use_afd_draft_replay(
    runner: AFDNPUAttentionModelRunnerV2,
    manager: DFlashAclGraphManager,
    real_tokens: int,
) -> Iterator[None]:
    """Preserve native replay while staging the draft's real and padded rows."""
    state = manager.__dict__
    had_override = "run_fullgraph" in state
    previous_override = state.get("run_fullgraph")
    original_replay = manager.run_fullgraph

    # Upstream: Ascend bd42eea0 DFlashAclGraphManager.run_fullgraph.
    # Signature matches upstream. Delegate graph replay and attention updates;
    # add only AFD's model-local metadata. Remove when the native manager has
    # an instance-local pre-replay callback.
    def run_fullgraph(
        self,
        desc: BatchExecutionDescriptor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # ### PATCH START: install draft-local replay metadata.
        previous = (
            runner._afd_pending_metadata,
            runner._afd_suppress_metadata_send,
            runner._is_warmup,
            runner._afd_is_graph_capturing,
            runner._afd_is_graph_replaying,
        )
        try:
            metadata = runner.build_afd_metadata(None, real_tokens)
            metadata.tokens_lens = [desc.num_tokens]
            runner._afd_pending_metadata = metadata
            runner._afd_suppress_metadata_send = True
            runner._is_warmup = False
            runner._afd_is_graph_capturing = False
            runner._afd_is_graph_replaying = True
            return original_replay(desc)
        finally:
            (
                runner._afd_pending_metadata,
                runner._afd_suppress_metadata_send,
                runner._is_warmup,
                runner._afd_is_graph_capturing,
                runner._afd_is_graph_replaying,
            ) = previous
        # ### PATCH END: install draft-local replay metadata.

    try:
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
        yield
    finally:
        if had_override:
            manager.run_fullgraph = previous_override
        else:
            del manager.run_fullgraph
