# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Temporary CUDA DeepSeek-V4 DBO fixes for vLLM 0.32.0.

Upstream pin: 5df63a7203cdb9328d824a27b682e77c7b9928be.

Removal tracking:
* Input alignment: no published upstream PR. Remove the aligned stage storage
  when native compressor metadata accepts arbitrary int64-aligned views.
* Padding-only attention: no complete upstream fix. Related
  https://github.com/vllm-project/vllm/pull/63823 changes FULL-graph slicing,
  not the eager consumer; keep the empty-stage guard.
* Indexer SM budget: https://github.com/vllm-project/vllm/pull/60732.
* Workspace reservation: https://github.com/vllm-project/vllm/pull/61584 provides
  all-slot reservation, but V4 must also reserve its FlashMLA workspace there.

Delete a workaround only after the pinned runtime contains the complete fix
and its affected regression passes. These scopes retain the existing AFD-only
application points and restore native bindings on exit.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import WorkspaceManager, current_workspace_manager

from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context

if TYPE_CHECKING:
    from afd_plugin.v1.worker.attention_model_runner_v2 import AFDAttentionModelRunnerV2

_FLASHMLA_PROFILE_TOKENS = 1
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
def _use_v4_ubatch_preparation(
    runner: "AFDAttentionModelRunnerV2",
) -> Iterator[None]:
    """Prepare CUDA V4 stage inputs and indexer plans for native ubatching."""
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
    original_prepare_attn = cast(DefaultModelState, model_state).prepare_attn
    ubatch_state = vars(ubatch_runner)
    had_ubatch_prepare = "prepare" in ubatch_state
    previous_ubatch_prepare = ubatch_state.get("prepare")
    original_ubatch_prepare = ubatch_runner.prepare
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
        return original_prepare_attn(
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

    # Patch reason: native preparation uses physical-SM indexer plans and
    # rebuilds context slots from potentially unaligned microbatch views.
    # Patch functionality: lease the UBatchRunner compute budget to the native
    # indexer builders, restore it even on failure, and bind aligned live/capture
    # slots. Native slicing, metadata construction and FULL replay stay native.
    # Signature: matches UBatchRunner.prepare in vllm/v1/worker/gpu/ubatch_utils.py
    # at vLLM 5df63a7203cdb9328d824a27b682e77c7b9928be; indexer.py at the same pin
    # owns the physical-capacity buffers and planner refill. Delegation avoids
    # copying the full native preparation. Remove when native propagates its SM
    # budget (#60732) and provides stable aligned V4 stage inputs.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: V4 stage input and indexer preparation.
        indexer_budgets: list[tuple[DeepseekV32IndexerMetadataBuilder, int]] = []
        try:
            for groups in self.attn_groups:
                for group in groups:
                    for stage in range(self.num_ubatches):
                        builder = group.get_metadata_builder(stage)
                        if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                            indexer_budgets.append((builder, builder.num_sms))
                            builder.num_sms = self.sm_control.compute_sms
            prepared = original_ubatch_prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
        finally:
            # Reverse order also restores correctly if groups share a builder.
            for builder, num_sms in reversed(indexer_budgets):
                builder.num_sms = num_sms
        # FULL replay uses the context captured by the graph, not this fresh one.
        if cg_mode != CUDAGraphMode.FULL or for_capture:
            for stage, context in enumerate(prepared.forward_contexts):
                slots = inputs.slot_mappings[stage]
                if slots is not None:
                    context.slot_mapping = build_slot_mappings_by_layer(
                        slots[:, : prepared.slices[stage].num_tokens],
                        runner.kv_cache_config,
                    )
        return prepared
        # ### PATCH END: V4 stage input and indexer preparation.

    model_state.prepare_attn = MethodType(prepare_attn, model_state)
    ubatch_runner.begin_capturable_run = MethodType(begin_capturable_run, ubatch_runner)
    ubatch_runner.prepare = MethodType(prepare, ubatch_runner)
    try:
        yield
    finally:
        if had_ubatch_prepare:
            ubatch_runner.prepare = previous_ubatch_prepare
        else:
            del ubatch_runner.prepare
        if had_begin:
            ubatch_runner.begin_capturable_run = previous_begin
        else:
            del ubatch_runner.begin_capturable_run
        if had_prepare:
            model_state.prepare_attn = previous_prepare
        else:
            del model_state.prepare_attn


@contextmanager
def _use_flashmla_workspace_slot(
    manager: WorkspaceManager, slot: int
) -> Iterator[None]:
    """Select one real workspace owner only during startup reservation."""
    state = vars(manager)
    had_override = "_get_workspace_id" in state
    previous_override = state.get("_get_workspace_id")

    # Patch reason: native pre-KV profiling has no UBatch context and no public
    # workspace-slot selector. A real UBatch context would start handoff threads.
    # Patch functionality: route only this manager's allocation-only reservation
    # to a configured slot, then restore its exact instance/class lookup.
    # Signature: matches WorkspaceManager._get_workspace_id(self) -> int in
    # vllm/v1/worker/workspace.py, 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Remove with #61584 only after V4 uses its all-slot reservation API.
    def _get_workspace_id(self) -> int:
        return slot

    # ### PATCH START: select one startup workspace slot without DBO threads.
    manager._get_workspace_id = MethodType(_get_workspace_id, manager)
    try:
        yield
    finally:
        if had_override:
            manager._get_workspace_id = previous_override
        else:
            del manager._get_workspace_id
    # ### PATCH END: select one startup workspace slot without DBO threads.


def _reserve_flashmla_ubatch_workspace(model: nn.Module, config: VllmConfig) -> None:
    """Reserve native V4 prefill scratch before the normal profile peak."""
    attention_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, DeepseekV4FlashMLAAttention)
    ]
    if not attention_layers:
        return
    manager = current_workspace_manager()
    # Native forward_mqa's NoMeta branch (nvidia/flashmla.py:105-131, vLLM
    # 5df63a7203cdb9328d824a27b682e77c7b9928be) reserves its configured bound,
    # zeros only this small output, and returns before KV/kernels/Indexer/FFN.
    # Visit all real layers: the first SWA-only layer does not bound C4 demand.
    # Known manager geometry is the pin-local exception above, not a new policy.
    with set_forward_context(None, config):
        for layer in attention_layers:
            q = layer.attn_sink.new_empty(
                (_FLASHMLA_PROFILE_TOKENS, layer.padded_heads, layer.head_dim),
                dtype=torch.bfloat16,
            )
            output = torch.empty_like(q)
            kv = q.new_empty((_FLASHMLA_PROFILE_TOKENS, layer.head_dim))
            positions = torch.empty(
                (_FLASHMLA_PROFILE_TOKENS,), dtype=torch.int64, device=q.device
            )
            for slot in range(manager._num_ubatches * manager._num_lanes):
                with _use_flashmla_workspace_slot(manager, slot):
                    layer.forward_mqa(q, kv, positions, output)


# Patch reason: native V4 attention asserts on a padding-only decode stage
# because no tile scheduler exists for zero actual tokens (vLLM 5df63a7,
# vllm/models/deepseek_v4/nvidia/flashmla.py, forward_mqa/_forward_decode).
# Patch functionality: retain the existing AFD stage-extent guard while the
# decoder continues its mHC and remote FFN protocol. This call adapter takes the
# attention instance explicitly; it does not replace a native method signature.
# Removal: #63823 is related FULL-graph work, not an eager consumer fix. Remove
# when native attention supports zero-token padded stages in the supported modes.
def forward_v4_attention(
    attention: DeepseekV4Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # ### PATCH START: preserve padded-stage protocol with zero attention.
    afd_metadata = get_afd_metadata_from_forward_context()
    if (
        afd_metadata is not None
        and afd_metadata.num_stages > 1
        and afd_metadata.tokens_unpadded_lens == [0]
    ):
        return torch.zeros_like(hidden_states)
    # ### PATCH END: preserve padded-stage protocol with zero attention.
    return attention(positions, hidden_states, None)
