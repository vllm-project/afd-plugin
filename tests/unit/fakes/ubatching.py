# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU doubles for stage-input and indexer-plan ownership, not device execution."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchState,
    create_ubatch_slices,
    slice_model_inputs,
)

TEST_PHYSICAL_SMS = 8
TEST_COMPUTE_SMS = 6


class IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """CPU planner stand-in exposing the concrete native builder's budget seam."""

    def __init__(self):
        self.num_sms = TEST_PHYSICAL_SMS

    def build_schedule(self, positions):
        # Expose the leased budget at the metadata boundary; no kernel simulation.
        return torch.full((self.num_sms + 1,), positions[0], dtype=torch.int32)


class AttentionMetadataState:
    def __init__(self):
        self.prepare_error: Exception | None = None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        if self.prepare_error is not None:
            raise self.prepare_error
        metadata = {}
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            builder = attn_groups[group_index][0].get_metadata_builder(ubatch_idx)
            group_metadata = SimpleNamespace(
                positions=input_batch.positions,
                slot_mapping=slot_mappings[group_index],
                decode=SimpleNamespace(
                    schedule_metadata=builder.build_schedule(input_batch.positions)
                ),
            )
            metadata.update({layer: group_metadata for layer in group.layer_names})
        return metadata


class PreparedUBatchRunner:
    """Exercise the native prepare/begin interfaces without CUDA threads."""

    num_ubatches = 2

    def __init__(self, model_state, kv_cache_config):
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.stage = 1
        self.sm_control = SimpleNamespace(compute_sms=TEST_COMPUTE_SMS)
        self.attn_groups = []
        for _ in kv_cache_config.kv_cache_groups:
            builders = [IndexerMetadataBuilder() for _ in range(self.num_ubatches)]
            self.attn_groups.append(
                [SimpleNamespace(get_metadata_builder=builders.__getitem__)]
            )

    def prepare(
        self,
        input_batch,
        block_tables,
        slot_mappings,
        cg_mode,
        for_capture=False,
    ):
        slices = create_ubatch_slices(input_batch, self.num_ubatches)
        contexts = []
        for stage, ubatch_slice in enumerate(slices):
            slots = slot_mappings[:, ubatch_slice.token_slice]
            metadata = self.model_state.prepare_attn(
                replace(
                    input_batch,
                    positions=input_batch.positions[ubatch_slice.token_slice],
                ),
                cg_mode,
                block_tables,
                slots,
                self.attn_groups,
                self.kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=stage,
            )
            contexts.append(
                ForwardContext(
                    no_compile_layers={},
                    attn_metadata=metadata,
                    slot_mapping=build_slot_mappings_by_layer(
                        slots, self.kv_cache_config
                    ),
                    additional_kwargs={"platform_stage": stage},
                )
            )
        return UBatchState(slices, contexts)

    def begin_capturable_run(
        self,
        model,
        model_inputs,
        ubatch_state,
        for_capture=False,
    ):
        # A retained callable proves input-address ownership/freshness only;
        # this is deliberately not a fake CUDA graph or scheduling test.
        def finish():
            return model(
                **slice_model_inputs(
                    model_inputs,
                    ubatch_state.slices[self.stage].token_slice,
                )
            )

        return finish


class LegacyNPUAttentionMetadataState:
    """Expose the vLLM 0.30 attention-preparation signature on CPU."""

    def __init__(self):
        self.prepare_error = None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
    ):
        if self.prepare_error is not None:
            raise self.prepare_error
        return {
            "kv0": SimpleNamespace(
                positions=input_batch.positions,
                slot_mapping=slot_mappings[0],
                stage=ubatch_idx,
            )
        }


class LegacyNPUUBatchRunner:
    """Build native stage contexts without CUDA state or scheduling."""

    num_ubatches = 2

    def __init__(self, model_state, kv_cache_config):
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config

    def prepare(
        self, input_batch, block_tables, slot_mappings, cg_mode, for_capture=False
    ):
        slices = create_ubatch_slices(input_batch, self.num_ubatches)
        contexts = []
        for stage, ubatch_slice in enumerate(slices):
            slots = slot_mappings[:, ubatch_slice.token_slice]
            metadata = self.model_state.prepare_attn(
                replace(
                    input_batch,
                    positions=input_batch.positions[ubatch_slice.token_slice],
                ),
                cg_mode,
                block_tables,
                slots,
                [],
                self.kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=stage,
            )
            contexts.append(
                ForwardContext(
                    no_compile_layers={},
                    attn_metadata=metadata,
                    slot_mapping={"kv0": slots[0]},
                    additional_kwargs={"platform_stage": stage},
                )
            )
        return UBatchState(slices, contexts)
