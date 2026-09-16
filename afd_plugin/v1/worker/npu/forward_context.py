# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Forward-context helpers for plugin-owned Ascend ubatching."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, DPMetadata, ForwardContext
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

from afd_plugin.compat.patches.npu.mla_graph import (
    AFD_MLA_GRAPH_PARAMS_KEY,
)
from afd_plugin.v1.worker.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
)

if TYPE_CHECKING:
    from vllm_ascend.compilation.acl_graph import GraphParams


def create_ascend_forward_context(
    cur_forward_context: ForwardContext,
    attn_metadata,
    vllm_config: VllmConfig,
    ubatch_slices: UBatchSlices,
    ubatch_num: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    batch_descriptor: BatchDescriptor | None = None,
    skip_compiled: bool = False,
    mla_graph_params: GraphParams | None = None,
) -> ForwardContext:
    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = CUDAGraphMode.NONE

    parent_kwargs = dict(cur_forward_context.additional_kwargs or {})
    afd_metadata = parent_kwargs.get("afd_metadata")
    if afd_metadata is not None:
        parent_kwargs = build_ubatch_additional_kwargs(
            parent_kwargs,
            build_ubatch_afd_metadata(afd_metadata, ubatch_slices, ubatch_num),
        )
    if mla_graph_params is not None:
        parent_kwargs[AFD_MLA_GRAPH_PARAMS_KEY] = mla_graph_params

    ubatch_slice = ubatch_slices[ubatch_num]
    is_padding = (
        cur_forward_context.is_padding[ubatch_slice.token_slice]
        if cur_forward_context.is_padding is not None
        else None
    )
    new_forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        all_moe_layers=cur_forward_context.all_moe_layers,
        attn_metadata=attn_metadata,
        slot_mapping={},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=parent_kwargs,
        is_padding=is_padding,
    )

    num_tokens = ubatch_slice.num_tokens
    tp_world_size = get_tensor_model_parallel_world_size()
    dp_world_size = get_dp_group().world_size

    new_forward_context.moe_comm_type = cur_forward_context.moe_comm_type
    new_forward_context.moe_comm_method = get_moe_comm_method(
        new_forward_context.moe_comm_type
    )
    new_forward_context.in_profile_run = cur_forward_context.in_profile_run
    new_forward_context.capturing = (
        mla_graph_params is not None or cur_forward_context.capturing
    )
    new_forward_context.mmrs_fusion = cur_forward_context.mmrs_fusion
    new_forward_context.num_tokens = num_tokens
    new_forward_context.ubatch_idx = int(ubatch_num)
    new_forward_context.num_ubatches = len(ubatch_slices)
    new_forward_context.device_metadata_executor = (
        cur_forward_context.device_metadata_executor
    )
    new_forward_context.is_decode_only_node = cur_forward_context.is_decode_only_node
    new_forward_context.use_mega_moe = cur_forward_context.use_mega_moe
    new_forward_context.is_first_layer = cur_forward_context.is_first_layer
    new_forward_context.layer_idx = cur_forward_context.layer_idx
    new_forward_context.prefetch_mlp_gate_up_proj = (
        cur_forward_context.prefetch_mlp_gate_up_proj
    )
    new_forward_context.prefetch_mlp_down_proj = (
        cur_forward_context.prefetch_mlp_down_proj
    )
    new_forward_context.model_instance = cur_forward_context.model_instance
    new_forward_context.is_draft_model = cur_forward_context.is_draft_model
    new_forward_context.is_draft_model_prefill = (
        cur_forward_context.is_draft_model_prefill
    )
    new_forward_context.draft_attn_metadatas = cur_forward_context.draft_attn_metadatas
    new_forward_context.max_tokens_across_pcp = (
        cur_forward_context.max_tokens_across_pcp
    )
    new_forward_context.sinks = cur_forward_context.sinks
    if cur_forward_context.input_ids is not None:
        new_forward_context.input_ids = cur_forward_context.input_ids[
            ubatch_slice.token_slice
        ]
    else:
        new_forward_context.input_ids = None
    new_forward_context.eplb_heat_collection_status = (
        cur_forward_context.eplb_heat_collection_status
    )

    if dp_world_size > 1 and dp_metadata is not None:
        max_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.max().item()
    else:
        max_tokens_across_dp = num_tokens
    new_forward_context.max_tokens_across_dp = max_tokens_across_dp
    new_forward_context.padded_length = (
        math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
    )
    new_forward_context.padded_num_tokens = new_forward_context.padded_length
    cur_mc2_mask = cur_forward_context.mc2_mask
    if cur_mc2_mask is not None:
        mc2_mask = torch.zeros(
            (new_forward_context.padded_num_tokens,),
            dtype=cur_mc2_mask.dtype,
            device=cur_mc2_mask.device,
        )
        mc2_mask[:num_tokens] = True
        mc2_mask[num_tokens:] = False
        new_forward_context.mc2_mask = mc2_mask

    new_forward_context.dbo_enabled = True
    return new_forward_context


__all__ = ["create_ascend_forward_context"]
