# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 model-owned Async CAM ubatch execution."""

from __future__ import annotations

from copy import copy
from itertools import islice
from typing import Final

import torch
from vllm.distributed import get_pp_group
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm_ascend.models.deepseek_v4 import model as native

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    AsyncMoeUbatchMetadata,
    CAMDispatchLayout,
    CAMDispatchPayload,
    build_async_moe_stage_inputs,
    prepare_cam_dispatch_payload,
    restore_async_moe_stage_outputs,
    restore_cam_dispatch_output,
)
from afd_plugin.model_executor.models.npu.deepseek_v4 import (
    AFDDeepseekV4AttentionGateRemoteMoE,
    AFDDeepseekV4DecoderLayer,
    AFDDeepseekV4Model,
    _run_two_stage_async_moe_schedule,
)

_ASYNC_MOE_STAGE_COUNT: Final[int] = 2
_PAD_INPUT_ID: Final[int] = -1


def _pad_stage_input_ids(
    input_ids: torch.Tensor,
    metadata: AsyncMoeUbatchMetadata,
) -> list[torch.Tensor]:
    """Build the token IDs that correspond to each physical AFD stage."""
    flat_input_ids = input_ids.reshape(-1)
    actual_parent_tokens = max(int(stage.token_slice.stop) for stage in metadata.stages)
    if int(flat_input_ids.numel()) < actual_parent_tokens:
        raise ValueError(
            "DSV4 async MoE input_ids do not cover the staged tokens: "
            f"input_ids={int(flat_input_ids.numel())}, "
            f"staged_tokens={actual_parent_tokens}",
        )
    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    stage_input_ids: list[torch.Tensor] = []
    for stage in metadata.stages:
        ids = flat_input_ids[stage.token_slice]
        physical_tokens = int(stage.input_tokens)
        if int(ids.numel()) < physical_tokens:
            ids = torch.nn.functional.pad(
                ids,
                (0, physical_tokens - int(ids.numel())),
                value=_PAD_INPUT_ID,
            )
        if metadata.use_sequence_parallel:
            if physical_tokens % tp_size != 0:
                raise ValueError(
                    "DSV4 async MoE stage is not TP divisible: "
                    f"tokens={physical_tokens}, tp_size={tp_size}",
                )
            local_tokens = physical_tokens // tp_size
            local_start = tp_rank * local_tokens
            ids = ids[local_start : local_start + local_tokens]
        stage_input_ids.append(ids)
    return stage_input_ids


def run_async_moe_ubatch_forward(
    model: AFDDeepseekV4Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    metadata: AsyncMoeUbatchMetadata,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor | IntermediateTensors:
    """Run DSV4's two AFD-owned CAM stages in one model invocation."""
    if len(metadata.stages) != _ASYNC_MOE_STAGE_COUNT:
        raise ValueError(
            "DSV4 async MoE currently requires exactly two AFD stages, got "
            f"{len(metadata.stages)}",
        )
    pp_group = get_pp_group()
    if pp_group.is_first_rank:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else model.embed_input_ids(input_ids)
        )
        hidden_states = hidden_states.unsqueeze(1).repeat(1, model.hc_mult, 1)
    else:
        if intermediate_tensors is None:
            raise ValueError("DSV4 pipeline stage requires intermediate tensors")
        hidden_states = intermediate_tensors["hidden_states"]

    parent_context = get_forward_context()
    if model.use_sequence_parallel_moe != metadata.use_sequence_parallel:
        raise RuntimeError(
            "DSV4 async MoE stage layout does not match model sequence parallelism: "
            f"layout_sequence_parallel={metadata.use_sequence_parallel}, "
            f"use_sequence_parallel_moe={model.use_sequence_parallel_moe}",
        )
    afd_metadata = get_afd_metadata_from_forward_context(parent_context)
    if afd_metadata is None:
        raise RuntimeError("DSV4 async MoE requires AFD forward metadata")

    stage_inputs = build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
    )
    stage_hidden_states = stage_inputs.hidden_states
    stage_positions = stage_inputs.positions
    stage_input_ids = _pad_stage_input_ids(input_ids, metadata)
    stage_dispatch_layouts: list[CAMDispatchLayout | None] = [
        None for _ in metadata.stages
    ]
    stage_dispatch_refs: list[torch.Tensor | None] = [None for _ in metadata.stages]
    stage_shared_outputs: list[torch.Tensor | None] = [None for _ in metadata.stages]
    stage_pending_dispatches: list[CAMDispatchPayload | None] = [
        None for _ in metadata.stages
    ]
    stage_ffn_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
        None for _ in metadata.stages
    ]

    def stage_context(stage_idx: int) -> ForwardContext:
        stage = metadata.stages[stage_idx]
        context = copy(parent_context)
        context.attn_metadata = metadata.attn_metadata[stage_idx]
        context.additional_kwargs = dict(parent_context.additional_kwargs or {})
        context.ubatch_idx = stage_idx
        context.num_ubatches = len(metadata.stages)
        context.dbo_enabled = False
        context.input_ids = stage_input_ids[stage_idx]
        context.num_tokens = stage.actual_tokens
        context.is_padding = None
        return context

    def compute_stage_attention(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
    ) -> None:
        if not isinstance(layer.mlp, AFDDeepseekV4AttentionGateRemoteMoE):
            raise RuntimeError("DSV4 async MoE requires Attention-side expert routing")
        with override_forward_context(stage_context(stage_idx)):
            stage_hidden = stage_hidden_states[stage_idx]
            attn_residual = stage_hidden.clone()
            stage_hidden, attn_post, attn_comb = layer.hc_pre(
                stage_hidden,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
            )
            stage_hidden = layer.input_layernorm(stage_hidden)
            if metadata.use_sequence_parallel and not layer.enable_dsa_cp:
                stage_hidden = native.sp_all_gather(stage_hidden)[
                    : metadata.stages[stage_idx].actual_tokens
                ]
            stage_hidden = layer.self_attn(
                positions=stage_positions[stage_idx],
                hidden_states=stage_hidden,
                llama_4_scaling=None,
            )
            if metadata.use_sequence_parallel and not layer.enable_dsa_cp:
                stage_hidden = native.sp_reduce_scatter(stage_hidden)
            stage_hidden = layer.hc_post(
                stage_hidden,
                attn_residual,
                attn_post,
                attn_comb,
            )
            ffn_residual = stage_hidden.clone()
            stage_hidden, ffn_post, ffn_comb = layer.hc_pre(
                stage_hidden,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
            )
            stage_hidden, stage_hidden_fp32 = layer.rms_norm_cast(stage_hidden)
            from afd_plugin.model_executor.models.npu import (
                deepseek_v4_attention_gate,
            )

            topk_weights, topk_ids = (
                deepseek_v4_attention_gate.compute_attention_gate_topk(
                    layer.mlp,
                    stage_hidden,
                    input_ids=stage_input_ids[stage_idx],
                    hidden_states_fp32=stage_hidden_fp32,
                )
            )
            dispatch = prepare_cam_dispatch_payload(
                stage_hidden,
                topk_weights,
                topk_ids,
                None,
                use_sequence_parallel=metadata.use_sequence_parallel,
            )
        stage_shared_outputs[stage_idx] = (
            layer.mlp.shared_experts(stage_hidden)
            if layer.mlp.shared_experts is not None
            else None
        )
        stage_pending_dispatches[stage_idx] = dispatch
        stage_ffn_state[stage_idx] = (ffn_residual, ffn_post, ffn_comb)

    def send_stage_attention(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
    ) -> None:
        dispatch = stage_pending_dispatches[stage_idx]
        if dispatch is None:
            raise RuntimeError(
                f"DSV4 async MoE stage {stage_idx} has no computed Attention"
            )
        transfer_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch.hidden_states.shape[0]),
        )
        afd_metadata.connector.send_attn_output(
            dispatch.hidden_states,
            AFDTransferContext(metadata=transfer_metadata),
            topk_weights=dispatch.topk_weights,
            topk_ids=dispatch.topk_ids,
        )
        stage_dispatch_layouts[stage_idx] = dispatch.layout
        stage_dispatch_refs[stage_idx] = dispatch.hidden_states
        stage_pending_dispatches[stage_idx] = None

    def receive_and_complete(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
    ) -> None:
        layout = stage_dispatch_layouts[stage_idx]
        dispatch_ref = stage_dispatch_refs[stage_idx]
        ffn_state = stage_ffn_state[stage_idx]
        if layout is None or dispatch_ref is None or ffn_state is None:
            raise RuntimeError(
                f"DSV4 async MoE stage {stage_idx} has no pending FFN work"
            )
        local_output = afd_metadata.connector.recv_ffn_output(
            ref_tensor=dispatch_ref,
            ubatch_idx=stage_idx,
        )
        ffn_output = restore_cam_dispatch_output(local_output, layout)
        shared_output = stage_shared_outputs[stage_idx]
        if shared_output is not None:
            ffn_output = ffn_output + shared_output
        stage_shared_outputs[stage_idx] = None
        ffn_residual, ffn_post, ffn_comb = ffn_state
        with override_forward_context(stage_context(stage_idx)):
            stage_hidden_states[stage_idx] = layer.hc_post(
                ffn_output,
                ffn_residual,
                ffn_post,
                ffn_comb,
            )
        stage_dispatch_layouts[stage_idx] = None
        stage_dispatch_refs[stage_idx] = None
        stage_ffn_state[stage_idx] = None

    layers = list(islice(model.layers, model.start_layer, model.end_layer))
    if layers:
        _run_two_stage_async_moe_schedule(
            layers,
            compute_stage_attention,
            send_stage_attention,
            receive_and_complete,
        )

    restored_hidden_states = restore_async_moe_stage_outputs(
        stage_hidden_states,
        metadata,
    )

    if model._needs_mtp_hidden_states:
        if model._mtp_hidden_buffer is None:
            model._mtp_hidden_buffer = torch.empty(
                model._mtp_buffer_shape,
                dtype=model._mtp_buffer_dtype,
                device=model.device,
            )
        hidden_flat = restored_hidden_states.flatten(1)
        model._mtp_hidden_buffer[: hidden_flat.shape[0]].copy_(hidden_flat)

    if not pp_group.is_last_rank:
        return IntermediateTensors({"hidden_states": restored_hidden_states})
    output = model.hc_head(
        restored_hidden_states,
        model.hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
    )
    return model.norm(output)


__all__ = ["run_async_moe_ubatch_forward"]
