# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 async CAM forward orchestration helpers."""

from __future__ import annotations

from copy import copy
from itertools import islice
from typing import TYPE_CHECKING

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    AsyncMoeUbatchMetadata,
    CAMDispatchLayout,
    build_async_moe_stage_inputs,
    get_async_moe_ubatch_metadata_from_forward_context,
    log_async_moe_stage_attention,
    prepare_cam_dispatch_payload,
    restore_async_moe_stage_outputs,
    restore_cam_dispatch_output,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        AFDDeepseekV2Model,
    )


def run_model_forward(
    model: AFDDeepseekV2Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    """Run the pinned Model fragment around the AFD-owned async schedule."""

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "Either input_ids or inputs_embeds must be provided "
                    "to AFDDeepseekV2Model.forward",
                )
            hidden_states = model.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    if model.aux_hidden_state_layers:
        raise RuntimeError(
            "AFD DeepSeekV2 async CAM does not support aux hidden state capture",
        )
    forward_context = get_forward_context()
    afd_metadata = get_afd_metadata_from_forward_context(forward_context)
    if afd_metadata is None:
        raise RuntimeError("async CAM requires AFD forward metadata")
    llama_4_scaling = model._get_llama_4_scaling(positions)
    async_moe_ubatch_metadata = get_async_moe_ubatch_metadata_from_forward_context(
        forward_context
    )
    if async_moe_ubatch_metadata is None:
        hidden_states, residual = run_attention_gate_afd_forward(
            model,
            hidden_states,
            residual,
            positions,
            afd_metadata,
            llama_4_scaling,
        )
    else:
        hidden_states, residual = run_async_moe_ubatch_afd_forward(
            model,
            hidden_states,
            residual,
            positions,
            afd_metadata,
            async_moe_ubatch_metadata,
            llama_4_scaling,
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual},
        )
    hidden_states, _ = model.norm(hidden_states, residual)
    return hidden_states


def run_attention_gate_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Attention-side gate AFD path used by async CAM."""

    afd_connector = afd_metadata.connector
    forward_context = get_forward_context()
    stage_idx = int(afd_metadata.stage_idx)
    pending_ffn_recv = False
    pending_dispatch_layout: CAMDispatchLayout | None = None
    pending_dispatch_ref: torch.Tensor | None = None

    for layer_offset, layer in enumerate(
        islice(model.layers, model.start_layer, model.end_layer),
    ):
        if layer_offset > 0 and pending_ffn_recv:
            if pending_dispatch_layout is None or pending_dispatch_ref is None:
                raise RuntimeError("Async CAM receive is missing its dispatch layout")
            local_ffn_output = afd_connector.recv_ffn_output(
                ref_tensor=pending_dispatch_ref,
                ubatch_idx=stage_idx,
            )
            hidden_states = restore_cam_dispatch_output(
                local_ffn_output,
                pending_dispatch_layout,
            )
            pending_ffn_recv = False
            pending_dispatch_layout = None
            pending_dispatch_ref = None

        if not layer.is_moe_layer:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            continue

        (
            hidden_states,
            residual,
            topk_weights,
            topk_ids,
            router_logits,
        ) = layer.compute_attn_output(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )

        # vLLM v0.26 profiles Attention with a local dummy forward and does not
        # start the connector-driven FFN loop for a matching profile request.
        # Launching CAM collectives here would therefore block on unmatched
        # synthetic routing metadata; only the local Attention profile is run.
        if forward_context.in_profile_run:
            continue

        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=forward_context.flash_comm_v1_enabled,
        )
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_connector.send_attn_output(
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        pending_ffn_recv = True
        pending_dispatch_layout = dispatch_payload.layout
        pending_dispatch_ref = dispatch_payload.hidden_states
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )

    if pending_ffn_recv:
        if pending_dispatch_layout is None or pending_dispatch_ref is None:
            raise RuntimeError("Async CAM receive is missing its dispatch layout")
        local_ffn_output = afd_connector.recv_ffn_output(
            ref_tensor=pending_dispatch_ref,
            ubatch_idx=stage_idx,
        )
        hidden_states = restore_cam_dispatch_output(
            local_ffn_output,
            pending_dispatch_layout,
        )
    return hidden_states, residual


def run_async_moe_ubatch_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    async_moe_ubatch_metadata: AsyncMoeUbatchMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the two-stage async MoE ubatch pipeline used by async CAM."""

    forward_context = get_forward_context()
    runtime_sequence_parallel = bool(forward_context.flash_comm_v1_enabled)
    if runtime_sequence_parallel != async_moe_ubatch_metadata.use_sequence_parallel:
        raise RuntimeError(
            "Async CAM stage layout does not match the current FlashComm1 "
            "mode: "
            f"layout_sequence_parallel="
            f"{async_moe_ubatch_metadata.use_sequence_parallel}, "
            f"flash_comm_v1_enabled={runtime_sequence_parallel}",
        )
    afd_connector = afd_metadata.connector
    model_layers = list(islice(model.layers, model.start_layer, model.end_layer))
    first_moe_offset = next(
        (
            layer_offset
            for layer_offset, layer in enumerate(model_layers)
            if layer.is_moe_layer
        ),
        len(model_layers),
    )
    dense_prefix_layers = model_layers[:first_moe_offset]
    moe_layers = model_layers[first_moe_offset:]
    if any(not layer.is_moe_layer for layer in moe_layers):
        raise RuntimeError(
            "async_moe_ubatching requires a dense prefix followed by "
            "contiguous MoE layers",
        )

    for layer in dense_prefix_layers:
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
    if not moe_layers:
        return hidden_states, residual

    stage_inputs = build_async_moe_stage_inputs(
        hidden_states,
        residual,
        positions,
        llama_4_scaling,
        async_moe_ubatch_metadata,
    )
    stage_hidden_states = stage_inputs.hidden_states
    stage_residual = stage_inputs.residuals
    stage_positions = stage_inputs.positions
    stage_llama_4_scaling = stage_inputs.llama_4_scaling
    stage_dispatch_layouts: list[CAMDispatchLayout | None] = [
        None for _ in stage_hidden_states
    ]
    stage_dispatch_refs: list[torch.Tensor | None] = [None for _ in stage_hidden_states]

    def compute_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        stage = async_moe_ubatch_metadata.stages[stage_idx]
        tp_size = get_tensor_model_parallel_world_size()
        if (
            async_moe_ubatch_metadata.use_sequence_parallel
            and int(stage.input_tokens) % tp_size != 0
        ):
            raise RuntimeError(
                "Async CAM sequence-parallel stage is not TP divisible: "
                f"stage={stage_idx}, input_tokens={int(stage.input_tokens)}, "
                f"tp_size={tp_size}",
            )
        expected_local_tokens = int(stage.input_tokens) // tp_size
        if not async_moe_ubatch_metadata.use_sequence_parallel:
            expected_local_tokens = int(stage.input_tokens)
        actual_local_tokens = int(stage_hidden_states[stage_idx].shape[0])
        if actual_local_tokens != expected_local_tokens:
            raise RuntimeError(
                "Async CAM stage input does not match its physical layout: "
                f"stage={stage_idx}, actual_tokens={stage.actual_tokens}, "
                f"input_tokens={int(stage.input_tokens)}, "
                f"expected_local_tokens={expected_local_tokens}, "
                f"actual_local_tokens={actual_local_tokens}, "
                f"sequence_parallel="
                f"{async_moe_ubatch_metadata.use_sequence_parallel}",
            )
        stage_forward_context = copy(forward_context)
        stage_forward_context.attn_metadata = async_moe_ubatch_metadata.attn_metadata[
            stage_idx
        ]
        stage_forward_context.additional_kwargs = dict(
            forward_context.additional_kwargs or {},
        )
        stage_forward_context.ubatch_idx = stage_idx
        stage_forward_context.num_ubatches = len(
            async_moe_ubatch_metadata.stages,
        )
        stage_forward_context.dbo_enabled = False
        if async_moe_ubatch_metadata.use_sequence_parallel:
            # FlashComm gathers the physical TP-local stage, removes its
            # trailing pad before attention, then restores that pad before
            # reduce-scatter.
            stage_forward_context.num_tokens = stage.actual_tokens
            stage_forward_context.pad_size = (
                int(stage.input_tokens) - stage.actual_tokens
            )
        else:
            stage_forward_context.num_tokens = int(stage.input_tokens)
            stage_forward_context.pad_size = 0
        expected_tokens = int(stage_hidden_states[stage_idx].shape[0])
        log_async_moe_stage_attention(
            stage_idx,
            stage,
            expected_tokens,
            stage_forward_context,
        )
        with override_forward_context(stage_forward_context):
            (
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                topk_weights,
                topk_ids,
                router_logits,
            ) = layer.compute_attn_output(
                stage_positions[stage_idx],
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                stage_llama_4_scaling[stage_idx],
            )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError(
                "async_moe_ubatching requires Attention-side topk payloads",
            )
        if int(stage_hidden_states[stage_idx].shape[0]) != expected_tokens:
            raise RuntimeError(
                "async_moe_ubatching stage output token count mismatch: "
                f"expected {expected_tokens}, got "
                f"{int(stage_hidden_states[stage_idx].shape[0])}",
            )
        return topk_weights, topk_ids, router_logits

    def send_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor | None,
    ) -> None:
        dispatch_payload = prepare_cam_dispatch_payload(
            stage_hidden_states[stage_idx],
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=async_moe_ubatch_metadata.use_sequence_parallel,
        )
        stage_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        stage_context = AFDTransferContext(metadata=stage_metadata)
        afd_connector.send_attn_output(
            dispatch_payload.hidden_states,
            stage_context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        stage_dispatch_layouts[stage_idx] = dispatch_payload.layout
        stage_dispatch_refs[stage_idx] = dispatch_payload.hidden_states

    def recv_stage_ffn(stage_idx: int) -> None:
        dispatch_layout = stage_dispatch_layouts[stage_idx]
        dispatch_ref = stage_dispatch_refs[stage_idx]
        if dispatch_layout is None or dispatch_ref is None:
            raise RuntimeError(
                f"Async CAM stage {stage_idx} receive has no pending dispatch",
            )
        local_ffn_output = afd_connector.recv_ffn_output(
            ref_tensor=dispatch_ref,
            ubatch_idx=stage_idx,
        )
        stage_hidden_states[stage_idx] = restore_cam_dispatch_output(
            local_ffn_output,
            dispatch_layout,
        )
        stage_dispatch_layouts[stage_idx] = None
        stage_dispatch_refs[stage_idx] = None

    last_moe_layer_offset = len(moe_layers) - 1
    first_layer = moe_layers[0]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        first_layer,
        0,
    )
    send_stage_attention(
        first_layer,
        0,
        topk_weights,
        topk_ids,
        router_logits,
    )

    for moe_layer_offset in range(last_moe_layer_offset):
        current_layer = moe_layers[moe_layer_offset]
        next_layer = moe_layers[moe_layer_offset + 1]

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            current_layer,
            1,
        )
        recv_stage_ffn(0)
        send_stage_attention(
            current_layer,
            1,
            topk_weights,
            topk_ids,
            router_logits,
        )

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            next_layer,
            0,
        )
        recv_stage_ffn(1)
        send_stage_attention(
            next_layer,
            0,
            topk_weights,
            topk_ids,
            router_logits,
        )

    last_layer = moe_layers[last_moe_layer_offset]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        last_layer,
        1,
    )
    recv_stage_ffn(0)
    send_stage_attention(
        last_layer,
        1,
        topk_weights,
        topk_ids,
        router_logits,
    )
    recv_stage_ffn(1)
    return _restore_async_moe_stage_state(
        stage_hidden_states,
        stage_residual,
        async_moe_ubatch_metadata,
    )


def _restore_async_moe_stage_state(
    stage_hidden_states: list[torch.Tensor],
    stage_residual: list[torch.Tensor | None],
    metadata: AsyncMoeUbatchMetadata,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if all(stage_output is None for stage_output in stage_residual):
        return restore_async_moe_stage_outputs(stage_hidden_states, metadata), None
    if any(stage_output is None for stage_output in stage_residual):
        raise RuntimeError(
            "Async CAM stages returned inconsistent residual layouts",
        )
    residuals = [
        stage_output for stage_output in stage_residual if stage_output is not None
    ]
    hidden_width = int(stage_hidden_states[0].shape[-1])
    residual_width = int(residuals[0].shape[-1])
    combined_states = restore_async_moe_stage_outputs(
        [
            torch.cat((stage_hidden, stage_residual), dim=-1)
            for stage_hidden, stage_residual in zip(
                stage_hidden_states,
                residuals,
                strict=True,
            )
        ],
        metadata,
    )
    hidden_states, residual = combined_states.split(
        (hidden_width, residual_width),
        dim=-1,
    )
    return hidden_states, residual


__all__ = [
    "run_async_moe_ubatch_afd_forward",
    "run_attention_gate_afd_forward",
    "run_model_forward",
]
