# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 async CAM forward orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable
from itertools import islice
from typing import TYPE_CHECKING, cast

import torch
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import (
    get_forward_context,
)
from vllm.sequence import IntermediateTensors

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_dbo import (
    send_async_cam_remote_call,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    CAMDispatchLayout,
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)
from afd_plugin.profiling import runtime_profiling_enabled
from afd_plugin.profiling.model import (
    attention_profile_annotation,
    begin_profile_call,
    begin_profile_step,
    transition,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        AFDDeepseekV2Model,
    )
    from afd_plugin.profiling.tracing import Execution


def run_model_forward(
    model: AFDDeepseekV2Model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    """Run one pinned Model fragment with Async CAM remote MoE calls."""

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
    hidden_states, residual = run_attention_gate_afd_forward(
        model,
        hidden_states,
        residual,
        positions,
        afd_metadata,
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

    afd_connector = cast("CAMAsyncAFDConnector", afd_metadata.connector)
    if runtime_profiling_enabled():
        afd_metadata.profile_pipeline = begin_profile_step(afd_connector)
    stage_idx = afd_metadata.stage_idx
    pending_shared_output: torch.Tensor | None = None
    pending_ffn_recv = False
    pending_dispatch_layout: CAMDispatchLayout | None = None
    pending_receive: Callable[[], torch.Tensor] | None = None
    pending_profile: Execution | None = None
    input_is_sequence_parallel = False

    def receive_pending_ffn() -> torch.Tensor:
        if pending_dispatch_layout is None or pending_receive is None:
            raise RuntimeError("Async CAM receive is missing its dispatch layout")
        routed = pending_receive()
        transition(pending_profile, "COMPUTE")
        output = restore_cam_dispatch_output(routed, pending_dispatch_layout)
        if pending_shared_output is not None:
            output = output + pending_shared_output
        if pending_profile is not None:
            pending_profile.tracer.release_execution(pending_profile)
        return output

    # Async CAM profile forwards are a distributed startup contract: every
    # Attention rank pairs CAM I/O with the FFN daemon to initialize resources.
    for layer_offset, layer in enumerate(
        islice(model.layers, model.start_layer, model.end_layer),
    ):
        if layer_offset > 0 and pending_ffn_recv:
            hidden_states = receive_pending_ffn()
            pending_shared_output = None
            pending_ffn_recv = False
            pending_dispatch_layout = None
            pending_receive = None

        if not layer.is_moe_layer:
            if input_is_sequence_parallel:
                hidden_states, residual = _gather_sequence_parallel_state(
                    hidden_states,
                    residual,
                    int(positions.shape[0]),
                )
                input_is_sequence_parallel = False
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            continue

        execution = begin_profile_call(afd_metadata)
        transition(execution, "COMPUTE")
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
            already_sequence_parallel=input_is_sequence_parallel,
        )
        input_is_sequence_parallel = layer.use_sequence_parallel_moe

        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=layer.use_sequence_parallel_moe,
        )
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        context = AFDTransferContext(
            metadata=metadata,
            profile=execution,
            profile_annotation=attention_profile_annotation(
                afd_metadata, layer.layer_idx
            )
            if execution is not None
            else "",
        )
        pending_profile = execution
        pending_receive = send_async_cam_remote_call(
            afd_connector,
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        if layer.mlp.shared_experts is not None:
            transition(execution, "COMPUTE")
            pending_shared_output = compute_shared_output(layer, hidden_states)
        pending_ffn_recv = True
        pending_dispatch_layout = dispatch_payload.layout
        transition(execution, "WAIT")
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )

    if pending_ffn_recv:
        hidden_states = receive_pending_ffn()
    if input_is_sequence_parallel:
        hidden_states, residual = _gather_sequence_parallel_state(
            hidden_states,
            residual,
            int(positions.shape[0]),
        )
    return hidden_states, residual


def _gather_sequence_parallel_state(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Restore both model streams with one collective at a full-token boundary."""
    if residual is None:
        return tensor_model_parallel_all_gather(hidden_states, 0)[:num_tokens], None
    hidden_width = int(hidden_states.shape[-1])
    combined = tensor_model_parallel_all_gather(
        torch.cat((hidden_states, residual), dim=-1),
        0,
    )[:num_tokens]
    hidden_states, residual = combined.split(hidden_width, dim=-1)
    return hidden_states, residual.contiguous()


def compute_shared_output(
    layer: AFDDeepseekV2DecoderLayer,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Evaluate shared experts on the model-local tokens."""
    if layer.mlp.shared_experts is None:
        return None
    output = layer.mlp.shared_experts(hidden_states)
    # Match native DeepSeek's FP16 overflow-avoidance convention. Routed
    # outputs are unscaled in FP16; the decoder restores the common scale.
    if hidden_states.dtype == torch.float16:
        output = output / layer.routed_scaling_factor
    return output


__all__ = [
    "run_attention_gate_afd_forward",
    "run_model_forward",
]
