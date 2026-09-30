# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 async CAM forward orchestration helpers."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from itertools import islice
from typing import TYPE_CHECKING

import torch
import vllm.forward_context as forward_context_module
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    get_forward_context,
)
from vllm.sequence import IntermediateTensors

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    AsyncMoeUbatchMetadata,
    build_async_moe_stage_inputs,
    get_async_moe_ubatch_metadata_from_forward_context,
    log_async_moe_stage_attention,
    restore_async_moe_stage_outputs,
)
from afd_plugin.model_executor.npu.async_cam_execution import (
    CAM_ASYNC_EXECUTION_KEY,
    CAM_ASYNC_SCHEDULER_KEY,
    CAMAsyncExecutionContext,
    CAMAsyncRuntimeContext,
    require_cam_async_execution_context,
)

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.deepseek_v2 import (
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

    execution = require_cam_async_execution_context()
    if (
        execution.num_stages != 1
        or execution.stage_idx != 0
        or execution.stage_idx != afd_metadata.stage_idx
    ):
        raise RuntimeError("CAMAsync single-stage context does not match metadata")

    forward_context = get_forward_context()
    scheduler = forward_context.additional_kwargs[CAM_ASYNC_SCHEDULER_KEY]
    try:
        with scheduler.single_stage(
            use_sequence_parallel=execution.use_sequence_parallel
        ) as current_execution:
            forward_context.additional_kwargs[CAM_ASYNC_EXECUTION_KEY] = (
                current_execution
            )
            # Profile/startup still pair every MoE dispatch/combine with the FFN daemon.
            for layer in islice(model.layers, model.start_layer, model.end_layer):
                if not layer.is_moe_layer:
                    hidden_states, residual = layer(
                        positions, hidden_states, residual, llama_4_scaling
                    )
                    continue
                hidden_states, residual = layer.compute_attn_output(
                    positions, hidden_states, residual, llama_4_scaling
                )
                hidden_states = layer.mlp(hidden_states)
                current_execution.layer_done(layer.layer_idx)
    finally:
        forward_context.additional_kwargs[CAM_ASYNC_EXECUTION_KEY] = execution
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
    scheduler = forward_context.additional_kwargs[CAM_ASYNC_SCHEDULER_KEY]
    stage_contexts = []
    stages = async_moe_ubatch_metadata.stages
    tp_size = get_tensor_model_parallel_world_size()
    for stage_idx, stage in enumerate(stages):
        input_tokens = int(stage.input_tokens)
        if runtime_sequence_parallel and input_tokens % tp_size != 0:
            raise RuntimeError("Async CAM sequence-parallel stage is not TP divisible")
        expected_tokens = input_tokens
        if runtime_sequence_parallel:
            expected_tokens //= tp_size
        if int(stage_inputs.hidden_states[stage_idx].shape[0]) != expected_tokens:
            raise RuntimeError(
                "Async CAM stage input does not match its physical layout"
            )
        stage_context = copy(forward_context)
        stage_context.attn_metadata = async_moe_ubatch_metadata.attn_metadata[stage_idx]
        stage_context.additional_kwargs = dict(forward_context.additional_kwargs)
        stage_context.ubatch_idx = stage_idx
        stage_context.num_ubatches = len(stages)
        stage_context.dbo_enabled = False
        stage_context.num_tokens = (
            stage.actual_tokens if runtime_sequence_parallel else input_tokens
        )
        stage_context.pad_size = input_tokens - stage_context.num_tokens
        stage_context.additional_kwargs["afd_metadata"] = replace(
            afd_metadata,
            stage_idx=stage_idx,
            num_stages=len(stages),
            tokens_start_loc=[item.token_slice.start for item in stages],
            requests_start_loc=[item.request_slice.start for item in stages],
            tokens_lens=[int(item.input_tokens) for item in stages],
            tokens_unpadded_lens=[item.actual_tokens for item in stages],
        )
        stage_context.additional_kwargs.pop(CAM_ASYNC_EXECUTION_KEY, None)
        stage_contexts.append(stage_context)

    runtime = CAMAsyncRuntimeContext(stage_contexts, hidden_states.device)

    def run_stage(
        execution: CAMAsyncExecutionContext,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stage_idx = execution.stage_idx
        stage_hidden = stage_inputs.hidden_states[stage_idx]
        stage_residual = stage_inputs.residuals[stage_idx]
        expected_tokens = int(stage_hidden.shape[0])
        for layer in moe_layers:
            log_async_moe_stage_attention(
                stage_idx,
                async_moe_ubatch_metadata.stages[stage_idx],
                expected_tokens,
                stage_contexts[stage_idx],
            )
            stage_hidden, stage_residual = layer.compute_attn_output(
                stage_inputs.positions[stage_idx],
                stage_hidden,
                stage_residual,
                stage_inputs.llama_4_scaling[stage_idx],
            )
            stage_hidden = layer.mlp(stage_hidden)
            if int(stage_hidden.shape[0]) != expected_tokens:
                raise RuntimeError(
                    "async_moe_ubatching stage output token count mismatch"
                )
            execution.layer_done(layer.layer_idx)
        return stage_hidden, stage_residual

    try:
        outputs = scheduler.run(
            run_stage,
            [layer.layer_idx for layer in moe_layers],
            use_sequence_parallel=runtime_sequence_parallel,
            activate=runtime.activate,
            thread_context=runtime.thread_context,
        )
    finally:
        # Workers never nest process-global context managers. Restore only after
        # neither call stack can access the model or change the active context.
        if scheduler.quiescent:
            forward_context_module._forward_context = forward_context
    return _restore_async_moe_stage_state(
        [output[0] for output in outputs],
        [output[1] for output in outputs],
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
