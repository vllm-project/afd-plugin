# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention TP token-layout conversion for Async CAM MoE stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

from afd_plugin.model_executor.npu.async_cam_ubatching import AsyncMoeStage

ASYNC_MOE_UBATCH_METADATA_KEY: Final[str] = "afd_async_moe_ubatch_metadata"
ASYNC_MOE_LAYOUT_LOG_ENV: Final[str] = "AFD_ASYNC_MOE_LAYOUT_LOG"
_TRUE_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

logger = init_logger(__name__)


@dataclass(frozen=True)
class AsyncMoeUbatchMetadata:
    """Execution plan for one two-stage Async CAM model forward.

    Each stage's token slice uses the parent real-token coordinate space, while
    ``stage.input_tokens`` includes its minimum physical padding. Under sequence
    parallelism each physical stage is divided evenly across TP ranks by the
    model-side layout helper. ``parent_input_tokens`` records the original
    padded layout restored after both stages.
    """

    attn_metadata: list[dict[str, AttentionMetadata]]
    stages: tuple[AsyncMoeStage, ...]
    parent_input_tokens: int
    use_sequence_parallel: bool

    def __post_init__(self) -> None:
        num_stages = len(self.stages)
        if not num_stages or len(self.attn_metadata) != num_stages:
            raise ValueError(
                "Async CAM execution-plan fields must describe the same "
                f"non-empty stage count: attention={len(self.attn_metadata)}, "
                f"slices={num_stages}",
            )

        expected_token_start = 0
        for stage in self.stages:
            token_slice = stage.token_slice
            token_start = int(token_slice.start)
            token_stop = int(token_slice.stop)
            actual_tokens = token_stop - token_start
            input_tokens = int(stage.input_tokens)
            if token_start != expected_token_start or actual_tokens <= 0:
                raise ValueError(
                    "Async CAM stage token slices must be contiguous, ordered, "
                    f"and non-empty: token_slice={token_slice}, "
                    f"expected_start={expected_token_start}",
                )
            if not 0 < actual_tokens <= input_tokens:
                raise ValueError(
                    "Async CAM stage actual-token count must fit its physical "
                    f"extent: actual={actual_tokens}, input={input_tokens}",
                )
            expected_token_start = token_stop
        if self.parent_input_tokens < expected_token_start:
            raise ValueError(
                "Async CAM parent input extent must cover every real token: "
                f"parent_input_tokens={self.parent_input_tokens}, "
                f"actual_tokens={expected_token_start}",
            )


def get_async_moe_ubatch_metadata_from_forward_context(
    forward_context: ForwardContext | None = None,
) -> AsyncMoeUbatchMetadata | None:
    """Return the Async CAM execution plan from the current context."""

    if forward_context is None:
        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    metadata: AsyncMoeUbatchMetadata | None = additional_kwargs.get(
        ASYNC_MOE_UBATCH_METADATA_KEY,
    )
    return metadata


@dataclass
class AsyncMoeStageInputs:
    """TP-local tensors for the two global MoE stages."""

    hidden_states: list[torch.Tensor]
    residuals: list[torch.Tensor | None]
    positions: list[torch.Tensor]
    llama_4_scaling: list[torch.Tensor | None]


@dataclass(frozen=True, slots=True)
class CAMDispatchLayout:
    """Mapping between a model tensor and its rank-local CAM payload."""

    parent_tokens: int
    padded_tokens: int
    local_token_slice: slice
    requires_tp_all_gather: bool
    use_sequence_parallel: bool
    tp_rank: int
    tp_size: int

    @property
    def local_tokens(self) -> int:
        return int(self.local_token_slice.stop) - int(self.local_token_slice.start)


@dataclass(frozen=True, slots=True)
class CAMDispatchPayload:
    """Rank-local tensors passed to one CAM dispatch operation."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None
    layout: CAMDispatchLayout


def prepare_cam_dispatch_payload(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor | None,
    *,
    use_sequence_parallel: bool,
) -> CAMDispatchPayload:
    """Convert the model token layout to one rank-local CAM payload.

    FlashComm1 already leaves each Attention TP rank with a disjoint token
    shard, so that layout passes through unchanged. Plain TP keeps a replicated
    token dimension after its tensor-parallel collectives; only the CAM
    boundary shards that dimension so each global token is dispatched once.
    """

    parent_tokens = int(hidden_states.shape[0])
    _require_matching_first_dim(
        topk_weights,
        parent_tokens,
        tensor_name="topk_weights",
    )
    _require_matching_first_dim(
        topk_ids,
        parent_tokens,
        tensor_name="topk_ids",
    )
    if router_logits is not None:
        _require_matching_first_dim(
            router_logits,
            parent_tokens,
            tensor_name="router_logits",
        )

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    if use_sequence_parallel or tp_size <= 1:
        layout = CAMDispatchLayout(
            parent_tokens=parent_tokens,
            padded_tokens=parent_tokens,
            local_token_slice=slice(0, parent_tokens),
            requires_tp_all_gather=False,
            use_sequence_parallel=use_sequence_parallel,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        _log_cam_layout("dispatch", layout)
        return CAMDispatchPayload(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            layout=layout,
        )

    local_tokens = (parent_tokens + tp_size - 1) // tp_size
    padded_tokens = local_tokens * tp_size
    local_start = tp_rank * local_tokens
    local_token_slice = slice(local_start, local_start + local_tokens)
    layout = CAMDispatchLayout(
        parent_tokens=parent_tokens,
        padded_tokens=padded_tokens,
        local_token_slice=local_token_slice,
        requires_tp_all_gather=True,
        use_sequence_parallel=False,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    _log_cam_layout("dispatch", layout)
    return CAMDispatchPayload(
        hidden_states=_pad_first_dim(hidden_states, padded_tokens)[local_token_slice],
        topk_weights=_pad_first_dim(topk_weights, padded_tokens)[local_token_slice],
        topk_ids=_pad_first_dim(topk_ids, padded_tokens)[local_token_slice],
        router_logits=(
            None
            if router_logits is None
            else _pad_first_dim(router_logits, padded_tokens)[local_token_slice]
        ),
        layout=layout,
    )


def restore_cam_dispatch_output(
    local_output: torch.Tensor,
    layout: CAMDispatchLayout,
) -> torch.Tensor:
    """Restore one CAM result to the token layout expected by the model."""

    if int(local_output.shape[0]) != layout.local_tokens:
        raise RuntimeError(
            "CAM output does not match the dispatched rank-local token count: "
            f"expected={layout.local_tokens}, actual={int(local_output.shape[0])}",
        )
    if not layout.requires_tp_all_gather:
        _log_cam_layout("restore", layout)
        return local_output

    global_output = _all_gather_rows(
        local_output,
        expected_rows=layout.padded_tokens,
    )
    _log_cam_layout("restore", layout)
    return global_output[: layout.parent_tokens]


def build_async_moe_stage_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    metadata: AsyncMoeUbatchMetadata,
) -> AsyncMoeStageInputs:
    """Convert the full model layout into per-stage Attention layouts."""

    if not metadata.use_sequence_parallel:
        return _build_replicated_stage_inputs(
            hidden_states,
            residual,
            positions,
            llama_4_scaling,
            metadata,
        )

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    global_input_tokens = metadata.parent_input_tokens
    if tp_size <= 1 or global_input_tokens % tp_size != 0:
        raise ValueError(
            "Invalid sequence-parallel Async CAM layout: "
            f"global_input_tokens={global_input_tokens}, tp_size={tp_size}",
        )
    local_full_tokens = global_input_tokens // tp_size
    if int(hidden_states.shape[0]) != local_full_tokens:
        raise ValueError(
            "Async CAM expected a TP-local full-batch hidden-state shard: "
            f"expected_rows={local_full_tokens}, "
            f"actual_rows={int(hidden_states.shape[0])}",
        )
    if residual is not None and int(residual.shape[0]) != local_full_tokens:
        raise ValueError(
            "Async CAM residual must use the same TP-local layout as hidden "
            f"states: expected_rows={local_full_tokens}, "
            f"actual_rows={int(residual.shape[0])}",
        )

    if residual is None:
        global_hidden_states = _all_gather_rows(
            hidden_states,
            expected_rows=global_input_tokens,
        )
        global_residual = None
    else:
        hidden_width = int(hidden_states.shape[-1])
        residual_width = int(residual.shape[-1])
        combined_states = _all_gather_rows(
            torch.cat((hidden_states, residual), dim=-1),
            expected_rows=global_input_tokens,
        )
        global_hidden_states, global_residual = combined_states.split(
            (hidden_width, residual_width),
            dim=-1,
        )
    positions_token_dim = _require_global_token_dim(
        positions,
        global_input_tokens,
        tensor_name="positions",
    )
    scaling_token_dim = (
        None
        if llama_4_scaling is None
        else _optional_global_token_dim(
            llama_4_scaling,
            global_input_tokens,
            preferred_dim=positions_token_dim,
        )
    )

    stage_hidden_states: list[torch.Tensor] = []
    stage_residuals: list[torch.Tensor | None] = []
    stage_positions: list[torch.Tensor] = []
    stage_scaling: list[torch.Tensor | None] = []
    for stage_slice in metadata.stages:
        stage_input_tokens = int(stage_slice.input_tokens)
        if stage_input_tokens % tp_size != 0:
            raise ValueError(
                "Async CAM stage extent must be TP divisible: "
                f"token_slice={stage_slice.token_slice}, tp_size={tp_size}",
            )
        local_stage_tokens = stage_input_tokens // tp_size
        stage_hidden = _slice_and_pad_token_dim(
            global_hidden_states,
            0,
            stage_slice.token_slice,
            stage_input_tokens,
        )
        stage_residual = (
            None
            if global_residual is None
            else _slice_and_pad_token_dim(
                global_residual,
                0,
                stage_slice.token_slice,
                stage_input_tokens,
            )
        )
        stage_position_tensor = _slice_and_pad_token_dim(
            positions,
            positions_token_dim,
            stage_slice.token_slice,
            stage_input_tokens,
        )
        stage_scaling_tensor = _slice_and_pad_optional_token_tensor(
            llama_4_scaling,
            scaling_token_dim,
            stage_slice.token_slice,
            stage_input_tokens,
        )
        local_stage_slice = slice(
            tp_rank * local_stage_tokens,
            (tp_rank + 1) * local_stage_tokens,
        )

        stage_hidden_states.append(stage_hidden[local_stage_slice])
        stage_residuals.append(
            None if stage_residual is None else stage_residual[local_stage_slice],
        )
        stage_positions.append(
            _slice_token_dim(
                stage_position_tensor,
                positions_token_dim,
                local_stage_slice,
            ),
        )
        stage_scaling.append(
            (
                None
                if stage_scaling_tensor is None or scaling_token_dim is None
                else _slice_token_dim(
                    stage_scaling_tensor,
                    scaling_token_dim,
                    local_stage_slice,
                )
            ),
        )
    return AsyncMoeStageInputs(
        hidden_states=stage_hidden_states,
        residuals=stage_residuals,
        positions=stage_positions,
        llama_4_scaling=stage_scaling,
    )


def restore_async_moe_stage_outputs(
    stage_outputs: list[torch.Tensor],
    metadata: AsyncMoeUbatchMetadata,
) -> torch.Tensor:
    """Restore stage-local outputs to the parent model's TP-local layout."""

    if len(stage_outputs) != len(metadata.stages):
        raise ValueError(
            "Async CAM stage output count does not match its execution plan: "
            f"outputs={len(stage_outputs)}, "
            f"stages={len(metadata.stages)}",
        )
    if not metadata.use_sequence_parallel:
        return _pad_first_dim(
            torch.cat(
                _trim_real_stage_outputs(
                    stage_outputs,
                    metadata,
                ),
                dim=0,
            ),
            metadata.parent_input_tokens,
        )

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    global_stage_outputs = [
        _all_gather_rows(
            stage_output,
            expected_rows=int(stage_slice.input_tokens),
        )
        for stage_output, stage_slice in zip(
            stage_outputs,
            metadata.stages,
            strict=True,
        )
    ]
    global_output = _pad_first_dim(
        torch.cat(
            _trim_real_stage_outputs(
                global_stage_outputs,
                metadata,
            ),
            dim=0,
        ),
        metadata.parent_input_tokens,
    )
    if int(global_output.shape[0]) % tp_size != 0:
        raise ValueError(
            "Restored Async CAM output is not TP divisible: "
            f"rows={int(global_output.shape[0])}, tp_size={tp_size}",
        )
    local_full_tokens = int(global_output.shape[0]) // tp_size
    local_start = tp_rank * local_full_tokens
    return global_output[local_start : local_start + local_full_tokens]


def _build_replicated_stage_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    metadata: AsyncMoeUbatchMetadata,
) -> AsyncMoeStageInputs:
    global_input_tokens = metadata.parent_input_tokens
    if int(hidden_states.shape[0]) != global_input_tokens:
        raise ValueError(
            "Non-SP Async CAM hidden states must contain the full stageable "
            f"batch: expected_rows={global_input_tokens}, "
            f"actual_rows={int(hidden_states.shape[0])}",
        )
    if residual is not None and int(residual.shape[0]) != global_input_tokens:
        raise ValueError(
            "Non-SP Async CAM residual must match hidden states: "
            f"expected_rows={global_input_tokens}, "
            f"actual_rows={int(residual.shape[0])}",
        )
    positions_token_dim = _require_global_token_dim(
        positions,
        global_input_tokens,
        tensor_name="positions",
    )
    scaling_token_dim = (
        None
        if llama_4_scaling is None
        else _optional_global_token_dim(
            llama_4_scaling,
            global_input_tokens,
            preferred_dim=positions_token_dim,
        )
    )
    return AsyncMoeStageInputs(
        hidden_states=[
            _slice_and_pad_token_dim(
                hidden_states,
                0,
                stage.token_slice,
                int(stage.input_tokens),
            )
            for stage in metadata.stages
        ],
        residuals=[
            (
                None
                if residual is None
                else _slice_and_pad_token_dim(
                    residual,
                    0,
                    stage.token_slice,
                    int(stage.input_tokens),
                )
            )
            for stage in metadata.stages
        ],
        positions=[
            _slice_and_pad_token_dim(
                positions,
                positions_token_dim,
                stage.token_slice,
                int(stage.input_tokens),
            )
            for stage in metadata.stages
        ],
        llama_4_scaling=[
            _slice_and_pad_optional_token_tensor(
                llama_4_scaling,
                scaling_token_dim,
                stage.token_slice,
                int(stage.input_tokens),
            )
            for stage in metadata.stages
        ],
    )


def _all_gather_rows(
    tensor: torch.Tensor,
    *,
    expected_rows: int,
) -> torch.Tensor:
    gathered = tensor_model_parallel_all_gather(tensor.contiguous(), 0)
    if int(gathered.shape[0]) != expected_rows:
        raise RuntimeError(
            "Async CAM TP all-gather returned an unexpected row count: "
            f"expected={expected_rows}, actual={int(gathered.shape[0])}",
        )
    return gathered


def _require_matching_first_dim(
    tensor: torch.Tensor,
    expected_tokens: int,
    *,
    tensor_name: str,
) -> None:
    if int(tensor.shape[0]) != expected_tokens:
        raise ValueError(
            f"Async CAM {tensor_name} must match hidden states on axis 0: "
            f"expected_rows={expected_tokens}, actual_rows={int(tensor.shape[0])}",
        )


def _log_cam_layout(
    phase: str,
    layout: CAMDispatchLayout,
) -> None:
    if os.environ.get(ASYNC_MOE_LAYOUT_LOG_ENV, "").lower() not in _TRUE_ENV_VALUES:
        return
    logger.warning(
        "AFD Async CAM layout; phase=%s sequence_parallel=%s "
        "tp_rank=%s tp_size=%s "
        "parent_tokens=%s padded_tokens=%s local_slice=(%s,%s) "
        "local_tokens=%s tp_all_gather=%s",
        phase,
        layout.use_sequence_parallel,
        layout.tp_rank,
        layout.tp_size,
        layout.parent_tokens,
        layout.padded_tokens,
        layout.local_token_slice.start,
        layout.local_token_slice.stop,
        layout.local_tokens,
        layout.requires_tp_all_gather,
    )


def log_async_moe_stage_attention(
    stage_idx: int,
    stage: AsyncMoeStage,
    local_tokens: int,
    forward_context: ForwardContext,
) -> None:
    """Log the real, physical, and rank-local stage token extents."""

    if os.environ.get(ASYNC_MOE_LAYOUT_LOG_ENV, "").lower() not in _TRUE_ENV_VALUES:
        return
    logger.warning(
        "AFD Async CAM stage attention; stage=%s actual_tokens=%s "
        "input_tokens=%s local_tokens=%s sequence_parallel=%s "
        "context_num_tokens=%s context_pad_size=%s",
        stage_idx,
        stage.actual_tokens,
        int(stage.input_tokens),
        local_tokens,
        bool(forward_context.flash_comm_v1_enabled),
        int(forward_context.num_tokens),
        int(forward_context.pad_size),
    )


def _trim_real_stage_outputs(
    stage_outputs: list[torch.Tensor],
    metadata: AsyncMoeUbatchMetadata,
) -> list[torch.Tensor]:
    real_stage_outputs = []
    for stage_output, stage in zip(
        stage_outputs,
        metadata.stages,
        strict=True,
    ):
        actual_rows = int(stage_output.shape[0])
        expected_rows = int(stage.input_tokens)
        if actual_rows != expected_rows:
            raise RuntimeError(
                "Async CAM stage output does not match its physical plan: "
                f"expected_rows={expected_rows}, actual_rows={actual_rows}",
            )
        real_stage_outputs.append(stage_output[: stage.actual_tokens])
    return real_stage_outputs


def _require_global_token_dim(
    tensor: torch.Tensor,
    num_tokens: int,
    *,
    tensor_name: str,
) -> int:
    token_dim = _optional_global_token_dim(tensor, num_tokens)
    if token_dim is None:
        raise ValueError(
            f"Async CAM {tensor_name} must expose {num_tokens} tokens on "
            f"axis 0 or 1, got shape={tuple(tensor.shape)}",
        )
    return token_dim


def _optional_global_token_dim(
    tensor: torch.Tensor,
    num_tokens: int,
    *,
    preferred_dim: int | None = None,
) -> int | None:
    if (
        preferred_dim is not None
        and tensor.dim() > preferred_dim
        and int(tensor.shape[preferred_dim]) == num_tokens
    ):
        return preferred_dim
    # vLLM uses [N] for ordinary positions and [axes, N] for multi-axis
    # positions. Prefer axis 1 for a higher-dimensional tensor so an
    # accidental square shape does not reinterpret the position axes as
    # tokens.
    if tensor.dim() > 1 and int(tensor.shape[1]) == num_tokens:
        return 1
    if tensor.dim() > 0 and int(tensor.shape[0]) == num_tokens:
        return 0
    return None


def _slice_token_dim(
    tensor: torch.Tensor,
    token_dim: int,
    token_slice: slice,
) -> torch.Tensor:
    if token_dim == 0:
        return tensor[token_slice]
    return tensor[:, token_slice]


def _pad_first_dim(
    tensor: torch.Tensor,
    output_tokens: int,
) -> torch.Tensor:
    current_tokens = int(tensor.shape[0])
    if current_tokens > output_tokens:
        raise ValueError(
            "Async CAM stage tensor exceeds its physical input extent: "
            f"current_tokens={current_tokens}, output_tokens={output_tokens}",
        )
    if current_tokens == output_tokens:
        return tensor
    padding_shape = (output_tokens - current_tokens, *tensor.shape[1:])
    return torch.cat(
        (tensor, tensor.new_zeros(padding_shape)),
        dim=0,
    )


def _slice_and_pad_token_dim(
    tensor: torch.Tensor,
    token_dim: int,
    token_slice: slice,
    output_tokens: int,
) -> torch.Tensor:
    sliced_tensor = _slice_token_dim(tensor, token_dim, token_slice)
    current_tokens = int(sliced_tensor.shape[token_dim])
    if current_tokens > output_tokens:
        raise ValueError(
            "Async CAM token tensor exceeds its physical input extent: "
            f"current_tokens={current_tokens}, output_tokens={output_tokens}",
        )
    if current_tokens == output_tokens:
        return sliced_tensor
    padding_shape = list(sliced_tensor.shape)
    padding_shape[token_dim] = output_tokens - current_tokens
    return torch.cat(
        (sliced_tensor, sliced_tensor.new_zeros(padding_shape)),
        dim=token_dim,
    )


def _slice_and_pad_optional_token_tensor(
    tensor: torch.Tensor | None,
    token_dim: int | None,
    token_slice: slice,
    output_tokens: int,
) -> torch.Tensor | None:
    if tensor is None or token_dim is None:
        return tensor
    return _slice_and_pad_token_dim(
        tensor,
        token_dim,
        token_slice,
        output_tokens,
    )


__all__ = [
    "ASYNC_MOE_LAYOUT_LOG_ENV",
    "ASYNC_MOE_UBATCH_METADATA_KEY",
    "AsyncMoeStageInputs",
    "AsyncMoeUbatchMetadata",
    "CAMDispatchLayout",
    "CAMDispatchPayload",
    "build_async_moe_stage_inputs",
    "get_async_moe_ubatch_metadata_from_forward_context",
    "log_async_moe_stage_attention",
    "prepare_cam_dispatch_payload",
    "restore_cam_dispatch_output",
    "restore_async_moe_stage_outputs",
]
