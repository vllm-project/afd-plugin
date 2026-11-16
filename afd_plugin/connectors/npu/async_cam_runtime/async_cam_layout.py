# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention TP token-layout conversion at the Async CAM boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

ASYNC_MOE_LAYOUT_LOG_ENV: Final[str] = "AFD_ASYNC_MOE_LAYOUT_LOG"
_TRUE_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})

logger = init_logger(__name__)


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

    Model-owned sequence parallelism leaves each Attention TP rank with a disjoint token
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


__all__ = [
    "ASYNC_MOE_LAYOUT_LOG_ENV",
    "CAMDispatchLayout",
    "CAMDispatchPayload",
    "prepare_cam_dispatch_payload",
    "restore_cam_dispatch_output",
]
