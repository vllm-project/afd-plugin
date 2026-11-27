# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Mega Attention-side dispatch payload building.

Mega selects and shards experts on the Attention side on top of its own operator
protocol. This module keeps the Mega-specific TP token slicing and the UMDK
padding route-id rule out of the shared CAM layout path. It reuses only that
module's public payload/layout types; padding lives here so the Mega package
does not reach into CAM's private helpers.
"""

from __future__ import annotations

import torch

from afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout import (
    CAMDispatchLayout,
    CAMDispatchPayload,
)


def plain_tp_layout(
    parent_tokens: int, tp_rank: int, tp_size: int
) -> CAMDispatchLayout:
    """Rank-local slice of a replicated plain-TP token dimension."""
    local_tokens = (parent_tokens + tp_size - 1) // tp_size
    padded_tokens = local_tokens * tp_size
    local_start = tp_rank * local_tokens
    return CAMDispatchLayout(
        parent_tokens=parent_tokens,
        padded_tokens=padded_tokens,
        local_token_slice=slice(local_start, local_start + local_tokens),
        requires_tp_all_gather=True,
        use_sequence_parallel=False,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


def build_mega_dispatch_payload(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layout: CAMDispatchLayout,
) -> CAMDispatchPayload:
    """Pad already-routed local rows; routing belongs to the model adapter."""
    tokens = hidden_states.shape[0]
    if topk_weights.shape[0] != tokens or topk_ids.shape != topk_weights.shape:
        raise ValueError("Mega routes must match the local token rows and top-k")
    padded_tokens = layout.local_tokens
    padded_ids = _pad_dispatch_rows(topk_ids, padded_tokens)
    if padded_tokens > tokens:
        # Each padding route must address a distinct expert slot.
        padded_ids[tokens:] = torch.arange(
            topk_ids.shape[1], device=topk_ids.device, dtype=topk_ids.dtype
        )
    return CAMDispatchPayload(
        hidden_states=_pad_dispatch_rows(hidden_states, padded_tokens),
        topk_weights=_pad_dispatch_rows(topk_weights, padded_tokens),
        topk_ids=padded_ids,
        router_logits=None,
        layout=layout,
    )


def _pad_dispatch_rows(tensor: torch.Tensor, padded_tokens: int) -> torch.Tensor:
    """Pad the token dimension to the TP slice rounded up to ``tp_size``."""
    current_tokens = int(tensor.shape[0])
    if current_tokens > padded_tokens:
        raise ValueError(
            "Mega dispatch tensor exceeds its padded token extent: "
            f"current_tokens={current_tokens}, padded_tokens={padded_tokens}",
        )
    if current_tokens == padded_tokens:
        return tensor
    padding_shape = (padded_tokens - current_tokens, *tensor.shape[1:])
    return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=0)
