# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous remote FFN exchange shared by model proxies."""

import torch
from vllm.forward_context import get_forward_context

from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


def send_and_receive_remote_ffn(
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    **send_kwargs: torch.Tensor,
) -> torch.Tensor:
    """Send one layer input, yield to the peer ubatch, then receive its output."""
    forward_context = get_forward_context()
    afd_metadata = get_afd_metadata_from_forward_context(forward_context)
    if afd_metadata is None:
        raise RuntimeError("Remote FFN exchange requires AFD forward metadata")
    stage_idx = int(
        getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
    )
    afd_metadata.stage_idx = stage_idx
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=layer_idx,
        stage_idx=stage_idx,
        seq_len=int(hidden_states.shape[0]),
    )
    context = AFDTransferContext(metadata=metadata)
    afd_metadata.connector.send_attn_output(
        hidden_states,
        context,
        **send_kwargs,
    )
    hidden_states = maybe_apply_dbo_yield(
        hidden_states,
        role="attention",
    )
    return afd_metadata.connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=stage_idx,
    )
