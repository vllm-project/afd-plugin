# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opaque CUDA async dispatch/receive boundaries for compiled model forwards."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _dispatch(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_idx: int,
) -> None:
    # Deferred imports avoid a connector -> models -> connector import cycle.
    from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
    from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context

    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("AFD async dispatch requires AFD forward metadata")
    afd_metadata.connector.send_attn_output(
        hidden_states,
        AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx,
                stage_idx=afd_metadata.stage_idx,
                seq_len=hidden_states.shape[0],
            )
        ),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _dispatch_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_idx: int,
) -> None:
    return None


def _receive(hidden_states: torch.Tensor) -> torch.Tensor:
    from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context

    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("AFD async receive requires AFD forward metadata")
    return afd_metadata.connector.recv_ffn_output(
        ref_tensor=hidden_states, ubatch_idx=afd_metadata.stage_idx
    )


def _receive_fake(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


# Dispatch changes remote memory. Mark its input as a conservative ordering
# dependency so compilation cannot discard/reorder a void transport operation.
direct_register_custom_op(
    op_name="afd_async_gpu_dispatch",
    op_func=_dispatch,
    fake_impl=_dispatch_fake,
    mutates_args=["hidden_states"],
    dispatch_key="CompositeExplicitAutograd",
)
direct_register_custom_op(
    op_name="afd_async_gpu_receive",
    op_func=_receive,
    fake_impl=_receive_fake,
    mutates_args=[],
    dispatch_key="CompositeExplicitAutograd",
)
