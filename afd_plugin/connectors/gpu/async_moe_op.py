# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dynamo-opaque ops for the async GPU connector's MoE dispatch and receive.

Enabling CUDA graphs on the Attention side puts vLLM's AOT compilation in
front of the model, and tracing used to run straight through the MoE proxy
into the connector -- raw NVSHMEM pointer views, host-side caches, ctypes
driver calls -- and abort. The data path itself is capture-safe (the flag
protocol keeps replays correct; see the ``async_gpu`` module docs); the
obstacle was purely Python visibility.

Two ops carve the round trip at the points the model already defers across:
``afd_async_dispatch`` sends one layer's tokens and stashes the payload the
receive will need; ``afd_async_recv`` consumes the stash, waits for the reply
on the stream, and restores the model layout. Dynamo splits at both, and the
Python between them -- the deferred-receive bookkeeping -- is plain control
flow it can trace. During cooperative capture the two ubatch threads run the
impls once each, and the kernel order their alternation produced is what the
graph replays.
"""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_DISPATCH_OP_NAME = "afd_async_dispatch"
_RECV_OP_NAME = "afd_async_recv"
_REGISTERED = False


def _resolve_stage_idx(afd_metadata) -> int:
    # vLLM tracks the ubatch by thread, not on the forward context; under
    # cooperative capture each thread resolves its own stage here, and the
    # kernel order that produces is what the captured graph replays.
    from afd_plugin.v1.worker.dbo import current_dbo_ubatch_id

    dbo_ubatch_id = current_dbo_ubatch_id()
    return afd_metadata.stage_idx if dbo_ubatch_id is None else int(dbo_ubatch_id)


def _dispatch_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    # Deferred imports: the connector sits below the models that trace this
    # op, and the forward-context helpers pull the model package in.
    from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
    from afd_plugin.model_executor.models import (
        get_afd_metadata_from_forward_context,
    )
    from afd_plugin.model_executor.models.npu.async_cam_layout import (
        prepare_cam_dispatch_payload,
    )

    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("afd_async_dispatch requires AFD forward metadata")
    stage_idx = _resolve_stage_idx(afd_metadata)
    afd_metadata.stage_idx = stage_idx
    connector = afd_metadata.connector
    # FlashComm1 token sharding is Ascend-only: the CUDA forward context
    # carries no such field, so the dispatch always sees a replicated token
    # dimension and the payload passes through unchanged.
    payload = prepare_cam_dispatch_payload(
        hidden_states,
        topk_weights,
        topk_ids,
        None,
        use_sequence_parallel=False,
    )
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=layer_idx,
        stage_idx=stage_idx,
        seq_len=int(payload.hidden_states.shape[0]),
    )
    connector.send_attn_output(
        payload.hidden_states,
        AFDTransferContext(metadata=metadata),
        topk_weights=payload.topk_weights,
        topk_ids=payload.topk_ids,
    )
    connector.pending_cam_dispatches[stage_idx] = payload  # type: ignore[attr-defined]
    return payload.hidden_states


def _dispatch_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    return hidden_states


def _recv_impl(hidden_states: torch.Tensor) -> torch.Tensor:
    from afd_plugin.model_executor.models import (
        get_afd_metadata_from_forward_context,
    )
    from afd_plugin.model_executor.models.npu.async_cam_layout import (
        restore_cam_dispatch_output,
    )

    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("afd_async_recv requires AFD forward metadata")
    stage_idx = _resolve_stage_idx(afd_metadata)
    connector = afd_metadata.connector
    payload = connector.pending_cam_dispatches.pop(  # type: ignore[attr-defined]
        stage_idx, None
    )
    if payload is None:
        raise RuntimeError(
            f"AFD async receive on stage {stage_idx} has no pending dispatch",
        )
    local_ffn_output = connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=stage_idx,
    )
    return restore_cam_dispatch_output(local_ffn_output, payload.layout)


def _recv_fake(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _register_once(op_name: str, op_func, fake_impl) -> None:
    try:
        direct_register_custom_op(
            op_name=op_name,
            op_func=op_func,
            mutates_args=[],
            fake_impl=fake_impl,
            # The body is plain Python that hands the tensors to the connector,
            # which does its own device work, so there is nothing
            # backend-specific to specialise. Registering under the platform
            # dispatch key instead would leave the op undefined for CPU, where
            # the NPU forward's unit tests exercise this path.
            dispatch_key="CompositeExplicitAutograd",
        )
    except RuntimeError as exc:
        # A prior import of this module can leave the op in torch's
        # process-global registry while this module's flag is back to False;
        # reuse the registered op instead of redefining it.
        if "already" not in str(exc).lower():
            raise


def register_async_moe_ops() -> tuple:
    """Register both ops once and return their callable handles."""
    global _REGISTERED
    if not _REGISTERED:
        _register_once(_DISPATCH_OP_NAME, _dispatch_impl, _dispatch_fake)
        _register_once(_RECV_OP_NAME, _recv_impl, _recv_fake)
        _REGISTERED = True
    return (
        getattr(torch.ops.vllm, _DISPATCH_OP_NAME),
        getattr(torch.ops.vllm, _RECV_OP_NAME),
    )


register_async_moe_ops()

__all__ = ["register_async_moe_ops"]
