# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Receive ownership for shared and per-stage CAM windows."""

from collections.abc import Callable

import torch
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
)

from afd_plugin.connectors import AFDConnectorBase, AFDTransferContext


def send_async_cam_remote_call(
    connector: AFDConnectorBase,
    hidden_states: torch.Tensor,
    context: AFDTransferContext,
    *,
    independent_windows: bool = False,
    **send_kwargs: torch.Tensor | None,
) -> Callable[[], torch.Tensor]:
    """Send once. Only shared windows require the peer to submit receive before send.

    The returned closure belongs to this call. Layout restoration, TP
    collectives and shared/HC/scaling remain with the resumed owner.
    """
    use_peer_hook = dbo_enabled() and not independent_windows
    if use_peer_hook:
        dbo_maybe_run_recv_hook()
    connector.send_attn_output(hidden_states, context, **send_kwargs)
    owner_stage = context.metadata.stage_idx
    receive_started = False
    output: torch.Tensor | None = None

    def receive() -> torch.Tensor:
        nonlocal receive_started, output
        if not receive_started:
            receive_started = True
            output = connector.recv_ffn_output(
                ref_tensor=hidden_states, ubatch_idx=owner_stage, context=context
            )
        if output is None:
            raise RuntimeError("Async CAM receive failed; restart the A/F worker group")
        return output

    if use_peer_hook:
        dbo_register_recv_hook(receive)
    return receive
