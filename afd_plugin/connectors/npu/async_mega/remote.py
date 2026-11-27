# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Send attention output and build the FFN receive closure for the Mega connector.

Both ubatches share one window and one sequence ring. The connector runs the
native DBO receive hook inside send_attn_output before it writes the next input,
so this helper only sends once and returns the closure that takes this ubatch's
result back.
"""

from collections.abc import Callable

import torch

from afd_plugin.connectors import AFDConnectorBase, AFDTransferContext


def send_and_build_receive(
    connector: AFDConnectorBase,
    hidden_states: torch.Tensor,
    context: AFDTransferContext,
    **send_kwargs: torch.Tensor | None,
) -> Callable[[], torch.Tensor]:
    """Send once and return the closure that receives the FFN output.

    The returned closure belongs to this call. Layout restoration, TP
    collectives and shared/HC/scaling remain with the resumed owner.
    """
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
            raise RuntimeError(
                "Async Mega receive failed; restart the A/F worker group"
            )
        return output

    return receive
