# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side MoE runners for Ascend CAM connectors."""

import torch
from torch import nn

from afd_plugin.model_executor.ffn.remote import AFDRemoteMoERunner


class AFDCAMP2PMoERunner(AFDRemoteMoERunner):
    """Send the V2 activation or the V4 activation with its IDs or routes."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        routing: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        if routing is not None:
            return self._exchange(
                hidden_states, routing=routing, shared_experts=shared_experts
            )
        if input_ids is not None:
            return self._exchange(hidden_states, input_ids=input_ids)
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )
