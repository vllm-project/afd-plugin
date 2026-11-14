# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU upstream expert seam for already-dispatched model-row tests."""

import torch


class LocalExpertMap:
    def get_local_expert_ids(self) -> tuple[int, ...]:
        return (3, 7)


class LocalExpertRows:
    def __init__(self):
        self.afd_global_expert_ids = torch.tensor([3, 7], dtype=torch.int32)

    def forward_modular(self, x, topk_weights, topk_ids):
        # Independent stand-in for the native per-row expert kernel.
        return x * topk_ids.to(x.dtype) * topk_weights.to(x.dtype)


class SingleExpertRouter:
    def select_experts(self, hidden_states, router_logits):
        return (
            torch.ones(hidden_states.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden_states.shape[0], 1, dtype=torch.int32),
        )
