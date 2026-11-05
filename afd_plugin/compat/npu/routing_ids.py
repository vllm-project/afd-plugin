# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Opaque boundary for the profile-dependent expert-ID balancing."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _afd_in_profile_run() -> bool:
    """Read the flag at execution time; never snapshot it at import time."""
    # Lazy: vllm_ascend is unavailable on hosts that only collect unit tests.
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    return bool(_EXTRA_CTX.in_profile_run)


def _force_balanced_topk_ids(
    topk_ids: torch.Tensor, *, num_logical_experts: int
) -> torch.Tensor:
    """Overwrite ``topk_ids`` with a balanced logical-ID layout, in place."""
    balanced = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=torch.int64)
    topk_ids.copy_(
        balanced.reshape(topk_ids.shape)
        .remainder(num_logical_experts)
        .to(topk_ids.dtype)
    )
    return topk_ids


def afd_balance_routing_ids_(
    topk_ids: torch.Tensor,
    num_experts: int,
    force_balance: bool,
) -> None:
    """Balance ``topk_ids`` in place, keeping the profile read inside the op."""
    if force_balance or _afd_in_profile_run():
        _force_balanced_topk_ids(topk_ids, num_logical_experts=num_experts)


def _afd_balance_routing_ids_fake(
    topk_ids: torch.Tensor,
    num_experts: int,
    force_balance: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="afd_balance_routing_ids_",
    op_func=afd_balance_routing_ids_,
    mutates_args=["topk_ids"],
    fake_impl=_afd_balance_routing_ids_fake,
)
