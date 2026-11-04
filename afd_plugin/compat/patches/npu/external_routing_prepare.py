# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Let the ALLGATHER prepare path consume routing computed on Attention.

The FFN side normally computes its own routing right after ``prepare`` has padded
and gathered the activation, so routing and activation line up by construction.
AFD moves the gate to the Attention side, which means the routing arrives over the
transport instead. It then has to be padded and gathered by exactly the rule that
``_prepare_with_dp_group`` already applies to the activation, or a token row is
weighted by another row's experts.

The heavy lifting lives in :class:`AFDExternalRoutingInputs`; this patch only wires
it into the upstream sequence at the two points where the activation is padded and
gathered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.distributed.parallel_state import get_pcp_group
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAllGather,
)
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    # Type-only: the patch calls methods on whatever object the caller attached,
    # and importing the connectors package pulls vLLM in through its __init__.
    from afd_plugin.connectors.npu.external_routing import AFDExternalRoutingInputs

# Declared as a class default so the patched function reads a plain attribute
# instead of guarding with getattr, and so upstream code paths keep working when
# the plugin never assigns it.
PrepareAndFinalizeWithAllGather.afd_external_routing = None


# Patch reason: upstream always pads and gathers ``router_logits`` and reads the
# routing back on the FFN side, because upstream computes the routing there.
# Patch functionality: when Attention supplied the routing, pad and gather it by
# the same rule as the activation, and tolerate a missing ``router_logits``.
# Without the attribute the function behaves exactly like upstream.
# Signature: matches upstream; no added parameters.
def prepare_with_dp_group(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    replace_allreduce: bool = False,
    quant_type=QuantType.NONE,
) -> MoEPrepareOutput:
    """
    Preparation steps:
      1. Fetch max token count across DP group from forward context.
      2. Pad local tensors to that size.
      3. All-gather across DP group to form global input tensor.

    Returns:
        MoEPrepareOutput with global tensors.
    """
    # ### PATCH START: routing may be precomputed on the Attention side
    external = self.afd_external_routing
    # The routing only describes this rank's rows, so it has to start from the
    # same row count as the activation. Mismatched counts pad to different
    # totals and the gathered ids silently describe the wrong tokens.
    if external is not None and external.token_count != hidden_states.shape[0]:
        raise ValueError(
            "external routing covers a different token count than the activation: "
            f"{external.token_count} routing rows for "
            f"{hidden_states.shape[0]} activation rows",
        )
    # ### PATCH END: routing may be precomputed on the Attention side
    if self.moe_config.dp_size > 1:
        max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

        self.num_tokens = hidden_states.shape[0]
        pad_size = max_tokens_across_dp - self.num_tokens
        if pad_size > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
            # ### PATCH START: no router logits when Attention owns the gate
            if router_logits is not None:
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
            # ### PATCH END: no router logits when Attention owns the gate
            # ### PATCH START: pad the routing by the same row count
            if external is not None:
                external = external.padded(pad_size)
            # ### PATCH END: pad the routing by the same row count

        # All-gather across DP group
        hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
        # ### PATCH START: gather the routing with the activation's collective
        if router_logits is not None:
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)
        if external is not None:
            external = external.gathered(
                lambda tensor: self.moe_config.dp_group.all_gather(tensor, 0),
            )
        # ### PATCH END: gather the routing with the activation's collective

    if self.moe_config.pcp_size > 1:
        max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

        self.num_tokens_pcp = hidden_states.shape[0]
        pad_size = max_tokens_across_pcp - self.num_tokens_pcp
        if pad_size > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
            # ### PATCH START: no router logits when Attention owns the gate
            if router_logits is not None:
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
            # ### PATCH END: no router logits when Attention owns the gate
            # ### PATCH START: pad the routing by the same row count
            if external is not None:
                external = external.padded(pad_size)
            # ### PATCH END: pad the routing by the same row count

        hidden_states = get_pcp_group().all_gather(
            hidden_states,
            dim=0,
        )
        # ### PATCH START: gather the routing with the activation's collective
        if router_logits is not None:
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )
        if external is not None:
            external = external.gathered(
                lambda tensor: get_pcp_group().all_gather(tensor, dim=0),
            )
        # ### PATCH END: gather the routing with the activation's collective

    # ### PATCH START: publish the gathered routing
    if external is not None:
        self.afd_external_routing = external
    # ### PATCH END: publish the gathered routing

    return MoEPrepareOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )


def apply_afd_external_routing_prepare_patch() -> None:
    """Install the ALLGATHER prepare patch.

    Call once during runtime initialisation, not per forward. A repeat call is a no-op:
    the class attribute already holds this function.
    """

    if PrepareAndFinalizeWithAllGather._prepare_with_dp_group is prepare_with_dp_group:
        return
    PrepareAndFinalizeWithAllGather._prepare_with_dp_group = prepare_with_dp_group


__all__ = [
    "apply_afd_external_routing_prepare_patch",
    "prepare_with_dp_group",
]
