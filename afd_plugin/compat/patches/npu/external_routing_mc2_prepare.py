# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Let the MC2 prepare path consume routing computed on Attention.

MC2 differs from ALLGATHER in every part that matters for alignment: it pads to
``padded_num_tokens`` instead of the DP-wide token count, it shards the token
dimension across TP ranks instead of gathering it, it carries ``mc2_mask``, and it
finalizes through the All2All implementation. So the routing cannot reuse the
ALLGATHER path's bookkeeping; it has to follow the same branch the activation
takes in each of those cases.

The heavy lifting is still :class:`AFDExternalRoutingInputs`; this patch only puts
it through the same pad and split the activation goes through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.moe_utils import _pad_tokens_with_cat
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    # Type-only: importing the connectors package pulls vLLM in through its
    # __init__, and the patch only calls methods on the caller's object.
    from afd_plugin.connectors.npu.external_routing import AFDExternalRoutingInputs

# Declared as a class default so the patched function reads a plain attribute
# instead of guarding with getattr, and so upstream code paths keep working when
# the plugin never assigns it.
PrepareAndFinalizeWithMC2.afd_external_routing = None


# Patch reason: upstream pads and shards ``router_logits`` unconditionally and
# reads the routing back on the FFN side, because upstream computes the routing
# there.
# Patch functionality: when Attention supplied the routing, pad and shard it by
# the same rule as the activation in each branch, and tolerate a missing
# ``router_logits``. Without the attribute the function behaves exactly like
# upstream.
# Signature: matches upstream; no added parameters.
def prepare(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    replace_allreduce: bool = False,
    quant_type=QuantType.NONE,
) -> MoEPrepareOutput:
    """
    Preparation steps:
      1. Fetch `mc2_mask` and target padding length from forward context.
      2. Pad `hidden_states` and `router_logits` to target length if needed.
      3. If TP > 1, split tensors along token dimension and select current TP
         rank's slice.
      4. Split and return corresponding `mc2_mask`.

    With `replace_allreduce`, inputs are already TP-sharded. Pad only the local
    shard to the DP-wide MC2 length, preserving its original mask.

    Returns:
        MoEPrepareOutput, possibly sliced/padded.
    """
    # ### PATCH START: routing may be precomputed on the Attention side
    external = self.afd_external_routing
    # The routing describes this rank's rows, so it must start from the same row
    # count as the activation. Mismatched counts pad to different totals and the
    # ids then describe the wrong tokens with no error.
    if external is not None and external.token_count != hidden_states.shape[0]:
        raise ValueError(
            "external routing covers a different token count than the activation: "
            f"{external.token_count} routing rows for "
            f"{hidden_states.shape[0]} activation rows",
        )
    # ### PATCH END: routing may be precomputed on the Attention side
    self.replace_allreduce = replace_allreduce
    mc2_mask = _EXTRA_CTX.mc2_mask
    if self.replace_allreduce:
        # SP shards use the local token count, not the largest DP batch.
        # Select valid bits before adding padding for uniform MC2 batches.
        self.num_tokens = hidden_states.shape[0]
        start = self.tp_rank * self.num_tokens
        mc2_mask = mc2_mask[start : start + self.num_tokens]
        pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
        if pad_size > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
            # ### PATCH START: no router logits when Attention owns the gate
            if router_logits is not None:
                router_logits = nn.functional.pad(
                    router_logits,
                    (0, 0, 0, pad_size),
                )
            # ### PATCH END: no router logits when Attention owns the gate
            # ### PATCH START: pad the routing by the same row count
            if external is not None:
                external = external.padded(pad_size)
            # ### PATCH END: pad the routing by the same row count
            mc2_mask = nn.functional.pad(mc2_mask, (0, pad_size), value=False)
    elif self.tp_size > 1:
        # Also slice mc2_mask
        split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
        mc2_mask = split_mc2_mask[self.tp_rank]

    padded_hidden_states_shape = hidden_states.shape
    if not self.replace_allreduce:
        self.num_tokens, _ = hidden_states.shape
        target_pad_length = _EXTRA_CTX.padded_num_tokens
        pad_size = target_pad_length - self.num_tokens

        if pad_size > 0:
            hidden_states = _pad_tokens_with_cat(hidden_states, target_pad_length)
            # ### PATCH START: no router logits when Attention owns the gate
            if router_logits is not None:
                router_logits = _pad_tokens_with_cat(
                    router_logits,
                    target_pad_length,
                )
            # ### PATCH END: no router logits when Attention owns the gate
            # ### PATCH START: pad the routing to the same total length
            if external is not None:
                external = external.padded(pad_size)
            # ### PATCH END: pad the routing to the same total length
            padded_hidden_states_shape = hidden_states.shape

        # Slice across TP ranks
        if self.tp_size > 1:
            split_hidden_states = torch.tensor_split(
                hidden_states,
                self.tp_size,
                dim=0,
            )
            # ### PATCH START: no router logits when Attention owns the gate
            if router_logits is not None:
                split_router_logits = torch.tensor_split(
                    router_logits,
                    self.tp_size,
                    dim=0,
                )
                router_logits = split_router_logits[self.tp_rank]
            # ### PATCH END: no router logits when Attention owns the gate
            hidden_states = split_hidden_states[self.tp_rank]
            # ### PATCH START: shard the routing with the activation
            if external is not None:
                external = external.split_rows(
                    lambda tensor: torch.tensor_split(
                        tensor,
                        self.tp_size,
                        dim=0,
                    )[self.tp_rank],
                )
            # ### PATCH END: shard the routing with the activation

    # ### PATCH START: publish the prepared routing
    if external is not None:
        self.afd_external_routing = external
    # ### PATCH END: publish the prepared routing

    return MoEPrepareOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        mc2_mask=mc2_mask,
        padded_hidden_states_shape=padded_hidden_states_shape,
        pertoken_scale=None,
    )


def apply_afd_external_routing_mc2_prepare_patch() -> bool:
    """Install the MC2 prepare patch.

    Call once during runtime initialisation, not per forward. The FFN-side
    consumer owns the per-call state: it writes the stash before ``prepare`` reads
    the gathered routing back after it, and clears it in a ``finally``.
    """

    PrepareAndFinalizeWithMC2.prepare = prepare
    return True


__all__ = [
    "apply_afd_external_routing_mc2_prepare_patch",
    "prepare",
]
