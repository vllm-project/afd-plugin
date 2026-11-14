# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector-independent remote dense/MoE proxies; no concrete model imports."""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.utils import sequence_parallel_chunk

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.forward_context import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

UNROUTED_EXPERT_ID = -1


def _unroute_tp_padding(topk_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Route the padding rows of this TP rank's ``sequence_parallel_chunk``
    slice of ``num_tokens`` tokens to ``UNROUTED_EXPERT_ID``."""
    local_tokens = topk_ids.shape[0]
    token_index = get_tensor_model_parallel_rank() * local_tokens
    token_index = token_index + torch.arange(local_tokens, device=topk_ids.device)
    return topk_ids.masked_fill(
        (token_index >= num_tokens).unsqueeze(1), UNROUTED_EXPERT_ID
    )


class RemoteFFNProxy(nn.Module):
    """FFN stage executed through the AFD connector.

    By default the proxy is parameter-free: it sends the hidden states and the
    FFN role runs the complete MLP or MoE block. A model whose connector routes
    on Attention (an M2N routed-experts connector such as DeepEP M2N) sets
    ``gate``, ``top_k``, ``renormalize`` and ``tp_size`` after construction,
    from its model and parallel config. The proxy then computes the gate and
    vLLM's default fused top-k (``FusedTopKRouter``; EPLB is not supported)
    and sends ``topk_ids`` and ``topk_weights``; combine returns the top-k
    weighted sum, which replaces the native MoE block's output.

    With Attention TP > 1 every TP rank holds the same tokens after the
    attention all-reduce; dispatching them from every rank would make the FFN
    run each token TP times. Each rank instead pads the tokens to a multiple
    of ``tp_size`` (``sequence_parallel_chunk``), routes and dispatches only
    its slice, and the TP group all-gathers the combined slices, as vLLM's
    sequence-parallel MoE does. Padding rows get ``UNROUTED_EXPERT_ID`` and
    reach no FFN rank; they are dropped after the all-gather.
    """

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.gate: ReplicatedLinear | None = None
        self.top_k = 0
        self.renormalize = False
        self.tp_size = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return self._send_and_receive(hidden_states)
        num_tokens = hidden_states.shape[0]
        if self.tp_size > 1:
            hidden_states = sequence_parallel_chunk(hidden_states)
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        if self.tp_size > 1:
            topk_ids = _unroute_tp_padding(topk_ids, num_tokens)
        output = self._send_and_receive(
            hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[:num_tokens]
        return output

    def _send_and_receive(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        stage_idx = self._send(hidden_states, **send_kwargs)
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return self._receive(hidden_states, stage_idx)

    def _send(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> int:
        """Send this layer's Attention output; return the stage to receive."""
        afd_metadata = self._afd_metadata()
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            **send_kwargs,
        )
        return stage_idx

    def _receive(self, hidden_states: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Receive the FFN output; a connector may write it into
        ``hidden_states``."""
        return self._afd_metadata().connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )

    @staticmethod
    def _afd_metadata() -> AFDForwardContextMetadata:
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteFFNProxy requires AFD forward metadata")
        return afd_metadata


class AFDAttentionFusedMoE(RemoteFFNProxy):
    """Parameter-free native-MoE experts proxy for the Attention runtime."""

    def __init__(
        self,
        *,
        layer_idx: int,
        external_routing: bool,
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        # vLLM 5df63a72 fused_moe/runner/moe_runner.py::MoERunner.forward
        # enters _forward_impl, which consumes transported router_logits only
        # when its gate is None (external routing); a held gate routes internally
        # and ignores the forwarded logits argument.
        self.external_routing = external_routing

    # The native MoERunner boundary passes router_logits and input_ids; only
    # the proxy's transport is reused, not its kernel execution contract.
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        send_kwargs: dict[str, torch.Tensor] = {}
        if self.external_routing:
            send_kwargs["router_logits"] = router_logits
        return self._send_and_receive(hidden_states, **send_kwargs)

    def update_expert_map(self) -> None:
        """Satisfy the native MoERunner EPLB interface without local experts."""


__all__ = ["RemoteFFNProxy", "AFDAttentionFusedMoE"]
