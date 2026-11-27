# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend Attention-side remote experts and CAM MoE runners."""

from typing import Any

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_dbo import (
    send_async_cam_remote_call,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout import (
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)
from afd_plugin.model_executor.ffn.remote import (
    AFDRemoteMoERunner,
    AFDRemoteRoutedExperts,
    _current_afd_metadata,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


class AFDAscendRemoteRoutedExperts(AFDRemoteRoutedExperts, AscendRoutedExperts):
    """Preserve AFD's weightless method through Ascend initialization."""

    def _replace_quant_method(self, quant_method: FusedMoEMethodBase) -> None:
        """Ascend 7dcdb690 installs a local BF16 method after weight creation.
        The remote method owns post-load processing for these weightless experts.
        """


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


class AFDAsyncCAMMoERunner(AFDRemoteMoERunner):
    """Dispatch local routes and combine on the microbatch that owns the call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_sequence_parallel_moe = (
            get_current_vllm_config().parallel_config.use_sequence_parallel_moe
        )

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
        assert routing is not None, "Async CAM requires Attention top-k routes"
        topk_ids, topk_weights = routing
        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            use_sequence_parallel=self.use_sequence_parallel_moe,
        )
        afd_metadata = _current_afd_metadata()
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=afd_metadata.stage_idx,
                seq_len=int(dispatch_payload.hidden_states.shape[0]),
            )
        )
        receive = send_async_cam_remote_call(
            afd_metadata.connector,
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
        )
        shared_output = (
            shared_experts(hidden_states) if shared_experts is not None else None
        )
        maybe_apply_dbo_yield(hidden_states, role="attention")
        output = receive()
        output = restore_cam_dispatch_output(output, dispatch_payload.layout)
        if shared_output is not None:
            output = output + shared_output
        return output
