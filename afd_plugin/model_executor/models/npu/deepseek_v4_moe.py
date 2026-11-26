# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-role MoE block for Ascend DeepSeek V4."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_dbo import (
    send_async_cam_remote_call,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_layout import (
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)
from afd_plugin.forward_context import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.ffn.remote import build_expert
from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
    compute_attention_gate_topk,
    local_hash_input_ids,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


class AFDDeepseekV4MoE(nn.Module):
    """Attention's local router and shared MLP with remote routed experts."""

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        prefix: str,
        vllm_config: VllmConfig,
        compute_gate_on_attention: bool,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_sequence_parallel_moe = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.gate: ReplicatedLinear | None = None
        self.shared_experts: nn.Module | None = None
        if compute_gate_on_attention:
            self.top_k = int(config.num_experts_per_tok)
            self.n_routed_experts = int(config.n_routed_experts)
            self.renormalize = bool(config.norm_topk_prob)
            self.scoring_func = getattr(config, "scoring_func", "softmax")
            self.num_expert_group = int(getattr(config, "n_group", 1))
            self.topk_group = int(getattr(config, "topk_group", 1))
            self.routed_scaling_factor = float(
                getattr(config, "routed_scaling_factor", 1.5),
            )
            # Match native DSV4's explicit shared-output reduction. Ascend keeps
            # weights TP-sharded with SP unless shared-expert DP is enabled.
            if config.n_shared_experts:
                from afd_plugin.model_executor.models.npu import (
                    deepseek_v4_shared_experts,
                )

                self.shared_experts = (
                    deepseek_v4_shared_experts.AFDDeepseekV4SharedExperts(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.moe_intermediate_size
                        * config.n_shared_experts,
                        hidden_act=config.hidden_act,
                        swiglu_limit=getattr(config, "swiglu_limit", None),
                        quant_config=vllm_config.quant_config,
                        reduce_results=False,
                        is_sequence_parallel=self.use_sequence_parallel_moe,
                        prefix=f"{prefix}.shared_experts",
                    )
                )
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.n_routed_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
            self.gate.precast_fp32_weight = True
            # Match DeepseekV4MoE in Ascend 7dcdb690. Transformers may place Hash
            # MoE at nonconsecutive layers and omits the legacy count.
            if is_draft_layer:
                uses_hash = False
            else:
                mlp_layer_types = getattr(config, "mlp_layer_types", None)
                if mlp_layer_types is not None:
                    uses_hash = (
                        layer_idx < len(mlp_layer_types)
                        and mlp_layer_types[layer_idx] == "hash_moe"
                    )
                else:
                    uses_hash = layer_idx < getattr(config, "num_hash_layers", 0)
            if uses_hash:
                self.gate.tid2eid = nn.Parameter(
                    (
                        torch.arange(config.vocab_size, dtype=torch.int32).unsqueeze(1)
                        + torch.arange(config.num_experts_per_tok, dtype=torch.int32)
                    )
                    % config.n_routed_experts,
                    requires_grad=False,
                )
                self.gate.e_score_correction_bias = None
            else:
                self.gate.tid2eid = None
                self.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32),
                )

        self.experts = None
        if (
            parse_afd_config(vllm_config, validate=False).connector
            != AFD_ASYNC_CONNECTOR
        ):
            self.experts = build_expert(
                vllm_config,
                gate=self.gate,
                transport_layer_idx=layer_idx,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
                prefix=f"{prefix}.experts",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gate is None:
            # FFN opens the IDs channel on every V4 layer, including non-Hash MoE.
            ids = local_hash_input_ids(
                input_ids=get_forward_context().input_ids
                if input_ids is None
                else input_ids,
                router_tokens=hidden_states.shape[0],
            )
            return self.experts(
                hidden_states, router_logits=hidden_states, input_ids=ids
            )

        weights, ids = compute_attention_gate_topk(
            self,
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )
        if self.experts is not None:
            return self.experts(
                hidden_states,
                router_logits=hidden_states,
                routing=(ids, weights),
                shared_experts=self.shared_experts,
            )
        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            weights,
            ids,
            None,
            use_sequence_parallel=self.use_sequence_parallel_moe,
        )
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("Async CAM requires AFD forward metadata")
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
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )
        maybe_apply_dbo_yield(hidden_states, role="attention")
        output = receive()
        output = restore_cam_dispatch_output(output, dispatch_payload.layout)
        if shared_output is not None:
            output = output + shared_output
        return output
