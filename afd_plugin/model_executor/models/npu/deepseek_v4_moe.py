# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-role MoE block for Ascend DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear

from afd_plugin.model_executor.ffn.remote import build_expert
from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
    compute_attention_gate_topk,
    local_hash_input_ids,
)

if TYPE_CHECKING:
    from transformers import DeepseekV2Config


class AFDDeepseekV4MoE(nn.Module):
    """Attention's local router and shared MLP with remote routed experts."""

    def __init__(
        self,
        *,
        config: DeepseekV2Config,
        layer_idx: int,
        prefix: str,
        vllm_config: VllmConfig,
        compute_gate_on_attention: bool,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()
        self.gate: ReplicatedLinear | None = None
        self.shared_experts: nn.Module | None = None
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        if compute_gate_on_attention:
            self.top_k = int(config.num_experts_per_tok)
            self.renormalize = bool(config.norm_topk_prob)
            self.scoring_func = getattr(config, "scoring_func", "softmax")
            self.num_expert_group = int(getattr(config, "n_group", 1))
            self.topk_group = int(getattr(config, "topk_group", 1))
            # Match native DSV4's explicit shared-output reduction. Ascend keeps
            # weights TP-sharded with SP unless shared-expert DP is enabled.
            if config.n_shared_experts:
                from .deepseek_v4_shared_experts import AFDDeepseekV4SharedExperts

                self.shared_experts = AFDDeepseekV4SharedExperts(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size
                    * config.n_shared_experts,
                    hidden_act=config.hidden_act,
                    swiglu_limit=getattr(config, "swiglu_limit", None),
                    quant_config=vllm_config.quant_config,
                    reduce_results=False,
                    is_sequence_parallel=(
                        vllm_config.parallel_config.use_sequence_parallel_moe
                    ),
                    prefix=f"{prefix}.shared_experts",
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

        self.experts = build_expert(
            vllm_config,
            gate=self.gate,
            transport_layer_idx=layer_idx,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
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
        return self.experts(
            hidden_states,
            router_logits=hidden_states,
            routing=(ids, weights),
            shared_experts=self.shared_experts,
        )
