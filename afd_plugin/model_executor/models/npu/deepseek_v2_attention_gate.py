# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 attention-gate MoE helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch

from afd_plugin.envs import force_balanced_topk_ids_enabled
from afd_plugin.forward_context import get_afd_metadata_from_forward_context

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe import FusedMoERouter
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        _DeepseekAdapterConfig,
    )
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4DecoderLayer,
    )


def prepare_attention_gate_w8a8_scales(
    layers: Iterable[torch.nn.Module],
) -> None:
    """Retain the GMM scale on its routed owner after native quant finalize."""
    from vllm_ascend.quantization.quant_type import QuantType

    # Model hooks select FFN decoders; ModuleList exposes their type as Module.
    for module in layers:
        layer = cast("AFDDeepseekV4DecoderLayer", module)
        if not layer.is_moe_layer:
            continue
        experts = cast("AscendMoERunner", layer.mlp.experts)
        if experts.quant_type != QuantType.W8A8:
            continue
        owner = cast("AscendRoutedExperts", experts.routed_experts)
        if owner.dynamic_eplb:
            continue
        # bd42eea's MegaMoE finalizer deletes this ordinary derived Tensor,
        # retaining w13_weight_scale on the same owner. This is a cache-lifetime
        # check, not runtime-version detection; preserve every existing value.
        if "w13_weight_scale_fp32" not in owner.__dict__:
            owner.w13_weight_scale_fp32 = cast(
                torch.Tensor, owner.w13_weight_scale
            ).data.to(torch.float32)


def create_gate_router(
    *,
    gate: torch.nn.Module,
    vllm_config: VllmConfig,
    config: _DeepseekAdapterConfig,
    top_k: int,
) -> FusedMoERouter:
    """Create the Ascend router used by one Attention-side MoE gate."""
    from vllm_ascend.ops.fused_moe.router import (
        create_ascend_fused_moe_router,
    )

    mix_placement = bool(
        getattr(vllm_config, "additional_config", {}).get(
            "mix_placement",
            False,
        ),
    )
    return create_ascend_fused_moe_router(
        top_k=top_k,
        global_num_experts=config.n_routed_experts,
        use_grouped_topk=True,
        renormalize=getattr(config, "norm_topk_prob", True),
        scoring_func=getattr(config, "scoring_func", "softmax"),
        num_expert_group=getattr(config, "n_group", 1),
        topk_group=getattr(config, "topk_group", 1),
        routed_scaling_factor=(
            getattr(config, "routed_scaling_factor", 1.0) if mix_placement else 1.0
        ),
        e_score_correction_bias=gate.e_score_correction_bias,
    )


def compute_attention_gate_topk(
    layer: AFDDeepseekV2DecoderLayer,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute router logits and top-k payloads for Attention-side gate."""

    return compute_gate_topk(
        gate=layer.mlp.gate,
        vllm_config=layer.vllm_config,
        config=layer.config,
        top_k=layer.top_k,
        hidden_states=hidden_states,
        gate_router=layer.mlp.gate_router,
    )


def compute_gate_topk(
    *,
    gate: torch.nn.Module,
    vllm_config: VllmConfig,
    config: _DeepseekAdapterConfig,
    top_k: int,
    hidden_states: torch.Tensor,
    gate_router: FusedMoERouter | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute routing payloads for a native-path gate proxy."""
    # Keep optional Ascend imports at the NPU execution boundary.

    router_logits, _ = gate(hidden_states)
    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError(
            "AFD connector required for compute_gate_on_attention "
            "but not found in forward context",
        )
    afd_connector = cast("CAMAsyncAFDConnector", afd_metadata.connector)
    mix_placement = bool(
        getattr(vllm_config, "additional_config", {}).get(
            "mix_placement",
            False,
        ),
    )
    num_redundant_experts = (
        vllm_config.parallel_config.eplb_config.num_redundant_experts
    )
    if mix_placement:
        num_experts = (
            config.n_shared_experts + config.n_routed_experts + num_redundant_experts
        )
    else:
        num_experts = config.n_routed_experts + num_redundant_experts
    routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
    topk_weights, topk_ids = afd_connector.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=True,
        renormalize=getattr(config, "norm_topk_prob", True),
        scoring_func=getattr(config, "scoring_func", "softmax"),
        num_expert_group=getattr(config, "n_group", 1),
        topk_group=getattr(config, "topk_group", 1),
        routed_scaling_factor=(routed_scaling_factor if mix_placement else 1.0),
        e_score_correction_bias=gate.e_score_correction_bias,
        mix_placement=mix_placement,
        num_logical_experts=router_logits.shape[1],
        num_shared_experts=config.n_shared_experts,
        num_experts=num_experts,
        router=gate_router,
    )
    # This proxy bypasses AscendRoutedExperts, which balances dummy profile
    # routing. Reuse the logical-ID layout without changing live routing.
    # The profile flag is read inside the custom op so tracing cannot
    # specialize the branch on the profile-time value.
    torch.ops.vllm.afd_balance_routing_ids_(
        topk_ids,
        router_logits.shape[1],
        force_balanced_topk_ids_enabled(),
    )
    topk_weights = topk_weights.to(torch.float32)
    return topk_weights, topk_ids, router_logits


__all__ = [
    "compute_attention_gate_topk",
    "prepare_attention_gate_w8a8_scales",
]
