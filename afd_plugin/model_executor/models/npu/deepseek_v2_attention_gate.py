# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 attention-gate MoE helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors import AFDF2ATransferPayload
from afd_plugin.envs import force_balanced_topk_ids_enabled
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context

try:
    from vllm_ascend.ascend_config import get_ascend_config
except ImportError:
    get_ascend_config = None

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        _DeepseekAdapterConfig,
    )


def _get_expert_parameter(experts: torch.nn.Module, name: str):
    """Read MoE weights across vLLM-Ascend EPLB API generations."""

    getter = getattr(experts, "get_eplb_parameter", None)
    if getter is not None:
        return getter(name)
    try:
        return getattr(experts, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Ascend MoE does not expose required expert parameter {name!r}"
        ) from exc


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
    )


def compute_gate_topk(
    *,
    gate: torch.nn.Module,
    vllm_config: VllmConfig,
    config: _DeepseekAdapterConfig,
    top_k: int,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute routing payloads for a native-path gate proxy."""

    # Native DSV4 computes router logits through DeviceOperator, which
    # performs the gate matmul in fp32.  Calling ReplicatedLinear directly
    # can return bf16 logits while e_score_correction_bias remains fp32;
    # the fused top-k operator requires these dtypes to match.
    if getattr(config, "model_type", None) == "deepseek_v4":
        from vllm_ascend.device.device_op import DeviceOperator

        router_logits = DeviceOperator.compute_gate_logits(
            hidden_states,
            gate.weight,
        )
    else:
        router_logits, _ = gate(hidden_states)
    routing_bias = getattr(gate, "e_score_correction_bias", None)
    if routing_bias is not None and routing_bias.dtype != router_logits.dtype:
        # The fused selectors require x and bias to have identical dtypes.
        # Native DSV4 normally produces fp32 logits; retain a defensive cast
        # for runtimes whose DeviceOperator returns another supported dtype.
        routing_bias = routing_bias.to(router_logits.dtype)
    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError(
            "AFD connector required for compute_gate_on_attention "
            "but not found in forward context",
        )
    afd_connector = afd_metadata.connector
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
    renormalize = getattr(config, "norm_topk_prob", True)
    scoring_func = getattr(config, "scoring_func", "softmax")
    if tid2eid is not None:
        # The native fused selector tries to obtain an MoE communication
        # method from the forward context for hash routing.  An AFD Attention
        # rank owns only the gate and remote-dispatch path, so it has no local
        # MoE communication method.  Run the hash selector directly on the
        # local token ids instead.
        if input_ids is None:
            raise RuntimeError("DSV4 hash routing requires input_ids")
        input_ids = input_ids.to(torch.int64)
        input_ids = torch.where(input_ids == -1, 0, input_ids)
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=routing_bias,
            input_ids=input_ids,
            tid2eid=tid2eid.to(torch.int32),
            k_group=getattr(config, "topk_group", 1),
            group_count=getattr(config, "n_group", 1),
            routed_scaling_factor=(routed_scaling_factor if mix_placement else 1.0),
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
    else:
        topk_weights, topk_ids = afd_connector.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=True,
            renormalize=renormalize,
            scoring_func=scoring_func,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=(routed_scaling_factor if mix_placement else 1.0),
            e_score_correction_bias=routing_bias,
            mix_placement=mix_placement,
            num_logical_experts=router_logits.shape[1],
            num_shared_experts=config.n_shared_experts,
            num_experts=num_experts,
            input_ids=input_ids,
            tid2eid=None,
        )
    if force_balanced_topk_ids_enabled():
        topk_ids = _force_balanced_topk_ids(
            topk_ids,
            num_logical_experts=router_logits.shape[1],
        )
    topk_weights = topk_weights.to(torch.float32)
    return topk_weights, topk_ids, router_logits


def compute_attention_gate_moe_ffn(
    layer: AFDDeepseekV2DecoderLayer,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    dynamic_scales: torch.Tensor | None,
    topk_scales: torch.Tensor | None,
    group_list_type: int,
    expand_x_shared: torch.Tensor | None = None,
    dynamic_scales_shared: torch.Tensor | None = None,
) -> AFDF2ATransferPayload:
    """Compute FFN output for MoE layers whose gate ran on Attention ranks."""

    if not hasattr(layer.mlp, "experts"):
        shared_experts = getattr(layer.mlp, "shared_experts", None)
        if shared_experts is None:
            raise RuntimeError("Window shared FFN rank has no shared expert module")
        if hidden_states.dtype == torch.int8:
            shared_output = _compute_w8a8_shared_experts_from_int8(
                shared_experts,
                hidden_states,
                dynamic_scales,
                output_dtype=torch.bfloat16,
            )
        else:
            shared_output = shared_experts(hidden_states)
        return AFDF2ATransferPayload(
            routed_output=shared_output,
            shared_output=None,
        )

    from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEMlpComputeInput,
        MoEWeights,
    )
    from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams
    from vllm_ascend.quantization.quant_type import QuantType

    experts = layer.mlp.experts
    quant_type = experts.quant_type
    moe_quant_params = MoEQuantParams(quant_type=quant_type)
    swiglu_limit = 0.0
    if quant_type == QuantType.NONE:
        moe_weights = MoEWeights(
            w1=_get_expert_parameter(experts, "w13_weight"),
            w2=_get_expert_parameter(experts, "w2_weight"),
            w1_bias=(
                _get_expert_parameter(experts, "w13_bias")
                if experts.moe_config.has_bias
                else None
            ),
            w2_bias=(
                _get_expert_parameter(experts, "w2_bias")
                if experts.moe_config.has_bias
                else None
            ),
        )
    elif quant_type == QuantType.W8A8:
        if experts.dynamic_eplb:
            moe_weights = MoEWeights(
                w1=_get_expert_parameter(experts, "w13_weight_list"),
                w2=_get_expert_parameter(experts, "w2_weight_list"),
                w1_scale=_get_expert_parameter(
                    experts,
                    "w13_weight_scale_fp32_list",
                ),
                w2_scale=_get_expert_parameter(experts, "w2_weight_scale_list"),
            )
        else:
            moe_weights = MoEWeights(
                w1=[_get_expert_parameter(experts, "w13_weight")],
                w2=[_get_expert_parameter(experts, "w2_weight")],
                w1_scale=[
                    _get_expert_parameter(experts, "w13_weight_scale_fp32"),
                ],
                w2_scale=[_get_expert_parameter(experts, "w2_weight_scale")],
            )
    elif quant_type == QuantType.W4A8MXFP:
        if experts.dynamic_eplb:
            raise RuntimeError("DSV4 Window W4A8MXFP does not support EPLB")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "DSV4 Window W4A8MXFP requires non-quantized FP16/BF16 "
                f"batching output, got {hidden_states.dtype}"
            )

        import torch_npu
        from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE
        from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams

        moe_weights = MoEWeights(
            w1=_get_expert_parameter(experts, "w13_weight"),
            w2=_get_expert_parameter(experts, "w2_weight"),
            w1_scale=_get_expert_parameter(experts, "w13_weight_scale"),
            w2_scale=_get_expert_parameter(experts, "w2_weight_scale"),
        )
        moe_quant_params = MoEQuantParams(
            quant_type=quant_type,
            mxfp=MoEMxfpParams(
                act_quant_type=torch.float8_e4m3fn,
                weight_quant_type=torch_npu.float4_e2m1fn_x2,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                use_bf16=hidden_states.dtype == torch.bfloat16,
            ),
        )
        # Non-quantized Window batching returns no usable activation scale.
        # Let the native MXFP MLP dynamically quantize the FP16/BF16 input.
        dynamic_scales = None
        swiglu_limit = getattr(experts, "swiglu_limit", 0.0) or 0.0
    else:
        raise RuntimeError(
            "compute_gate_on_attention currently supports only unquantized "
            f"W8A8, or W4A8MXFP Ascend MoE experts, got {quant_type}",
        )
    use_gmmswigluquant_fusion = (
        quant_type in (QuantType.W8A8, getattr(QuantType, "MXFP8", None))
        and _gmmswigluquant_fusion_enabled()
    )

    # Preserve the existing DSV2 path, where routed and shared experts are
    # colocated inside one FusedMoE.  DSV4 Window routed ranks have no embedded
    # shared module and therefore skip this block.
    shared_output = None
    shared_experts = getattr(experts, "_shared_experts", None)
    if shared_experts is not None:
        if expand_x_shared is None:
            raise RuntimeError("shared-expert FFN input is missing")
        if expand_x_shared.dtype == torch.int8 and quant_type == QuantType.W8A8:
            shared_output = _compute_w8a8_shared_experts_from_int8(
                shared_experts,
                expand_x_shared,
                dynamic_scales_shared,
                output_dtype=torch.bfloat16,
            )
        else:
            shared_input = _dequantize_int8_activation(
                expand_x_shared,
                dynamic_scales_shared,
                output_dtype=torch.bfloat16,
            )
            shared_output = shared_experts(shared_input)

    routed_output, _ = unified_apply_mlp(
        mlp_compute_input=MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=group_list,
            group_list_type=int(group_list_type),
            dynamic_scale=dynamic_scales,
            topk_scales=topk_scales,
            weights=moe_weights,
            quant=moe_quant_params,
            fusion=use_gmmswigluquant_fusion,
            activation=experts.activation,
            need_trans=False,
            dynamic_eplb=experts.dynamic_eplb,
            swiglu_limit=swiglu_limit,
        ),
    )

    if shared_output is None or hidden_states.dtype != torch.float16:
        routed_output *= layer.mlp.routed_scaling_factor
    else:
        # Retain the established DSV2 FP16 scaling convention.
        shared_output *= 1.0 / layer.mlp.routed_scaling_factor
    return AFDF2ATransferPayload(
        routed_output=routed_output,
        shared_output=shared_output,
    )


def _dequantize_int8_activation(
    hidden_states: torch.Tensor,
    dynamic_scales: torch.Tensor | None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_states.dtype != torch.int8:
        return hidden_states
    if dynamic_scales is None:
        raise RuntimeError("INT8 AFD shared experts input requires dynamic_scales")

    scales = dynamic_scales.to(torch.float32)
    while scales.dim() < hidden_states.dim():
        scales = scales.unsqueeze(-1)
    return (hidden_states.to(torch.float32) * scales).to(dtype=output_dtype)


def _compute_w8a8_shared_experts_from_int8(
    shared_experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    dynamic_scales: torch.Tensor | None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if dynamic_scales is None:
        raise RuntimeError("INT8 AFD shared experts fast path requires dynamic_scales")

    import torch_npu

    quantized_input = hidden_states
    pertoken_scale = dynamic_scales
    unsqueeze_output = False
    if (
        pertoken_scale.dim() == 2
        and quantized_input.dim() == 3
        and quantized_input.shape[1] == 1
    ):
        quantized_input = quantized_input.squeeze(dim=1)
        pertoken_scale = pertoken_scale.squeeze(dim=1)
        unsqueeze_output = True
    elif pertoken_scale.dim() == 2 and pertoken_scale.shape[1] == 1:
        pertoken_scale = pertoken_scale.squeeze(dim=1)
    quantized_input = quantized_input.clone()
    pertoken_scale = pertoken_scale.clone()

    gate_up = torch_npu.npu_quant_matmul(
        quantized_input,
        shared_experts.gate_up_proj.weight,
        shared_experts.gate_up_proj.weight_scale,
        pertoken_scale=pertoken_scale,
        bias=None,
        output_dtype=output_dtype,
    )
    if unsqueeze_output:
        gate_up = gate_up.unsqueeze(dim=1)

    shared_act = shared_experts.act_fn(gate_up)
    shared_output, _ = shared_experts.down_proj(shared_act)
    return shared_output


def _gmmswigluquant_fusion_enabled() -> bool:
    if get_ascend_config is None:
        return False
    ascend_config = get_ascend_config()
    fusion_config = getattr(ascend_config, "ascend_fusion_config", None)
    return bool(getattr(fusion_config, "fusion_ops_gmmswigluquant", False))


def _force_balanced_topk_ids(
    topk_ids: torch.Tensor,
    *,
    num_logical_experts: int,
) -> torch.Tensor:
    balanced_topk_ids = torch.arange(
        topk_ids.numel(),
        device=topk_ids.device,
        dtype=torch.int64,
    ).reshape(topk_ids.shape)
    balanced_topk_ids = balanced_topk_ids.remainder(num_logical_experts).to(
        dtype=topk_ids.dtype,
    )
    topk_ids.copy_(balanced_topk_ids)
    return topk_ids


__all__ = [
    "compute_attention_gate_moe_ffn",
    "compute_attention_gate_topk",
]
