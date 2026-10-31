# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4 Window global FFN helpers for Ascend MXFP weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


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


@dataclass(frozen=True)
class WindowGlobalMXFPWeights:
    """Layer-major packed weights consumed by one Window FFN transaction."""

    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    routed_scaling_factor: float
    swiglu_limit: float
    is_routed: bool


def _stack_mxfp_weights(parts: list[torch.Tensor]) -> torch.Tensor:
    """Stack FP8 logical weights without using unsupported aclnnStack."""

    first = parts[0]
    packed = torch.empty(
        (len(parts), *first.shape),
        dtype=first.dtype,
        device=first.device,
    )
    for index, part in enumerate(parts):
        packed[index].copy_(part)
    return packed


def build_window_global_mxfp_weights(
    layers: Iterable[torch.nn.Module],
) -> WindowGlobalMXFPWeights:
    """Build routed MXFP4 or shared MXFP8 layer-major weights."""

    w1_parts: list[torch.Tensor] = []
    w1_scale_parts: list[torch.Tensor] = []
    w2_parts: list[torch.Tensor] = []
    w2_scale_parts: list[torch.Tensor] = []
    is_routed: bool | None = None
    routed_scaling_factor = 1.0
    swiglu_limit = 0.0

    for layer in layers:
        mlp = layer.mlp
        experts = getattr(mlp, "experts", None)
        layer_is_routed = experts is not None
        if is_routed is None:
            is_routed = layer_is_routed
        elif is_routed != layer_is_routed:
            raise RuntimeError(
                "Window global FFN cannot mix routed and shared-only layers"
            )

        if layer_is_routed:
            from vllm_ascend.quantization.quant_type import QuantType

            if experts.quant_type != QuantType.W4A8MXFP:
                raise RuntimeError(
                    "Window global FFN currently requires W4A8MXFP experts, "
                    f"got {experts.quant_type}"
                )
            layer_w1 = _get_expert_parameter(experts, "w13_weight")
            layer_w1_scale = _get_expert_parameter(experts, "w13_weight_scale")
            layer_w2 = _get_expert_parameter(experts, "w2_weight")
            layer_w2_scale = _get_expert_parameter(experts, "w2_weight_scale")
            w1_parts.append(layer_w1)
            w1_scale_parts.append(layer_w1_scale)
            w2_parts.append(layer_w2)
            w2_scale_parts.append(layer_w2_scale)
            routed_scaling_factor = float(mlp.routed_scaling_factor)
            swiglu_limit = float(getattr(experts, "swiglu_limit", 0.0) or 0.0)
        else:
            from vllm_ascend.quantization.methods.w8a8_mxfp8 import (
                AscendW8A8MXFP8DynamicLinearMethod,
            )

            shared = getattr(mlp, "shared_experts", None)
            if shared is None:
                raise RuntimeError(
                    "Window global FFN layer has neither routed nor shared experts"
                )
            for projection, weights, scales in (
                (shared.gate_up_proj, w1_parts, w1_scale_parts),
                (shared.down_proj, w2_parts, w2_scale_parts),
            ):
                adapter = getattr(projection, "quant_method", None)
                scheme = getattr(adapter, "quant_method", adapter)
                if not isinstance(scheme, AscendW8A8MXFP8DynamicLinearMethod):
                    raise RuntimeError(
                        "Window global shared FFN requires W8A8MXFP8 weights, "
                        f"got {type(scheme).__name__}"
                    )
                if (
                    projection.weight.dtype != torch.float8_e4m3fn
                    or projection.weight_scale.dtype != torch.uint8
                    or projection.weight_scale.ndim != 3
                ):
                    raise RuntimeError(
                        "Window global shared FFN weights must be initialized "
                        "after native MXFP8 post-load processing"
                    )
                weights.append(projection.weight)
                scales.append(projection.weight_scale)
            swiglu_limit = float(
                getattr(shared.act_fn, "swiglu_limit", 0.0) or 0.0
            )

    if not w1_parts or is_routed is None:
        raise RuntimeError("Window global FFN found no local expert weights")
    if is_routed:
        # Routed weights are packed from raw ND checkpoint tensors in
        # ``load_weights()`` before vLLM converts every layer to WeightNZ.
        raw_w1 = torch.cat(w1_parts, dim=0)
        raw_w1_scale = torch.cat(w1_scale_parts, dim=0)
        raw_w2 = torch.cat(w2_parts, dim=0)
        raw_w2_scale = torch.cat(w2_scale_parts, dim=0)
        import torch_npu

        # Match AscendW4A8MXFPDynamicFusedMoEMethod post-load processing after
        # flattening layer and local-expert dimensions into one group axis.
        w1 = torch_npu.npu_format_cast(
            raw_w1,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        ).transpose(1, 2)
        w2 = torch_npu.npu_format_cast(
            raw_w2,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        ).transpose(1, 2)

        group_num, n, k = raw_w1_scale.shape
        w1_scale = raw_w1_scale.reshape(group_num, n, k // 2, 2).transpose(
            -3, -2
        )
        group_num, n, k = raw_w2_scale.shape
        w2_scale = raw_w2_scale.reshape(group_num, n, k // 2, 2).transpose(
            -3, -2
        )
    else:
        # Native per-layer W8A8MXFP8 processing has already transposed weights
        # and expanded scales. Only add the global layer/group dimension.
        w1 = _stack_mxfp_weights(w1_parts)
        w1_scale = torch.stack(w1_scale_parts, dim=0)
        w2 = _stack_mxfp_weights(w2_parts)
        w2_scale = torch.stack(w2_scale_parts, dim=0)

    group_num = int(w1.shape[0])
    if group_num > 1024:
        raise RuntimeError(
            "Window global FFN exceeds the GMM group limit: "
            f"groups={group_num} limit=1024"
        )
    if not (
        w1.shape[0]
        == w1_scale.shape[0]
        == w2.shape[0]
        == w2_scale.shape[0]
    ):
        raise RuntimeError("Window global FFN packed weights have inconsistent groups")

    return WindowGlobalMXFPWeights(
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
        is_routed=is_routed,
    )


def compute_window_global_mxfp_ffn(
    *,
    hidden_states: torch.Tensor,
    compact_group_list: torch.Tensor,
    dynamic_scale: torch.Tensor | None,
    weights: WindowGlobalMXFPWeights,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run all ready Window layers with one layer-major MXFP MoE MLP."""

    if hidden_states.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise RuntimeError(
            "Window global MXFP FFN requires FP16/BF16 or FP8 E4M3FN input, "
            f"got {hidden_states.dtype}"
        )
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Window global MXFP FFN requires FP16/BF16 output, "
            f"got {output_dtype}"
        )
    group_num = int(weights.w1.shape[0])
    if tuple(compact_group_list.shape) != (group_num, 2):
        raise RuntimeError(
            "Window global FFN requires a type-2 group list with shape "
            f"{(group_num, 2)}, got {tuple(compact_group_list.shape)}"
        )

    import torch_npu
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE
    from vllm_ascend.quantization.quant_type import QuantType

    input_dtype = output_dtype
    weight_quant_type = (
        torch_npu.float4_e2m1fn_x2
        if weights.is_routed
        else torch.float8_e4m3fn
    )
    mxfp_quant_dtype = (
        QuantType.W4A8MXFP if weights.is_routed else QuantType.MXFP8
    )
    if hidden_states.dtype in (torch.float16, torch.bfloat16):
        quantized_states, input_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=torch.float8_e4m3fn,
            use_mxfp_quant=True,
        )
    else:
        if hidden_states.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "Window global MXFP FFN requires FP16/BF16 or FP8 E4M3FN "
                f"input, got {hidden_states.dtype}"
            )
        if dynamic_scale is None:
            raise RuntimeError("Window MXFP8 input requires dynamic_scale")
        expected_scale_shape = (
            hidden_states.shape[0],
            (hidden_states.shape[1] + 31) // 32,
        )
        if tuple(dynamic_scale.shape) != expected_scale_shape:
            raise RuntimeError(
                "Window MXFP8 scale has unexpected shape: "
                f"got={tuple(dynamic_scale.shape)} "
                f"expected={expected_scale_shape}"
            )
        if dynamic_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
            raise RuntimeError(
                "Window MXFP8 scale requires E8M0 or its uint8 byte view, "
                f"got {dynamic_scale.dtype}"
            )
        quantized_states = hidden_states
        input_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(
            dynamic_scale
        )
    if weights.is_routed:
        gate_up_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=quantized_states,
            weight=weights.w1,
            weight_scale=weights.w1_scale,
            per_token_scale=input_scale,
            group_list=compact_group_list,
            group_list_type=2,
            input_dtype=torch.bfloat16,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=weight_quant_type,
            scale_type=FLOAT8_E8M0FNU_DTYPE,
            per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
            use_bf16=True,
            use_mxfp_quant=True,
            fallback_output_dtype=torch.bfloat16,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
        activated_states, activated_scale, _ = (
            torch.ops._C_ascend.npu_swiglu_group_quant(
                gate_up_states,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=weights.swiglu_limit,
            )
        )
        activated_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(
            activated_scale
        )
    else:
        # Preserve the native shared-MLP quantization boundaries:
        # gate_up_proj -> BF16 activation -> down_proj.  GMM executes every
        # ready layer at once, while SwiGLU remains an elementwise operation.
        gate_up_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=quantized_states,
            weight=weights.w1,
            weight_scale=weights.w1_scale,
            per_token_scale=input_scale,
            group_list=compact_group_list,
            group_list_type=2,
            input_dtype=input_dtype,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=weight_quant_type,
            scale_type=FLOAT8_E8M0FNU_DTYPE,
            per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
            use_bf16=input_dtype == torch.bfloat16,
            use_mxfp_quant=True,
            fallback_output_dtype=input_dtype,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
        activated_states, activated_scale, _ = (
            torch.ops._C_ascend.npu_swiglu_group_quant(
                gate_up_states,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=weights.swiglu_limit,
            )
        )
        activated_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(
            activated_scale
        )
    output = DeviceOperator.npu_grouped_matmul_gmm2(
        hidden_states=activated_states,
        weight=weights.w2,
        weight_scale=weights.w2_scale,
        per_token_scale=activated_scale,
        group_list=compact_group_list,
        group_list_type=2,
        input_dtype=input_dtype,
        act_quant_type=torch.float8_e4m3fn,
        weight_quant_type=weight_quant_type,
        scale_type=FLOAT8_E8M0FNU_DTYPE,
        per_token_scale_type=FLOAT8_E8M0FNU_DTYPE,
        use_bf16=input_dtype == torch.bfloat16,
        use_mxfp_quant=True,
        fallback_output_dtype=input_dtype,
        mxfp_quant_dtype=mxfp_quant_dtype,
    )
    if weights.is_routed:
        output = output * weights.routed_scaling_factor
    return output


__all__ = [
    "WindowGlobalMXFPWeights",
    "build_window_global_mxfp_weights",
    "compute_window_global_mxfp_ffn",
]
