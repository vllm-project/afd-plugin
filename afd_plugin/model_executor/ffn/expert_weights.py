# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loaded expert tensors and shared quantized layout contracts; no native ops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

import torch

INT4_PER_INT32 = 8


@dataclass(frozen=True)
class QuantizedExpertWeights:
    layer_idx: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_bias: torch.Tensor | None
    w2_bias: torch.Tensor | None
    per_channel: bool
    swiglu_limit: float
    routed_scaling_factor: float
    # Async CAM quantizes to INT8; the existing W4A8 MLP returns BF16.
    output_dtype: torch.dtype = torch.bfloat16
    quantization: Literal["W4A8", "W8A8"] = "W4A8"

    def signature(self) -> tuple:
        return (
            tuple(
                (tuple(t.shape), t.dtype, t.stride(), t.device)
                if t is not None
                else None
                for t in (
                    self.w13,
                    self.w2,
                    self.w13_scale,
                    self.w2_scale,
                    self.w13_bias,
                    self.w2_bias,
                )
            ),
            self.per_channel,
            self.swiglu_limit,
            self.routed_scaling_factor,
            self.output_dtype,
            self.quantization,
        )


def validate_expert_weights(
    layers: list[QuantizedExpertWeights],
) -> list[QuantizedExpertWeights]:
    layers = sorted(layers, key=lambda layer: layer.layer_idx)
    if not layers:
        raise ValueError("layered GMM requires at least one quantized MoE layer")
    layer_ids = tuple(layer.layer_idx for layer in layers)
    if layer_ids[0] < 0 or len(set(layer_ids)) != len(layers):
        raise ValueError("layered GMM requires unique nonnegative model layer IDs")
    if any(layer.signature() != layers[0].signature() for layer in layers):
        raise ValueError("layered GMM requires homogeneous quantized layers")
    for layer in layers:
        _validate_layer(layer)
    return layers


def expert_geometry(layer: QuantizedExpertWeights) -> tuple[int, int, int]:
    """Return (experts, hidden, intermediate) from the loaded weight layout.

    INT4 weights pack eight nibbles per INT32, so the stored last dimension
    is a quarter of the gate/up width; INT8 stores that width directly.
    Callers that need only the logical geometry must not re-derive it.
    """
    experts, hidden, last = layer.w13.shape
    if layer.quantization == "W8A8":
        if last % 2:
            raise ValueError("layered GMM requires an even W13 gate/up width")
        return experts, hidden, last // 2
    return experts, hidden, last * INT4_PER_INT32 // 2


def expert_weight_signature(layers: list[QuantizedExpertWeights], capacity: int) -> str:
    """Preserve the negotiated layout, including Layered's W8A8 zero bias."""
    layers = validate_expert_weights(layers)
    first = layers[0]
    tensors = [
        first.w13,
        first.w2,
        first.w13_scale,
        first.w2_scale,
        first.w13_bias,
        first.w2_bias,
    ]
    if first.quantization == "W8A8":
        # Meta describes zeros_like's layout without allocating a device buffer.
        tensors[-1] = torch.empty_like(first.w2_scale, dtype=torch.int32, device="meta")
    return json.dumps(
        {
            "capacity": capacity,
            "layers": tuple(layer.layer_idx for layer in layers),
            "weights": [
                (tuple(t.shape), str(t.dtype), t.stride()) if t is not None else None
                for t in tensors
            ],
            "per_channel": first.per_channel,
            "quantization": first.quantization,
            "limit": first.swiglu_limit,
            "routed_scaling": first.routed_scaling_factor,
            "output_dtype": str(first.output_dtype),
        },
        sort_keys=True,
    )


def _validate_layer(layer: QuantizedExpertWeights) -> None:
    prefix = f"layered GMM layer {layer.layer_idx}: "
    if layer.quantization == "W8A8":
        _validate_w8a8_layer(layer)
        return
    if layer.quantization != "W4A8":
        raise ValueError(prefix + "requires W4A8 or W8A8 dynamic quantization")
    if layer.w13.dim() != 3 or layer.w2.dim() != 3:
        raise ValueError(prefix + "weights must be packed [E, K, N/8] tensors")
    experts, hidden, packed_intermediate = layer.w13.shape
    intermediate = packed_intermediate * INT4_PER_INT32 // 2
    if layer.w2.shape != (experts, intermediate, hidden // INT4_PER_INT32):
        raise ValueError(prefix + "W13/W2 hidden and intermediate dimensions disagree")
    for name, tensor in (("w13", layer.w13), ("w2", layer.w2)):
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(
                prefix + name + " requires contiguous INT32-packed INT4 storage"
            )
    for name, scale, width, k in (
        ("w13_scale", layer.w13_scale, 2 * intermediate, hidden),
        ("w2_scale", layer.w2_scale, hidden, intermediate),
    ):
        expected_rank = 2 if name == "w13_scale" and layer.per_channel else 3
        if (
            scale.dtype != torch.int64
            or scale.dim() != expected_rank
            or scale.shape[0] != experts
            or scale.shape[-1] != width
        ):
            raise ValueError(
                prefix + name + " has incompatible shape or encoded INT64 dtype"
            )
        if scale.dim() == 3:
            groups = scale.shape[1]
            if groups <= 0 or k % groups or (layer.per_channel and groups != 1):
                raise ValueError(
                    prefix + name + " has incompatible quantization groups"
                )
    for name, bias, width in (
        ("w13_scale_bias", layer.w13_bias, 2 * intermediate),
        ("w2_scale_bias", layer.w2_bias, hidden),
    ):
        if (
            bias is None
            or bias.dtype != torch.float32
            or bias.shape != (experts, width)
        ):
            raise ValueError(prefix + name + " requires [E, N] FP32 compensation")
    tensors = (
        layer.w2,
        layer.w13_scale,
        layer.w2_scale,
        cast(torch.Tensor, layer.w13_bias),
        cast(torch.Tensor, layer.w2_bias),
    )
    if any(t.device != layer.w13.device for t in tensors):
        raise ValueError(
            prefix + "all weights, scales and compensation must share a device"
        )


def _validate_w8a8_layer(layer: QuantizedExpertWeights) -> None:
    prefix = f"layered GMM layer {layer.layer_idx}: "
    if layer.w13.dim() != 3 or layer.w2.dim() != 3:
        raise ValueError(prefix + "W8A8 weights require [E, K, N] tensors")
    experts, hidden, gate_up = layer.w13.shape
    if gate_up % 2 or layer.w2.shape != (experts, gate_up // 2, hidden):
        raise ValueError(prefix + "W13/W2 hidden and intermediate dimensions disagree")
    if any(
        t.dtype != torch.int8 or not t.is_contiguous() for t in (layer.w13, layer.w2)
    ):
        raise ValueError(prefix + "W8A8 requires contiguous INT8 weights")
    if not layer.per_channel or layer.w13_scale.dtype != torch.float32:
        raise ValueError(prefix + "W8A8 requires FP32 per-channel W13 scales")
    if layer.w13_scale.shape != (experts, gate_up):
        raise ValueError(prefix + "W8A8 w13_scale requires [E, 2I]")
    if layer.w2_scale.shape != (experts, hidden):
        raise ValueError(prefix + "W8A8 w2_scale requires [E, H] scales")
    if layer.w2_scale.dtype != torch.bfloat16 or layer.output_dtype != torch.bfloat16:
        raise ValueError(prefix + "W8A8 requires BF16 W2 scales and output")
    if layer.w13_bias is not None or layer.w2_bias is not None:
        raise ValueError(prefix + "W8A8 does not use W4A8 compensation")
    if any(
        t.device != layer.w13.device
        for t in (layer.w2, layer.w13_scale, layer.w2_scale)
    ):
        raise ValueError(prefix + "all weights and scales must share a device")
