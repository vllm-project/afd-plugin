# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static W4A8 weights and device-controlled async CAM FFN execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

INT4_PER_INT32 = 8
CAM_LAYER_INDEX = 2


@dataclass(frozen=True)
class W4A8LayerWeights:
    layer_idx: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_bias: torch.Tensor
    w2_bias: torch.Tensor
    per_channel: bool
    swiglu_limit: float
    routed_scaling_factor: float
    # Async CAM quantizes to INT8; the existing W4A8 MLP returns BF16.
    output_dtype: torch.dtype = torch.bfloat16

    def signature(self) -> tuple:
        return (
            tuple(
                (tuple(t.shape), t.dtype, t.stride(), t.device)
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
        )


class AsyncCAMW4A8Executor:
    """Reuse layer tensors; all per-work-item control data stays on device."""

    def __init__(self, layers: list[W4A8LayerWeights]) -> None:
        layers = sorted(layers, key=lambda layer: layer.layer_idx)
        if not layers:
            raise ValueError("layered GMM requires at least one W4A8 MoE layer")
        self.layer_ids = tuple(layer.layer_idx for layer in layers)
        if self.layer_ids[0] < 0 or len(set(self.layer_ids)) != len(layers):
            raise ValueError("layered GMM requires unique nonnegative model layer IDs")
        first = layers[0]
        if any(layer.signature() != first.signature() for layer in layers):
            raise ValueError("layered GMM requires homogeneous W4A8 layers")
        for layer in layers:
            self._validate_layer(layer)
        self.w13 = [layer.w13 for layer in layers]
        self.w2 = [layer.w2 for layer in layers]
        self.w13_scale = [layer.w13_scale for layer in layers]
        self.w2_scale = [layer.w2_scale for layer in layers]
        self.w13_bias = [layer.w13_bias for layer in layers]
        self.w2_bias = [layer.w2_bias for layer in layers]
        self.per_channel = first.per_channel
        self.output_dtype = first.output_dtype
        self.routed_scaling_factor = first.routed_scaling_factor
        self.layer_id_to_slot = None
        if self.layer_ids != tuple(range(len(layers))):
            mapping = [-1] * (self.layer_ids[-1] + 1)
            for slot, layer_idx in enumerate(self.layer_ids):
                mapping[layer_idx] = slot
            self.layer_id_to_slot = torch.tensor(
                mapping,
                dtype=torch.int64,
                device=first.w13.device,
            )
        # Resolve both operators once, before dispatch can receive work.
        self.w13_op = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered
        self.w2_op = torch.ops.afd_ascend.grouped_matmul_layered

    @staticmethod
    def _validate_layer(layer: W4A8LayerWeights) -> None:
        prefix = f"layered GMM layer {layer.layer_idx}: "
        if layer.w13.dim() != 3 or layer.w2.dim() != 3:
            raise ValueError(prefix + "weights must be packed [E, K, N/8] tensors")
        experts, hidden, packed_intermediate = layer.w13.shape
        intermediate = packed_intermediate * INT4_PER_INT32 // 2
        if layer.w2.shape != (experts, intermediate, hidden // INT4_PER_INT32):
            raise ValueError(
                prefix + "W13/W2 hidden and intermediate dimensions disagree"
            )
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
            if bias.dtype != torch.float32 or bias.shape != (experts, width):
                raise ValueError(prefix + name + " requires [E, N] FP32 compensation")
        tensors = (
            layer.w2,
            layer.w13_scale,
            layer.w2_scale,
            layer.w13_bias,
            layer.w2_bias,
        )
        if any(t.device != layer.w13.device for t in tensors):
            raise ValueError(
                prefix + "all weights, scales and compensation must share a device"
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dynamic_scales: torch.Tensor,
        group_list: torch.Tensor,
        batch_info: torch.Tensor,
    ) -> torch.Tensor:
        layer_index = batch_info[CAM_LAYER_INDEX : CAM_LAYER_INDEX + 1]
        if self.layer_id_to_slot is not None:
            layer_index = self.layer_id_to_slot.index_select(0, layer_index)
        else:
            # The CANN tiling path checks the storage shape. A slice still
            # carries batch_info's larger storage, so materialize one element.
            layer_index = layer_index.clone()
        activation, scale = self.w13_op(
            hidden_states,
            self.w13,
            self.w13_scale,
            self.w13_bias,
            dynamic_scales,
            group_list,
            layer_index,
            dequant_mode=0 if self.per_channel else 1,
            quant_mode=0,
            group_list_type=1,
        )
        output = self.w2_op(
            [activation],
            self.w2,
            self.w2_bias,
            self.w2_scale,
            layer_index,
            group_list,
            per_token_scale=scale,
            group_list_type=1,
            split_item=3,
            output_dtype=self.output_dtype,
        )[0]
        if self.routed_scaling_factor != 1.0:
            output.mul_(self.routed_scaling_factor)
        return output
