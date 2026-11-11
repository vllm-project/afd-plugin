# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Async CAM compute: independent Layered and Mega native calls."""

from __future__ import annotations

from typing import cast

import torch

from afd_plugin.model_executor.ffn.npu.expert_weights import (
    QuantizedExpertWeights,
    validate_expert_weights,
)

CAM_LAYER_INDEX = 2


class AsyncCAMLayeredExecutor:
    """Device-controlled Layered FFN; retains only compute tensors and mappings."""

    def __init__(self, layers: list[QuantizedExpertWeights]) -> None:
        layers = validate_expert_weights(layers)
        self.layer_ids = tuple(layer.layer_idx for layer in layers)
        first = layers[0]
        self.w13 = [layer.w13 for layer in layers]
        self.w2 = [layer.w2 for layer in layers]
        self.w13_scale = [layer.w13_scale for layer in layers]
        self.w2_scale = [layer.w2_scale for layer in layers]
        self.quantization = first.quantization
        self.w13_bias: list[torch.Tensor]
        self.w2_bias: list[torch.Tensor]
        if first.quantization == "W8A8":
            self.w13_bias = []
            # The existing W2 binding requires a bias for each layer. Symmetric
            # W8A8 has none: share one zero INT32 tensor across the entire table.
            # Adding zero preserves the integer accumulator without changing
            # the native binding or allocating compensation per model layer.
            zero_bias = torch.zeros_like(first.w2_scale, dtype=torch.int32)
            self.w2_bias = [zero_bias] * len(layers)
        else:
            self.w13_bias = [cast(torch.Tensor, layer.w13_bias) for layer in layers]
            self.w2_bias = [cast(torch.Tensor, layer.w2_bias) for layer in layers]
        self.per_channel = first.per_channel
        # W4A8 applies the model limit; W8A8 keeps its unclipped computation.
        self.swiglu_limit = first.swiglu_limit
        self.applied_swiglu_limit = (
            first.swiglu_limit if first.quantization == "W4A8" else 0.0
        )
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

    def __call__(
        self,
        hidden_states: torch.Tensor,
        dynamic_scales: torch.Tensor,
        group_list: torch.Tensor,
        batch_info: torch.Tensor,
        *,
        retained_tensors: list[torch.Tensor] | None = None,
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
            swiglu_limit=self.applied_swiglu_limit,
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
        if retained_tensors is not None:
            # A graph session owns these capture-time addresses through its
            # last replay. Eager calls retain only the usual stream lifetimes.
            retained_tensors.extend((layer_index, activation, scale, output))
        if self.routed_scaling_factor != 1.0:
            output.mul_(self.routed_scaling_factor)
        return output
