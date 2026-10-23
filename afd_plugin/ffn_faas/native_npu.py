# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone W4A8 executor using the pinned native Ascend compute operators.

PREPARE retires before the host selects one borrowed layer's parameters.
Native quantization/GMM operations then run on the session stream; FINISH
publishes results and advances the source only after their output copy. This
backend is deliberately nonfused and makes no device-only scheduling claim.
"""

from __future__ import annotations

import math

import torch
import torch_npu

from afd_plugin.ffn_faas.npu import (
    BF16_BYTES,
    CACHE_LINE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WINDOW_MB,
    EXPERT_TILE_ROWS,
    INT32_BYTES,
    MILLISECONDS_PER_SECOND,
    SCRATCH_HEADER_WORDS,
    NativeDeviceError,
    NativeWeightView,
    NPUExecutor,
)
from afd_plugin.ffn_faas.protocol import Mode, Result, SessionSpec
from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)


class NativeW4A8Executor(NPUExecutor):
    """Borrow quantized Model parameters and own only execution resources."""

    WEIGHT_QUANTIZATION = "w4a8_per_channel"

    def __init__(
        self,
        spec: SessionSpec,
        *,
        rank: int,
        device: int,
        host: str,
        port: int,
        weights: tuple[W4A8LayerWeights, ...],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        window_mb: int = DEFAULT_WINDOW_MB,
        startup_timeout: float | None = None,
        auto_initialize: bool = True,
    ) -> None:
        if rank < spec.attention_ranks:
            raise ValueError("native W4A8 compute executor belongs to F ranks")
        if tuple(layer.layer_idx for layer in weights) != spec.layer_indices:
            raise ValueError("native W4A8 weights differ from negotiated layers")
        target = torch.device(f"npu:{device}")
        for layer in weights:
            AsyncCAMW4A8Executor._validate_layer(layer)
            if (
                layer.w13.device != target
                or not layer.per_channel
                or layer.swiglu_limit != spec.swiglu_limit
                or layer.routed_scaling_factor != 1.0
                or layer.output_dtype != torch.bfloat16
            ):
                raise ValueError("native W4A8 weights differ from session semantics")
        if any(layer.signature() != weights[0].signature() for layer in weights):
            raise ValueError("native W4A8 backend requires homogeneous geometry")
        layer_map = torch.full((spec.model_layers,), -1, dtype=torch.int32)
        for compact, layer_id in enumerate(spec.layer_indices):
            layer_map[layer_id] = compact
        anchor = torch.zeros((1,), dtype=torch.bfloat16, device=target)
        # The shared control op never projects with these tiny placeholders.
        # Quantized weights remain owned by Model and are borrowed below.
        super().__init__(
            spec,
            rank=rank,
            device=device,
            host=host,
            port=port,
            weights=NativeWeightView(anchor, anchor, layer_map.to(target)),
            timeout=timeout,
            window_mb=window_mb,
            startup_timeout=startup_timeout,
            auto_initialize=False,
        )
        self._layers = {layer.layer_idx: layer for layer in weights}
        self._maximum_rows = (
            spec.attention_tp * spec.capacity * spec.top_k
            + (EXPERT_TILE_ROWS - 1) * spec.local_experts
        )
        prefix = SCRATCH_HEADER_WORDS + spec.local_experts
        words = self.scratch.view(torch.int32)
        self._expert_prefix = words[prefix : prefix + spec.local_experts + 1]
        metadata_bytes = (
            SCRATCH_HEADER_WORDS
            + 3 * spec.local_experts
            + 1
            + spec.attention_tp
            + self._maximum_rows
        ) * INT32_BYTES
        packed_offset = (
            (metadata_bytes + CACHE_LINE_BYTES - 1)
            // CACHE_LINE_BYTES
            * CACHE_LINE_BYTES
        )
        packed_bytes = self._maximum_rows * spec.hidden_size * BF16_BYTES
        self._packed = (
            self.scratch[packed_offset : packed_offset + packed_bytes]
            .view(torch.bfloat16)
            .view(self._maximum_rows, spec.hidden_size)
        )
        down_offset = (
            packed_offset
            + self._maximum_rows
            * (spec.hidden_size + 3 * spec.intermediate_size)
            * BF16_BYTES
        )
        self._down = (
            self.scratch[down_offset : down_offset + packed_bytes]
            .view(torch.bfloat16)
            .view(self._maximum_rows, spec.hidden_size)
        )
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError(
                "native W4A8 FFN requires the pinned Ascend custom operators"
            )
        self._gmm1 = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
        if auto_initialize:
            self.initialize()

    def _submit_native(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        probabilities: torch.Tensor,
        request: torch.Tensor,
        mode: Mode,
        config: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in (Mode.EXECUTE, Mode.SERVE):
            return super()._submit_native(
                x, expert_ids, probabilities, request, mode, config
            )
        if mode == Mode.SERVE:
            raise ValueError(
                "native W4A8 backend executes one transaction per submission"
            )
        prepared = super()._submit_native(
            x, expert_ids, probabilities, request, Mode.PREPARE, config
        )
        self._untracked_retained += tuple(prepared)
        code, layer_index, rows = torch.ops.afd_ascend.ffn_faas_wait_selected(
            prepared[1], math.ceil(self.timeout * MILLISECONDS_PER_SECOND)
        )
        if Result(code) != Result.OK:
            return prepared
        if layer_index not in self._layers or not 0 <= rows <= self._maximum_rows:
            raise NativeDeviceError("native FFN selected invalid layer or row count")
        if rows:
            layer = self._layers[layer_index]
            # Native W4A8 converts cumulative boundaries to counts before
            # both GMMs. Preserve that exact public-operator contract.
            groups = torch.diff(self._expert_prefix).to(torch.int64)
            # Match pinned native _quant_hidden_states, including its scale
            # rounding. This is not claimed bit-identical to old CAM dispatch.
            quantized, dynamic_scale = torch_npu.npu_dynamic_quant(
                self._packed[:rows], dst_type=torch.int8
            )
            self._untracked_retained += (groups, quantized, dynamic_scale)
            activation, activation_scale = self._gmm1(
                x=quantized,
                weight=[layer.w13],
                weight_scale=[layer.w13_scale],
                x_scale=dynamic_scale,
                group_list=groups,
                weight_assist_matrix=[layer.w13_bias],
                dequant_mode=0,
                group_list_type=1,
                swiglu_limit=layer.swiglu_limit,
            )
            self._untracked_retained += (activation, activation_scale)
            output = torch_npu.npu_grouped_matmul(
                x=[activation],
                weight=[layer.w2],
                scale=[layer.w2_scale],
                bias=[layer.w2_bias],
                per_token_scale=[activation_scale],
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=groups,
                output_dtype=torch.bfloat16,
            )[0]
            self._untracked_retained += (output,)
            self._down[:rows].copy_(output)
        # Zero routed rows still finish the source transaction. No native GMM
        # is called and no uninitialized packed/output storage is consumed.
        return super()._submit_native(
            x, expert_ids, probabilities, request, Mode.FINISH, config
        )

    def close(self, *, abort: bool = False) -> None:
        super().close(abort=abort)
        self._layers = {}
