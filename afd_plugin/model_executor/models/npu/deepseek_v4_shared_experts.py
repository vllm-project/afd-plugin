# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local shared-expert computation for DSV4 Async CAM."""

import torch
import torch.nn.functional as functional
import torch_npu
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm_ascend.models.deepseek_v4.model import DeepseekV2MLP
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod,
)
from vllm_ascend.utils import shared_expert_dp_enabled


class AFDDeepseekV4SharedExperts(DeepseekV2MLP):
    """Run Ascend's fused W8A8 shared MLP with its native TP/SP layout.

    Async CAM invokes this MLP outside FusedMoE._forward_shared_experts, so it
    must also perform that runner's TP collectives around the fused arithmetic.
    This adapter can be removed when upstream exposes standalone execution.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            reduce_results=reduce_results,
            is_sequence_parallel=is_sequence_parallel,
            prefix=prefix,
        )
        self.is_sequence_parallel = is_sequence_parallel
        self.weights_replicated = shared_expert_dp_enabled()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return self._run_local_mlp(x)

        if self.weights_replicated:
            if self.is_sequence_parallel:
                return self._run_local_mlp(x)
            # Native shared-expert DP splits full tokens across TP ranks,
            # then gathers the disjoint outputs without summing them.
            original_num_tokens = x.shape[0]
            pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
            if pad_size:
                x = functional.pad(x, (0, 0, 0, pad_size))
            x = torch.tensor_split(x, tp_size, dim=0)[get_tp_group().rank_in_group]
            output = self._run_local_mlp(x)
            return tensor_model_parallel_all_gather(output, dim=0)[:original_num_tokens]

        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, dim=0)
            x = x[: get_forward_context().num_tokens]

        output = self._run_local_mlp(x)
        if not self.is_sequence_parallel:
            return tensor_model_parallel_all_reduce(output)

        pad_size = (tp_size - output.shape[0] % tp_size) % tp_size
        if pad_size:
            output = functional.pad(output, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(output, dim=0)

    def _run_local_mlp(self, x: torch.Tensor) -> torch.Tensor:
        for projection in (self.gate_up_proj, self.down_proj):
            method = projection.quant_method
            if not isinstance(method, AscendLinearMethod):
                return super().forward(x)
            scheme = method.quant_method
            if (
                not isinstance(scheme, AscendW8A8DynamicLinearMethod)
                or scheme.act_quant_type != torch.int8
            ):
                return super().forward(x)

        # Match the actual activation, including nondefault sigmoid/up terms.
        clamp_limit, glu_alpha, glu_bias = 0.0, 1.0, 0.0
        if isinstance(self.act_fn, SiluAndMulWithClamp):
            clamp_limit = self.act_fn.swiglu_limit
            glu_alpha = self.act_fn.alpha
            glu_bias = self.act_fn.beta

        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        gate_up = torch_npu.npu_quant_matmul(
            quantized_x,
            self.gate_up_proj.weight,
            self.gate_up_proj.weight_scale,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32,
        )
        quantized_activation, activation_scale = (
            torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=gate_up,
                weight_scale=self.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=clamp_limit,
                glu_alpha=glu_alpha,
                glu_bias=glu_bias,
            )
        )
        return torch_npu.npu_quant_matmul(
            quantized_activation,
            self.down_proj.weight,
            self.down_proj.weight_scale,
            pertoken_scale=activation_scale,
            bias=None,
            output_dtype=x.dtype,
        )
