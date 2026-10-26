# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime-only routed parameter owner using the pinned native V4 loader.

No native CausalLM/decoder constructor or forward is called. Only routed MoE
parameter modules are constructed; the native checkpoint loader and quantizer
keep ownership of name mapping, expert sharding and packed-weight conversion.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm_ascend.models.deepseek_v4 import model as native

from afd_plugin.ffn_faas.platforms.npu.settings import validate_static_experts
from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription
from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)


class RoutedParameterContainer(nn.Module):
    """Factory hook that retains native parameters without a forward runner.

    The explicit keyword contract matches vLLM ced6857's FusedMoEFactory runner
    hook. Changes fail at construction instead of disappearing in **kwargs.
    No MoERunner, MC2 dispatcher or opposite-role module is constructed.
    """

    def __init__(
        self,
        *,
        layer_name: str,
        moe_config: FusedMoEConfig,
        routed_experts: RoutedExperts,
        router: FusedMoERouter,
        enable_dbo: bool,
        gate: nn.Module | None,
        shared_expert_gate: nn.Module | None,
        shared_experts: nn.Module | None,
        routed_input_transform: nn.Module | None,
        routed_output_transform: nn.Module | None,
        routed_scaling_factor: float,
    ) -> None:
        super().__init__()
        if (
            enable_dbo
            or routed_scaling_factor != 1.0
            or any(
                module is not None
                for module in (
                    gate,
                    shared_expert_gate,
                    shared_experts,
                    routed_input_transform,
                    routed_output_transform,
                )
            )
        ):
            raise ValueError("routed-only ownership cannot accept execution modules")
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.routed_experts = routed_experts
        self.router = router


class V4RoutedWeightOwner(nn.Module):
    """Preserve native parameter paths without constructing opposite-role data."""

    def __init__(
        self, checkpoint: V4CheckpointDescription, vllm_config: VllmConfig
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        parallel = vllm_config.parallel_config
        validate_static_experts(
            enable_eplb=parallel.enable_eplb,
            enable_elastic_ep=parallel.enable_elastic_ep,
            redundant_experts=parallel.eplb_config.num_redundant_experts,
        )
        self.num_redundant_experts = parallel.eplb_config.num_redundant_experts
        self.model = nn.Module()
        self.model.start_layer = 0
        self.model.end_layer = checkpoint.model_layers
        self.model.layers = nn.ModuleDict()

    def create_layer(self, layer_index: int) -> nn.Module:
        if layer_index not in self.checkpoint.layer_indices:
            raise ValueError("V4 FFN weights must belong to a main-model layer")
        layers = self.model.layers
        key = str(layer_index)
        if key in layers:
            raise RuntimeError("V4 FFN layer weights are already constructed")
        config = self.config
        layer = nn.Module()
        layer.layer_idx = layer_index
        layer.is_moe_layer = True
        layer.mlp = nn.Module()
        layer.mlp.swiglu_limit = self.checkpoint.swiglu_limit
        # Same routed parameter factory and prefixes as native DeepseekV4MoE.
        # A already owns the gate, shared experts and the scaled probabilities.
        layer.mlp.experts = FusedMoEFactory(
            runner_cls=RoutedParameterContainer,
            shared_experts=None,
            gate=None,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=self.vllm_config.quant_config,
            prefix=f"model.layers.{layer_index}.mlp.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=1.0,
            swiglu_limit=config.swiglu_limit,
            e_score_correction_bias=None,
            enable_eplb=self.vllm_config.parallel_config.enable_eplb,
            num_redundant_experts=self.num_redundant_experts,
            is_sequence_parallel=False,
            n_shared_experts=0,
            hash_indices_table=None,
        )
        layers[key] = layer
        return layer

    # Upstream: vLLM-Ascend 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6,
    # vllm_ascend/models/deepseek_v4/model.py, AscendDeepseekV4ForCausalLM.
    # Adaptation reason: reuse its expert-aware name mapping/weight_loader
    # without constructing a CausalLM. The receiver contract is hf config,
    # model.start_layer/end_layer/layers, named_parameters and redundant-expert
    # count. Native factory construction supplies RoutedExperts' own mapping.
    # Delegation exception: do not copy the large native loader. Keep this
    # exact-signature adapter until upstream exposes a routed-only loader API;
    # qualify dependency upgrades with real checkpoint loading and numerics.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: use the native loader on the routed-only owner.
        return native.AscendDeepseekV4ForCausalLM.load_weights(self, weights)
        # ### PATCH END: use the native loader on the routed-only owner.

    def finish_layer(self, layer_index: int) -> None:
        routed = self.routed_experts(layer_index)
        routed.quant_method.process_weights_after_loading(routed)

    def routed_experts(self, layer_index: int) -> RoutedExperts:
        """Keep the native V4 module path inside the model adapter."""
        return self.model.layers[str(layer_index)].mlp.experts.routed_experts

    def weight_views(self) -> tuple[W4A8LayerWeights, ...]:
        from vllm_ascend.quantization.quant_type import QuantType

        weights = []
        for layer in self.model.layers.values():
            routed = layer.mlp.experts.routed_experts
            parallel = self.vllm_config.parallel_config
            validate_static_experts(
                enable_eplb=parallel.enable_eplb,
                enable_elastic_ep=parallel.enable_elastic_ep,
                redundant_experts=parallel.eplb_config.num_redundant_experts,
                dynamic_eplb=routed.dynamic_eplb,
            )
            scheme = routed.quant_method.quant_method
            if scheme.quant_type != QuantType.W4A8:
                raise ValueError("V4 FFN requires native W4A8 routed parameters")
            view = W4A8LayerWeights(
                layer_idx=layer.layer_idx,
                w13=routed.w13_weight,
                w2=routed.w2_weight,
                w13_scale=routed.w13_weight_scale,
                w2_scale=routed.w2_weight_scale,
                w13_bias=routed.w13_scale_bias,
                w2_bias=routed.w2_scale_bias,
                per_channel=scheme.is_per_channel_weight,
                swiglu_limit=self.checkpoint.swiglu_limit,
                routed_scaling_factor=1.0,
            )
            AsyncCAMW4A8Executor._validate_layer(view)
            weights.append(view)
        if not weights or any(w.signature() != weights[0].signature() for w in weights):
            raise ValueError("V4 FFN requires homogeneous native W4A8 layers")
        return tuple(weights)
