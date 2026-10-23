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
from vllm_ascend.models.deepseek_v4 import model as native

from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
from afd_plugin.model_executor.npu.async_cam_w4a8 import (
    AsyncCAMW4A8Executor,
    W4A8LayerWeights,
)


class RoutedParameterContainer(nn.Module):
    """Factory hook that retains native parameters without a forward runner.

    The native factory supplies execution-only arguments alongside the routed
    module. Accepting them here avoids constructing MoERunner's unused MC2 and
    dispatcher resources in the standalone parameter owner.
    """

    def __init__(
        self, *, routed_experts: nn.Module, router: nn.Module, **execution_args: object
    ) -> None:
        super().__init__()
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
        self.num_redundant_experts = 0
        self.vllm_config = vllm_config
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
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=False,
            n_shared_experts=0,
            hash_indices_table=None,
        )
        layers[key] = layer
        return layer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse the exact V4 raw-name mapping and expert-aware weight_loader.
        # Its self contract is config, model.layers, named_parameters and the
        # redundant-expert count, all supplied by this routed-only owner.
        return native.AscendDeepseekV4ForCausalLM.load_weights(self, weights)

    def finish_layer(self, layer_index: int) -> None:
        layer = self.model.layers[str(layer_index)]
        routed = layer.mlp.experts.routed_experts
        routed.quant_method.process_weights_after_loading(routed)

    def weight_views(self) -> tuple[W4A8LayerWeights, ...]:
        from vllm_ascend.quantization.quant_type import QuantType

        weights = []
        for layer in self.model.layers.values():
            routed = layer.mlp.experts.routed_experts
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
