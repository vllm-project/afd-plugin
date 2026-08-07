# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3.5/3.6 MoE AFD wrapper for the native vLLM lifecycle."""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models import qwen3_5 as native
from vllm.model_executor.models import qwen3_next as next_native

from afd_plugin.config import parse_optional_afd_config
from afd_plugin.connectors import AFDExpertRoutingSpec
from afd_plugin.model_executor.models.deepseek_v2 import AFDAttentionFusedMoE

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLES = frozenset()


def _weight_layer_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
    """Return ``(layer index, stage, remainder)`` for a decoder weight."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2], tuple(parts[marker_idx + 3 :])
    return None


def _checkpoint_weight_roles(
    name: str,
    *,
    compute_gate_on_attention: bool,
) -> frozenset[str]:
    """Classify one native Qwen3.5/3.6 checkpoint path by AFD owner."""
    parts = name.split(".")
    if "visual" in parts or "mtp" in parts:
        return _NO_ROLES

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        # Embeddings, final norm, and lm_head are required only by Attention.
        return _ATTENTION_ROLE

    _layer_idx, stage, remainder = layer_path
    if stage in ("linear_attn", "self_attn"):
        return _ATTENTION_ROLE
    if stage != "mlp":
        return _ATTENTION_ROLE
    if remainder and remainder[0] == "gate":
        return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE
    if remainder and remainder[0] in (
        "experts",
        "shared_expert",
        "shared_expert_gate",
    ):
        return _FFN_ROLE
    raise RuntimeError(f"unclassified Qwen MoE checkpoint weight: {name}")


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str | None,
    compute_gate_on_attention: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only this role's paths."""
    for name, loaded_weight in weights:
        if role is None or role in _checkpoint_weight_roles(
            name,
            compute_gate_on_attention=compute_gate_on_attention,
        ):
            yield name, loaded_weight


class AFDQwen3_5RemoteExpertsMoE(  # noqa: N801
    native.Qwen3NextSparseMoeBlock,
):
    """Native Qwen MoE shell with a parameter-free remote experts runner."""

    # Patch reason: native Qwen3NextSparseMoeBlock allocates routed and shared
    # experts on every rank.
    # Patch functionality: preserve its native forward while keeping a
    # parameter-free experts proxy and, when selected, the Attention router.
    # Signature: AFD-owned; layer_idx is required for correlation metadata.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_next.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
        compute_gate_on_attention: bool,
    ) -> None:
        # ### PATCH START: construct a remote-experts native MoE shell.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support SP MoE")
        if parallel_config.enable_eplb:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support EPLB")

        self.tp_size = next_native.get_tensor_model_parallel_world_size()
        self.ep_group = next_native.get_ep_group().device_group
        self.ep_rank = next_native.get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = int(config.num_experts)
        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )
        self.gate = (
            next_native.ReplicatedLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
            if compute_gate_on_attention
            else None
        )
        self.shared_expert = None
        self.shared_expert_gate = None
        self.experts = AFDAttentionFusedMoE(
            layer_idx=layer_idx,
            is_internal_router=not compute_gate_on_attention,
            routing_spec=(
                AFDExpertRoutingSpec(
                    router_logits_width=self.n_routed_experts,
                    router_logits_dtype=self.gate.weight.dtype,
                )
                if compute_gate_on_attention
                else None
            ),
        )
        # ### PATCH END: construct a remote-experts native MoE shell.


class MissingAttentionStage(nn.Module):
    """Parameter-free FFN-role placeholder for Attention/GDN modules."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("Attention is not constructed on the AFD FFN role")


class AFDQwen3_5DecoderLayer(native.Qwen3_5DecoderLayer):  # noqa: N801
    """Native Qwen decoder forward with role-aware construction."""

    # Patch reason: native Qwen3_5DecoderLayer constructs Attention/GDN and MoE.
    # Patch functionality: allocate only the modules owned by the active role.
    # Signature: matches upstream.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_5.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        # ### PATCH START: require the experts-boundary Qwen AFD contract.
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen DecoderLayer requires AFD activation")
        nn.Module.__init__(self)
        # ### PATCH END

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        if config.model_type != "qwen3_5_moe_text":
            raise RuntimeError(
                f"AFD Qwen adapter requires qwen3_5_moe_text, got {config.model_type}",
            )
        if parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support SP MoE")

        self.layer_type = layer_type
        self.layer_idx = native.extract_layer_index(prefix)
        self.use_attn_reduce_scatter_for_moe = False
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        self.uses_remote_experts = True

        # ### PATCH START: construct only the active execution stage.
        if self.afd_role == "attention":
            if self.layer_type == "linear_attention":
                self.linear_attn = native.QwenGatedDeltaNetAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.linear_attn",
                    gqa_interleaved_layout=False,
                    reduce_results=True,
                )
            elif self.layer_type == "full_attention":
                self.self_attn = native.Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    reduce_results=True,
                )
            else:
                raise ValueError(f"Invalid layer_type {self.layer_type}")
            self.mlp = AFDQwen3_5RemoteExpertsMoE(
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                prefix=f"{prefix}.mlp",
                compute_gate_on_attention=afd_config.compute_gate_on_attention,
            )
            self.input_layernorm = native.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = native.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            if self.layer_type == "linear_attention":
                self.linear_attn = MissingAttentionStage()
            elif self.layer_type == "full_attention":
                self.self_attn = MissingAttentionStage()
            else:
                raise ValueError(f"Invalid layer_type {self.layer_type}")
            self.mlp = native.Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
            if afd_config.compute_gate_on_attention:
                # The native external-router path executes routed and shared
                # experts while consuming Attention-provided router logits.
                self.mlp.experts.gate = None
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: construct only the active execution stage.

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale and self.afd_role == "attention":
            self.attn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
            self.ffn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the native external-router FusedMoE on the FFN role."""
        if self.afd_role != "ffn":
            raise RuntimeError("native Qwen experts are owned by the FFN role")
        if not isinstance(self.mlp, native.Qwen3NextSparseMoeBlock):
            raise RuntimeError("FFN role does not own native Qwen MoE")
        if self.mlp.experts.is_internal_router:
            raise RuntimeError("FFN native runner must use external routing")
        return self.mlp.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the native internal-router MoE on the FFN role."""
        if self.afd_role != "ffn":
            raise RuntimeError("native Qwen MoE is owned by the FFN role")
        if not isinstance(self.mlp, native.Qwen3NextSparseMoeBlock):
            raise RuntimeError("FFN role does not own native Qwen MoE")
        if not self.mlp.experts.is_internal_router:
            raise RuntimeError(
                "Attention-side gate must call compute_experts_output",
            )
        return self.mlp(hidden_states)


@native.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AFDQwen3_5Model(native.Qwen3_5Model):  # noqa: N801
    """Native Qwen model lifecycle with AFD role-aware decoder layers."""

    fall_back_to_pt_during_load = False

    # Patch reason: native Qwen3_5Model always creates native decoder layers.
    # Patch functionality: use role-aware layers without replacing its forward
    # or load_weights implementation.
    # Signature: matches upstream.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_5.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require the minimal experts boundary.
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen model requires AFD activation")
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support SP MoE")
        if vllm_config.parallel_config.enable_eplb:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support EPLB")
        nn.Module.__init__(self)
        self.afd_config = afd_config
        # ### PATCH END

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.num_redundant_experts = 0
        self.vocab_size = config.vocab_size

        if afd_config.role == "attention":
            self.embed_tokens = native.VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        def get_layer(prefix: str) -> AFDQwen3_5DecoderLayer:
            return AFDQwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[native.extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size,
            )
        )
        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = native.PPMissingLayer()
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_experts_output(
            hidden_states,
            router_logits,
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.end_layer))

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        """Return this layer's cross-rank router-logits wire contract."""
        if not self.afd_config.compute_gate_on_attention:
            raise RuntimeError(
                "Qwen router logits are only transferred for Attention-side gate",
            )
        layer = self.layers[layer_idx]
        if not isinstance(layer, AFDQwen3_5DecoderLayer):
            raise RuntimeError(f"layer {layer_idx} is not an AFD Qwen layer")
        if not isinstance(layer.mlp, native.Qwen3NextSparseMoeBlock):
            raise RuntimeError(f"layer {layer_idx} does not own native Qwen MoE")
        gate = layer.mlp.gate
        return AFDExpertRoutingSpec(
            router_logits_width=int(layer.mlp.n_routed_experts),
            router_logits_dtype=gate.weight.dtype,
        )


class AFDQwen3_5MoeForCausalLM(  # noqa: N801
    native.Qwen3_5MoeForCausalLM,
):
    """Text-generation shell that owns the role-aware native Qwen model."""

    # Patch reason: the native shell hard-codes Qwen3_5Model construction.
    # Patch functionality: construct AFDQwen3_5Model while preserving inherited
    # forward, logits, state, and loader methods.
    # Signature: matches upstream.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_5.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen causal LM requires AFD activation")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5/3.6 requires --mamba-cache-mode=align",
            )
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.afd_config = afd_config
        self.model = AFDQwen3_5Model(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )

        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = native.ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=native.maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.compute_experts_output(
            hidden_states,
            layer_idx,
            router_logits,
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        return self.model.get_experts_routing_spec(layer_idx)


class AFDQwen3_5MoeForConditionalGeneration(  # noqa: N801
    native.Qwen3_5MoeForConditionalGeneration,
):
    """Qwen3.5/3.6 checkpoint wrapper for text-only AFD execution."""

    # Patch reason: the native multimodal shell hard-codes the native causal LM.
    # Patch functionality: retain native tower staging but construct the AFD
    # language model. Forward and multimodal interfaces stay native.
    # Signature: matches upstream.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/qwen3_5.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen conditional model requires AFD activation")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = False
        self.afd_config = afd_config
        self.afd_role = afd_config.role

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = native.Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = AFDQwen3_5MoeForCausalLM(
                vllm_config=vllm_config,
                prefix=native.maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.compute_experts_output(
            hidden_states,
            layer_idx,
            router_logits,
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        return self.language_model.get_experts_routing_spec(layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
            ),
        )


__all__ = ["AFDQwen3_5MoeForConditionalGeneration"]
