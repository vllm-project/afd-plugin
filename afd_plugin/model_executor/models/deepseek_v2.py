# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 AFD model wrapper.

The wrapper constructs and loads only the model components required by each
AFD role. Shared embedding, normalization, and output components remain
available where required by the model lifecycle. The forward path transfers
hidden states between the Attention and FFN roles through the AFD connector.
"""

from collections.abc import Iterable
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config, GlmMoeDsaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.models import deepseek_v2 as native

from afd_plugin.config import DEEPEP_M2N_CONNECTORS, parse_afd_config
from afd_plugin.connectors import AFDExpertRoutingSpec
from afd_plugin.model_executor.ffn.remote import RemoteFFNProxy, build_expert
from afd_plugin.model_executor.models._weight_utils import (
    ATTENTION_ROLE,
    BOTH_ROLES,
    FFN_ROLE,
    filter_role_weights,
    layer_weight_path,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

_DeepseekAdapterConfig: TypeAlias = (
    DeepseekV2Config | DeepseekV3Config | GlmMoeDsaConfig
)


def _is_moe_layer(config: _DeepseekAdapterConfig, layer_idx: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % moe_layer_freq == 0
    )


def _checkpoint_weight_roles(
    name: str,
    config: _DeepseekAdapterConfig,
    *,
    compute_gate_on_attention: bool,
    attention_shared_experts: bool = False,
) -> frozenset[str]:
    """Classify one native checkpoint path by its AFD execution owner."""
    layer_path = layer_weight_path(name)
    if layer_path is None:
        return BOTH_ROLES

    layer_idx, stage, remainder = layer_path
    if stage == "self_attn":
        return ATTENTION_ROLE
    if stage != "mlp":
        return BOTH_ROLES

    if not _is_moe_layer(config, layer_idx):
        return ATTENTION_ROLE if compute_gate_on_attention else FFN_ROLE

    if remainder and remainder[0] == "shared_experts" and attention_shared_experts:
        return ATTENTION_ROLE
    is_moe_gate = bool(remainder) and remainder[0] == "gate"
    if is_moe_gate and compute_gate_on_attention:
        return BOTH_ROLES
    return FFN_ROLE


class AFDDeepseekV2RemoteExpertsMoE(native.DeepseekV2MoE):
    """Native DeepSeek MoE forward with remote experts and no expert weights."""

    # Patch reason: native DeepseekV2MoE constructs local routed/shared experts
    # through FusedMoEFactory.
    # Patch functionality: keep the native MoE forward. Construct only the
    # gate owned by Attention, and build the experts through FusedMoEFactory
    # with a runner whose experts run on the FFN role and allocate no expert
    # weights. As native, the runner holds the gate and applies it. When the
    # connector sends routes (DeepEP M2N), Attention also builds the shared
    # experts, and with TP > 1 each TP rank sends only its slice.
    # Signature: AFD-owned; takes vllm_config for the factory, adds
    # compute_gate_on_attention and sends_routes, and omits quant_config
    # because no local expert kernel is constructed.
    # Upstream: vLLM v0.32.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def __init__(
        self,
        *,
        config: _DeepseekAdapterConfig,
        vllm_config: VllmConfig,
        prefix: str,
        compute_gate_on_attention: bool,
        sends_routes: bool,
    ) -> None:
        # ### PATCH START: construct a remote-experts native MoE shell.
        nn.Module.__init__(self)
        parallel_config = vllm_config.parallel_config
        # Every Attention TP rank holds the same tokens; sending routes from
        # each would make the FFN run every token TP times.
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe or (
            sends_routes and parallel_config.tensor_parallel_size > 1
        )

        router_dtype = native._get_moe_router_dtype(config)
        self.router_dtype = router_dtype
        # The experts runner holds this same gate, as native does. Assign it
        # before the experts so its parameters keep their checkpoint names
        # (mlp.gate.*).
        if compute_gate_on_attention:
            self.gate = native.GateLinear(
                config.hidden_size,
                config.n_routed_experts,
                out_dtype=router_dtype,
                prefix=f"{prefix}.gate",
            )
            if getattr(config, "topk_method", None) == "noaux_tc":
                self.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32),
                )
            else:
                self.gate.e_score_correction_bias = None
        else:
            self.gate = None

        ep_size = native.get_ep_group().device_group.size()
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = int(config.n_shared_experts)
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_size
        # Shared experts are never fused into this shell's remote experts.
        self.is_fused_shared_expert_enabled = False
        # With routes, the FFN runs only the routed experts, so Attention runs
        # the shared experts; otherwise the FFN runs the whole MoE.
        self.shared_experts = (
            native.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if sends_routes and config.n_shared_experts
            else None
        )
        self.experts = build_expert(
            vllm_config,
            gate=self.gate,
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            e_score_correction_bias=(
                None if self.gate is None else self.gate.e_score_correction_bias
            ),
            shared_expert_prefix=f"{prefix}.shared_experts",
            router_logits_dtype=self.router_dtype,
        )
        # ### PATCH END: construct a remote-experts native MoE shell.


class AFDDeepseekV2DecoderLayer(native.DeepseekV2DecoderLayer):
    """DeepSeek decoder layer with separable Attention and FFN execution."""

    # Patch reason: native DeepSeek constructs both Attention and FFN modules.
    # Patch functionality: construct only the modules owned by the active AFD role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        index_group_builder: native.SparseMLAIndexGroupBuilder | None = None,
    ) -> None:
        # ### PATCH START: require an explicit AFD role before allocation.
        afd_config = parse_afd_config(vllm_config, validate=False)
        afd_role = afd_config.role

        torch.nn.Module.__init__(self)
        # ### PATCH END

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        # TODO(wentao): enable SP MoE with PP after the PP boundary logic can safely
        # send/receive sequence-parallel hidden_states across stages.
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        # ### PATCH START: construct only the stage owned by this AFD role.
        self.is_moe_layer = is_moe_layer
        self.compute_gate_on_attention = bool(
            afd_config.compute_gate_on_attention,
        )
        device_type = native.current_platform.device_type
        if self.compute_gate_on_attention and device_type != "cuda":
            raise RuntimeError(
                "DeepSeekV2 compute_gate_on_attention requires CUDA",
            )

        if afd_role == "attention":
            attn_cls = (
                native.DeepseekAttention
                if use_mha
                else (
                    native.DeepseekV2MLAAttention
                    if model_config.use_mla
                    else native.DeepseekV2Attention
                )
            )
            attn_kwargs = (
                {"index_group_builder": index_group_builder}
                if attn_cls is native.DeepseekV2MLAAttention
                else {}
            )
            self.self_attn = attn_cls(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                kv_lora_rank=kv_lora_rank,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                reduce_results=not self.use_sequence_parallel_moe,
                **attn_kwargs,
            )

            if device_type == "cuda" and self.is_moe_layer:
                self.mlp = AFDDeepseekV2RemoteExpertsMoE(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.mlp",
                    compute_gate_on_attention=self.compute_gate_on_attention,
                    sends_routes=afd_config.connector in DEEPEP_M2N_CONNECTORS,
                )
            elif self.compute_gate_on_attention:
                self.mlp = native.DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = RemoteFFNProxy(layer_idx=layer_idx)

        elif afd_role == "ffn":
            self.self_attn = native.PPMissingLayer()
            if self.compute_gate_on_attention and not self.is_moe_layer:
                self.mlp = native.PPMissingLayer()
            elif self.is_moe_layer:
                # vLLM models bind FusedMoEFactory at module import time. AFD
                # can import native DeepSeek before vLLM-Ascend patches the
                # package factory, so refresh that binding after platform
                # initialization and before constructing the NPU FFN MoE.
                if device_type == "npu":
                    native.FusedMoEFactory = fused_moe.FusedMoEFactory
                moe_config = config
                if afd_config.connector in DEEPEP_M2N_CONNECTORS:
                    moe_config = copy(config)
                    # HF validates this public field as int, but pinned native
                    # MoE uses only None to omit shared construction (0 builds
                    # a zero-width MLP). Apply its sentinel to this private
                    # construction copy; preserve the real model config.
                    object.__setattr__(moe_config, "n_shared_experts", None)
                self.mlp = native.DeepseekV2MoE(
                    config=moe_config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    # aiter applies routed_scaling_factor internally
                    apply_routed_scale_to_output=(
                        not native.rocm_aiter_ops.is_fused_moe_enabled()
                    ),
                )
                if self.compute_gate_on_attention:
                    # Keep the gate weights loadable, but with the runner's
                    # gate cleared it consumes the Attention-side router
                    # logits instead of computing its own (gate is None means
                    # external routing in the v0.32 runner contract).
                    self.mlp.experts.gate = None
            else:
                self.mlp = native.DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
        # ### PATCH END

        self.input_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.compute_gate_on_attention and not self.is_moe_layer:
            raise RuntimeError(
                "Dense DeepSeek layers are computed on the Attention side "
                "when compute_gate_on_attention=true",
            )
        if self.compute_gate_on_attention:
            raise RuntimeError(
                "GPU Attention-side gate must call compute_experts_output",
            )
        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the native external-router runner on the FFN role."""
        if not self.compute_gate_on_attention or not self.is_moe_layer:
            raise RuntimeError(
                "compute_experts_output requires an Attention-side-gate MoE layer",
            )
        if not isinstance(self.mlp, native.DeepseekV2MoE):
            raise RuntimeError("FFN role does not own a native DeepSeek MoE")
        # v0.32 runner contract: gate is None means the runner consumes the
        # router logits transported from Attention (external routing).
        if self.mlp.experts.gate is not None:
            raise RuntimeError("FFN native runner must use external routing")
        return self.mlp.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


@native.support_torch_compile
class AFDDeepseekV2Model(native.DeepseekV2Model):
    """DeepSeek model wrapper that routes Attention outputs through AFD."""

    fall_back_to_pt_during_load = False

    # Patch reason: native DeepSeek always creates native Decoder layers.
    # Patch functionality: create role-aware AFD layers without full allocation.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.34.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require AFD activation and avoid native allocation.
        afd_config = parse_afd_config(vllm_config, validate=False)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise RuntimeError(
                "AFD DeepSeek does not support sequence-parallel MoE",
            )
        torch.nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        # ### PATCH START: allocate the Indexer buffer only on Attention.
        if self.is_v32 and afd_config.role == "attention":
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
        index_group_builder = (
            native.SparseMLAIndexGroupBuilder(
                topk_indices_buffer,
                native.get_sparse_mla_index_group_max_rows(vllm_config),
            )
            if topk_indices_buffer is not None
            else None
        )
        # ### PATCH END

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        # ### PATCH START: use the pinned role-aware DecoderLayer constructor.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV2DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                index_group_builder=index_group_builder,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END

        self.fused_shared_expert_layers = native.get_fused_shared_expert_layers(
            self,
            native.DeepseekV2MoE,
            enable_eplb=vllm_config.parallel_config.enable_eplb,
        )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                self.hidden_size,
            )
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

        # Needed by load_weights
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(
            hidden_states,
            **kwargs,
        )

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

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if isinstance(layer, AFDDeepseekV2DecoderLayer) and layer.is_moe_layer
        )

    def get_routed_experts(self, layer_idx: int) -> "RoutedExperts":
        """FFN only: the vLLM module holding one MoE layer's local experts.

        Connectors that route on Attention run these experts themselves. The
        FFN never applies ``routed_scaling_factor``: Attention's runner scales
        the combined output.
        """
        return self.layers[layer_idx].mlp.experts.routed_experts

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        """Return the static native-router contract for graph capture."""
        layer = self.layers[layer_idx]
        gate = layer.mlp.gate
        return AFDExpertRoutingSpec(
            router_logits_width=int(layer.mlp.n_routed_experts),
            router_logits_dtype=gate.out_dtype or gate.weight.dtype,
        )


class AFDDeepseekV2ForCausalLM(native.DeepseekV2ForCausalLM):
    """DeepSeekV2 causal LM wrapper for AFD execution."""

    model_cls = AFDDeepseekV2Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

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

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_routed_experts(self, layer_idx: int) -> "RoutedExperts":
        return self.model.get_routed_experts(layer_idx)

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        return self.model.get_experts_routing_spec(layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            filter_role_weights(
                weights,
                role=self.afd_role,
                classify=partial(
                    _checkpoint_weight_roles,
                    config=self.config,
                    compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
                    attention_shared_experts=self.afd_config.connector
                    in DEEPEP_M2N_CONNECTORS,
                ),
            )
        )


class AFDDeepseekForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDDeepseekV3ForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDGlmMoeDsaForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


__all__ = [
    "AFDDeepseekForCausalLM",
    "AFDDeepseekV2ForCausalLM",
    "AFDDeepseekV3ForCausalLM",
    "AFDGlmMoeDsaForCausalLM",
]
