# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 AFD model wrapper.

The wrapper constructs and loads only the model components required by each
AFD role. Shared embedding, normalization, and output components remain
available where required by the model lifecycle. The forward path transfers
hidden states between the Attention and FFN roles through the AFD connector.
"""

from collections.abc import Iterable, Iterator
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config, GlmMoeDsaConfig
from vllm.config import ParallelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import deepseek_v2 as native

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.connectors import (
    AFDExpertRoutingSpec,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

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


_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))


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
    config: _DeepseekAdapterConfig,
    *,
    compute_gate_on_attention: bool,
) -> frozenset[str]:
    """Classify one native checkpoint path by its AFD execution owner."""
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    layer_idx, stage, remainder = layer_path
    if stage == "self_attn":
        return _ATTENTION_ROLE
    if stage != "mlp":
        return _BOTH_ROLES

    if not _is_moe_layer(config, layer_idx):
        return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE

    is_moe_gate = bool(remainder) and remainder[0] == "gate"
    if is_moe_gate and compute_gate_on_attention:
        return _BOTH_ROLES
    return _FFN_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    config: _DeepseekAdapterConfig,
    compute_gate_on_attention: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(
            name,
            config,
            compute_gate_on_attention=compute_gate_on_attention,
        ):
            yield name, loaded_weight


class RemoteFFNProxy(nn.Module):
    """Parameter-free FFN stage executed through the AFD connector."""

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._send_and_receive(hidden_states)

    def _send_and_receive(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteFFNProxy requires AFD forward metadata")
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            **send_kwargs,
        )
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )


class AFDAttentionFusedMoE(RemoteFFNProxy):
    """Parameter-free native-MoE experts proxy for the Attention runtime."""

    def __init__(
        self,
        *,
        layer_idx: int,
        is_internal_router: bool,
        routing_spec: AFDExpertRoutingSpec | None = None,
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        self.is_internal_router = is_internal_router
        self.routing_spec = routing_spec

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        send_kwargs = (
            {}
            if self.is_internal_router
            else {
                "router_logits": router_logits,
                "routing_spec": self.routing_spec,
            }
        )
        return self._send_and_receive(hidden_states, **send_kwargs)

    def update_expert_map(self) -> None:
        """Satisfy the native EPLB model interface without local experts."""


class GateOnlyRemoteMoE(RemoteFFNProxy):
    """Attention-side MoE gate with experts delegated to the FFN role."""

    def __init__(
        self,
        *,
        config: _DeepseekAdapterConfig,
        layer_idx: int,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        self.vllm_config = vllm_config
        self.config = config
        self.top_k = int(config.num_experts_per_tok)
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
            )
        else:
            self.gate.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from afd_plugin.model_executor.models.npu import (
            deepseek_v2_attention_gate,
        )

        topk_weights, topk_ids, router_logits = (
            deepseek_v2_attention_gate.compute_gate_topk(
                gate=self.gate,
                vllm_config=self.vllm_config,
                config=self.config,
                top_k=self.top_k,
                hidden_states=hidden_states,
            )
        )
        return self._send_and_receive(
            hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )


class AFDDeepseekV2RemoteExpertsMoE(native.DeepseekV2MoE):
    """Native DeepSeek MoE forward with parameter-free remote experts."""

    # Patch reason: native DeepseekV2MoE constructs local routed/shared experts.
    # Patch functionality: preserve the native MoE forward contract while
    # constructing only the gate owned by Attention and a parameter-free proxy.
    # Signature: AFD-owned; adds layer_idx and compute_gate_on_attention and omits
    # quant_config because no local expert kernel is constructed.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        *,
        config: _DeepseekAdapterConfig,
        parallel_config: ParallelConfig,
        layer_idx: int,
        prefix: str,
        compute_gate_on_attention: bool,
    ) -> None:
        # ### PATCH START: construct a remote-experts native MoE shell.
        nn.Module.__init__(self)
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        router_dtype = native._get_moe_router_dtype(config)
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
        self.experts = AFDAttentionFusedMoE(
            layer_idx=layer_idx,
            is_internal_router=not compute_gate_on_attention,
            routing_spec=(
                AFDExpertRoutingSpec(
                    router_logits_width=int(config.n_routed_experts),
                    router_logits_dtype=router_dtype,
                )
                if compute_gate_on_attention
                else None
            ),
        )
        # ### PATCH END: construct a remote-experts native MoE shell.


class AFDDeepseekV2DecoderLayer(native.DeepseekV2DecoderLayer):
    """DeepSeek decoder layer with separable Attention and FFN execution."""

    # Patch reason: native DeepSeek constructs both Attention and FFN modules.
    # Patch functionality: construct only the modules owned by the active AFD role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
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
        self.vllm_config = vllm_config
        self.config = config
        self.is_moe_layer = is_moe_layer
        self.compute_gate_on_attention = bool(
            afd_config.compute_gate_on_attention,
        )
        device_type = native.current_platform.device_type
        self.uses_remote_experts = device_type == "cuda" and self.is_moe_layer
        if (
            afd_role == "attention"
            and self.uses_remote_experts
            and parallel_config.enable_eplb
        ):
            raise RuntimeError(
                "CUDA remote experts do not support EPLB on the Attention role",
            )
        if self.compute_gate_on_attention and device_type not in ("cuda", "npu"):
            raise RuntimeError(
                "DeepSeekV2 compute_gate_on_attention requires CUDA or NPU",
            )
        self.top_k = int(config.num_experts_per_tok)

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
            )

            if self.uses_remote_experts:
                self.mlp = AFDDeepseekV2RemoteExpertsMoE(
                    config=config,
                    parallel_config=parallel_config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                    compute_gate_on_attention=self.compute_gate_on_attention,
                )
            elif self.compute_gate_on_attention and self.is_moe_layer:
                self.mlp = GateOnlyRemoteMoE(
                    config=config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                    vllm_config=vllm_config,
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
                # vLLM models bind FusedMoE at module import time. AFD can import
                # native DeepSeek before vLLM-Ascend patches the package factory,
                # so refresh that binding after platform initialization and before
                # constructing the NPU FFN MoE.
                if device_type == "npu":
                    native.FusedMoE = fused_moe.FusedMoE
                self.mlp = native.DeepseekV2MoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    # aiter applies routed_scaling_factor internally
                    apply_routed_scale_to_output=(
                        not native.rocm_aiter_ops.is_fused_moe_enabled()
                    ),
                )
                if self.compute_gate_on_attention and device_type == "cuda":
                    # Keep the native gate parameter and path for loader/model
                    # compatibility, but configure the runner to consume the
                    # router logits transferred from Attention.
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

    def compute_attn_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs: dict[str, torch.Tensor | None] = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, native.DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        topk_weights = None
        topk_ids = None
        router_logits = None
        # NPU-only: Attention-side gate/topk is implemented in the NPU helper.
        if self.compute_gate_on_attention and self.is_moe_layer:
            from afd_plugin.model_executor.models.npu import (
                deepseek_v2_attention_gate,
            )

            topk_weights, topk_ids, router_logits = (
                deepseek_v2_attention_gate.compute_attention_gate_topk(
                    self,
                    hidden_states,
                )
            )
        return hidden_states, residual, topk_weights, topk_ids, router_logits

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        group_list: torch.Tensor | None = None,
        dynamic_scales: torch.Tensor | None = None,
        expand_x_shared: torch.Tensor | None = None,
        dynamic_scales_shared: torch.Tensor | None = None,
        topk_scales: torch.Tensor | None = None,
        group_list_type: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        if self.compute_gate_on_attention and not self.is_moe_layer:
            raise RuntimeError(
                "Dense DeepSeek layers are computed on the Attention side "
                "when compute_gate_on_attention=true",
            )
        if (
            self.compute_gate_on_attention
            and native.current_platform.device_type == "npu"
        ):
            if group_list is None:
                raise RuntimeError(
                    "compute_gate_on_attention FFN MoE compute requires group_list",
                )
            # NPU-only: gated MoE FFN compute consumes Attention-side topk payloads.
            from afd_plugin.model_executor.models.npu import (
                deepseek_v2_attention_gate,
            )

            output = deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
                self,
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                expand_x_shared=expand_x_shared,
                dynamic_scales_shared=dynamic_scales_shared,
                topk_scales=topk_scales,
                group_list_type=group_list_type,
            )
            return output
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
        if self.mlp.experts.is_internal_router:
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
    # Upstream: vLLM v0.26.0, vllm/model_executor/models/deepseek_v2.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require AFD activation and avoid native allocation.
        afd_config = parse_afd_config(vllm_config, validate=False)
        if bool(
            getattr(
                vllm_config.parallel_config,
                "use_sequence_parallel_moe",
                False,
            ),
        ):
            raise RuntimeError(
                "AFD DeepSeek does not support sequence-parallel MoE",
            )
        torch.nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.afd_config = afd_config
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
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END

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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        if self.afd_config.connector == AFD_ASYNC_CONNECTOR:
            from afd_plugin.model_executor.models.npu import (
                deepseek_v2_async_cam_forward,
            )

            return deepseek_v2_async_cam_forward.run_model_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
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
            if isinstance(layer, AFDDeepseekV2DecoderLayer)
            and layer.uses_remote_experts
        )

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

    def _get_llama_4_scaling(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is None:
            return None
        return native._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
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
    ) -> torch.Tensor | AFDF2ATransferPayload:
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

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        return self.model.get_experts_routing_spec(layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                config=self.config,
                compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
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
