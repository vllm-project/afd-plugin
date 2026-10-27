# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3.5/3.6 MoE AFD wrapper for the native vLLM lifecycle."""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.models import qwen3_5 as native
from vllm.model_executor.models import qwen3_next as next_native
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import FASTAFD_CONNECTOR, parse_optional_afd_config
from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector
from afd_plugin.model_executor.models.deepseek_v2 import AFDAttentionFusedMoE
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLES: frozenset[str] = frozenset()


def _validate_qwen_text_only(model_config: ModelConfig) -> None:
    """Reject multimodal Qwen execution before constructing the visual path."""
    multimodal_config = model_config.multimodal_config
    if multimodal_config.language_model_only is not True:
        raise ValueError(
            "AFD Qwen3.5/3.6 currently supports text-only execution only; "
            "pass --language-model-only",
        )


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
    """Classify one native Qwen3.5/3.6 checkpoint path by AFD owner.

    With the gate on Attention, Attention also owns the gated shared expert;
    only the routed experts belong to FFN.
    """
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
    if remainder and remainder[0] in ("gate", "shared_expert", "shared_expert_gate"):
        return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE
    if remainder and remainder[0] == "experts":
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


def fastafd_expert_tensor_name(
    layer_idx: int,
    expert: int,
    projection: str,
    kind: str,
) -> str:
    """Checkpoint name of one routed-expert tensor of a Qwen3.5/3.6 checkpoint.

    ``projection`` is ``gate_proj``, ``up_proj`` or ``down_proj``; ``kind`` is
    ``weight`` or ``weight_scale_inv``.
    """
    return (
        f"model.language_model.layers.{layer_idx}.mlp.experts.{expert}."
        f"{projection}.{kind}"
    )


class AFDQwen3_5RemoteExpertsMoE(  # noqa: N801
    native.Qwen3NextSparseMoeBlock,
):
    """Native Qwen MoE shell with a parameter-free remote experts runner."""

    # Patch reason: native Qwen3NextSparseMoeBlock allocates routed and shared
    # experts on every rank.
    # Patch functionality: preserve its native forward while keeping a
    # parameter-free experts proxy with a local FFN router.
    # Signature: AFD-owned; layer_idx is required for correlation metadata.
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/qwen3_next.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
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
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = int(config.num_experts)
        self.is_sequence_parallel = False
        # No local shared-expert module and no fused shared-expert slot.
        self.is_fused_shared_expert_enabled = False
        self.replicate_shared_expert = False
        self.enable_eplb = False
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.gate: nn.Module | None = None
        self.shared_expert: nn.Module | None = None
        self.shared_expert_gate: nn.Module | None = None
        self.experts = AFDAttentionFusedMoE(
            layer_idx=layer_idx,
            external_routing=False,
        )
        # ### PATCH END: construct a remote-experts native MoE shell.


class AFDQwen3_5AttentionRoutedMoE(AFDQwen3_5RemoteExpertsMoE):  # noqa: N801
    """Attention-side Qwen MoE for M2N connectors such as FastAFDAFDConnector.

    The router and the gated shared expert run locally with the native
    modules. Routed experts run on the FFN ranks through the connector's
    dispatch and combine kernels, one pair per layer and microbatch lane; the
    shared expert runs between them.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__(vllm_config=vllm_config, layer_idx=layer_idx, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.layer_idx = layer_idx
        self.top_k = int(config.num_experts_per_tok)
        # Same default as the native Qwen3NextSparseMoeBlock router.
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.gate = next_native.ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = next_native.ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = next_native.Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            expert_gate=self.shared_expert_gate,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        assert self.gate is not None and self.shared_expert is not None
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits,
            self.top_k,
            self.renormalize,
        )
        # Data dependencies order the three steps for torch.compile: the
        # shared expert reads the copy that dispatch returns, and combine
        # takes the shared output, so the shared expert runs while the FFN
        # ranks compute the routed experts.
        local_input = torch.ops.vllm.afd_fastafd_dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        shared_output = self.shared_expert(local_input)
        routed_output = torch.ops.vllm.afd_fastafd_combine(
            shared_output,
            hidden_states,
            topk_ids,
            topk_weights,
        )
        routed_output = maybe_apply_dbo_yield(routed_output, role="attention")
        return routed_output + shared_output


def _fastafd_connector_and_lane() -> tuple[FastAFDAFDConnector, int]:
    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("Attention-routed Qwen MoE needs AFD forward metadata")
    connector = afd_metadata.connector
    if not isinstance(connector, FastAFDAFDConnector):
        raise RuntimeError(
            f"Attention-routed Qwen MoE requires {FASTAFD_CONNECTOR}, got "
            f"{type(connector).__name__}",
        )
    lane = int(getattr(get_forward_context(), "ubatch_idx", afd_metadata.stage_idx))
    return connector, lane


def _fastafd_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Send the routed tokens; return a copy of the input for local work."""
    connector, lane = _fastafd_connector_and_lane()
    connector.dispatch(hidden_states, topk_ids, topk_weights, lane)
    return hidden_states.clone()


def _fastafd_dispatch_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _fastafd_combine(
    local_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Wait for the routed-expert outputs; ``local_output`` only orders it."""
    connector, lane = _fastafd_connector_and_lane()
    return connector.combine(hidden_states, topk_ids, topk_weights, lane)


def _fastafd_combine_fake(
    local_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


for _op_name, _op_func, _fake_impl in (
    ("afd_fastafd_dispatch", _fastafd_dispatch, _fastafd_dispatch_fake),
    ("afd_fastafd_combine", _fastafd_combine, _fastafd_combine_fake),
):
    if not hasattr(torch.ops.vllm, _op_name):
        direct_register_custom_op(
            op_name=_op_name,
            op_func=_op_func,
            mutates_args=[],
            fake_impl=_fake_impl,
        )


class MissingAttentionStage(nn.Module):
    """Parameter-free FFN-role placeholder for Attention/GDN modules."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("Attention is not constructed on the AFD FFN role")


class AFDQwen3_5DecoderLayer(native.Qwen3_5DecoderLayer):  # noqa: N801
    """Native Qwen decoder forward with role-aware construction."""

    # Patch reason: native Qwen3_5DecoderLayer constructs Attention/GDN and MoE.
    # Patch functionality: allocate only the modules owned by the active role.
    # Signature: matches upstream.
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/qwen3_5.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
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
        # ### PATCH END

        nn.Module.__init__(self)

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
        # Native v0.30.0 derives the MoE reduce-scatter mode from the SP
        # configuration; AFD rejects SP MoE above, so this is always False.
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        # ### PATCH START: record the role that owns this decoder's modules.
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        self.uses_remote_experts = True
        # ### PATCH END

        # ### PATCH START: construct only the active execution stage.
        if self.afd_role == "attention":
            if self.layer_type == "linear_attention":
                self.linear_attn = native.QwenGatedDeltaNetAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.linear_attn",
                    gqa_interleaved_layout=False,
                    reduce_results=not self.use_attn_reduce_scatter_for_moe,
                )
            elif self.layer_type == "full_attention":
                self.self_attn = native.Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    reduce_results=not self.use_attn_reduce_scatter_for_moe,
                )
            else:
                raise ValueError(f"Invalid layer_type {self.layer_type}")
            moe_cls = (
                AFDQwen3_5AttentionRoutedMoE
                if afd_config.connector == FASTAFD_CONNECTOR
                else AFDQwen3_5RemoteExpertsMoE
            )
            self.mlp = moe_cls(
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                prefix=f"{prefix}.mlp",
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
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: construct only the active execution stage.

        self.layer_scale = getattr(config, "layer_scale", False)
        # ### PATCH START: Attention alone owns Qwen layer-scale parameters.
        if self.layer_scale and self.afd_role == "attention":
            self.attn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
            self.ffn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size),
            )
        # ### PATCH END

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the native internal-router MoE on the FFN role."""
        if self.afd_role != "ffn":
            raise RuntimeError("native Qwen MoE is owned by the FFN role")
        if not isinstance(self.mlp, native.Qwen3NextSparseMoeBlock):
            raise RuntimeError("FFN role does not own native Qwen MoE")
        # v0.30 runner contract: a held gate routes internally, consuming no
        # transported router logits.
        if self.mlp.experts.gate is None:
            raise RuntimeError("FFN native runner must use its local router")
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
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/qwen3_5.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require the minimal experts boundary.
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen model requires AFD activation")
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support SP MoE")
        if vllm_config.parallel_config.enable_eplb:
            raise RuntimeError("AFD Qwen3.5/3.6 does not support EPLB")
        self.afd_config = afd_config
        # ### PATCH END

        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        # ### PATCH START: remote experts do not participate in EPLB.
        self.num_redundant_experts = 0
        # ### PATCH END
        self.vocab_size = config.vocab_size

        # ### PATCH START: construct embeddings only on the Attention role.
        if afd_config.role == "attention":
            self.embed_tokens = native.VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        # ### PATCH END

        # ### PATCH START: build role-aware decoder layers.
        def get_layer(prefix: str) -> AFDQwen3_5DecoderLayer:
            return AFDQwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[native.extract_layer_index(prefix)],
                prefix=prefix,
            )

        # ### PATCH END

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # Needed by the inherited native loader's fused shared-expert routing;
        # role-local shells never enable the fused path.
        self.is_fused_shared_expert_enabled = (
            native.is_model_fused_shared_expert_compatible(
                self.layers,
                native.Qwen3NextSparseMoeBlock,
                "mlp",
            )
        )
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size,
            )
        )
        # ### PATCH START: construct the final norm only on the Attention role.
        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = native.PPMissingLayer()
        # ### PATCH END
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_layer, self.end_layer))


class AFDQwen3_5MoeForCausalLM(  # noqa: N801
    native.Qwen3_5MoeForCausalLM,
):
    """Text-generation shell that owns the role-aware native Qwen model."""

    # Patch reason: the native shell hard-codes Qwen3_5Model construction.
    # Patch functionality: construct AFDQwen3_5Model while preserving inherited
    # forward, logits, state, and loader methods.
    # Signature: matches upstream.
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/qwen3_5.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require AFD activation for the replacement shell.
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen causal LM requires AFD activation")
        # ### PATCH END

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
        # ### PATCH START: retain the active AFD role for weight ownership.
        self.afd_config = afd_config
        # ### PATCH END

        # ### PATCH START: replace the native model with role-aware construction.
        self.model = AFDQwen3_5Model(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        # ### PATCH END

        # ### PATCH START: restrict LM-head ownership to the Attention role.
        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = native.PPMissingLayer()
        # ### PATCH END
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()


class AFDQwen3_5MoeForConditionalGeneration(  # noqa: N801
    native.Qwen3_5MoeForConditionalGeneration,
):
    """Qwen3.5/3.6 checkpoint wrapper for text-only AFD execution."""

    # Patch reason: the native multimodal shell hard-codes the native causal LM.
    # Patch functionality: retain native tower staging but construct the AFD
    # language model. Forward and multimodal interfaces stay native.
    # Signature: matches upstream.
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/qwen3_5.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # ### PATCH START: enforce AFD's supported Qwen GPU configuration.
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is None:
            raise RuntimeError("AFD Qwen conditional model requires AFD activation")
        if afd_config.compute_gate_on_attention != (
            afd_config.connector == FASTAFD_CONNECTOR
        ):
            raise ValueError(
                "AFD Qwen3.5/3.6 CUDA routes on Attention exactly when the "
                f"connector is {FASTAFD_CONNECTOR}",
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "AFD Qwen3.5/3.6 CUDA does not support speculative decoding",
            )
        if vllm_config.lora_config is not None:
            raise ValueError("AFD Qwen3.5/3.6 CUDA does not support LoRA")
        _validate_qwen_text_only(vllm_config.model_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise ValueError("AFD Qwen3.5/3.6 CUDA does not support SP MoE")
        if parallel_config.enable_eplb:
            raise ValueError("AFD Qwen3.5/3.6 CUDA does not support EPLB")
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "AFD Qwen3.5/3.6 CUDA supports pipeline_parallel_size=1 only",
            )
        # ### PATCH END

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = native.cached_tokenizer_from_config(vllm_config.model_config)

        # Attributes needed by EVS-related functions inherited from Qwen3-VL.
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        # ### PATCH START: retain AFD role metadata for filtered loading.
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        # ### PATCH END

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = native.Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "visual"),
            )
        # ### PATCH START: replace only the native language-model child.
        with self._mark_language_model(vllm_config):
            self.language_model = AFDQwen3_5MoeForCausalLM(
                vllm_config=vllm_config,
                prefix=native.maybe_prefix(prefix, "language_model"),
            )
        # ### PATCH END
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    # Patch reason: native loading allocates every Qwen checkpoint path locally.
    # Patch functionality: filter checkpoint paths to the owning AFD role.
    # Signature: matches upstream.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: load each checkpoint path only on its owning role.
        loaded = super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
            ),
        )
        # ### PATCH END
        return loaded


__all__ = [
    "AFDQwen3_5MoeForConditionalGeneration",
    "fastafd_expert_tensor_name",
]
