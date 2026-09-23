# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Kimi K3 AFD wrapper for the native vLLM lifecycle.

Kimi K3 routes its experts through a *latent* MoE: the decoder layer
down-projects the post-attention hidden states into a
``routed_expert_hidden_size`` latent (3584 for Kimi-K3), the routed experts
run entirely in that latent space, and a replicated up-projection maps the
reduced latent back to the hidden size. This adapter splits that contract at
the routed-experts boundary:

- Attention owns the router gate, the latent down-projection, the shared
  experts (they consume the un-transformed hidden states, not the latent),
  the latent RMSNorm, and the up-projection. It sends only the latent plus
  the router logits to the FFN role, which halves the transferred activation
  volume versus the hidden states.
- FFN owns the routed experts and returns the TP-reduced routed latent; the
  native ``KimiRoutedOutputTransform`` on Attention then applies
  norm -> up-proj -> shared-expert add, matching the native latent-MoE math.
"""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.models.kimi_k3.nvidia import model as native
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from afd_plugin.config import AFDConfig, parse_optional_afd_config
from afd_plugin.connectors import AFDExpertRoutingSpec
from afd_plugin.model_executor.models.deepseek_v2 import AFDAttentionFusedMoE

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))
_NO_ROLES: frozenset[str] = frozenset()

# Latent-MoE stages that stay on the Attention role inside block_sparse_moe.
_ATTENTION_MOE_STAGES = frozenset(
    (
        "routed_expert_down_proj",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "shared_experts",
    )
)
# Residual/norm modules written before the MoE call in the native layer.
_ATTENTION_NORM_STAGES = frozenset(
    (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attention_res_norm",
        "self_attention_res_proj",
        "mlp_res_norm",
        "mlp_res_proj",
    )
)


def _is_moe_layer(config: KimiLinearConfig, layer_idx: int) -> bool:
    """Mirror the native KimiDecoderLayer MoE-layer predicate."""
    return (
        config.is_moe
        and config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def _require_kimi_k3_afd_config(
    vllm_config: VllmConfig,
) -> tuple[KimiLinearConfig, AFDConfig]:
    """Fail closed on AFD, model, and topology combinations outside this
    adapter, and return ``(text config, afd config)``."""
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    if afd_config is None:
        raise RuntimeError("AFD Kimi K3 requires AFD activation")
    if not afd_config.compute_gate_on_attention:
        raise ValueError(
            "AFD Kimi K3 supports compute_gate_on_attention=true only: the "
            "router gate must run on Attention because only the latent is "
            "transferred to FFN",
        )
    model_config: ModelConfig = vllm_config.model_config
    config = model_config.hf_text_config
    if config.routed_expert_hidden_size is None:
        raise ValueError(
            "AFD Kimi K3 requires a latent MoE config "
            "(routed_expert_hidden_size); non-latent KimiLinear models are "
            "not supported",
        )
    parallel_config = vllm_config.parallel_config
    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        raise ValueError("AFD Kimi K3 does not support the deep_gemm_mega_moe backend")
    if vllm_config.speculative_config is not None:
        raise ValueError("AFD Kimi K3 does not support speculative decoding")
    if vllm_config.lora_config is not None:
        raise ValueError("AFD Kimi K3 does not support LoRA")
    if parallel_config.enable_eplb:
        raise ValueError("AFD Kimi K3 does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("AFD Kimi K3 supports pipeline_parallel_size=1 only")
    # Native enables sequence parallelism for EP + TP>1 + DP>1; the remote
    # latent transfer assumes plain TP attention ranks.
    if (
        parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.data_parallel_size > 1
    ):
        raise ValueError(
            "AFD Kimi K3 does not support sequence-parallel topologies "
            "(EP + TP>1 + DP>1)",
        )
    if not native.current_platform.is_cuda():
        raise ValueError("AFD Kimi K3 currently supports the CUDA platform only")
    return config, afd_config


def _validate_kimi_text_only(model_config: ModelConfig) -> None:
    """Reject multimodal Kimi execution before constructing the visual path."""
    multimodal_config = model_config.multimodal_config
    if multimodal_config.language_model_only is not True:
        raise ValueError(
            "AFD Kimi K3 currently supports text-only execution only; "
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
    config: KimiLinearConfig,
) -> frozenset[str]:
    """Classify one native Kimi K3 checkpoint path by its AFD owner."""
    parts = name.split(".")
    if "vision_tower" in parts or "mm_projector" in parts:
        return _NO_ROLES

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        # Embeddings, output attention-residual, final norm, and lm_head are
        # required only by Attention.
        return _ATTENTION_ROLE

    layer_idx, stage, remainder = layer_path
    if stage == "self_attn" or stage in _ATTENTION_NORM_STAGES:
        return _ATTENTION_ROLE
    if stage == "mlp":
        # Dense layers run on Attention under the gate-on-attention contract.
        return _ATTENTION_ROLE
    if stage != "block_sparse_moe":
        raise RuntimeError(f"unclassified Kimi K3 checkpoint weight: {name}")
    if not _is_moe_layer(config, layer_idx):
        raise RuntimeError(f"unexpected block_sparse_moe weight: {name}")

    moe_stage = remainder[0] if remainder else ""
    if moe_stage == "gate":
        # Attention computes the routing; FFN still constructs the native
        # gate module so the routing correction bias loads through the native
        # path and feeds the FusedMoE grouped-topk router.
        return _BOTH_ROLES
    if moe_stage in _ATTENTION_MOE_STAGES:
        return _ATTENTION_ROLE
    if moe_stage == "experts":
        return _FFN_ROLE
    raise RuntimeError(f"unclassified Kimi K3 checkpoint weight: {name}")


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    config: KimiLinearConfig,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name, config):
            yield name, loaded_weight


def _make_native_gate(config: KimiLinearConfig, prefix: str) -> nn.Module:
    """Build the native fp32 routing gate with its grouped-topk bias."""
    assert config.num_experts is not None
    gate = native.GateLinear(
        input_size=config.hidden_size,
        output_size=config.num_experts,
        bias=False,
        out_dtype=torch.float32,
        prefix=f"{prefix}.gate",
    )
    gate.e_score_correction_bias = nn.Parameter(
        torch.empty(config.num_experts, dtype=torch.float32)
    )
    return gate


def _activation_situ_betas(
    config: KimiLinearConfig,
) -> tuple[float | None, float | None]:
    """SITU activation betas; ``None`` for other activations (native rule)."""
    if config.hidden_act != "situ":
        return None, None
    return config.activation_situ_beta, config.activation_situ_linear_beta


class AFDKimiK3AttentionMoE(native.KimiMoE):  # noqa: N801
    """Attention-side latent MoE with remote routed experts."""

    # Patch reason: native KimiMoE constructs local routed experts and runs
    # the whole latent MoE (gate, down-proj, experts, shared experts, latent
    # norm, up-proj) inside one forward call.
    # Patch functionality: construct the Attention-owned projections and gate
    # exactly as upstream, delegate the routed experts to a parameter-free
    # remote proxy that transfers the latent plus router logits, and apply
    # the native output transform to the received routed latent. The shared
    # experts stay on Attention because they consume the un-transformed
    # hidden states; upstream hands them to the MoE runner, which is not
    # constructed on this role. The inherited native gate/down-proj overlap
    # (``_maybe_overlap_router_and_down_proj``) is preserved, so only the
    # attributes it reads are mirrored here.
    # Signature: AFD-owned; drops use_sequence_parallel and run_gemm_rs,
    # which AFD rejects.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py (KimiMoE)
    # Commit: 2cf0a6915c
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_idx: int = 0,
    ) -> None:
        # ### PATCH START: construct the Attention-owned latent MoE stages.
        nn.Module.__init__(self)
        assert config.routed_expert_hidden_size is not None
        assert config.moe_intermediate_size is not None
        latent_size = config.routed_expert_hidden_size
        self.use_mega_moe = False

        self.gate = _make_native_gate(config, prefix)
        situ_beta, situ_linear_beta = _activation_situ_betas(config)
        self.shared_experts = (
            native.KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.num_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                # Upstream passes reduce_results=False because the MoE runner
                # fuses the shared reduction with the latent all-reduce; that
                # runner is not constructed on Attention, so the shared
                # partials must reduce here over the Attention TP group.
                reduce_results=True,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
            )
            if config.num_shared_experts
            else None
        )
        self.routed_expert_down_proj = native.ReplicatedLinear(
            config.hidden_size,
            latent_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_norm = (
            native.RMSNorm(latent_size, eps=config.rms_norm_eps)
            if config.latent_moe_use_norm
            else None
        )
        self.routed_expert_up_proj = native.ReplicatedLinear(
            latent_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.routed_output_transform = native.KimiRoutedOutputTransform(
            self.routed_expert_norm, self.routed_expert_up_proj
        )
        self._down_proj_stream: torch.cuda.Stream | None = native.aux_stream()
        self._down_proj_events = (torch.cuda.Event(), torch.cuda.Event())

        self.experts = AFDAttentionFusedMoE(
            layer_idx=layer_idx,
            is_internal_router=False,
        )
        # ### PATCH END: construct the Attention-owned latent MoE stages.

    # Patch reason: the native forward runs the routed experts locally inside
    # the MoE runner, which does not exist on the Attention role.
    # Patch functionality: compute the gate and latent down-projection with
    # the inherited native overlap helper, transfer the latent plus router
    # logits through the AFD connector (yielding the DBO ubatch between send
    # and receive), then map the received TP-reduced routed latent back to
    # the hidden size with the native output transform. The math matches the
    # upstream latent-MoE tail: norm(all-reduce(latent)) -> up-proj, plus the
    # shared-expert output.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py (KimiMoE)
    # Commit: 2cf0a6915c
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # ### PATCH START: remote routed experts over the latent boundary.
        routed_hidden_states, router_logits, _topk_ids = (
            self._maybe_overlap_router_and_down_proj(hidden_states)
        )
        routed_latent = self.experts(routed_hidden_states, router_logits)
        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )
        final_hidden_states = self.routed_output_transform(
            routed_latent,
            residual=shared_output,
        )
        # ### PATCH END: remote routed experts over the latent boundary.
        return final_hidden_states.view(num_tokens, hidden_size)


class AFDKimiK3FFNMoE(native.KimiMoE):  # noqa: N801
    """FFN-side latent MoE owning only the routed experts."""

    # Patch reason: native KimiMoE constructs gate, latent projections,
    # shared experts, and the output transform next to the routed experts.
    # On the FFN role only the routed experts execute.
    # Patch functionality: construct the native routed experts (FusedMoE) in
    # the latent space exactly as upstream, keep the native gate module so
    # the routing correction bias loads through the native path and feeds
    # the grouped-topk router, and drop the shared experts and latent output
    # transform, which are Attention-owned. The runner reduces the routed
    # latent across the FFN TP group before it is returned, matching the
    # reduction point of the upstream latent tail.
    # Signature: AFD-owned; drops use_sequence_parallel, run_gemm_rs, and
    # layer_idx, which are unused on this role.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py (KimiMoE)
    # Commit: 2cf0a6915c
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # ### PATCH START: construct the FFN-owned routed experts.
        nn.Module.__init__(self)
        assert config.routed_expert_hidden_size is not None
        assert config.moe_intermediate_size is not None
        self.use_mega_moe = False

        # The gate never executes on FFN; it exists so the native checkpoint
        # path ``block_sparse_moe.gate.*`` resolves and the routing
        # correction bias feeds the FusedMoE grouped-topk router below.
        self.gate = _make_native_gate(config, prefix)
        self.shared_experts = None
        self.routed_expert_down_proj = None
        self.routed_expert_norm = None
        self.routed_expert_up_proj = None
        self.routed_output_transform = None

        tp_size = native.get_tensor_model_parallel_world_size()
        min_per_partition = getattr(config, "min_moe_intermediate_per_partition", 256)
        padded_intermediate_size = config.moe_intermediate_size
        if (
            tp_size > 1
            and not vllm_config.parallel_config.enable_expert_parallel
            and padded_intermediate_size < min_per_partition * tp_size
        ):
            padded_intermediate_size = min_per_partition * tp_size

        situ_beta, situ_linear_beta = _activation_situ_betas(config)
        self.experts = native.FusedMoEFactory(
            # Attention owns the latent projections, the shared experts, and
            # the latent output transform; FFN receives pre-routed latents
            # and returns the reduced routed latent.
            shared_experts=None,
            routed_input_transform=None,
            routed_output_transform=None,
            is_sequence_parallel=False,
            runner_cls=None,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.routed_expert_hidden_size,
            intermediate_size=padded_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            renormalize=config.moe_renormalize,
            quant_config=vllm_config.quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        if padded_intermediate_size != config.moe_intermediate_size:
            for shard_id in ("w13", "w2"):
                weight = getattr(self.experts, f"{shard_id}_weight", None)
                if weight is None:
                    weight = getattr(self.experts, f"{shard_id}_weight_packed", None)
                if weight is not None:
                    weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // tp_size
            )
        # ### PATCH END: construct the FFN-owned routed experts.


class AFDKimiK3DecoderLayer(native.KimiDecoderLayer):  # noqa: N801
    """Native Kimi K3 decoder forward with role-aware construction."""

    # Patch reason: native KimiDecoderLayer constructs attention, MoE/MLP,
    # norms, and attention-residual modules on every rank.
    # Patch functionality: allocate only the modules owned by the active AFD
    # role while keeping the native forward, which calls ``self.mlp(...)`` at
    # the latent-MoE boundary, unchanged.
    # Signature: matches upstream.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py
    # (KimiDecoderLayer), Commit: 2cf0a6915c
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        run_gemm_rs: bool = False,
    ) -> None:
        # ### PATCH START: require an explicit AFD role before allocation.
        _config, afd_config = _require_kimi_k3_afd_config(vllm_config)
        afd_role = afd_config.role

        nn.Module.__init__(self)
        # ### PATCH END

        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_moe_layer = _is_moe_layer(config, self.layer_idx)
        self.use_sequence_parallel = False

        # ### PATCH START: construct only the stage owned by this AFD role.
        if afd_role == "attention":
            if config.is_kda_layer(self.layer_idx):
                kda_config = config.linear_attn_config
                assert kda_config is not None
                # This class also serves standalone Kimi-Linear; only
                # Kimi-K3's full-rank gate uses the private KDA path.
                if kda_config.get("use_full_rank_gate", False):
                    self.self_attn = native.KimiK3DeltaAttention(
                        config,
                        vllm_config,
                        prefix=f"{prefix}.self_attn",
                        run_gemm_rs=run_gemm_rs,
                    )
                    self._self_attn_writes_output = False
                else:
                    self.self_attn = native.KimiLinearGatedDeltaNetAttention(
                        config,
                        vllm_config,
                        prefix=f"{prefix}.self_attn",
                    )
                    self._self_attn_writes_output = True
            else:
                assert config.mla_use_nope, (
                    "Kimi-K3 MLA (MultiHeadLatentAttention) is NoPE-only"
                )
                self.self_attn = native.MultiHeadLatentAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    use_output_gate=bool(config.mla_use_output_gate),
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.self_attn",
                    aux_stream=aux_stream,
                    run_gemm_rs=run_gemm_rs,
                )
                self._self_attn_writes_output = False

            if self.is_moe_layer:
                self.block_sparse_moe = AFDKimiK3AttentionMoE(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.block_sparse_moe",
                    layer_idx=self.layer_idx,
                )
                self.mlp = self.block_sparse_moe
            else:
                self.mlp = native.KimiMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.mlp",
                    activation_situ_beta=config.activation_situ_beta,
                    activation_situ_linear_beta=config.activation_situ_linear_beta,
                )
            self.input_layernorm = native.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.self_attn = native.PPMissingLayer()
            if self.is_moe_layer:
                self.block_sparse_moe = AFDKimiK3FFNMoE(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.block_sparse_moe",
                )
                self.mlp = self.block_sparse_moe
            else:
                # Dense layers run on Attention; the FFN runner only visits
                # expert layers under the gate-on-attention contract.
                self.mlp = native.PPMissingLayer()
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: construct only the stage owned by this AFD role.

        # ### PATCH START: Attention alone owns attention-residual modules.
        attn_res_block_size = config.attn_res_block_size
        self.use_attn_res = attn_res_block_size is not None and afd_role == "attention"
        if self.use_attn_res:
            assert attn_res_block_size is not None
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = self.layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = self.layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = native.cdiv(
                self.layer_idx, self.attn_res_block_size
            )
            self.self_attention_res_norm = native.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = native.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attention_res_proj = native.ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = native.ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )
        # ### PATCH END: Attention alone owns attention-residual modules.

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the native external-router latent experts on the FFN role."""
        if not self.is_moe_layer:
            raise RuntimeError("compute_experts_output requires an AFD Kimi MoE layer")
        if not isinstance(self.block_sparse_moe, AFDKimiK3FFNMoE):
            raise RuntimeError("FFN role does not own the native Kimi experts")
        return self.block_sparse_moe.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


class AFDKimiLinearModel(native.KimiLinearModel):  # noqa: N801
    """Native Kimi model lifecycle with AFD role-aware decoder layers."""

    fall_back_to_pt_during_load = False

    # Patch reason: native KimiLinearModel always creates native decoder
    # layers, the final norm, and attention-residual output modules.
    # Patch functionality: use role-aware layers and restrict embeddings and
    # output-side modules to the Attention role without replacing the native
    # forward or load_weights implementation.
    # Signature: matches upstream.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py
    # (KimiLinearModel), Commit: 2cf0a6915c
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require AFD activation and the supported config.
        config, afd_config = _require_kimi_k3_afd_config(vllm_config)

        nn.Module.__init__(self)
        # ### PATCH END

        self.config = config
        self.afd_config = afd_config
        self.attn_res_block_size: int | None = config.attn_res_block_size
        self.use_attn_res = self.attn_res_block_size is not None
        self.vocab_size = config.vocab_size
        # The inherited native forward gates its sequence-parallel collectives
        # on this flag; AFD rejects SP topologies, so it is always False.
        self.use_sequence_parallel = False

        # ### PATCH START: construct embeddings and the aux attention stream
        # only on the Attention role.
        if afd_config.role == "attention":
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
            aux_stream: torch.cuda.Stream | None = torch.cuda.Stream()
        else:
            self.embed_tokens = native.PPMissingLayer()
            aux_stream = None
        # ### PATCH END

        # ### PATCH START: build role-aware decoder layers.
        # GEMM-RS requires sequence parallelism, which AFD rejects.
        def get_layer(prefix: str) -> AFDKimiK3DecoderLayer:
            return AFDKimiK3DecoderLayer(
                config,
                vllm_config,
                prefix,
                aux_stream=aux_stream,
                run_gemm_rs=False,
            )

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END
        self.num_attn_res_blocks = (
            native.cdiv(self.end_layer, self.attn_res_block_size)
            if self.attn_res_block_size is not None
            else 0
        )

        # ### PATCH START: construct output-side modules only on Attention.
        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.use_attn_res:
                self.output_attn_res_norm = native.RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = native.ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = native.PPMissingLayer()
            if self.use_attn_res:
                self.output_attn_res_norm = native.PPMissingLayer()
                self.output_attn_res_proj = native.PPMissingLayer()
        # ### PATCH END

        world_size = native.get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

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

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer)
            if _is_moe_layer(self.config, layer_idx)
        )

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        """Return the static native-router contract for graph capture."""
        gate = self.layers[layer_idx].block_sparse_moe.gate
        return AFDExpertRoutingSpec(
            router_logits_width=int(gate.weight.shape[0]),
            router_logits_dtype=gate.out_dtype,
        )


class AFDKimiLinearForCausalLM(native.KimiLinearForCausalLM):  # noqa: N801
    """Text-generation shell that owns the role-aware native Kimi model."""

    # Patch reason: the native shell hard-codes KimiLinearModel construction
    # and the lm_head.
    # Patch functionality: construct AFDKimiLinearModel and restrict lm_head
    # ownership to the Attention role while preserving the inherited forward,
    # logits, state, and loader methods.
    # Signature: matches upstream.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py
    # (KimiLinearForCausalLM), Commit: 2cf0a6915c
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: require the supported AFD Kimi K3 contract.
        _config, afd_config = _require_kimi_k3_afd_config(vllm_config)
        self.afd_role = afd_config.role

        nn.Module.__init__(self)
        # ### PATCH END
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        # ### PATCH START: replace the native model with role-aware layers.
        self.model = AFDKimiLinearModel(
            vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model")
        )
        # ### PATCH END
        # ### PATCH START: restrict LM-head ownership to the Attention role.
        if afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        # ### PATCH END
        native.enable_kimi_k3_low_latency_gemm(self, self.model_config.dtype)
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = native.LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

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

    # Patch reason: native loading allocates every Kimi checkpoint path
    # locally.
    # Patch functionality: filter checkpoint paths to the owning AFD role and
    # delegate the filtered iterator to the native loader unchanged.
    # Signature: matches upstream.
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # ### PATCH START: load each checkpoint path only on its owner.
        return super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                config=self.config,
            ),
        )
        # ### PATCH END


class AFDKimiK3ForConditionalGeneration(  # noqa: N801
    native.KimiK3ForConditionalGeneration,
):
    """Kimi K3 checkpoint wrapper for text-only AFD execution."""

    # Patch reason: the native multimodal shell hard-codes the native causal
    # LM as its language model and registers a flash-attention-4 vision
    # warmup.
    # Patch functionality: enforce text-only execution, skip the vision
    # warmup (the vision path never runs), and construct the AFD language
    # model; every other interface, including forward, logits, state, and
    # loading, stays inherited from upstream.
    # Signature: matches upstream.
    # Upstream: vLLM v0.28.0, vllm/models/kimi_k3/nvidia/model.py
    # (KimiK3ForConditionalGeneration), Commit: 2cf0a6915c
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: enforce AFD's supported Kimi K3 configuration.
        _config, afd_config = _require_kimi_k3_afd_config(vllm_config)
        self.afd_role = afd_config.role
        _validate_kimi_text_only(vllm_config.model_config)
        # ### PATCH END

        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.model_config = model_config
        quant_config = vllm_config.quant_config

        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = native.is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = native.current_platform.current_device()

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = native.MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=native.maybe_prefix(prefix, "vision_tower"),
            )
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            self.mm_projector = native.KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=native.maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        # ### PATCH START: construct the AFD role-aware language model.
        # The native shell resolves ``KimiLinearForCausalLM`` through the
        # model registry; the AFD class is not registered there, so build it
        # directly with the text-only config view.
        with self._mark_language_model(vllm_config):
            self.language_model = AFDKimiLinearForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=native.maybe_prefix(prefix, "language_model"),
            )
        # ### PATCH END
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

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

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        return self.language_model.get_experts_routing_spec(layer_idx)

    # Patch reason: native loading allocates every Kimi K3 checkpoint path
    # locally.
    # Patch functionality: filter checkpoint paths to the owning AFD role and
    # delegate the filtered iterator to the native loader unchanged.
    # Signature: matches upstream.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # ### PATCH START: load each checkpoint path only on its owner.
        return super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                config=self.config.text_config,
            ),
        )
        # ### PATCH END


__all__ = [
    "AFDKimiK3AttentionMoE",
    "AFDKimiK3DecoderLayer",
    "AFDKimiK3FFNMoE",
    "AFDKimiK3ForConditionalGeneration",
    "AFDKimiLinearForCausalLM",
    "AFDKimiLinearModel",
]
