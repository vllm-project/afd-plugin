# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MiniMax-M3 AFD model wrapper for the Ascend native implementation.

The Ascend MiniMax-M3 decoder layer owns block-sparse attention, the dense MLP
or sparse MoE block, and the Gemma residual/norm order.  This adapter keeps the
native layer forward on the Attention role and replaces only the FFN module
with the AFD remote proxy.  The FFN role constructs the native dense MLP or
``MiniMaxM3MoE`` block and exposes it through the runner-facing
``compute_ffn_output`` hook.  Only the active role's large modules are
constructed; the opposite role's module is never allocated.

The adapter requires synchronous ``CAMP2pAFDConnector`` with
``compute_gate_on_attention=false``: the MoE gate stays with the native FFN
block, so Attention sends only post-normalization hidden states.  Multimodal
checkpoints run through ``--language-model-only``; the vision tower is staged
by vLLM and receives no AFD weights.
"""

from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.platforms import current_platform

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy

try:
    from vllm_ascend.models.minimax_m3 import minimax_m3 as native
    from vllm_ascend.models.minimax_m3 import minimax_m3_vl as native_vl
except ImportError as exc:  # pragma: no cover - only reachable off Ascend.
    raise ImportError(
        "MiniMax-M3 AFD support requires the vLLM-Ascend native MiniMax-M3 model"
    ) from exc

_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
_ATTENTION_ROLES = frozenset((_ATTENTION_ROLE,))
_FFN_ROLES = frozenset((_FFN_ROLE,))
_BOTH_ROLES = frozenset((_ATTENTION_ROLE, _FFN_ROLE))
_NO_ROLES: frozenset[str] = frozenset()
# Vision parameters are staged by the native multimodal wrapper on both roles
# but never executed by the text-only AFD path.
_MULTIMODAL_NAME_PARTS = frozenset(
    ("vision_tower", "multi_modal_projector", "patch_merge_mlp"),
)


def _weight_layer_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
    """Return ``(layer index, stage, remainder)`` for a native decoder path."""
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


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify one native MiniMax-M3 checkpoint path by its AFD owner."""
    parts = name.split(".")
    if _MULTIMODAL_NAME_PARTS.intersection(parts):
        return _NO_ROLES
    # Speculative layers are never executed by the AFD split; the native loader
    # would skip them anyway, so do not route them to a role that did not
    # construct them.
    if "mtp" in parts:
        return _NO_ROLES

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    _layer_idx, stage, _remainder = layer_path
    if stage in ("self_attn", "input_layernorm", "post_attention_layernorm"):
        return _ATTENTION_ROLES
    if stage in ("block_sparse_moe", "mlp"):
        return _FFN_ROLES
    raise RuntimeError(f"unclassified MiniMax-M3 checkpoint weight: {name}")


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


def _validate_supported_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Reject MiniMax-M3 modes not covered by the initial Ascend contract."""
    if current_platform.device_type != "npu":
        raise RuntimeError("AFD MiniMax-M3 currently supports Ascend NPU only")
    # The native FFN block owns the MoE gate, so Attention sends only the
    # post-Attention normalized hidden states.
    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD MiniMax-M3 requires compute_gate_on_attention=false",
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError("AFD MiniMax-M3 does not support sequence-parallel MoE")
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD MiniMax-M3 does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("AFD MiniMax-M3 does not support pipeline parallelism")
    if vllm_config.speculative_config is not None:
        raise RuntimeError("AFD MiniMax-M3 does not support speculative decoding")
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD MiniMax-M3 does not support LoRA")


def _validate_text_only(model_config: ModelConfig) -> None:
    """Reject multimodal MiniMax-M3 execution before building the tower path."""
    multimodal_config = model_config.multimodal_config
    if multimodal_config.language_model_only is not True:
        raise ValueError(
            "AFD MiniMax-M3 currently supports text-only execution only; "
            "pass --language-model-only",
        )


class AFDMiniMaxM3DecoderLayer(native.MiniMaxM3DecoderLayer):
    """MiniMax-M3 decoder layer with separable Attention and FFN execution."""

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/minimax_m3/minimax_m3.py::MiniMaxM3DecoderLayer.__init__.
    # Patch reason: the native decoder allocates Attention, normalization, and
    # the dense or sparse FFN block on every rank.
    # Patch functionality: construct only the modules owned by the AFD role and
    # replace the FFN slot with a parameter-free remote proxy on Attention.
    # Signature: matches upstream plus the AFD-owned ``afd_config`` keyword,
    # which is passed by AFDMiniMaxM3Model instead of re-parsing the vLLM
    # config inside every layer.
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        parallel_config: Any,
        cache_config: Any = None,
        quant_config: Any = None,
        *,
        afd_config: AFDConfig,
    ) -> None:
        # ### PATCH START: initialize the role-aware layer without native allocation.
        nn.Module.__init__(self)
        self.afd_role = afd_config.role
        # ### PATCH END: initialize the role-aware layer without native allocation.
        self.hidden_size = config.hidden_size
        max_position_embeddings = native._get_max_position_embeddings(config)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])

        self.layer_idx = layer_idx

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        # ``is_layer_sparse`` means "this layer's MLP is a sparse MoE".
        self.is_layer_sparse = (
            moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True
        )
        # Sequence-parallel MoE moves residual layout across the split boundary
        # and is rejected by AFD validation.
        self.use_sequence_parallel = parallel_config.use_sequence_parallel_moe

        sparse_attention_config = getattr(config, "sparse_attention_config", None)

        if sparse_attention_config is not None:
            is_sparse_attention_layer = layer_idx in native._sparse_attention_layer_ids(
                config
            )
            disable_index_value = (
                sparse_attention_config["sparse_disable_index_value"][layer_idx] == 1
            )
        else:
            is_sparse_attention_layer = False
            disable_index_value = False

        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=native._get_rope_parameters(config),
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            reduce_results=not self.use_sequence_parallel,
        )
        # ### PATCH START: allocate only the active role's execution stage.
        if self.afd_role == _ATTENTION_ROLE:
            if is_sparse_attention_layer:
                self.self_attn = native.MiniMaxM3SparseAttention(
                    **attn_kwargs,
                    sparse_cfg=sparse_attention_config,
                    disable_index_value=disable_index_value,
                )
            else:
                self.self_attn = native.MiniMaxM3Attention(**attn_kwargs)
            # The proxy forwards the post-Attention normalized hidden states and
            # receives the complete native dense MLP or MoE block output.
            remote_ffn = RemoteFFNProxy(layer_idx=layer_idx)
            if self.is_layer_sparse:
                self.block_sparse_moe = remote_ffn
            else:
                self.mlp = remote_ffn
            self.input_layernorm = native.GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = native.GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        elif self.afd_role == _FFN_ROLE:
            self.self_attn = native.PPMissingLayer()
            if self.is_layer_sparse:
                self.block_sparse_moe = native.MiniMaxM3MoE(
                    config=config,
                    quant_config=quant_config,
                    parallel_config=parallel_config,
                    prefix=f"{prefix}.block_sparse_moe",
                    is_sequence_parallel=self.use_sequence_parallel,
                )
            else:
                self.mlp = native.MiniMaxM3MLP(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    intermediate_size=config.dense_intermediate_size,
                    is_sequence_parallel=self.use_sequence_parallel,
                )
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: allocate only the active role's execution stage.

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute this layer's complete native FFN block on the FFN role."""
        if self.afd_role != _FFN_ROLE:
            raise RuntimeError("MiniMax-M3 FFN compute requires the AFD FFN role")
        if self.is_layer_sparse:
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states)


@native.support_torch_compile
class AFDMiniMaxM3Model(native.MiniMaxM3Model):
    """MiniMax-M3 model with role-local Attention/FFN allocations."""

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/minimax_m3/minimax_m3.py::MiniMaxM3Model.__init__.
    # Patch reason: the native model hard-codes MiniMaxM3DecoderLayer in its
    # layer factory.
    # Patch functionality: inject the role-aware decoder layer while keeping
    # the native embedding, PP placeholder, final-norm, and auxiliary-state
    # lifecycle.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize the role-local model.
        afd_config = parse_afd_config(vllm_config, validate=False)
        nn.Module.__init__(self)
        # ### PATCH END: initialize the role-local model.

        config = native._get_text_config(vllm_config)
        text_config = config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # ### PATCH START: retain AFD role metadata.
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        # ### PATCH END: retain AFD role metadata.
        self.config = config
        self._enable_eagle3_aux_hidden_states = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.method == "eagle3"
        )

        self.vocab_size = text_config.vocab_size
        self.num_hidden_layers = text_config.num_hidden_layers
        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        # ### PATCH START: role-aware decoder factory.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            text_config.num_hidden_layers,
            lambda prefix: AFDMiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                parallel_config=vllm_config.parallel_config,
                cache_config=cache_config,
                quant_config=quant_config,
                afd_config=afd_config,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END: role-aware decoder factory.

        if native.get_pp_group().is_last_rank:
            self.norm = native.GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.norm = native.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            native.make_pp_empty_intermediate_tensors(
                self,
                native.make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"],
                    config.hidden_size,
                ),
            )
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer.layer_idx
            for layer in self.layers
            if isinstance(layer, AFDMiniMaxM3DecoderLayer) and layer.is_layer_sparse
        )


class AFDMiniMaxM3SparseForCausalLM(native.MiniMaxM3SparseForCausalLM):
    """MiniMax-M3 causal LM wrapper for AFD Ascend execution."""

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/minimax_m3/minimax_m3.py::
    # MiniMaxM3SparseForCausalLM.__init__.
    # Patch reason: the native constructor hard-codes MiniMaxM3Model and reads
    # MoE metadata from the locally constructed MoE modules.
    # Patch functionality: construct AFDMiniMaxM3Model and report
    # configuration-derived MoE metadata on Attention, where no local experts
    # exist.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: validate and initialize the role-local causal LM.
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_supported_config(vllm_config, afd_config)
        nn.Module.__init__(self)
        # ### PATCH END: validate and initialize the role-local causal LM.
        config = native._get_text_config(vllm_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # ### PATCH START: retain AFD role metadata for filtered loading.
        self.afd_config = afd_config
        self.afd_role = afd_config.role
        # ### PATCH END: retain AFD role metadata for filtered loading.
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        # ### PATCH START: inject the role-aware model.
        self.model = AFDMiniMaxM3Model(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        # ### PATCH END: inject the role-aware model.
        if native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = native.LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = native.PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self._set_moe_parameters()

    def _set_moe_parameters(self) -> None:
        """Report MoE metadata for the modules this role actually owns."""
        if self.afd_role == _FFN_ROLE:
            super()._set_moe_parameters()
            return
        # ### PATCH START: Attention owns no local experts.
        config = self.config
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[nn.Module] = []
        self.num_moe_layers = 0
        self.num_logical_experts = config.num_local_experts
        self.num_physical_experts = config.num_local_experts
        self.num_local_physical_experts = 0
        self.num_routed_experts = config.num_local_experts
        self.num_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        self.num_redundant_experts = 0
        # ### PATCH END: Attention owns no local experts.

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    # Patch reason: the native loader would materialize both execution stages
    # even though each AFD role constructs only one stage's decoder modules.
    # Patch functionality: retain only role-owned checkpoint paths, then use the
    # native loader unchanged for mapping, packing, and loaded-parameter results.
    # Signature: matches upstream; no added parameters.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: filter the checkpoint stream by AFD execution role.
        role_weights = _iter_role_weights(weights, role=self.afd_role)
        # ### PATCH END: filter the checkpoint stream by AFD execution role.
        return super().load_weights(role_weights)


class AFDMiniMaxM3VLModel(native_vl.MiniMaxM3VLModel):
    """MiniMax-M3 VL model with the AFD language model."""

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/minimax_m3/minimax_m3_vl.py::MiniMaxM3VLModel.__init__.
    # Patch reason: the native VL wrapper hard-codes MiniMaxM3SparseForCausalLM.
    # Patch functionality: construct the AFD language model and keep the native
    # vision-tower staging.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # ### PATCH START: initialize the native VL shell without the native LM.
        nn.Module.__init__(self)
        # ### PATCH END: initialize the native VL shell without the native LM.
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise ValueError("MiniMax-M3 VL requires config.vision_config.")

        if isinstance(vision_config, dict):
            vision_config = native_vl.PretrainedConfig.from_dict(vision_config)

        projector_hidden_size = getattr(config, "projector_hidden_size", None)
        self.vision_tower = native_vl.MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=native_vl.maybe_prefix(prefix, "vision_tower"),
        )
        # ### PATCH START: construct the role-aware language model.
        self.language_model = AFDMiniMaxM3SparseForCausalLM(
            vllm_config=vllm_config,
            prefix=native_vl.maybe_prefix(prefix, "language_model"),
        )
        # ### PATCH END: construct the role-aware language model.


class AFDMiniMaxM3SparseForConditionalGeneration(
    native_vl.MiniMaxM3SparseForConditionalGeneration,
):
    """MiniMax-M3 multimodal checkpoint wrapper for text-only AFD execution."""

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/minimax_m3/minimax_m3_vl.py::
    # MiniMaxM3SparseForConditionalGeneration.__init__.
    # Patch reason: the native multimodal shell hard-codes the native VL model.
    # Patch functionality: retain native tower staging but construct the AFD
    # language model. Forward and multimodal interfaces stay native.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: enforce the text-only AFD contract.
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_text_only(vllm_config.model_config)
        nn.Module.__init__(self)
        # ### PATCH END: enforce the text-only AFD contract.
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = (
            self.multimodal_config is not None
            and self.multimodal_config.mm_encoder_tp_mode == "data"
        )
        # ### PATCH START: inject the AFD VL model.
        with (
            self._mark_language_model(
                vllm_config,
                targets=AFDMiniMaxM3SparseForCausalLM,
            ),
            self._mark_tower_model(
                vllm_config,
                {"image", "video"},
                targets=native_vl.MiniMaxVLVisionModel,
            ),
        ):
            self.model = AFDMiniMaxM3VLModel(
                vllm_config=vllm_config,
                prefix=native_vl.maybe_prefix(prefix, "model"),
            )
        # ### PATCH END: inject the AFD VL model.

        self.afd_config = afd_config
        self.afd_role = afd_config.role
        self.vision_tower = self.model.vision_tower
        self.language_model = self.model.language_model
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self._sync_moe_parameters()

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    # Patch reason: native loading allocates every MiniMax-M3 checkpoint path
    # locally, including the vision tower's.
    # Patch functionality: filter checkpoint paths to the owning AFD role.
    # Signature: matches upstream; no added parameters.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: load each checkpoint path only on its owning role.
        role_weights = _iter_role_weights(weights, role=self.afd_role)
        # ### PATCH END: load each checkpoint path only on its owning role.
        return super().load_weights(role_weights)


__all__ = [
    "AFDMiniMaxM3DecoderLayer",
    "AFDMiniMaxM3Model",
    "AFDMiniMaxM3SparseForCausalLM",
    "AFDMiniMaxM3SparseForConditionalGeneration",
    "AFDMiniMaxM3VLModel",
]
