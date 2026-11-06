# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4.1 AFD model wrapper for the Ascend native implementation.

The Ascend DSV4.1 implementation is not a DeepSeek V2 layer with different
dimensions.  Its decoder layer owns the source-shared DSA attention path, the
hyper-connection (HC) state transitions around attention and FFN, and optional
Engram n-gram lookup tables.  This adapter therefore keeps the native layer
forward path on the Attention role and replaces only the FFN module with the
AFD remote proxy.  The FFN role constructs the native ``DeepseekV41MoE`` block
and exposes its weights to the async layered executor. Synchronous CAMP2P
uses the runner-facing ``compute_ffn_output`` hook.

Engram tables, candidate buffers, and decoder replay belong to the Attention
role: only that role executes the backbone forward.  The FFN role declares
``has_engram = False`` and drops every ``engram`` checkpoint path.

The router shell and the CAM exchange are parameter-identical to DeepSeek V4,
so this adapter reuses ``AFDDeepseekV4SharedExperts`` and the V4 Attention-side
routing helpers instead of duplicating them.
"""

from collections.abc import Iterable, Iterator
from copy import copy
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers import fused_moe

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.connectors import AFDExpertRoutingSpec
from afd_plugin.model_executor.models.npu.deepseek_v4 import (
    AFDDeepseekV4AttentionGateRemoteMoE,
    AFDDeepseekV4RemoteMoE,
)

try:
    from vllm_ascend.models.deepseek_v41 import model as native
except ImportError as exc:  # pragma: no cover - only reachable off Ascend.
    raise ImportError(
        "DSV4.1 AFD support requires the vLLM-Ascend native DSV4.1 model"
    ) from exc

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
    from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
    from vllm_ascend.quantization.methods.w4a8.w4a8 import (
        AscendW4A8DynamicFusedMoEMethod,
    )

    from afd_plugin.model_executor.npu.async_cam_layered import LayeredFFNWeights


_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
_ATTENTION_ROLES = frozenset((_ATTENTION_ROLE,))
_FFN_ROLES = frozenset((_FFN_ROLE,))
_BOTH_ROLES = frozenset((_ATTENTION_ROLE, _FFN_ROLE))
_NO_ROLES: frozenset[str] = frozenset()
# Vision and draft tensors are never executed by the text-only AFD split.
_MULTIMODAL_NAME_PREFIXES = ("vision.", "aligner.", "image_")
_DEFERRED_NAME_PREFIXES = ("mtp.",)


def _refresh_ascend_fused_moe() -> None:
    """Bind native DSV4.1 MoE construction to the Ascend implementation."""
    # vLLM-Ascend applies this replacement during platform initialization,
    # while the DSV4.1 module keeps a module-level FusedMoEFactory binding.
    # Refresh it before either AFD role constructs its local model.
    if native.current_platform.device_type == "npu":
        native.FusedMoEFactory = fused_moe.FusedMoEFactory


def _weight_layer_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
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
    attn_owns_gate: bool = True,
    attention_shared_experts: bool = False,
    attention_owns_engram: bool = False,
) -> frozenset[str]:
    """Return the AFD owner for a DeepSeek-V4.1 checkpoint path.

    DSV4.1 checkpoints use ``attn``/``ffn`` names while the Ascend runtime
    model exposes ``self_attn``/``mlp``.  The native loader performs that name
    conversion later, so filtering must understand both spellings here.

    ``attn_owns_gate`` describes whether the Attention role built a router for
    the current configuration.  Handing a role a path it never registered is
    not a harmless no-op: the upstream Ascend loader indexes its parameter dict
    by name without a membership check, so it raises ``KeyError`` instead of
    skipping.

    ``attention_owns_engram`` is False on the FFN role, which never allocates
    the n-gram tables; the FFN role must not receive their checkpoint paths.
    """

    name = name.removeprefix("model.")
    if name.startswith(_MULTIMODAL_NAME_PREFIXES) or name.startswith(
        _DEFERRED_NAME_PREFIXES
    ):
        return _NO_ROLES

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    _, stage, remainder = layer_path
    if stage in ("attn", "self_attn"):
        return _ATTENTION_ROLES
    if stage == "engram":
        return _ATTENTION_ROLES if attention_owns_engram else _NO_ROLES
    if stage in ("ffn", "mlp"):
        if attention_shared_experts and remainder and remainder[0] == "shared_experts":
            return _ATTENTION_ROLES
        if remainder and remainder[0] == "gate":
            # The vision router bias is registered only by the native MoE, so
            # an Attention-side router shell has no slot for it even when it
            # owns the rest of the gate.
            if len(remainder) > 1 and remainder[1] == "bias_vl":
                return _FFN_ROLES
            # Every other router parameter, including the Hash id table,
            # belongs to each role that built a router.  With gate-on-Attention
            # the Attention shell registers ``tid2eid`` for Hash layers and
            # routes from it, while the native FFN MoE registers its own copy.
            # With the gate on FFN the Attention MoE slot is parameter-free.
            return _BOTH_ROLES if attn_owns_gate else _FFN_ROLES
        return _FFN_ROLES
    if stage in (
        "attn_norm",
        "ffn_norm",
        "input_layernorm",
        "post_attention_layernorm",
    ):
        return _BOTH_ROLES
    if stage.startswith("hc_"):
        # Hyper-connection state surrounds both stages and is required by both
        # role-local model instances.
        return _BOTH_ROLES
    raise RuntimeError(f"unclassified DeepSeek-V4.1 checkpoint weight: {name}")


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    attn_owns_gate: bool = True,
    attention_shared_experts: bool = False,
    attention_owns_engram: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
            attention_shared_experts=attention_shared_experts,
            attention_owns_engram=attention_owns_engram,
        ):
            yield name, loaded_weight


def _validate_supported_config(
    vllm_config: VllmConfig,
    afd_config: Any,
    config: Any,
) -> None:
    """Reject DeepSeek-V4.1 modes outside the initial Ascend AFD contract."""
    if native.current_platform.device_type != "npu":
        raise RuntimeError("AFD DeepSeek-V4.1 currently supports Ascend NPU only")
    model_config = cast(ModelConfig, vllm_config.model_config)
    multimodal_config = model_config.multimodal_config
    if (
        multimodal_config is not None
        and multimodal_config.language_model_only is not True
    ):
        raise RuntimeError(
            "AFD DeepSeek-V4.1 currently supports text-only execution only; "
            "pass --language-model-only",
        )
    if afd_config.connector == AFD_ASYNC_CONNECTOR and not (
        afd_config.compute_gate_on_attention
    ):
        raise RuntimeError(
            "DSV4.1 CAMAsyncAFDConnector requires compute_gate_on_attention=true",
        )

    parallel_config = vllm_config.parallel_config
    if parallel_config.use_sequence_parallel_moe and not (
        afd_config.compute_gate_on_attention
    ):
        # With Attention-side routing the CAM payload already carries this
        # rank's token shard.  The native FFN MoE would shard it a second time.
        raise RuntimeError(
            "AFD DeepSeek-V4.1 supports sequence-parallel MoE only with "
            "compute_gate_on_attention=true",
        )
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 does not support pipeline parallelism",
        )
    if parallel_config.prefill_context_parallel_size != 1:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 does not support prefill context parallelism",
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 does not support decode context parallelism",
        )
    if vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 does not support speculative decoding",
        )
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD DeepSeek-V4.1 does not support LoRA")
    if bool(getattr(vllm_config.cache_config, "swa_bounded_replay", False)):
        raise RuntimeError(
            "AFD DeepSeek-V4.1 does not support SWA bounded replay",
        )
    if vllm_config.engram_config is not None and (
        bool(parallel_config.enable_dbo) or bool(parallel_config.use_ubatching)
    ):
        # vLLM states this constraint during engine config creation. AFD binds
        # its DBO connector settings in the worker, after that check, so restate
        # the constraint where the model is actually built.
        raise RuntimeError(
            "DeepSeek-V4.1 Engram does not support DBO or microbatching; "
            "disable --enable-dbo and set --ubatch-size to 0",
        )
    if getattr(config, "num_experts_per_tok", None) is None:
        raise RuntimeError(
            "AFD DeepSeek-V4.1 requires a sparse MoE checkpoint",
        )


class AFDDeepseekV41RemoteMoE(AFDDeepseekV4RemoteMoE):
    """DSV4.1 gate-on-FFN shell.

    The transfer itself is identical to DeepSeek V4: Attention sends the
    activations plus the rank-local token ids that Hash layers route on.  The
    only V4.1 difference is the extra ``already_sequence_parallel`` argument
    the native decoder layer passes to every FFN module.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # The CAM payload carries exactly the shard the decoder produced, so
        # the flag needs no further action here.
        del already_sequence_parallel
        return super().forward(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )


class AFDDeepseekV41AttentionGateRemoteMoE(AFDDeepseekV4AttentionGateRemoteMoE):
    """DSV4.1 Attention-side router shell.

    The router, Hash id table, shared experts, and CAM exchange have the same
    parameter layout and operator contract as DeepSeek V4, so this shell reuses
    the V4 implementation and only widens ``forward`` with the V4.1 decoder's
    ``already_sequence_parallel`` argument.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        del already_sequence_parallel
        return super().forward(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    """Role-local DSV4.1 decoder layer.

    The inherited native ``forward`` is intentionally retained.  On the
    Attention role it executes the full native HC/DSA/Engram sequence and the
    ``RemoteFFNProxy`` makes the AFD transfer at exactly the native FFN
    boundary.  The FFN role is connector-driven and does not call this full
    forward method; async CAM uses the layered executor and synchronous CAMP2P
    invokes ``compute_ffn_output`` instead.
    """

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/deepseek_v41/model.py::DeepseekV41DecoderLayer.__init__.
    # Patch reason: allocate Attention, FFN, and Engram weights only on their
    # AFD role.
    # Patch functionality: keep the target HC/SP layer contract and native
    # forward, replacing the Attention role's MoE with a remote module and the
    # FFN role's attention with a missing layer.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: initialize role-local decoder
        afd_config = parse_afd_config(vllm_config, validate=False)
        nn.Module.__init__(self)
        # ### PATCH END: initialize role-local decoder

        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.prefill_context_parallel_size = (
            parallel_config.prefill_context_parallel_size
        )

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        # ### PATCH START: AFD decoder metadata
        self.vllm_config = vllm_config
        self.config = config
        self.afd_role = afd_config.role
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.is_moe_layer = True
        # ### PATCH END: AFD decoder metadata
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe

        # ### PATCH START: role-local Attention and remote experts
        self.mlp: (
            AFDDeepseekV41AttentionGateRemoteMoE
            | AFDDeepseekV41RemoteMoE
            | native.DeepseekV41MoE
        )
        if afd_config.role == _ATTENTION_ROLE:
            self.self_attn = self.attention_cls(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                reduce_results=not self.use_sequence_parallel_moe,
            )
            if self.compute_gate_on_attention:
                self.mlp = AFDDeepseekV41AttentionGateRemoteMoE(
                    config=config,
                    layer_idx=self.layer_idx,
                    prefix=f"{prefix}.mlp",
                    vllm_config=vllm_config,
                    is_draft_layer=is_draft_layer,
                )
            else:
                self.mlp = AFDDeepseekV41RemoteMoE(layer_idx=layer_idx)
        elif afd_config.role == _FFN_ROLE:
            self.self_attn = native.PPMissingLayer()
            _refresh_ascend_fused_moe()
            moe_config = config
            if afd_config.connector == AFD_ASYNC_CONNECTOR:
                moe_config = copy(config)
                # HF validates this public field as int, but pinned native
                # MoE uses only None to omit shared construction (0 builds a
                # zero-width MLP). Keep the original model config unchanged.
                object.__setattr__(moe_config, "n_shared_experts", None)
            self.mlp = native.DeepseekV41MoE(
                config=moe_config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )
        else:  # pragma: no cover - parse_afd_config validates role values.
            raise ValueError(f"Unsupported AFD role: {afd_config.role!r}")
        # ### PATCH END: role-local Attention and remote experts

        self.input_layernorm = native.RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=self.norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_sequence_parallel = parallel_config.use_sequence_parallel_moe
        # Leave the TP partial sums for the reduce-scatter below. The mHC
        # and MoE paths then stay sharded between attention calls.
        # ### PATCH START: FFN role has no attention projection
        if self.use_sequence_parallel and afd_config.role == _ATTENTION_ROLE:
            self.self_attn.wo_b.reduce_results = False
        # ### PATCH END: FFN role has no attention projection
        # ### PATCH START: Attention-only Engram tables
        has_engram = (
            afd_config.role == _ATTENTION_ROLE
            and native._engram_enabled_for_runtime(config, vllm_config)
        )
        self.engram: native.AscendEngram | None
        if (
            has_engram
            and not is_draft_layer
            and self.layer_idx in config.engram_layer_ids
        ):
            self.engram = native.AscendEngram(config, quant_config, f"{prefix}.engram")
        else:
            self.engram = None
        # ### PATCH END: Attention-only Engram tables

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if not isinstance(self.mlp, native.DeepseekV41MoE):
            raise RuntimeError(
                "DSV4.1 compute_ffn_output requires the FFN role to own the MoE"
            )
        if self.compute_gate_on_attention:
            raise RuntimeError("Async CAM FFN requires the layered executor")
        # Hash MoE layers take explicit token IDs from the CAM transfer.
        return self.mlp(hidden_states, input_ids=get_forward_context().input_ids)


@native.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AFDDeepseekV41Model(native.DeepseekV41Model):
    """DSV4.1 model with role-local Attention/FFN/Engram allocations."""

    fall_back_to_pt_during_load = False
    decoder_layer_cls = AFDDeepseekV41DecoderLayer

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/deepseek_v41/model.py::DeepseekV41Model.__init__.
    # Patch reason: construct role-local decoder layers and keep Engram,
    # candidate buffers, and decoder replay on the executing Attention role.
    # Patch functionality: initialize the target SP/PP/HC/Engram model contract
    # without allocating the native unsplit model through super().__init__.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize role-local model
        afd_config = parse_afd_config(vllm_config, validate=False)
        nn.Module.__init__(self)
        # ### PATCH END: initialize role-local model

        config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        _validate_supported_config(vllm_config, afd_config, config)
        quant_config = vllm_config.quant_config
        # ### PATCH START: AFD model metadata
        self.vllm_config = vllm_config
        self.afd_config = afd_config
        # ### PATCH END: AFD model metadata
        self.config = config
        # ### PATCH START: FFN role never builds Engram tables
        self.has_engram = (
            afd_config.role == _ATTENTION_ROLE
            and native._engram_enabled_for_runtime(config, vllm_config)
        )
        # ### PATCH END: FFN role never builds Engram tables
        self.device = native.current_platform.device_type
        self.use_sequence_parallel_moe = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        self.vocab_size = config.vocab_size
        # ### PATCH START: Attention-owned index buffer
        if afd_config.role == _ATTENTION_ROLE and hasattr(config, "index_topk"):
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
        # ### PATCH END: Attention-owned index buffer
        self.topk_indices_buffer = topk_indices_buffer

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )

        # Decoder layers run in order, so their OTP collectives can reuse the
        # same address-stable scratch storage during profiling and graph replay.
        self._oproj_buffer_pool = native.OProjBufferPool()
        for layer in self.layers:
            if isinstance(layer, AFDDeepseekV41DecoderLayer):
                # ### PATCH START: FFN has no Attention implementation
                if layer.afd_role != _ATTENTION_ROLE:
                    continue
                # ### PATCH END: FFN has no Attention implementation
                impl = layer.self_attn.dsa_attn.dsa_attn.impl
                if not isinstance(impl, native.AscendDSAImpl):
                    continue
                impl._oproj_buffer_pool = self._oproj_buffer_pool
                impl.reserve_oproj_buffers(
                    torch.device(self.device), vllm_config.model_config.dtype
                )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        self.hc_mult = config.hc_mult
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        self._mtp_hidden_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_mult * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
            if native.get_pp_group().is_last_rank and needs_mtp_hidden_states
            else None
        )
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        # ### PATCH START: Attention-only shared attention state
        self.candidate_indices_buffer: torch.Tensor | None = None
        self.candidate_lengths_buffer: torch.Tensor | None = None
        self.topk_lengths_buffer: torch.Tensor | None = None
        self.shared_attention_state: native.DeepseekV41SharedAttentionState | None = (
            None
        )
        if afd_config.role == _ATTENTION_ROLE:
            topology = native.build_layer_plan(self.config)
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            # The buffer device follows the index buffer's owner; fall back to
            # the platform device when the checkpoint declares no index_topk.
            buffer_device = (
                self.topk_indices_buffer.device
                if self.topk_indices_buffer is not None
                else self.device
            )
            candidate_buffer = torch.full(
                (max_tokens, 1, topology.candidate_topk_blocks),
                -1,
                dtype=torch.int32,
                device=buffer_device,
            )
            self.candidate_indices_buffer = candidate_buffer
            candidate_lengths = torch.zeros(
                (max_tokens, 1),
                dtype=torch.int32,
                device=buffer_device,
            )
            topk_lengths = torch.zeros_like(candidate_lengths)
            self.candidate_lengths_buffer = candidate_lengths
            self.topk_lengths_buffer = topk_lengths
            self.shared_attention_state = native.DeepseekV41SharedAttentionState(
                self.topk_indices_buffer,
                candidate_buffer,
                candidate_lengths,
                topk_lengths,
            )
            for layer in self.layers:
                if isinstance(layer, AFDDeepseekV41DecoderLayer):
                    layer.self_attn.shared_state = self.shared_attention_state
        # ### PATCH END: Attention-only shared attention state
        self.engram_root = vllm_config.model_config.model
        # ### PATCH START: Attention-only Engram allocation
        self.engram_weight_root = self.engram_root
        self.engram_dp_shared_memory = False
        self.engram_layout = (
            native.EngramLayout.from_config(config) if self.has_engram else None
        )
        if self.engram_layout is not None:
            # Head buckets use TP x DP x PCP when sharded, or TP alone when host
            # storage is shared across local DP x PCP replicas.
            # Fail on an unreadable checkpoint before the first table exists:
            # the allocation below is per-rank 24-51 GiB, and discovering a
            # missing index/key during weight iteration would mean paying for
            # it first.  `dummy` reads no checkpoint at all.
            if vllm_config.load_config.load_format != "dummy":
                native.preflight_engram_checkpoint(
                    self.engram_weight_root,
                    config.engram_layer_ids,
                    native.AscendParallelEngramEmbedding,
                )
            cpu_offload = native.engram_cpu_offload(vllm_config)
            self.engram_dp_shared_memory = native.resolve_dp_shared_memory(
                bool(
                    vllm_config.engram_config
                    and vllm_config.engram_config.dp_shared_memory
                )
            )
            native_mxfp8 = native.get_current_hardware_profile().supports(
                native.HardwareCapability.ENGRAM_MXFP8
            )
            for slot, (layer_id, rows) in enumerate(
                zip(
                    config.engram_layer_ids,
                    config.engram_num_embeddings,
                    strict=False,
                )
            ):
                head_sizes = tuple(
                    size for order in self.engram_layout.primes[slot] for size in order
                )
                storage_dtype = (
                    torch.bfloat16 if vllm_config.quant_config is None else torch.int8
                )
                if vllm_config.load_config.load_format != "dummy":
                    storage_dtype = native.engram_storage_dtype(
                        self.engram_weight_root, layer_id
                    )
                    if storage_dtype == torch.float8_e4m3fn and not native_mxfp8:
                        storage_dtype = torch.int8
                elif native_mxfp8 and vllm_config.quant_config is not None:
                    storage_dtype = torch.float8_e4m3fn
                embed = native.AscendParallelEngramEmbedding(
                    rows,
                    config.engram_head_dim,
                    head_sizes,
                    slot,
                    storage_dtype=storage_dtype,
                    cpu_offload=cpu_offload,
                    dp_shared_memory=self.engram_dp_shared_memory,
                )
                embed.bind_checkpoint(
                    self.engram_weight_root, f"layers.{layer_id}.engram.embed.weight"
                )
                self.layers[layer_id].engram.embed_tokens = embed
        # ### PATCH END: Attention-only Engram allocation
        self.engram_hash = None
        self._engram_input_buffers = None
        # ExternalEvent pairs for FULL graph capture. V1 keys by the forward
        # context's batch descriptor; V2 preparation runs outside any forward
        # context, so it keys by the padded token count (the FULL graph bucket).
        self._engram_graph_events = {}
        self._engram_prepare_stream = None
        self._engram_capture_stream = None
        self._engram_capture_events = None
        self._engram_overlap_enabled = (
            native.get_ascend_config().multistream_engram_overlap
        )
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        # ### PATCH START: Attention-only Engram hash state
        rotation_path = native.get_rotation_path(vllm_config)
        self.engram_rotated = rotation_path is not None and self.has_engram
        engram_rotation = torch.eye(32, device=self.device)
        if self.has_engram:
            if self.engram_rotated and vllm_config.load_config.load_format != "dummy":
                with torch.device("cpu"):
                    engram_rotation = native.load_engram_rotation_block(
                        self.engram_root, config.hidden_size, rotation_path
                    )
                engram_rotation = engram_rotation.to(device=self.device)
            # Upstream owns the n-gram history; the adapter only hands it the
            # Ascend SWA slot metadata (see engram/hash_state.py).
            swa_cache_layer = self.layers[
                config.engram_layer_ids[0]
            ].self_attn.dsa_attn.swa_cache_layer
            self.engram_hash = native.create_engram_hash_state(
                vllm_config, config, swa_cache_layer
            )
        # ### PATCH END: Attention-only Engram hash state
        self.register_buffer("engram_rotation", engram_rotation, persistent=False)

        self.decoder_replay_start = self.end_layer
        # ### PATCH START: decoder replay stays outside the AFD role split
        # Replay re-executes the late layers inside one custom op, which cannot
        # host the cross-role CAM exchange. Attention runs those layers eagerly.
        self.decoder_replay_layers = None
        self.decoder_replay_layer_name: str | None = None
        # ### PATCH END: decoder replay stays outside the AFD role split

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer.layer_idx
            for layer in self.layers
            if isinstance(layer, AFDDeepseekV41DecoderLayer)
            and isinstance(layer.mlp, native.DeepseekV41MoE)
        )

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        layer = self.layers[layer_idx]
        if not isinstance(layer.mlp, native.DeepseekV41MoE):
            raise RuntimeError("DSV4.1 layer does not own a native MoE")
        gate = layer.mlp.gate
        return AFDExpertRoutingSpec(
            router_logits_width=int(layer.mlp.n_routed_experts),
            router_logits_dtype=gate.out_dtype or gate.weight.dtype,
        )

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DSV4.1 AFD routes experts through Attention or the native FFN MoE"
        )


class AFDDeepseekV41ForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    """Ascend DSV4.1 causal LM wrapper for AFD.

    The native text class builds the backbone, LM head, and MoE metadata
    through ``model_cls``, so this wrapper only overrides the model class and
    adds the role-aware weight filter and runner-facing hooks.
    """

    model_cls = AFDDeepseekV41Model

    # DSV4.1 routes Hash layers by token identity and looks up Engram n-grams by
    # token ids. The FFN role holds neither, so the connector must transport ids
    # and the FFN runner installs them before the FFN compute.
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    # vLLM 5df63a7 model_loader/utils.py invokes this model hook after quant
    # finalize. AFD's CAM GMM path still needs the scale that native MegaMoE
    # drops.
    def process_weights_after_loading(self) -> None:
        # ### PATCH START: retain FFN gate-on-A W8A8 scales after quant finalize
        if (
            native.current_platform.device_type != "npu"
            or self.afd_role != _FFN_ROLE
            or not self.afd_config.compute_gate_on_attention
        ):
            return
        from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
            prepare_attention_gate_w8a8_scales,
        )

        prepare_attention_gate_w8a8_scales(self.model.layers)
        # ### PATCH END: retain FFN gate-on-A W8A8 scales after quant finalize

    def get_async_cam_layered_weights(self) -> "tuple[list[LayeredFFNWeights], str]":
        return _extract_async_cam_layered_weights(self.model.layers)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        return self.model.get_experts_routing_spec(layer_idx)

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Only the gate-on-Attention configuration gives Attention a router; with
        # the gate on FFN its MoE slot is a parameter-free transfer shell.
        role_weights = _iter_role_weights(
            weights,
            role=self.afd_role,
            attn_owns_gate=bool(self.afd_config.compute_gate_on_attention),
            attention_shared_experts=self.afd_config.connector == AFD_ASYNC_CONNECTOR,
            attention_owns_engram=bool(self.model.has_engram),
        )
        loaded = super().load_weights(role_weights)
        return loaded


def _extract_async_cam_layered_weights(
    layers: Iterable[AFDDeepseekV41DecoderLayer],
) -> "tuple[list[LayeredFFNWeights], str]":
    """Expose loaded DeepSeek V4.1 routed-expert weights to the layered executor."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_ascend.quantization.quant_type import QuantType

    from afd_plugin.model_executor.npu.async_cam_layered import LayeredFFNWeights

    layers = [layer for layer in layers if layer.is_moe_layer]
    # Only the FFN role exposes loaded experts; Ascend's factory returns its
    # runner while the routed container retains parameter ownership.
    quant_types = {
        cast("AscendMoERunner", layer.mlp.experts).quant_type for layer in layers
    }
    if quant_types not in ({QuantType.W4A8}, {QuantType.W8A8}):
        return [], f"requires homogeneous W4A8 or W8A8 quantization: {quant_types}"
    is_w4a8 = quant_types == {QuantType.W4A8}
    weights = []
    for layer in layers:
        moe = cast("native.DeepseekV41MoE", layer.mlp)
        experts = cast("AscendMoERunner", moe.experts)
        owner = cast("AscendRoutedExperts", experts.routed_experts)
        if owner.dynamic_eplb:
            raise ValueError(
                f"layered GMM layer {layer.layer_idx}: dynamic EPLB is unsupported"
            )
        if experts.activation != MoEActivation.SILU:
            raise ValueError(
                f"layered GMM layer {layer.layer_idx}: requires SiLU activation"
            )
        if experts._shared_experts is not None:
            raise ValueError(
                f"layered GMM layer {layer.layer_idx}: "
                "shared experts must run on Attention"
            )
        required_parameters: tuple[str, ...] = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
        if is_w4a8:
            required_parameters += ("w13_scale_bias", "w2_scale_bias")
        for name in required_parameters:
            if owner._parameters.get(name) is None:
                raise ValueError(
                    f"layered GMM layer {layer.layer_idx}: missing loaded {name}"
                )
        weights.append(
            LayeredFFNWeights(
                layer_idx=layer.layer_idx,
                # The presence checks above establish these dynamically
                # registered parameters as loaded tensors on the routed owner.
                w13=cast(torch.Tensor, owner.w13_weight),
                w2=cast(torch.Tensor, owner.w2_weight),
                w13_scale=cast(
                    torch.Tensor,
                    owner.w13_weight_scale if is_w4a8 else owner.w13_weight_scale_fp32,
                ),
                w2_scale=cast(torch.Tensor, owner.w2_weight_scale),
                w13_bias=cast(torch.Tensor, owner.w13_scale_bias) if is_w4a8 else None,
                w2_bias=cast(torch.Tensor, owner.w2_scale_bias) if is_w4a8 else None,
                per_channel=not is_w4a8
                or cast(
                    "AscendW4A8DynamicFusedMoEMethod",
                    cast("AscendFusedMoEMethod", owner.quant_method).quant_method,
                ).is_per_channel_weight,
                swiglu_limit=float(cast("float | None", moe.swiglu_limit) or 0.0),
                # DSV4.1 already applies routed scaling in Attention top-k.
                routed_scaling_factor=1.0,
                quantization="W4A8" if is_w4a8 else "W8A8",
            )
        )
    # One layered call shares static geometry, quantization, and scaling
    # across every selectable layer; heterogeneous layers are unsupported.
    if any(weight.signature() != weights[0].signature() for weight in weights):
        return [], "heterogeneous expert geometry, quantization or model semantics"
    return weights, ""


__all__ = [
    "AFDDeepseekV41DecoderLayer",
    "AFDDeepseekV41ForCausalLM",
    "AFDDeepseekV41Model",
]
