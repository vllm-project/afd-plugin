# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA AFD wrapper for the NVIDIA DeepSeek-V4 implementation.

The split is placed immediately around each decoder FFN. Attention retains all
mHC residual-stream state; only the normalized two-dimensional FFN activation
and token-aligned input IDs cross the Attention-to-FFN boundary. The returned
FFN activation is the sole FFN-to-Attention tensor.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.nvidia import model as native

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import (
    AFDA2FTransportSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))
_DEEPSEEK_V4_TRANSPORT_SPEC = AFDA2FTransportSpec()


def _weight_layer_path(name: str) -> tuple[int, str] | None:
    """Extract the decoder layer index and first layer-local path component."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2]
    return None


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify a native DeepSeek-V4 checkpoint path by execution owner."""
    if name in {
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
    }:
        return _ATTENTION_ROLE
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES
    _, stage = layer_path
    if stage == "ffn":
        return _FFN_ROLE
    return _ATTENTION_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only role-owned paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class RemoteDeepseekV4FFN(nn.Module):
    """Parameter-free FFN proxy carrying V4 hash-router token identifiers."""

    def __init__(
        self,
        *,
        layer_idx: int,
        transport_spec: AFDA2FTransportSpec,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.transport_spec = transport_spec

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 remote FFN requires input_ids")
        if input_ids.ndim != 1 or input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "DeepSeek-V4 input_ids must be one-dimensional and token-aligned",
            )

        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteDeepseekV4FFN requires AFD forward metadata")
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
            transport_spec=self.transport_spec,
            input_ids=input_ids.to(dtype=torch.int64),
        )
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )


class AFDDeepseekV4DecoderLayer(native.DeepseekV4DecoderLayer):
    """DeepSeek-V4 decoder layer with an FFN-boundary synchronous split."""

    # Patch reason: native DeepSeek-V4 always constructs Attention and FFN.
    # Patch functionality: allocate only the stage owned by the active AFD role.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        # ### PATCH START: construct a role-local decoder stage.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        layer_idx = int(prefix.rsplit(".", maxsplit=1)[-1])
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # ### PATCH START: replace the remote stage with a parameter-free proxy.
        if afd_config.role == "attention":
            self.attn = native._select_dsv4_attn_cls(vllm_config)(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.ffn = RemoteDeepseekV4FFN(
                layer_idx=layer_idx,
                transport_spec=_DEEPSEEK_V4_TRANSPORT_SPEC,
            )
        elif afd_config.role == "ffn":
            self.attn = native.PPMissingLayer()
            self.ffn = native.DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        else:
            raise ValueError(f"unsupported AFD role {afd_config.role!r}")
        # ### PATCH END

        # ### PATCH START: mHC state and normalization are Attention-owned.
        if afd_config.role == "ffn":
            return
        # ### PATCH END
        self.attn_norm = native.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = native.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

    # Patch reason: native forward directly invokes its locally allocated FFN.
    # Patch functionality: preserve native mHC state locally while the proxy
    # transfers only the two-dimensional FFN activation and input IDs.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ### PATCH START: prohibit accidental execution on the FFN worker.
        if isinstance(self.attn, native.PPMissingLayer):
            raise RuntimeError("DeepSeek-V4 decoder forward is Attention-owned")
        # ### PATCH END
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            if x.dim() == 2:
                assert self.hc_attn_fn_broadcast is not None
                residual, post_mix, res_mix, x = native.mhc_pre_broadcast_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                    fn_broadcast=self.hc_attn_fn_broadcast,
                )
            else:
                residual = x
                post_mix, res_mix, x = native.mhc_pre_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                )
        else:
            residual, post_mix, res_mix, x = native.mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        x = self.attn(positions, x, None)
        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = native.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )

        # ### PATCH START: this call enters the synchronous remote FFN proxy.
        x = self.ffn(x, input_ids)
        # ### PATCH END
        return x, residual, post_mix, res_mix

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute the complete native V4 MoE, including its native router."""
        if not isinstance(self.ffn, native.DeepseekV4MoE):
            raise RuntimeError("DeepSeek-V4 FFN compute is FFN-role only")
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 FFN compute requires input_ids")
        return self.ffn(hidden_states, input_ids)


class AFDDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware DeepSeek-V4 model retaining mHC exclusively on Attention."""

    # Patch reason: native DeepSeek-V4 allocates every decoder stage and stream.
    # Patch functionality: build role-aware layers and Attention-only resources.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: validate the deliberately narrow first release.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        if native.current_platform.device_type != "cuda":
            raise RuntimeError("AFD DeepSeek-V4 supports CUDA only")
        if self.afd_config.connector != "P2pNcclAFDConnector":
            raise RuntimeError(
                "AFD DeepSeek-V4 requires the synchronous P2pNcclAFDConnector",
            )
        if self.afd_config.compute_gate_on_attention:
            raise RuntimeError(
                "AFD DeepSeek-V4 does not support compute_gate_on_attention",
            )
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("AFD DeepSeek-V4 does not support PP")
        if parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD DeepSeek-V4 does not support SP MoE")
        if parallel_config.enable_eplb:
            raise RuntimeError("AFD DeepSeek-V4 does not support EPLB")
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = parallel_config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        # ### PATCH START: MegaMoE has unproven role-local finalization semantics.
        if self.use_mega_moe:
            raise RuntimeError("AFD DeepSeek-V4 does not support MegaMoE")
        # ### PATCH END
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # ### PATCH START: CUDA streams and sparse-index buffers are Attention-owned.
        if self.afd_config.role == "attention":
            aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
            )
        else:
            aux_stream_list = None
            self.topk_indices_buffer = None
        # ### PATCH END

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        # ### PATCH START: use the pinned role-aware layer constructor.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        # ### PATCH START: final mHC state is constructed only on Attention.
        if self.afd_config.role == "attention":
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None
        if self.afd_config.role == "attention" and native.get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None
        # ### PATCH END

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(
            hidden_states,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(int(self.config.num_hidden_layers)))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return native expert mappings only where real experts are owned."""
        if self.afd_config.role == "attention":
            return []
        return super().get_expert_mapping()

    def get_afd_transport_spec(
        self,
        layer_idx: int,
    ) -> AFDA2FTransportSpec:
        if layer_idx < 0 or layer_idx >= int(self.config.num_hidden_layers):
            raise IndexError(f"invalid DeepSeek-V4 layer index {layer_idx}")
        return _DEEPSEEK_V4_TRANSPORT_SPEC

    def finalize_mega_moe_weights(self) -> None:
        """MegaMoE is rejected before allocation, so no finalizer is needed."""

    def finalize_mhc_broadcast_weights(self) -> None:
        """Finalize only the Attention-owned first-layer broadcast matrix."""
        if self.afd_config.role == "ffn":
            return
        if (
            not native.get_pp_group().is_first_rank
            or self.start_layer >= self.end_layer
        ):
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, AFDDeepseekV4DecoderLayer):
            layer.hc_attn_fn_broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )


class AFDDeepseekV4ForCausalLM(native.DeepseekV4ForCausalLM):
    """DeepSeek-V4 causal LM exposing the GPU FFN-runner model contract."""

    model_cls = AFDDeepseekV4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(
            hidden_states,
            layer_idx,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_afd_transport_spec(
        self,
        layer_idx: int,
    ) -> AFDA2FTransportSpec:
        return self.model.get_afd_transport_spec(layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(weights, role=self.afd_role),
        )


__all__ = ["AFDDeepseekV4ForCausalLM"]
