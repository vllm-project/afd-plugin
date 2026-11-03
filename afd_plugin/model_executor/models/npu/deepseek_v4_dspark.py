# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role-local DSV4 DSpark models over the pinned Ascend implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm_ascend.models.deepseek_v4 import dspark as native
from vllm_ascend.models.deepseek_v4.dspark import _get_dspark_num_mtp_layers
from vllm_ascend.utils import get_rotation_path

from afd_plugin.config import parse_afd_config
from afd_plugin.model_executor.models.npu.deepseek_v4 import (
    AFDDeepseekV4DecoderLayer,
    _extract_async_cam_layered_weights,
)

if TYPE_CHECKING:
    from afd_plugin.model_executor.npu.async_cam_layered import LayeredFFNWeights


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, role: str
) -> Iterator[tuple[str, torch.Tensor]]:
    """Filter once before the native loader resolves its strict parameter names."""
    for name, weight in weights:
        parts = name.split(".")
        routed = (
            len(parts) > 4
            and parts[0] == "mtp"
            and parts[2] in ("ffn", "mlp")
            and parts[3] in ("experts", "gate")
        )
        expert = routed and parts[3] == "experts"
        if (role == "ffn" and routed) or (role == "attention" and not expert):
            yield name, weight


class AFDDeepseekV4DSparkModel(native.DeepseekV4DSparkModel):
    # Upstream: Ascend bd42eea0, DeepseekV4DSparkModel.__init__.
    # Patch reason: native hard-codes complete draft decoder construction.
    # Patch functionality: role-local layers/embedding and FFN-only allocation.
    # Signature matches upstream. Inherited forward, context-KV and heads remain
    # native. Remove this constructor when upstream exposes a layer factory.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: role-local draft construction
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END: role-local draft construction
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        # ### PATCH START: embedding belongs to Attention
        if self.afd_role == "attention":
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        # ### PATCH END: embedding belongs to Attention
        # ### PATCH START: use AFD draft layers with native checkpoint/KV names
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): AFDDeepseekV4DecoderLayer(
                    vllm_config,
                    config=config,
                    prefix=f"mtp.{idx}",
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )
        # ### PATCH END: use AFD draft layers with native checkpoint/KV names

        # ### PATCH START: FFN keeps only its role-local decoder layers
        if self.afd_role == "ffn":
            return
        # ### PATCH END: FFN keeps only its role-local decoder layers

        self._oproj_buffer_pool = native.OProjBufferPool()
        for layer in self.layers.values():
            impl = layer.self_attn.dsa_attn.dsa_attn.impl
            if not isinstance(impl, native.AscendDSAImpl):
                continue
            impl._oproj_buffer_pool = self._oproj_buffer_pool
            impl.reserve_oproj_buffers(
                torch.device(native.current_platform.device_type),
                cast(torch.dtype, vllm_config.model_config.dtype),
            )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = first_layer.use_sequence_parallel_moe

        self.context_wkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.head_dim] * self.num_dspark_layers,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix="mtp.0.self_attn.wkv",
            disable_tp=True,
        )

        _model_quant_cfg = getattr(config, "quantization_config", None)
        _main_proj_qconfig = (
            vllm_config.quant_config
            if _model_quant_cfg is not None
            and _model_quant_cfg.get("quant_method") == "fp8"
            else None
        )
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_main_proj_qconfig,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(
                prefix,
                f"layers.{last_layer_idx}.markov_head",
            ),
        )

        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head

        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
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
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps


# The native decorator owns __call__ even in eager mode. Reapply its constructor
# initialization after replacing the native role-agnostic constructor.
@support_torch_compile
class AFDDSparkDeepseekV4ForCausalLM(native.DSparkDeepseekV4ForCausalLM):
    """Keep native draft semantics; expose experts to the existing F runner."""

    afd_requires_input_ids = False

    # Upstream: Ascend bd42eea0, DSparkDeepseekV4ForCausalLM.__init__.
    # Patch reason: native hard-codes its inner model and allocates a head on F.
    # Patch functionality: substitute the role-local model and Attention head.
    # Signature matches upstream; native loading/remapping/forward are delegated.
    # Remove when native construction exposes model_cls and a role-local head.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: construct only this role's draft modules
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        self.rotation_path = (
            get_rotation_path(vllm_config)
            if vllm_config.quant_config is not None
            else None
        )
        self.model = AFDDeepseekV4DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self.afd_role == "attention"
            else native.PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()
        # ### PATCH END: construct only this role's draft modules

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(_iter_role_weights(weights, role=self.afd_role))
        native.logger.info(
            "AFD DSpark role=%s remote MoE layer IDs=%s loaded parameters=%d",
            self.afd_role,
            self.get_experts_layer_indices(),
            len(loaded),
        )
        return loaded

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.model.layers)

    def get_async_cam_layered_weights(self) -> tuple[list[LayeredFFNWeights], str]:
        return _extract_async_cam_layered_weights(self.model.layers.values())

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        group_list: torch.Tensor | None = None,
        dynamic_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.layers[str(layer_idx)].compute_ffn_output(
            hidden_states, group_list=group_list, dynamic_scales=dynamic_scales
        )
