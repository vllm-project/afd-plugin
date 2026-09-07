# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 AFD model wrapper for the Ascend native implementation.

The Ascend DSV4 implementation is not a DeepSeek V2 layer with different
dimensions.  Its decoder layer owns the DSA attention path and the
hyper-connection (HC) state transitions around both attention and FFN.  This
adapter therefore keeps the native layer forward path on the Attention role
and replaces only the FFN module with the AFD remote proxy.  The FFN role
constructs the native MoE module and exposes it through the runner-facing
``compute_ffn_output`` hook.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.sequence import IntermediateTensors

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.connectors import (
    AFDExpertRoutingSpec,
    AFDF2ATransferPayload,
)
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    get_async_moe_ubatch_metadata_from_forward_context,
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)

try:
    from vllm_ascend.models import deepseek_v4 as native
except ImportError as exc:  # pragma: no cover - only reachable off Ascend.
    raise ImportError(
        "DSV4 AFD support requires the vLLM-Ascend native DSV4 model"
    ) from exc


_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
_BOTH_ROLES = frozenset((_ATTENTION_ROLE, _FFN_ROLE))


def _refresh_ascend_fused_moe() -> None:
    """Bind native DSV4 MoE construction to the Ascend implementation."""
    # vLLM-Ascend applies this replacement during platform initialization,
    # while the DSV4 module keeps a module-level FusedMoE binding.  Refresh it
    # before either AFD role constructs its local model.
    if native.current_platform.device_type == "npu":
        native.FusedMoE = fused_moe.FusedMoE


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


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the AFD owner for a DSV4 checkpoint path.

    DSV4 checkpoints use ``attn``/``ffn`` names while the Ascend runtime
    model exposes ``self_attn``/``mlp``.  The native loader performs that name
    conversion later, so filtering must understand both spellings here.
    """

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    _, stage, remainder = layer_path
    if stage in ("attn", "self_attn"):
        return frozenset((_ATTENTION_ROLE,))
    if stage in ("ffn", "mlp"):
        if remainder and remainder[0] == "gate":
            return _BOTH_ROLES
        return frozenset((_FFN_ROLE,))
    # HC parameters and any future shared layer parameters are required by
    # both role-local model instances.
    return _BOTH_ROLES


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class AFDDeepseekV4AttentionGateRemoteMoE(RemoteFFNProxy):
    """DSV4 gate shell that routes local Attention tokens through Async CAM."""

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        self.top_k = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.n_routed_experts)
        self.renormalize = bool(config.norm_topk_prob)
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.num_expert_group = int(getattr(config, "n_group", 1))
        self.topk_group = int(getattr(config, "topk_group", 1))
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5),
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.precast_fp32_weight = True
        if layer_idx < config.num_hash_layers:
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Select DSV4 experts on Attention and exchange routed work via CAM."""
        from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate

        topk_weights, topk_ids = deepseek_v4_attention_gate.compute_attention_gate_topk(
            self,
            hidden_states,
        )
        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            use_sequence_parallel=get_forward_context().flash_comm_v1_enabled,
        )
        output = self._send_and_receive(
            dispatch_payload.hidden_states,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
        )
        return restore_cam_dispatch_output(output, dispatch_payload.layout)


class AFDDeepseekV4DecoderLayer(native.DeepseekV2DecoderLayer):
    """Role-local DSV4 decoder layer.

    The inherited native ``forward`` is intentionally retained.  On the
    Attention role it executes the full native HC/DSA sequence and the
    ``RemoteFFNProxy`` makes the AFD transfer at exactly the native FFN
    boundary.  The FFN role is connector-driven and does not call this full
    forward method; it invokes ``compute_ffn_output`` instead.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        if is_draft_layer:
            raise ValueError("AFD DSV4 decoder layers do not support draft layers")
        afd_config = parse_afd_config(vllm_config, validate=False)
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        layer_idx = int(prefix.split(sep=".")[-1])

        self.vllm_config = vllm_config
        self.config = config
        self.afd_role = afd_config.role
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.is_moe_layer = True
        self.use_sequence_parallel_moe = False

        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        if afd_config.role == _ATTENTION_ROLE:
            self.self_attn = native.DeepseekV4Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            if self.compute_gate_on_attention:
                self.mlp = AFDDeepseekV4AttentionGateRemoteMoE(
                    config=config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = RemoteFFNProxy(layer_idx=layer_idx)
        elif afd_config.role == _FFN_ROLE:
            self.self_attn = native.PPMissingLayer()
            _refresh_ascend_fused_moe()
            self.mlp = native.DeepseekV4MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=False,
            )
        else:  # pragma: no cover - parse_afd_config validates role values.
            raise ValueError(f"Unsupported AFD role: {afd_config.role!r}")

        self.input_layernorm = native.RMSNorm(
            config.hidden_size,
            eps=self.norm_eps,
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size,
            eps=self.norm_eps,
        )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

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
        **_: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        if not isinstance(self.mlp, native.DeepseekV4MoE):
            raise RuntimeError(
                "DSV4 compute_ffn_output requires the FFN role to own the MoE"
            )
        if self.compute_gate_on_attention:
            if group_list is None:
                raise RuntimeError(
                    "DSV4 Attention-side routing requires CAM group_list on FFN",
                )
            from afd_plugin.model_executor.models.npu import (
                deepseek_v2_attention_gate,
            )

            return deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
                self,
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                expand_x_shared=expand_x_shared,
                dynamic_scales_shared=dynamic_scales_shared,
                topk_scales=topk_scales,
                group_list_type=group_list_type,
                # DSV4's Attention-side CANN gate folds routed scaling into
                # topk_weights, which CAM applies during combine-recv.
                routed_scale_applied_in_topk=True,
            )
        return self.mlp(hidden_states)


@native.support_torch_compile
class AFDDeepseekV4Model(native.DeepseekV4Model):
    """DSV4 model with role-local Attention/FFN allocations."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        afd_config = parse_afd_config(vllm_config, validate=False)
        if (
            afd_config.connector == AFD_ASYNC_CONNECTOR
            and not afd_config.compute_gate_on_attention
        ):
            raise ValueError(
                "DSV4 CAMAsyncAFDConnector requires compute_gate_on_attention=true",
            )
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD DSV4 does not support sequence-parallel MoE")

        _refresh_ascend_fused_moe()
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.afd_config = afd_config
        self.config = config
        self.device = native.current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps

        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32 and afd_config.role == _ATTENTION_ROLE:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None
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
            lambda prefix, **_: AFDDeepseekV4DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self._mtp_hidden_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            hc_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    )
                }
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    # Upstream source: vllm-ascend commit 80d8c194f,
    # DeepseekV4Model.forward.
    # Patch reason: native DSV4 serializes Attention and FFN within each layer,
    # while Async CAM needs a model-owned two-stage schedule at the FFN boundary.
    # Patch functionality: delegate planned Async CAM execution to the isolated
    # DSV4 pipeline while preserving native forward for unsplit requests.
    # The native forward is large, so the non-AFD path stays exact through
    # inheritance rather than copying that implementation into this adapter.
    # Signature: matches upstream; no added parameters.
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ### PATCH START: DSV4 model-owned Async CAM pipeline
        async_moe_metadata = get_async_moe_ubatch_metadata_from_forward_context()
        if async_moe_metadata is not None:
            if input_ids is None or inputs_embeds is not None:
                raise NotImplementedError(
                    "DSV4 AFD async MoE ubatching requires token input_ids for "
                    "hash routing and does not support inputs_embeds"
                )
            from afd_plugin.model_executor.models.npu import (
                deepseek_v4_async_cam_forward,
            )

            return deepseek_v4_async_cam_forward.run_async_moe_ubatch_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                async_moe_metadata,
                inputs_embeds,
            )
        # ### PATCH END: DSV4 model-owned Async CAM pipeline
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer.layer_idx
            for layer in self.layers
            if isinstance(layer, AFDDeepseekV4DecoderLayer)
            and isinstance(layer.mlp, native.DeepseekV4MoE)
        )

    def get_experts_routing_spec(
        self,
        layer_idx: int,
    ) -> AFDExpertRoutingSpec:
        layer = self.layers[layer_idx]
        if not isinstance(layer.mlp, native.DeepseekV4MoE):
            raise RuntimeError("DSV4 layer does not own a native MoE")
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
            "DSV4 AFD does not support compute_gate_on_attention yet"
        )


def _run_two_stage_async_moe_schedule(
    layers: list[AFDDeepseekV4DecoderLayer],
    compute_stage_attention: Callable[[AFDDeepseekV4DecoderLayer, int], None],
    send_stage_attention: Callable[[AFDDeepseekV4DecoderLayer, int], None],
    receive_and_complete: Callable[[AFDDeepseekV4DecoderLayer, int], None],
) -> None:
    """Pipeline two AFD-owned stages through Attention and remote FFN.

    Once a send succeeds, any later exception is intentionally fatal: the
    EngineCore propagates model-forward failures and terminates the worker.
    Blindly draining CAM here would be unsafe because the remote receive point
    is unknown after an asynchronous operator failure.
    """

    if not layers:
        return
    compute_stage_attention(layers[0], 0)
    send_stage_attention(layers[0], 0)
    for layer_idx in range(len(layers) - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        compute_stage_attention(current_layer, 1)
        receive_and_complete(current_layer, 0)
        send_stage_attention(current_layer, 1)
        compute_stage_attention(next_layer, 0)
        receive_and_complete(current_layer, 1)
        send_stage_attention(next_layer, 0)
    last_layer = layers[-1]
    compute_stage_attention(last_layer, 1)
    receive_and_complete(last_layer, 0)
    send_stage_attention(last_layer, 1)
    receive_and_complete(last_layer, 1)


class AFDDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """Ascend DSV4 causal LM wrapper for AFD."""

    model_cls = AFDDeepseekV4Model

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
        return super().load_weights(_iter_role_weights(weights, role=self.afd_role))


__all__ = [
    "AFDDeepseekV4DecoderLayer",
    "AFDDeepseekV4ForCausalLM",
    "AFDDeepseekV4Model",
]
