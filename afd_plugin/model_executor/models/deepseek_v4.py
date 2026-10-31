# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4 AFD model wrapper for the pinned Ascend runtime."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.sequence import IntermediateTensors
from vllm_ascend.models import deepseek_v4 as native
from vllm_ascend.models import deepseek_v4_mtp as native_mtp
from vllm_ascend.utils import is_dspark_config

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDF2ATransferPayload
from afd_plugin.distributed.topology import (
    AFDWindowExpertLayout,
    build_window_expert_layout,
    resolve_role_rank,
)
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDRemoteFFNTransfer,
    RemoteFFNProxy,
)

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLE = frozenset()


def _checkpoint_weight_roles(
    name: str,
    *,
    compute_gate_on_attention: bool,
) -> frozenset[str]:
    """Return the DSV4 AFD role that owns one raw checkpoint key."""
    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return _NO_ROLE

    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        if parts[2] not in {"ffn", "mlp"}:
            return _ATTENTION_ROLE
        if compute_gate_on_attention and len(parts) >= 4 and parts[3] == "gate":
            return _ATTENTION_ROLE
        return _FFN_ROLE
    return _ATTENTION_ROLE


def _mtp_checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the strict AFD owner of one raw DSV4 MTP checkpoint key."""
    normalized = name.removeprefix("model.")
    parts = normalized.split(".")
    if len(parts) < 3 or parts[0] != "mtp" or not parts[1].isdigit():
        return _NO_ROLE
    return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE


def _iter_mtp_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume one MTP checkpoint iterator once for the active AFD role."""
    for name, loaded_weight in weights:
        if role in _mtp_checkpoint_weight_roles(name):
            yield name, loaded_weight


def _uses_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return (
        speculative_config is not None
        and getattr(speculative_config, "method", None) == "mtp"
        and not is_dspark_config(vllm_config)
    )


def _uses_dspark(vllm_config: VllmConfig) -> bool:
    return is_dspark_config(vllm_config)


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    compute_gate_on_attention: bool,
    window_ffn_kind: str | None,
    num_hidden_layers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain the active role's keys."""
    for name, loaded_weight in weights:
        normalized = name.removeprefix("model.")
        parts = normalized.split(".")
        if (
            len(parts) >= 2
            and parts[0] == "layers"
            and parts[1].isdigit()
            and int(parts[1]) >= num_hidden_layers
        ):
            continue
        if role not in _checkpoint_weight_roles(
            name,
            compute_gate_on_attention=compute_gate_on_attention,
        ):
            continue
        if role == "ffn" and window_ffn_kind is not None:
            is_shared = any(
                marker in f".{normalized}"
                for marker in (".ffn.shared_experts.", ".mlp.shared_experts.")
            )
            if window_ffn_kind == "shared" and not is_shared:
                continue
            if window_ffn_kind == "routed" and is_shared:
                continue
        yield name, loaded_weight


def _window_expert_layout(
    vllm_config: VllmConfig,
    role_rank: int,
) -> AFDWindowExpertLayout:
    afd_config = parse_afd_config(vllm_config, validate=False)
    if vllm_config.parallel_config.enable_eplb:
        raise ValueError("DSV4 Window AFD does not support EPLB")
    if bool(getattr(native.get_ascend_config(), "mix_placement", False)):
        raise ValueError(
            "DSV4 Window AFD with a dedicated shared-expert rank requires "
            "mix_placement=false"
        )
    shared_experts = int(vllm_config.model_config.hf_config.n_shared_experts)
    if shared_experts != 1:
        raise ValueError(
            "DSV4 Window AFD requires exactly one dedicated shared-expert rank"
        )
    return build_window_expert_layout(
        routed_expert_num=int(vllm_config.model_config.hf_config.n_routed_experts),
        ffn_size=afd_config.num_ffn_ranks,
        ffn_rank=role_rank,
    )


class AFDDeepseekV4RoutedMoE(native.DeepseekV4MoE):
    """Routed-only DSV4 MoE backed by rank-local expert weights."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: native.DeepseekV2Config,
        quant_config: Any,
        prefix: str,
        expert_layout: AFDWindowExpertLayout,
    ) -> None:
        nn.Module.__init__(self)
        if expert_layout.is_shared:
            raise ValueError("routed DSV4 MoE requires a routed expert layout")
        parallel_config = vllm_config.parallel_config
        self.tp_size = native.get_tensor_model_parallel_world_size()
        self.tp_rank = native.get_tensor_model_parallel_rank()
        self.layer_idx = int(prefix.split(sep=".")[-2])
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        # Window operators own cross-rank dispatch/combine.  Keep this FusedMoE
        # local so its EP group cannot accidentally include the shared-only FFN
        # rank from the surrounding vLLM data-parallel group.
        self.ep_rank = 0
        self.ep_size = 1
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = 0
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.enable_eplb = False
        self.n_redundant_experts = 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = expert_layout.local_expert_count
        self.physical_expert_start = expert_layout.local_expert_start
        self.physical_expert_end = (
            self.physical_expert_start + expert_layout.local_expert_count
        )
        self.gate = None
        self.shared_experts = None
        self.hash = False
        self.experts = native.FusedMoE(
            shared_experts=None,
            gate=None,
            num_experts=expert_layout.local_expert_count,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            tp_size=1,
            dp_size=1,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=None,
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=0,
        )
        # The checkpoint uses global routed expert IDs while this local-only
        # FusedMoE allocates only this rank's expert slots.  Install the
        # deterministic Window global-to-local map before weight loading.
        expert_map = torch.full(
            (self.n_routed_experts,),
            -1,
            dtype=torch.int32,
        )
        start = expert_layout.local_expert_start
        count = expert_layout.local_expert_count
        expert_map[start : start + count] = torch.arange(count, dtype=torch.int32)
        self.experts.expert_map_manager._expert_map = expert_map
        self.experts.expert_map_manager._local_num_experts = (
            expert_layout.local_expert_count
        )
        self.experts._expert_map = expert_map


class AFDDeepseekV4SharedMoE(nn.Module):
    """Shared-only DSV4 FFN placed on the first Window FFN rank."""

    def __init__(
        self,
        *,
        config: native.DeepseekV2Config,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
    ) -> None:
        super().__init__()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.shared_experts = native.DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            quant_config=quant_config,
            is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)


class AFDDeepseekV4RemoteMoEProxy(RemoteFFNProxy):
    """Parameter-free DSV4 MoE stage executed by the remote FFN role."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._send_and_receive(hidden_states)

    def dispatch_remote_ffn(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> AFDRemoteFFNTransfer:
        if self.phase == "decoder" and self.layer_idx == 0:
            input_ids = getattr(get_forward_context(), "input_ids", None)
            if input_ids is None:
                raise RuntimeError(
                    "DSV4 layer 0 requires input_ids in the forward context"
                )
            send_kwargs["input_ids"] = input_ids
        return super().dispatch_remote_ffn(hidden_states, **send_kwargs)


class AFDDeepseekV4DecoderLayer(native.DeepseekV2DecoderLayer):
    """DSV4 decoder layer that constructs only the active AFD role."""

    # Patch reason: native DSV4 constructs Attention, HC, and MoE for every role.
    # Patch functionality: construct Attention/HC or MoE, never both.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: native.DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
        attn_cls: type[nn.Module] | None = None,
    ) -> None:
        # ### PATCH START: role-selective DSV4 construction.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.vllm_config = vllm_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # The pinned vllm_cann DSV4 implementation constructs DeepseekV4MoE
        # unconditionally in every decoder layer.
        self.is_moe_layer = True
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.is_window_afd = afd_config.connector == "WindowAFDConnector"
        self.top_k = int(getattr(config, "num_experts_per_tok", 1))

        if self.afd_role == "attention":
            max_position_embeddings = config.rope_parameters[
                "original_max_position_embeddings"
            ]
            attention_class = attn_cls or native.DeepseekV4Attention
            self.self_attn = attention_class(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            self.mlp = AFDDeepseekV4RemoteMoEProxy(
                layer_idx=self.layer_idx,
                phase="mtp" if is_draft_layer else "decoder",
            )
            if self.compute_gate_on_attention and self.is_moe_layer:
                # Keep the native DSV4 checkpoint namespace (mlp.gate) even
                # though this proxy executes the gate on the Attention role.
                self.mlp.gate = native.ReplicatedLinear(
                    config.hidden_size,
                    config.n_routed_experts,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.mlp.gate",
                )
                self.mlp.gate.precast_fp32_weight = True
                self.mlp.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32),
                )
                num_hash_layers = int(getattr(config, "num_hash_layers", 0))
                if self.layer_idx < num_hash_layers and not is_draft_layer:
                    # Native DSV4 uses a token-id -> expert-id table for the
                    # hash-routing prefix.  Keep the same parameter on the
                    # Attention-side gate so checkpoint loading and routing
                    # follow the native contract.
                    self.mlp.gate.tid2eid = nn.Parameter(
                        torch.zeros(
                            config.vocab_size,
                            self.top_k,
                            dtype=torch.int32,
                        ),
                        requires_grad=False,
                    )
                    self.mlp.gate.e_score_correction_bias = None
                else:
                    self.mlp.gate.tid2eid = None
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.hc_mult = hc_mult = config.hc_mult
            self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
            self.hc_eps = config.hc_eps
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * config.hidden_size
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        else:
            self.self_attn = native.PPMissingLayer()
            if self.is_window_afd:
                role_rank = resolve_role_rank(vllm_config, afd_config)
                expert_layout = _window_expert_layout(vllm_config, role_rank)
                if expert_layout.is_shared:
                    self.mlp = AFDDeepseekV4SharedMoE(
                        config=config,
                        parallel_config=parallel_config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.mlp",
                    )
                else:
                    self.mlp = AFDDeepseekV4RoutedMoE(
                        vllm_config=vllm_config,
                        config=config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.mlp",
                        expert_layout=expert_layout,
                    )
            else:
                self.mlp = native.DeepseekV4MoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    is_draft_layer=is_draft_layer,
                )
        # ### PATCH END

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0),
            residual.unsqueeze(dim=0),
            post.unsqueeze(dim=0),
            comb.unsqueeze(dim=0),
        )
        return output.squeeze(dim=0)

    # Patch reason: native forward invokes the locally constructed MoE.
    # Patch functionality: the Attention role invokes a parameter-free remote proxy.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    # Patch reason: upstream forward always runs locally owned draft MoE.
    # Patch functionality: reject FFN full execution and identify the remote step.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ffn_output, continuation = self.forward_attention_to_remote_ffn(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
        hidden_states = self.complete_remote_ffn(ffn_output, continuation)
        return hidden_states, continuation[0]

    def forward_attention_to_remote_ffn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run through the remote MoE receive, deferring FFN HC post."""
        hidden_states, continuation = self.forward_attention_to_remote_ffn_input(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            return self.mlp(hidden_states), continuation
        transfer = self.dispatch_remote_ffn(hidden_states)
        return self.receive_remote_ffn(transfer), continuation

    def forward_attention_to_remote_ffn_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Compute the Attention-side input consumed by the remote MoE."""
        # ### PATCH START: reject accidental FFN full-model execution.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN layers are connector-driven")
        # ### PATCH END
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=llama_4_scaling,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, (residual, post, comb)

    def dispatch_remote_ffn(
        self,
        hidden_states: torch.Tensor,
    ) -> AFDRemoteFFNTransfer:
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            raise RuntimeError("DSV4 Attention layer requires its remote MoE proxy")
        send_kwargs: dict[str, torch.Tensor] = {}
        if self.compute_gate_on_attention:
            from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
                compute_gate_topk,
            )

            topk_weights, topk_ids, _ = compute_gate_topk(
                gate=self.mlp.gate,
                vllm_config=self.vllm_config,
                config=self.config,
                top_k=self.top_k,
                hidden_states=hidden_states,
                input_ids=(
                    getattr(get_forward_context(), "input_ids", None)
                    if self.layer_idx < int(getattr(self.config, "num_hash_layers", 0))
                    else None
                ),
                tid2eid=getattr(self.mlp.gate, "tid2eid", None),
            )
            send_kwargs["expert_ids"] = topk_ids
            send_kwargs["expert_scales"] = topk_weights
        return self.mlp.dispatch_remote_ffn(hidden_states, **send_kwargs)

    def receive_remote_ffn(
        self,
        transfer: AFDRemoteFFNTransfer,
    ) -> torch.Tensor:
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            raise RuntimeError("DSV4 Attention layer requires its remote MoE proxy")
        return self.mlp.receive_remote_ffn(transfer, layer_idx=self.layer_idx)

    def complete_remote_ffn(
        self,
        ffn_output: torch.Tensor,
        continuation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Complete the FFN HC post after the remote output becomes visible."""
        residual, post, comb = continuation
        return self.hc_post(ffn_output, residual, post, comb)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        group_list: torch.Tensor | None = None,
        dynamic_scales: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        if self.afd_role != "ffn":
            raise RuntimeError("DSV4 Attention role does not own local MoE weights")
        if self.compute_gate_on_attention and self.is_moe_layer:
            if group_list is None:
                raise RuntimeError("DSV4 Window FFN requires group_list from batching")
            from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
                compute_attention_gate_moe_ffn,
            )

            output = compute_attention_gate_moe_ffn(
                self,
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_scales=kwargs.get("topk_scales"),
                group_list_type=int(kwargs.get("group_list_type", 1)),
            )
            return output
        return self.mlp(hidden_states, input_ids=input_ids)


@native.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AFDDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware DSV4 model with a remote-MoE Attention forward path."""

    fall_back_to_pt_during_load = False

    # Patch reason: native DSV4 allocates embedding, every full layer, and head HC.
    # Patch functionality: build role-owned modules and omit the disabled MTP buffer.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize role-aware storage without native allocation.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        self._window_global_mxfp_weights = None
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        # vLLM 0.23 does not initialize this field for the Ascend DSV4 model,
        # while the newer forward path checks it even when Eagle/MTP is off.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        # ### PATCH START: DSA scratch data belongs only to Attention.
        if self.is_v32 and self.afd_role == "attention":
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None
        # ### PATCH END

        if self.afd_role == "attention" and native.get_pp_group().is_first_rank:
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
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        self.hc_mult = config.hc_mult

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> native.IntermediateTensors:
            return native.IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    )
                }
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors

        # ### PATCH START: head HC and normalization belong only to Attention.
        if self.afd_role == "attention":
            self.norm_eps = config.rms_norm_eps
            self.hc_eps = config.hc_eps
            hc_dim = self.hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mtp_enabled = self.afd_role == "attention" and _uses_mtp(vllm_config)
        if self.mtp_enabled:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        self.dspark_enabled = self.afd_role == "attention" and _uses_dspark(
            vllm_config
        )
        self._dspark_target_layer_ids = (
            list(getattr(config, "dspark_target_layer_ids", []) or [])
            if self.dspark_enabled
            else []
        )
        self._dspark_target_layer_id_set = frozenset(
            self._dspark_target_layer_ids
        )
        if self.dspark_enabled:
            if not self._dspark_target_layer_ids:
                raise RuntimeError(
                    "DeepSeek-V4 AFD DSpark requires dspark_target_layer_ids"
                )
            self._dspark_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                len(self._dspark_target_layer_ids) * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        # ### PATCH END

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN role does not own token embeddings")
        return self.embed_tokens(input_ids)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        flattened = x.flatten(1).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flattened, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        output = torch.sum(pre.unsqueeze(-1) * flattened.view(shape), dim=1)
        return output.to(dtype)

    # Patch reason: native forward runs native full layers and always updates MTP.
    # Patch functionality: run role-aware layers and omit disabled MTP state.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        # ### PATCH START: only Attention runs the complete DSV4 model.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN model execution is connector-driven")
        # ### PATCH END
        if native.get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            if intermediate_tensors is None:
                raise RuntimeError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        llama_4_scaling = None
        aux_hidden_states: list[torch.Tensor] = []
        dspark_hiddens: list[torch.Tensor] = []
        if native.get_pp_group().is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, _ = layer(
                positions,
                hidden_states,
                None,
                llama_4_scaling,
            )
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))
            if layer.layer_idx in self._dspark_target_layer_id_set:
                dspark_hiddens.append(hidden_states.mean(dim=1))

        if self.mtp_enabled:
            mtp_hidden = hidden_states.flatten(1)
            self._mtp_hidden_buffer[: mtp_hidden.shape[0]].copy_(mtp_hidden)
        self._store_dspark_hidden_states(dspark_hiddens)

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def forward_ubatches_layer_major(
        self,
        ubatch_metadata: list[Any],
    ) -> list[Any]:
        """Run HCCL U2 in layer-major, stage-minor order on one host thread."""
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN model execution is connector-driven")
        if len(ubatch_metadata) != 2:
            raise RuntimeError(
                "DSV4 layer-major execution requires exactly two stages; "
                f"got {len(ubatch_metadata)}"
            )

        stage_contexts = [item.context.forward_context for item in ubatch_metadata]
        connectors = []
        for stage_idx, (item, forward_context) in enumerate(
            zip(ubatch_metadata, stage_contexts, strict=True)
        ):
            afd_metadata = (forward_context.additional_kwargs or {}).get("afd_metadata")
            connector = getattr(afd_metadata, "connector", None)
            if connector is None:
                raise RuntimeError(
                    "DSV4 layer-major U2 requires AFD connector metadata"
                )
            if int(getattr(forward_context, "ubatch_idx", -1)) != stage_idx:
                raise RuntimeError(
                    "DSV4 layer-major stage context order is invalid: "
                    f"expected={stage_idx} actual="
                    f"{getattr(forward_context, 'ubatch_idx', None)}"
                )
            forward_context.input_ids = item.input_ids
            forward_context.afd_layer_major_u2 = True
            connectors.append(connector)

        connector = connectors[0]
        if any(stage_connector is not connector for stage_connector in connectors[1:]):
            raise RuntimeError("DSV4 layer-major U2 stages must share one connector")
        require_idle = getattr(connector, "require_attention_pipeline_idle", None)
        wait_for_receive = getattr(
            connector,
            "wait_for_attention_stage_receive",
            None,
        )
        reset_pipeline = getattr(connector, "reset_attention_pipeline_state", None)
        if not all(callable(method) for method in (require_idle, wait_for_receive)):
            raise RuntimeError("DSV4 layer-major U2 requires the HCCL stream connector")

        hidden_ubatches: list[torch.Tensor] = []
        pending_layers: list[AFDDeepseekV4DecoderLayer | None] = [None, None]
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ] = [None, None]
        aux_hidden_ubatches: list[list[torch.Tensor]] = [[], []]
        dspark_hidden_ubatches: list[list[torch.Tensor]] = [[], []]
        require_idle()
        try:
            for item, forward_context in zip(
                ubatch_metadata,
                stage_contexts,
                strict=True,
            ):
                with override_forward_context(forward_context):
                    if native.get_pp_group().is_first_rank:
                        hidden_states = (
                            item.inputs_embeds
                            if item.inputs_embeds is not None
                            else self.embed_input_ids(item.input_ids)
                        )
                        hidden_states = hidden_states.unsqueeze(1).repeat(
                            1,
                            self.hc_mult,
                            1,
                        )
                    else:
                        if item.intermediate_tensors is None:
                            raise RuntimeError(
                                "pipeline stage requires intermediate tensors"
                            )
                        hidden_states = item.intermediate_tensors["hidden_states"]
                    hidden_ubatches.append(hidden_states)

            graph_pipeline_active = getattr(
                connector,
                "attention_graph_compute_pipeline_active",
                None,
            )
            with override_forward_context(stage_contexts[0]):
                use_graph_compute = bool(
                    callable(graph_pipeline_active) and graph_pipeline_active()
                )
            if use_graph_compute:
                self._forward_ubatches_graph_compute_pipeline(
                    ubatch_metadata=ubatch_metadata,
                    stage_contexts=stage_contexts,
                    connector=connector,
                    hidden_ubatches=hidden_ubatches,
                    pending_layers=pending_layers,
                    pending_continuations=pending_continuations,
                    aux_hidden_ubatches=aux_hidden_ubatches,
                    dspark_hidden_ubatches=dspark_hidden_ubatches,
                )
            else:
                self._forward_ubatches_eager_pipeline(
                    ubatch_metadata=ubatch_metadata,
                    stage_contexts=stage_contexts,
                    connector=connector,
                    wait_for_receive=wait_for_receive,
                    hidden_ubatches=hidden_ubatches,
                    pending_layers=pending_layers,
                    pending_continuations=pending_continuations,
                    aux_hidden_ubatches=aux_hidden_ubatches,
                    dspark_hidden_ubatches=dspark_hidden_ubatches,
                )
        except BaseException:
            if callable(reset_pipeline):
                reset_pipeline()
            raise

        if not native.get_pp_group().is_last_rank:
            return [
                native.IntermediateTensors({"hidden_states": hidden_states})
                for hidden_states in hidden_ubatches
            ]

        if self.mtp_enabled:
            buffer_offset = 0
            for hidden_states in hidden_ubatches:
                mtp_hidden = hidden_states.flatten(1)
                next_offset = buffer_offset + mtp_hidden.shape[0]
                self._mtp_hidden_buffer[buffer_offset:next_offset].copy_(
                    mtp_hidden,
                )
                buffer_offset = next_offset

        if self.dspark_enabled:
            buffer_offset = 0
            for dspark_hiddens in dspark_hidden_ubatches:
                dspark_states = self._concat_dspark_hidden_states(dspark_hiddens)
                next_offset = buffer_offset + dspark_states.shape[0]
                self._dspark_hidden_buffer[buffer_offset:next_offset].copy_(
                    dspark_states
                )
                buffer_offset = next_offset

        outputs: list[Any] = []
        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                hidden_states = self.hc_head(
                    hidden_ubatches[stage_idx],
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                )
                hidden_states = self.norm(hidden_states)
                aux_hidden_states = aux_hidden_ubatches[stage_idx]
                outputs.append(
                    (hidden_states, aux_hidden_states)
                    if aux_hidden_states
                    else hidden_states
                )
        return outputs

    def _forward_ubatches_eager_pipeline(
        self,
        *,
        ubatch_metadata: list[Any],
        stage_contexts: list[Any],
        connector: Any,
        wait_for_receive: Any,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
        dspark_hidden_ubatches: list[list[torch.Tensor]],
    ) -> None:
        del connector
        llama_4_scaling = None
        for layer_offset, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            for stage_idx, (item, forward_context) in enumerate(
                zip(ubatch_metadata, stage_contexts, strict=True)
            ):
                with override_forward_context(forward_context):
                    if layer_offset > 0:
                        wait_for_receive(
                            stage_idx=stage_idx,
                            tensor=hidden_ubatches[stage_idx],
                        )
                        self._complete_pending_stage(
                            stage_idx,
                            hidden_ubatches,
                            pending_layers,
                            pending_continuations,
                            aux_hidden_ubatches,
                            dspark_hidden_ubatches,
                        )
                    hidden_states, continuation = layer.forward_attention_to_remote_ffn(
                        item.positions,
                        hidden_ubatches[stage_idx],
                        None,
                        llama_4_scaling,
                    )
                    hidden_ubatches[stage_idx] = hidden_states
                    pending_layers[stage_idx] = layer
                    pending_continuations[stage_idx] = continuation

        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                wait_for_receive(
                    stage_idx=stage_idx,
                    tensor=hidden_ubatches[stage_idx],
                )
                self._complete_pending_stage(
                    stage_idx,
                    hidden_ubatches,
                    pending_layers,
                    pending_continuations,
                    aux_hidden_ubatches,
                    dspark_hidden_ubatches,
                    final=True,
                )

    def _forward_ubatches_graph_compute_pipeline(
        self,
        *,
        ubatch_metadata: list[Any],
        stage_contexts: list[Any],
        connector: Any,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
        dspark_hidden_ubatches: list[list[torch.Tensor]],
    ) -> None:
        compute_scope = getattr(connector, "attention_graph_compute", None)
        wait_for_compute = getattr(
            connector,
            "wait_for_attention_graph_compute",
            None,
        )
        join_compute = getattr(
            connector,
            "join_attention_graph_compute",
            None,
        )
        hybrid_dag_active = getattr(
            connector,
            "attention_graph_hybrid_dag_active",
            None,
        )
        wait_for_receive = getattr(
            connector,
            "wait_for_attention_stage_receive",
            None,
        )
        wait_for_graph_receive = getattr(
            connector,
            "wait_for_attention_graph_receive",
            None,
        )
        is_window_connector = bool(
            getattr(connector, "is_window_connector", False)
        )
        if not all(
            callable(method)
            for method in (
                compute_scope,
                wait_for_compute,
                join_compute,
                wait_for_receive,
                hybrid_dag_active,
            )
        ) or (not is_window_connector and not callable(wait_for_graph_receive)):
            raise RuntimeError("DSV4 Graph U2 requires the HCCL compute pipeline")

        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        llama_4_scaling = None
        with override_forward_context(stage_contexts[0]):
            use_hybrid_dag = hybrid_dag_active()

        def enqueue_attention_compute(
            layer_offset: int,
            layer: AFDDeepseekV4DecoderLayer,
            stage_idx: int,
        ) -> None:
            item = ubatch_metadata[stage_idx]
            forward_context = stage_contexts[stage_idx]
            with (
                override_forward_context(forward_context),
                compute_scope(
                    layer_idx=layer.layer_idx,
                    stage_idx=stage_idx,
                    tensors=(hidden_ubatches[stage_idx], item.positions),
                    wait_for_receive_layer_idx=(
                        layers[layer_offset - 1].layer_idx
                        if use_hybrid_dag and layer_offset > 0
                        else None
                    ),
                ),
            ):
                if layer_offset > 0:
                    if not use_hybrid_dag and callable(wait_for_graph_receive):
                        wait_for_graph_receive(
                            layer_idx=layers[layer_offset - 1].layer_idx,
                            stage_idx=stage_idx,
                            tensor=hidden_ubatches[stage_idx],
                        )
                    wait_for_receive(
                        stage_idx=stage_idx,
                        tensor=hidden_ubatches[stage_idx],
                    )
                    self._complete_pending_stage(
                        stage_idx,
                        hidden_ubatches,
                        pending_layers,
                        pending_continuations,
                        aux_hidden_ubatches,
                        dspark_hidden_ubatches,
                    )
                hidden_states, continuation = (
                    layer.forward_attention_to_remote_ffn_input(
                        item.positions,
                        hidden_ubatches[stage_idx],
                        None,
                        llama_4_scaling,
                    )
                )
                hidden_ubatches[stage_idx] = hidden_states
                pending_continuations[stage_idx] = continuation

        # Host-side Graph construction remains layer-major. With the hybrid DAG,
        # each side-stream stage waits only for its own prior-layer F2A receive,
        # so S0 compute can execute while the parent stream finishes S1 exchange.
        first_layer = layers[0]
        for stage_idx in range(len(stage_contexts)):
            enqueue_attention_compute(0, first_layer, stage_idx)

        final_event_layer = int(self.config.num_hidden_layers)
        for layer_offset, layer in enumerate(layers):
            for stage_idx, forward_context in enumerate(stage_contexts):
                with override_forward_context(forward_context):
                    wait_for_compute(
                        layer_idx=layer.layer_idx,
                        stage_idx=stage_idx,
                        tensors=(hidden_ubatches[stage_idx],),
                    )
                    transfer = layer.dispatch_remote_ffn(hidden_ubatches[stage_idx])
                    hidden_ubatches[stage_idx] = layer.receive_remote_ffn(transfer)
                    pending_layers[stage_idx] = layer

            next_layer_offset = layer_offset + 1
            if next_layer_offset < len(layers):
                for stage_idx in range(len(stage_contexts)):
                    enqueue_attention_compute(
                        next_layer_offset,
                        layers[next_layer_offset],
                        stage_idx,
                    )
            else:
                for stage_idx, forward_context in enumerate(stage_contexts):
                    with (
                        override_forward_context(forward_context),
                        compute_scope(
                            layer_idx=final_event_layer,
                            stage_idx=stage_idx,
                            tensors=(hidden_ubatches[stage_idx],),
                            wait_for_receive_layer_idx=(
                                layer.layer_idx if use_hybrid_dag else None
                            ),
                        ),
                    ):
                        if not use_hybrid_dag and callable(wait_for_graph_receive):
                            wait_for_graph_receive(
                                layer_idx=layer.layer_idx,
                                stage_idx=stage_idx,
                                tensor=hidden_ubatches[stage_idx],
                            )
                        wait_for_receive(
                            stage_idx=stage_idx,
                            tensor=hidden_ubatches[stage_idx],
                        )
                        self._complete_pending_stage(
                            stage_idx,
                            hidden_ubatches,
                            pending_layers,
                            pending_continuations,
                            aux_hidden_ubatches,
                            dspark_hidden_ubatches,
                            final=True,
                        )

        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                join_compute(
                    layer_idx=final_event_layer,
                    stage_idx=stage_idx,
                    tensors=(hidden_ubatches[stage_idx],),
                )

    def _complete_pending_stage(
        self,
        stage_idx: int,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
        dspark_hidden_ubatches: list[list[torch.Tensor]],
        *,
        final: bool = False,
    ) -> None:
        pending_layer = pending_layers[stage_idx]
        continuation = pending_continuations[stage_idx]
        if pending_layer is None or continuation is None:
            qualifier = "final " if final else ""
            raise RuntimeError(
                f"DSV4 layer-major stage has no {qualifier}pending layer: "
                f"stage={stage_idx}"
            )
        hidden_ubatches[stage_idx] = pending_layer.complete_remote_ffn(
            hidden_ubatches[stage_idx],
            continuation,
        )
        if pending_layer.layer_idx + 1 in self.aux_hidden_state_layers:
            aux_hidden_ubatches[stage_idx].append(
                hidden_ubatches[stage_idx].mean(dim=1)
            )
        if pending_layer.layer_idx in self._dspark_target_layer_id_set:
            dspark_hidden_ubatches[stage_idx].append(
                hidden_ubatches[stage_idx].mean(dim=1)
            )

    def _concat_dspark_hidden_states(
        self,
        hidden_states: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(hidden_states) != len(self._dspark_target_layer_ids):
            raise RuntimeError(
                "DeepSeek-V4 AFD did not collect every DSpark target layer: "
                f"expected {self._dspark_target_layer_ids}, "
                f"collected {len(hidden_states)} hidden states"
            )
        return torch.cat(hidden_states, dim=-1)

    def _store_dspark_hidden_states(
        self,
        hidden_states: list[torch.Tensor],
    ) -> None:
        if not self.dspark_enabled:
            return
        dspark_states = self._concat_dspark_hidden_states(hidden_states)
        self._dspark_hidden_buffer[: dspark_states.shape[0]].copy_(dspark_states)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute every ready Window layer without host-side layer parsing."""

        if self._window_global_mxfp_weights is None:
            # Routed WeightNZ tensors must be packed before native post-load
            # processing and are initialized by ``load_weights()``.  Shared
            # MXFP8 tensors remain ND, so initialize them here after their
            # native per-layer post-load processing has completed.
            first_local_layer = next(
                layer
                for layer in self.layers
                if not isinstance(layer, native.PPMissingLayer)
            )
            if getattr(first_local_layer.mlp, "experts", None) is not None:
                raise RuntimeError(
                    "Window routed FFN weights were not initialized before "
                    "MXFP post-load processing"
                )
            self.initialize_window_global_mxfp_weights()
        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            compute_window_global_mxfp_ffn,
        )

        return compute_window_global_mxfp_ffn(
            hidden_states=hidden_states,
            compact_group_list=group_list,
            dynamic_scale=dynamic_scale,
            weights=self._window_global_mxfp_weights,
            output_dtype=self.vllm_config.model_config.dtype,
        )

    def initialize_window_global_mxfp_weights(self) -> None:
        """Pack Window FFN weights while checkpoint tensors are still ND."""

        if self._window_global_mxfp_weights is not None:
            return

        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            build_window_global_mxfp_weights,
        )

        local_layers = [
            layer
            for layer in self.layers
            if not isinstance(layer, native.PPMissingLayer)
        ]
        if len(local_layers) != int(self.config.num_hidden_layers):
            raise RuntimeError(
                "Window global FFN requires every decoder layer on the FFN rank: "
                f"local={len(local_layers)} total={self.config.num_hidden_layers}"
            )
        self._window_global_mxfp_weights = build_window_global_mxfp_weights(
            local_layers
        )


class AFDDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """DSV4 causal LM wrapper with strict role ownership."""

    model_cls = AFDDeepseekV4Model

    # Patch reason: native construction allocates the LM head for both roles.
    # Patch functionality: build the head only for Attention and register FFN MoE.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: establish the AFD role before allocating modules.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        self.vllm_config = vllm_config
        # ### PATCH END
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.num_moe_layers = config.num_hidden_layers
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if self.afd_role == "ffn":
            for layer in self.model.layers:
                if isinstance(layer, native.PPMissingLayer):
                    continue
                if isinstance(layer.mlp, native.DeepseekV4MoE):
                    example_moe = layer.mlp
                    self.moe_mlp_layers.append(layer.mlp)
                    self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model.compute_window_global_ffn_output(
            hidden_states,
            group_list,
            dynamic_scale,
        )

    def initialize_window_global_mxfp_weights(self) -> None:
        self.model.initialize_window_global_mxfp_weights()

    def forward_ubatches_layer_major(
        self,
        ubatch_metadata: list[Any],
    ) -> list[Any]:
        return self.model.forward_ubatches_layer_major(ubatch_metadata)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        if self.afd_role != "attention":
            return None
        if getattr(self.model, "dspark_enabled", False):
            return getattr(self.model, "_dspark_hidden_buffer", None)
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        window_ffn_kind = None
        if self.afd_role == "ffn" and self.afd_config.connector == "WindowAFDConnector":
            role_rank = resolve_role_rank(self.vllm_config, self.afd_config)
            window_ffn_kind = _window_expert_layout(
                self.vllm_config,
                role_rank,
            ).kind
        loaded_weights = super().load_weights(
            _iter_role_weights(
                weights,
                role=self.afd_role,
                compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
                window_ffn_kind=window_ffn_kind,
                num_hidden_layers=int(self.model.config.num_hidden_layers),
            )
        )
        if window_ffn_kind == "routed":
            self.model.initialize_window_global_mxfp_weights()
        return loaded_weights


class AFDDeepSeekMultiTokenPredictorLayer(native_mtp.DeepSeekMultiTokenPredictorLayer):
    """Role-selective DSV4 MTP layer for the pinned v0.23 Ascend stack."""

    # Patch reason: upstream MTP allocates Attention, HC, head, and MoE together.
    # Patch functionality: allocate only the active AFD role and remote the MoE.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        # ### PATCH START: role-selective MTP construction.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.device = native_mtp.current_platform.device_type
        self.is_v32 = hasattr(config, "index_topk")

        if self.afd_role == "attention":
            self.e_proj = native_mtp.ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.e_proj",
                return_bias=False,
            )
            self.h_proj = native_mtp.ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.h_proj",
                return_bias=False,
            )
            self.enorm = native_mtp.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.hnorm = native_mtp.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            topk_indices_buffer = (
                torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.index_topk,
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.is_v32
                else None
            )
            self.shared_head = native_mtp.SharedHead(
                config=config,
                prefix=prefix,
                quant_config=quant_config,
            )
            self.hc_eps = config.hc_eps
            self.hc_mult = hc_mult = config.hc_mult
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.norm_eps = config.rms_norm_eps
        else:
            topk_indices_buffer = None

        self.mtp_block = AFDDeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
            is_draft_layer=True,
        )
        # ### PATCH END

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only remote-MoE execution.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role is connector-driven")
        if inputs_embeds is None:
            raise RuntimeError("DSV4 MTP Attention requires inputs_embeds")
        # ### PATCH END
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            0,
            inputs_embeds,
        )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = previous_hidden_states.view(
            -1,
            self.hc_mult,
            self.config.hidden_size,
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(
            previous_hidden_states
        )
        # ### PATCH START: bind this proposal iteration to the MTP phase.
        proxy = self.mtp_block.mlp
        if isinstance(proxy, AFDDeepseekV4RemoteMoEProxy):
            proxy.speculative_step = spec_step_index
        # ### PATCH END
        hidden_states, _ = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        return hidden_states

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mtp_block.compute_ffn_output(hidden_states)


class AFDDeepSeekMultiTokenPredictor(native_mtp.DeepSeekMultiTokenPredictor):
    """Role-aware container for the single DSV4 MTP layer."""

    # Patch reason: upstream constructs embeddings and full MTP layers on both roles.
    # Patch functionality: construct role-owned modules and expose layer bounds.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: role-selective MTP container.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        # vLLM-Ascend caches the target model's layer-index capability and
        # subsequently reads these attributes from the draft model as well.
        self.start_layer = 0
        self.end_layer = self.num_mtp_layers
        self.layers = nn.ModuleDict(
            {
                str(index): AFDDeepSeekMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.{index}",
                )
                for index in range(self.num_mtp_layers)
            }
        )
        if self.afd_role == "attention":
            self.embed_tokens = native_mtp.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
            self.logits_processor = native_mtp.LogitsProcessor(config.vocab_size)
        else:
            self.embed_tokens = native.PPMissingLayer()
            self.logits_processor = None
        # ### PATCH END

    # Patch reason: upstream permits embedding lookup on every constructed role.
    # Patch functionality: enforce Attention ownership of draft embeddings.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # ### PATCH START: Attention-only embedding ownership.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role does not own embeddings")
        # ### PATCH END
        return self.embed_tokens(input_ids)

    # Patch reason: upstream permits full MTP execution on every constructed role.
    # Patch functionality: run draft orchestration only on Attention and keep
    # the proposal iteration separate from the modulo-selected MTP layer.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only draft orchestration.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role is connector-driven")
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # ### PATCH END
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            # ### PATCH START: preserve the protocol iteration.
            spec_step_idx,
            # ### PATCH END: preserve the protocol iteration.
        )

    # Patch reason: upstream exposes logits for every constructed role.
    # Patch functionality: enforce Attention ownership of draft logits.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only logits ownership.
        if self.afd_role != "attention" or self.logits_processor is None:
            raise RuntimeError("DSV4 MTP logits belong to the Attention role")
        # ### PATCH END
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        hidden_states = hidden_states.view(
            -1,
            mtp_layer.hc_mult,
            mtp_layer.config.hidden_size,
        )
        hidden_states = mtp_layer.hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
        )
        return self.logits_processor(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        return self.layers[str(layer_idx)].compute_ffn_output(hidden_states)


# ### PATCH START: eager multi-token drafts require a live Python step counter.
@native_mtp.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda config: (
        not (
            config.speculative_config.enforce_eager
            and config.speculative_config.num_speculative_tokens > 1
        )
    )
)
# ### PATCH END: eager multi-token drafts require a live Python step counter.
class AFDDeepSeekV4MTP(native_mtp.DeepSeekV4MTP):
    """Strict AFD role wrapper for the native DSV4 MTP model."""

    # Patch reason: upstream constructs a full draft model for every worker.
    # Patch functionality: use the role-aware MTP container.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: role-aware top-level MTP model.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # The merged proposer omits spec_step_idx on each call. Eager drafts
        # number those calls; graph drafts keep the static step-zero contract.
        speculative_config = vllm_config.speculative_config
        self._afd_num_speculative_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config.enforce_eager
            else 1
        )
        self._afd_next_speculative_step = 0
        self.model = AFDDeepSeekMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=native_mtp.maybe_prefix(prefix, "mtp"),
        )
        self.set_moe_parameters()
        # ### PATCH END

    # Patch reason: the pinned merged proposer does not pass proposal step IDs.
    # Patch functionality: number eager calls modulo N while graph drafts use 0.
    # Signature: matches upstream; no added parameters. The upstream default 0
    # denotes the implicit next step; explicit nonzero steps remain supported.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 11ee45653b199a097805b87011824a81ffa51b95
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: track implicit merged-draft iterations.
        if self._afd_num_speculative_tokens > 1:
            if spec_step_idx == 0:
                spec_step_idx = self._afd_next_speculative_step
            self._afd_next_speculative_step = (
                spec_step_idx + 1
            ) % self._afd_num_speculative_tokens
        # ### PATCH END: track implicit merged-draft iterations.
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.afd_role != "ffn":
            raise RuntimeError("DSV4 MTP Attention does not own local MoE weights")
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def attach_afd_connector(self, connector: object) -> None:
        if self.afd_role != "attention":
            return
        for layer in self.model.layers.values():
            proxy = layer.mtp_block.mlp
            if isinstance(proxy, AFDDeepseekV4RemoteMoEProxy):
                proxy.attach_connector(connector)

    # Patch reason: upstream loader receives both Attention and FFN checkpoint keys.
    # Patch functionality: filter original keys once, then reuse the native loader.
    # Signature: matches upstream; no added parameters. Native delegation is retained
    # because its quantized expert mappings must remain pinned to the target stack.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: one-shot role filtering before native load.
        role_weights = _iter_mtp_role_weights(weights, role=self.afd_role)
        # ### PATCH END
        return super().load_weights(role_weights)


__all__ = ["AFDDeepseekV4ForCausalLM", "AFDDeepSeekV4MTP"]
