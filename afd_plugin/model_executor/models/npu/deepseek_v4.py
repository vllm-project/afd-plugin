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

from collections.abc import Iterable, Iterator
from copy import copy
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from transformers import DeepseekV2Config
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.sequence import IntermediateTensors

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    CAMP2P_CONNECTOR,
    parse_afd_config,
)
from afd_plugin.connectors import (
    AFDExpertRoutingSpec,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy
from afd_plugin.model_executor.models.npu.async_cam_dbo import (
    send_async_cam_remote_call,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    prepare_cam_dispatch_payload,
    restore_cam_dispatch_output,
)
from afd_plugin.model_executor.models.npu.deepseek_v4_shared_experts import (
    AFDDeepseekV4SharedExperts,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

try:
    from vllm_ascend.models.deepseek_v4 import model as native
except ImportError as exc:  # pragma: no cover - only reachable off Ascend.
    raise ImportError(
        "DSV4 AFD support requires the vLLM-Ascend native DSV4 model"
    ) from exc

if TYPE_CHECKING:
    from transformers.modeling_rope_utils import RopeParameters
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
    from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
    from vllm_ascend.quantization.methods.w4a8.w4a8 import (
        AscendW4A8DynamicFusedMoEMethod,
    )

    from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights


_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"
_BOTH_ROLES = frozenset((_ATTENTION_ROLE, _FFN_ROLE))


def _refresh_ascend_fused_moe() -> None:
    """Bind native DSV4 MoE construction to the Ascend implementation."""
    # vLLM-Ascend applies this replacement during platform initialization,
    # while the DSV4 module keeps a module-level FusedMoEFactory binding.  Refresh it
    # before either AFD role constructs its local model.
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
) -> frozenset[str]:
    """Return the AFD owner for a DSV4 checkpoint path.

    DSV4 checkpoints use ``attn``/``ffn`` names while the Ascend runtime
    model exposes ``self_attn``/``mlp``.  The native loader performs that name
    conversion later, so filtering must understand both spellings here.

    ``attn_owns_gate`` describes whether the Attention role built a router for
    the current configuration. Handing a role a path it never registered is not
    a harmless no-op: the upstream Ascend loader indexes its parameter dict by
    name without a membership check, so it raises ``KeyError`` instead of
    skipping.
    """

    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES

    _, stage, remainder = layer_path
    if stage in ("attn", "self_attn"):
        return frozenset((_ATTENTION_ROLE,))
    if stage in ("ffn", "mlp"):
        if attention_shared_experts and remainder and remainder[0] == "shared_experts":
            return frozenset((_ATTENTION_ROLE,))
        if remainder and remainder[0] == "gate":
            # Every router parameter, including the Hash id table, belongs to
            # each role that built a router: with the gate on Attention, the
            # Attention gate shell registers ``tid2eid`` for Hash layers and
            # routes from it, and the native FFN MoE registers its own copy.
            # With the gate on FFN the Attention MoE slot is parameter-free, so
            # Attention must not receive any of them.
            if attn_owns_gate:
                return _BOTH_ROLES
            return frozenset((_FFN_ROLE,))
        return frozenset((_FFN_ROLE,))
    # HC parameters and any future shared layer parameters are required by
    # both role-local model instances.
    return _BOTH_ROLES


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    attn_owns_gate: bool = True,
    attention_shared_experts: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
            attention_shared_experts=attention_shared_experts,
        ):
            yield name, loaded_weight


class AFDDeepseekV4RemoteMoE(RemoteFFNProxy):
    """DSV4 gate-on-FFN shell that sends Hash ids alongside the activations.

    The FFN role owns the gate for this configuration. Its Hash layers route by
    token identity, and only Attention holds ``input_ids``, so Attention sends
    the rank-local ids that the FFN rank's tokens correspond to. Connectors that
    do not transport ids ignore the extra argument, which keeps this shell valid
    for gate-on-FFN configurations in general.

    Whether ids cross the boundary is a run-level decision the two roles share
    through the model's ``afd_requires_input_ids`` declaration, not a per-layer
    one, because the FFN role cannot tell Hash layers from non-Hash ones. A
    forward context with no ids is therefore an error here rather than a silent
    activations-only send.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
            local_hash_input_ids,
        )

        # The FFN rank asks for the operator's ids channel on every layer, so
        # this side must send ids rather than fall back to an activations-only
        # transfer. ``local_hash_input_ids`` raises if the forward context
        # cannot supply them.
        return self._send_and_receive(
            hidden_states,
            input_ids=local_hash_input_ids(
                input_ids=(
                    get_forward_context().input_ids if input_ids is None else input_ids
                ),
                router_tokens=int(hidden_states.shape[0]),
            ),
        )


class AFDDeepseekV4AttentionGateRemoteMoE(RemoteFFNProxy):
    """DSV4 gate shell that routes local Attention tokens through Async CAM."""

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(layer_idx=layer_idx)
        self.use_sequence_parallel_moe = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.top_k = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.n_routed_experts)
        self.renormalize = bool(config.norm_topk_prob)
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.num_expert_group = int(getattr(config, "n_group", 1))
        self.topk_group = int(getattr(config, "topk_group", 1))
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5),
        )
        # Match native DSV4's explicit shared-output reduction. Ascend keeps
        # weights TP-sharded with SP unless shared-expert DP is enabled.
        self.shared_experts = (
            AFDDeepseekV4SharedExperts(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                is_sequence_parallel=self.use_sequence_parallel_moe,
                prefix=f"{prefix}.shared_experts",
            )
            if config.n_shared_experts
            else None
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
                (
                    torch.arange(config.vocab_size, dtype=torch.int32).unsqueeze(1)
                    + torch.arange(config.num_experts_per_tok, dtype=torch.int32)
                )
                % config.n_routed_experts,
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select DSV4 experts on Attention and exchange routed work via CAM."""
        from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate

        topk_weights, topk_ids = deepseek_v4_attention_gate.compute_attention_gate_topk(
            self,
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )
        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            use_sequence_parallel=self.use_sequence_parallel_moe,
        )
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("Async CAM requires AFD forward metadata")
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=afd_metadata.stage_idx,
                seq_len=int(dispatch_payload.hidden_states.shape[0]),
            )
        )
        receive = send_async_cam_remote_call(
            afd_metadata.connector,
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
        )
        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )
        maybe_apply_dbo_yield(hidden_states, role="attention")
        output = receive()
        output = restore_cam_dispatch_output(output, dispatch_payload.layout)
        if shared_output is not None:
            output = output + shared_output
        return output


class AFDDeepseekV4SyncAttentionGateRemoteMoE(AFDDeepseekV4AttentionGateRemoteMoE):
    """DSV4 gate shell that keeps the router on Attention for the sync connector.

    Inheritance is deliberate: the gate construction, the FP32 precast weight, the
    hash-layer table and the shared-expert module are identical to the async shell,
    and only the transport differs. Duplicating that construction would let the two
    shells drift on routing parameters.

    The forward differs in three ways, all forced by the sync contract:

    - routing travels on the A2E wire slot instead of a CAM dispatch payload, so the
      send and the receive are separate calls;
    - the shared experts run *between* those calls, which is the whole point of
      moving them here: their work overlaps the FFN side's routed work;
    - no DBO yield, because the first prototype keeps the sync path free of the
      scheduling interaction the async shell needs.

    Reusing ``RemoteFFNProxy._send_and_receive`` is not possible here: it issues the
    send and the receive back to back, leaving no window for the shared experts.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate

        topk_weights, topk_ids = (
            deepseek_v4_attention_gate.compute_attention_gate_topk(
                self,
                hidden_states,
                input_ids=input_ids,
                hidden_states_fp32=hidden_states_fp32,
            )
        )
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("Sync Attention-side gate requires AFD forward metadata")
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=afd_metadata.stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
        )
        # Routing shares the wire slot with token ids, so this send carries routing
        # only. The FFN side is configured with recv_routing for the same reason.
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            routing=(topk_ids, topk_weights),
        )
        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )
        output = afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=afd_metadata.stage_idx,
        )
        if shared_output is not None:
            output = output + shared_output
        return output


def _routed_experts_of(moe: Any) -> Any:
    """Return the routed-experts module that owns apply and finalize state.

    The path is ``moe.experts.routed_experts`` in the pinned tag. Reaching it through
    a helper keeps the failure readable if upstream changes the nesting, instead of
    an AttributeError from deep inside the MoE.
    """

    try:
        return moe.experts.routed_experts
    except AttributeError as exc:
        raise RuntimeError(
            "DSV4 external routing needs the routed-experts module at "
            f"mlp.experts.routed_experts, which {type(moe).__name__} does not expose",
        ) from exc


class AFDDeepseekV4DecoderLayer(native.DeepseekV4DecoderLayer):
    """Role-local DSV4 decoder layer.

    The inherited native ``forward`` is intentionally retained.  On the
    Attention role it executes the full native HC/DSA sequence and the
    ``RemoteFFNProxy`` makes the AFD transfer at exactly the native FFN
    boundary.  The FFN role is connector-driven and does not call this full
    forward method; it invokes ``compute_ffn_output`` instead.
    """

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/deepseek_v4/model.py::DeepseekV4DecoderLayer.__init__.
    # Patch reason: allocate Attention and FFN weights only on their AFD role.
    # Patch functionality: keep target HC/SP attributes and native layer forward,
    # replacing the Attention role's MoE with a remote module.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: initialize role-local decoder
        if is_draft_layer:
            raise ValueError("AFD DSV4 decoder layers do not support draft layers")
        afd_config = parse_afd_config(vllm_config, validate=False)
        nn.Module.__init__(self)
        # ### PATCH END: initialize role-local decoder

        if config is None:
            config = vllm_config.model_config.hf_config
        # ### PATCH START: AFD decoder metadata
        self.vllm_config = vllm_config
        self.config = config
        self.afd_role = afd_config.role
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        # Which wire protocol this layer expects, decided from the configuration.
        # The payload only validates that expectation; it must not pick the path,
        # because a missing or mixed payload would otherwise silently reroute.
        self.uses_external_routing = bool(
            afd_config.compute_gate_on_attention
            and afd_config.connector == CAMP2P_CONNECTOR
        )
        self.is_moe_layer = True
        # ### PATCH END: AFD decoder metadata
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        # Both supported lanes accept flat and per-group checkpoint RoPE dicts.
        # Use compressor (CSA/HCA) YaRN values when the nested group is present.
        yarn_rope_parameters = cast("RopeParameters", config.rope_parameters)
        compress_rope_parameters = yarn_rope_parameters.get("compress")
        if isinstance(compress_rope_parameters, dict):
            yarn_rope_parameters = cast("RopeParameters", compress_rope_parameters)
        max_position_embeddings = yarn_rope_parameters.get(
            "original_max_position_embeddings",
            cast("DeepseekV2Config | DeepseekV4Config", config).max_position_embeddings,
        )
        # DecoderLayers are created with `native.make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = (
            native.enable_dsa_cp()
        )  # TODO: delete this when native.enable_dsa_cp is sunset.

        # ### PATCH START: role-local Attention and remote experts
        self.mlp: (
            AFDDeepseekV4AttentionGateRemoteMoE
            | AFDDeepseekV4RemoteMoE
            | native.DeepseekV4MoE
        )
        if afd_config.role == _ATTENTION_ROLE:
            self.self_attn = native.DeepseekV4Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                reduce_results=not self.use_sequence_parallel_moe,
                need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
            )
            if self.compute_gate_on_attention:
                # Both shells keep the router on Attention; only the transport
                # differs. Choosing by connector keeps the async CAM path unchanged.
                gate_shell = (
                    AFDDeepseekV4AttentionGateRemoteMoE
                    if afd_config.connector == AFD_ASYNC_CONNECTOR
                    else AFDDeepseekV4SyncAttentionGateRemoteMoE
                )
                self.mlp = gate_shell(
                    config=config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.mlp",
                    vllm_config=vllm_config,
                )
            else:
                self.mlp = AFDDeepseekV4RemoteMoE(layer_idx=layer_idx)
        elif afd_config.role == _FFN_ROLE:
            self.self_attn = native.PPMissingLayer()
            _refresh_ascend_fused_moe()
            moe_config = config
            if self.compute_gate_on_attention:
                # The shared experts live wherever the router lives, because they
                # run in the same window as the gate. That is true for either
                # connector, not just for the async one.
                moe_config = copy(config)
                # HF validates this public field as int, but pinned native
                # MoE uses only None to omit shared construction (0 builds a
                # zero-width MLP). Keep the original model config unchanged.
                object.__setattr__(moe_config, "n_shared_experts", None)
            self.mlp = native.DeepseekV4MoE(
                config=moe_config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=False,
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
        topk_ids: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        activation_scale: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        if not isinstance(self.mlp, native.DeepseekV4MoE):
            raise RuntimeError(
                "DSV4 compute_ffn_output requires the FFN role to own the MoE"
            )
        # The configuration decides which protocol this layer speaks; the payload
        # only has to satisfy it. Letting the payload choose would turn a missing or
        # mixed payload into a silent reroute to a different computation.
        cam_payload_present = group_list is not None
        routing_payload_present = topk_ids is not None or topk_weights is not None

        if self.uses_external_routing:
            if cam_payload_present:
                raise RuntimeError(
                    "DSV4 sync external routing received CAM dispatch data as well; "
                    "a layer speaks one protocol",
                )
            if topk_ids is None or topk_weights is None:
                raise RuntimeError(
                    "DSV4 sync external routing requires both topk_ids and "
                    f"topk_weights; got ids={topk_ids is not None}, "
                    f"weights={topk_weights is not None}",
                )
            return self._compute_external_routing_ffn(
                hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                activation_scale=activation_scale,
            )

        if self.compute_gate_on_attention:
            if routing_payload_present:
                raise RuntimeError(
                    "DSV4 async CAM routing received a wire-slot routing payload as "
                    "well; a layer speaks one protocol",
                )
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

        if routing_payload_present or cam_payload_present:
            raise RuntimeError(
                "DSV4 gate-on-FFN received Attention-side routing; the FFN side owns "
                "the gate for this configuration and must compute it itself",
            )
        # Target Hash MoE takes explicit token IDs from the CAM transfer.
        return self.mlp(hidden_states, input_ids=get_forward_context().input_ids)

    def _compute_external_routing_ffn(
        self,
        hidden_states: torch.Tensor,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the FFN on routing the Attention side computed.

        Mirrors ``AscendRoutedExperts.forward`` from the pinned vLLM-Ascend tag, with
        two differences: the top-k comes from the transport instead of a local gate,
        and both the padding/gathering of the routing and the scale it carries are
        supplied by the prepare patch.

        Routed scaling needs no separate step here. The Attention-side selector folds
        ``routed_scaling_factor`` into ``topk_weights``, and ``apply`` consumes those
        weights directly in the token combine, so the factor is applied exactly once.
        Applying it again here would scale the routed output a second time.

        Both ALLGATHER and MC2 are wired. They share this flow because their prepare
        patches both pad the routing, and MC2's additionally splits it for the TP
        sharding before writing the aligned version back. What differs is only whether
        finalize reduces across ranks.

        Any other communication method is rejected rather than silently taking one of
        these routes.
        """

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe.moe_comm_method import (
            AllGatherCommImpl,
            MC2CommImpl,
        )

        from vllm_ascend.ops.fused_moe.moe_utils import (
            maybe_normalize_mxfp_scale_layout,
        )

        from afd_plugin.connectors.npu.external_routing import (
            AFDExternalRoutingInputs,
            require_matching_activation_scale,
        )

        routed_experts = _routed_experts_of(self.mlp)
        comm_method = _EXTRA_CTX.moe_comm_method
        if not isinstance(comm_method, (AllGatherCommImpl, MC2CommImpl)):
            raise RuntimeError(
                "DSV4 external routing is wired for the ALLGATHER and MC2 MoE paths; "
                f"got {type(comm_method).__name__}. The all-to-all paths shard "
                "differently and are not verified yet.",
            )

        # The activation and its scale are a pair or nothing. A quantized activation without
        # its scale cannot be interpreted, and a scale without a quantized activation would
        # make the MoE skip quantizing a model-dtype tensor. The constructor also rejects a
        # malformed scale.
        activation_is_quantized = hidden_states.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        if activation_is_quantized != (activation_scale is not None):
            raise RuntimeError(
                "DSV4 external routing needs the quantized activation and its scale "
                f"together; got activation={hidden_states.dtype}, "
                f"scale={'present' if activation_scale is not None else 'absent'}",
            )
        if activation_scale is not None:
            # Only the target format is accepted. E5M2 would silently take the same route
            # while the weight path is defined for E4M3.
            if hidden_states.dtype is not torch.float8_e4m3fn:
                raise RuntimeError(
                    "DSV4 external routing expects an E4M3 activation for the MXFP8 "
                    f"pass-through; got {hidden_states.dtype}",
                )
            require_matching_activation_scale(hidden_states, activation_scale)

        # The ALLGATHER prepare patch is installed once during runtime start-up, not
        # here: re-installing it per layer forward only repeats work, since the patch
        # already behaves exactly like upstream when no routing was published.

        # The prepare patch reads the routing from this attribute and writes the
        # padded/gathered version back, so the two must be the same object.
        prepare_finalize = comm_method.prepare_finalize
        # The scale travels with the routing so the prepare patch pads, gathers or shards
        # it by exactly the same rule as the activation rows it belongs to.
        prepare_finalize.afd_external_routing = AFDExternalRoutingInputs(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            scale=activation_scale,
        )
        try:
            prepare_output = comm_method.prepare(
                hidden_states=hidden_states,
                # The FFN side owns no router in this configuration.
                router_logits=None,
                replace_allreduce=routed_experts.moe_config.is_sequence_parallel,
                quant_type=routed_experts.quant_type,
            )
            aligned = prepare_finalize.afd_external_routing
            if aligned is None:
                raise RuntimeError(
                    "DSV4 external routing vanished during prepare; the ALLGATHER "
                    "prepare patch did not run",
                )
            # The scale and mask belong to the experts module, which apply reads back.
            # What consumes it fixes the layout, and the paths do not agree:
            #
            #   ALLGATHER  the fused GMM path wants ``(rows, cols // 2, 2)``, which is what
            #              the quant method would have produced had it quantized here.
            #   MC2        the dispatcher's pre-quantized contract takes the scale
            #              two-dimensionally, so normalising here would hand it the wrong
            #              shape.
            #   BF16       no scale at all.
            #
            # The condition names ALLGATHER rather than testing for "not MC2": a path added
            # later should have to state its layout instead of inheriting one.
            pertoken_scale = prepare_output.pertoken_scale
            if pertoken_scale is not None and isinstance(comm_method, AllGatherCommImpl):
                pertoken_scale = maybe_normalize_mxfp_scale_layout(pertoken_scale)
            routed_experts.ascend_pertoken_scale = pertoken_scale
            routed_experts.ascend_mc2_mask = prepare_output.mc2_mask
            try:
                fused = routed_experts.quant_method.apply(
                    layer=routed_experts,
                    x=prepare_output.hidden_states,
                    topk_weights=aligned.topk_weights,
                    topk_ids=aligned.topk_ids,
                    shared_experts=None,
                    shared_experts_input=None,
                )
            finally:
                routed_experts.ascend_pertoken_scale = None
                routed_experts.ascend_mc2_mask = None
            return comm_method.finalize(
                hidden_states=fused.routed_out,
                reduce_results=isinstance(comm_method, AllGatherCommImpl),
                padded_hidden_states_shape=prepare_output.padded_hidden_states_shape,
            )
        finally:
            # The inner finally already clears these before finalize runs, matching
            # the native path's ordering. Clearing again here covers the paths that
            # never reach it: if prepare raises, no other block cleans up, and the
            # next call would start with another call's per-token state.
            prepare_finalize.afd_external_routing = None
            routed_experts.ascend_pertoken_scale = None
            routed_experts.ascend_mc2_mask = None


@native.support_torch_compile
class AFDDeepseekV4Model(native.DeepseekV4Model):
    """DSV4 model with role-local Attention/FFN allocations."""

    fall_back_to_pt_during_load = False

    # Upstream source: vllm-ascend bd42eea01d8312c4df469fabaed45f77a12df4d3,
    # vllm_ascend/models/deepseek_v4/model.py::DeepseekV4Model.__init__.
    # Patch reason: construct role-local decoder layers and retain AFD metadata.
    # Patch functionality: initialize the target SP/PP/HC/MTP model contract
    # without allocating the native unsplit model through super().__init__.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize role-local model
        afd_config = parse_afd_config(vllm_config, validate=False)
        if (
            afd_config.connector == AFD_ASYNC_CONNECTOR
            and not afd_config.compute_gate_on_attention
        ):
            raise ValueError(
                "DSV4 CAMAsyncAFDConnector requires compute_gate_on_attention=true",
            )
        _refresh_ascend_fused_moe()
        nn.Module.__init__(self)
        # ### PATCH END: initialize role-local model

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        # ### PATCH START: AFD model metadata
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.afd_config = afd_config
        self.hidden_size = config.hidden_size
        # ### PATCH END: AFD model metadata
        self.config = config
        self.device = native.current_platform.device_type
        self.use_sequence_parallel_moe = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        # ### PATCH START: Attention-owned index buffer
        if self.is_v32 and afd_config.role == _ATTENTION_ROLE:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # ### PATCH END: Attention-owned index buffer

        # Expose at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
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
        # ### PATCH START: role-local decoder factory
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END: role-local decoder factory

        # Decoder layers run in order, so their OTP collectives can reuse the
        # same address-stable scratch storage during profiling and graph replay.
        self._oproj_buffer_pool = native.OProjBufferPool()
        for layer in self.layers:
            if isinstance(layer, native.DeepseekV4DecoderLayer):
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
                    ),
                }
            )

        self.make_empty_intermediate_tensors = (
            native.make_pp_empty_intermediate_tensors(
                self,
                make_empty_intermediate_tensors,
            )
        )

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_norm = native.RMSNorm(
            hc_dim, eps=config.rms_norm_eps, has_weight=False, dtype=torch.float32
        )

        # Pre-hc_head residual stream buffer for the speculative draft
        # (MTP / DSpark / DFlash). Only needed when the decoder consumes
        # target-model hidden states; allocating it unconditionally would
        # permanently cost max_num_batched_tokens * hc_dim per rank.
        # Aligned with upstream DeepSeekV4 (see vllm PR #50312).
        spec_config = vllm_config.speculative_config
        self._mtp_hidden_buffer: torch.Tensor | None = None
        if (
            native.get_pp_group().is_last_rank
            and spec_config is not None
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
        ):
            # Bind the buffer before compilation: lazy allocation in forward
            # can leave FULL graph replay writing a different tensor from the
            # model-owned residual restored for the replicated PCP draft.
            self._mtp_hidden_buffer = torch.empty(
                (vllm_config.scheduler_config.max_num_batched_tokens, hc_dim),
                # ModelConfig.__post_init__ already resolves the model dtype.
                dtype=cast(torch.dtype, vllm_config.model_config.dtype),
                device=self.device,
            )

        # ### PATCH START: AFD model introspection
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        # ### PATCH END: AFD model introspection

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
            "DSV4 AFD never returns gate outputs through this method; the "
            "Attention-side gate keeps its top-k inside the gate shell"
        )


class AFDDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """Ascend DSV4 causal LM wrapper for AFD."""

    model_cls = AFDDeepseekV4Model

    # DSV4 Hash layers route by token identity. The FFN role does not hold
    # input_ids, so the connector must transport them and the FFN runner
    # installs them in the forward context before the FFN compute.
    afd_requires_input_ids = True

    # The wire slot carries either token ids or Attention-side routing, never both.
    # When the router lives on Attention with the sync connector, the FFN side needs
    # the routing instead of the ids, and the runner must ask for it by name.
    afd_requires_attention_routing = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # Per-instance, because the same model class serves both connectors and the
        # async CAM shell carries its routing inside its own dispatch payload.
        self.afd_requires_attention_routing = bool(
            self.afd_config.compute_gate_on_attention
            and self.afd_config.connector == CAMP2P_CONNECTOR
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    # vLLM 5df63a7 model_loader/utils.py invokes this model hook after quant
    # finalize. AFD's CAM GMM path still needs the scale that native MegaMoE drops.
    def process_weights_after_loading(self) -> None:
        # ### PATCH START: retain FFN gate-on-A W8A8 scales after quant finalize
        if (
            native.current_platform.device_type != "npu"
            or self.afd_role != "ffn"
            or not self.afd_config.compute_gate_on_attention
        ):
            return
        from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
            prepare_attention_gate_w8a8_scales,
        )

        prepare_attention_gate_w8a8_scales(self.model.layers)
        # ### PATCH END: retain FFN gate-on-A W8A8 scales after quant finalize

    def get_async_cam_w4a8_layers(self) -> "tuple[list[W4A8LayerWeights], str]":
        return _extract_async_cam_w4a8_layers(self.model.layers)

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
        # The shared experts follow the router, so their weights load on Attention
        # for either connector; keying this on the async connector alone would
        # construct the modules on the sync path without ever loading them.
        role_weights = _iter_role_weights(
            weights,
            role=self.afd_role,
            attn_owns_gate=bool(self.afd_config.compute_gate_on_attention),
            attention_shared_experts=bool(self.afd_config.compute_gate_on_attention),
        )
        loaded = super().load_weights(role_weights)
        return loaded


def _extract_async_cam_w4a8_layers(
    layers: Iterable[AFDDeepseekV4DecoderLayer],
) -> "tuple[list[W4A8LayerWeights], str]":
    """Expose loaded DeepSeek V4 routed-expert weights to the layered executor."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_ascend.quantization.quant_type import QuantType

    from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights

    layers = [layer for layer in layers if layer.is_moe_layer]
    # Only the FFN role exposes loaded experts; Ascend's factory returns its
    # runner while the routed container retains parameter ownership.
    quant_types = {
        cast("AscendMoERunner", layer.mlp.experts).quant_type for layer in layers
    }
    if quant_types != {QuantType.W4A8}:
        return [], f"non-W4A8 or mixed quantization: {quant_types}"
    weights = []
    for layer in layers:
        moe = cast("native.DeepseekV4MoE", layer.mlp)
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
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_scale_bias",
            "w2_scale_bias",
        ):
            if owner._parameters.get(name) is None:
                raise ValueError(
                    f"layered GMM layer {layer.layer_idx}: missing loaded {name}"
                )
        weights.append(
            W4A8LayerWeights(
                layer_idx=layer.layer_idx,
                # The presence checks above establish these dynamically
                # registered parameters as loaded tensors on the routed owner.
                w13=cast(torch.Tensor, owner.w13_weight),
                w2=cast(torch.Tensor, owner.w2_weight),
                w13_scale=cast(torch.Tensor, owner.w13_weight_scale),
                w2_scale=cast(torch.Tensor, owner.w2_weight_scale),
                w13_bias=cast(torch.Tensor, owner.w13_scale_bias),
                w2_bias=cast(torch.Tensor, owner.w2_scale_bias),
                per_channel=cast(
                    "AscendW4A8DynamicFusedMoEMethod",
                    cast("AscendFusedMoEMethod", owner.quant_method).quant_method,
                ).is_per_channel_weight,
                swiglu_limit=float(cast("float | None", moe.swiglu_limit) or 0.0),
                # DSV4 already applies routed scaling in Attention top-k.
                routed_scaling_factor=1.0,
            )
        )
    # One layered call shares static geometry, quantization, and scaling
    # across every selectable layer; heterogeneous layers use the legacy path.
    if any(weight.signature() != weights[0].signature() for weight in weights):
        return [], "heterogeneous W4A8 geometry, quantization or model semantics"
    return weights, ""


__all__ = [
    "AFDDeepseekV4DecoderLayer",
    "AFDDeepseekV4ForCausalLM",
    "AFDDeepseekV4Model",
]
