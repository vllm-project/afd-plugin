# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector-independent remote dense/MoE proxies and Attention-side MoE
runners; no concrete model imports."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.forward_context import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

UNROUTED_EXPERT_ID = -1
# The only placement validated for native-factory Attention MoE runners.
# DeepEP M2N routes on Attention through its own model shell instead.
ATTENTION_MOE_RUNNER_DEVICE = "cuda"
ATTENTION_MOE_RUNNER_CONNECTOR = "P2pNcclAFDConnector"


def _unroute_tp_padding(topk_ids: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Route the padding rows of this TP rank's ``sequence_parallel_chunk``
    slice of ``num_tokens`` tokens to ``UNROUTED_EXPERT_ID``."""
    local_tokens = topk_ids.shape[0]
    token_index = get_tensor_model_parallel_rank() * local_tokens
    token_index = token_index + torch.arange(local_tokens, device=topk_ids.device)
    return topk_ids.masked_fill(
        (token_index >= num_tokens).unsqueeze(1), UNROUTED_EXPERT_ID
    )


class RemoteFFNProxy(nn.Module):
    """FFN stage executed through the AFD connector.

    By default the proxy is parameter-free: it sends the hidden states and the
    FFN role runs the complete MLP or MoE block. A model whose connector routes
    on Attention (an M2N routed-experts connector such as DeepEP M2N) sets
    ``gate``, ``top_k``, ``renormalize`` and ``tp_size`` after construction,
    from its model and parallel config. The proxy then computes the gate and
    vLLM's default fused top-k (``FusedTopKRouter``; EPLB is not supported)
    and sends ``topk_ids`` and ``topk_weights``; combine returns the top-k
    weighted sum, which replaces the native MoE block's output.

    With Attention TP > 1 every TP rank holds the same tokens after the
    attention all-reduce; dispatching them from every rank would make the FFN
    run each token TP times. Each rank instead pads the tokens to a multiple
    of ``tp_size`` (``sequence_parallel_chunk``), routes and dispatches only
    its slice, and the TP group all-gathers the combined slices, as vLLM's
    sequence-parallel MoE does. Padding rows get ``UNROUTED_EXPERT_ID`` and
    reach no FFN rank; they are dropped after the all-gather.
    """

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.gate: ReplicatedLinear | None = None
        self.top_k = 0
        self.renormalize = False
        self.tp_size = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return self._send_and_receive(hidden_states)
        num_tokens = hidden_states.shape[0]
        if self.tp_size > 1:
            hidden_states = sequence_parallel_chunk(hidden_states)
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        if self.tp_size > 1:
            topk_ids = _unroute_tp_padding(topk_ids, num_tokens)
        output = self._send_and_receive(
            hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[:num_tokens]
        return output

    def _send_and_receive(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        stage_idx = self._send(hidden_states, **send_kwargs)
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return self._receive(hidden_states, stage_idx)

    def _send(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> int:
        """Send this layer's Attention output; return the stage to receive."""
        afd_metadata = self._afd_metadata()
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
        return stage_idx

    def _receive(self, hidden_states: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Receive the FFN output; a connector may write it into
        ``hidden_states``."""
        return self._afd_metadata().connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )

    @staticmethod
    def _afd_metadata() -> AFDForwardContextMetadata:
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteFFNProxy requires AFD forward metadata")
        return afd_metadata


class AFDRemoteMoEMethod(FusedMoEMethodBase):
    """Native MoE method lifecycle without local expert weights or kernels.

    ``apply`` and ``apply_monolithic`` keep the base ``NotImplementedError``,
    so computing experts on the Attention role fails loudly.
    """

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Allocate nothing: the FFN role owns the expert weights."""

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        # No local kernel or all2all buffer needs aligned sizes, so the
        # runner's FusedMoEConfig keeps the model's dimensions.
        return hidden_size, intermediate_size_per_partition


class AFDRemoteRoutedExperts(RoutedExperts):
    """Native routed-experts descriptor that allocates no expert parameters."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        return AFDRemoteMoEMethod(moe_config)


class AFDRemoteMoERunner(MoERunner):
    """Native MoE runner whose gate and experts run on the FFN role.

    The runner keeps the native layer registration but runs no local routing
    or expert kernel: it sends this layer's hidden states through the
    ``RemoteFFNProxy`` transport (send, DBO yield, receive) and returns the
    FFN output.
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: nn.Module | None = None,
        shared_experts: nn.Module | None = None,
        shared_expert_gate: nn.Module | None = None,
        routed_input_transform: nn.Module | None = None,
        routed_output_transform: nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__(
            layer_name=layer_name,
            moe_config=moe_config,
            router=router,
            routed_experts=routed_experts,
            enable_dbo=enable_dbo,
            gate=gate,
            shared_experts=shared_experts,
            shared_expert_gate=shared_expert_gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            routed_scaling_factor=routed_scaling_factor,
        )
        # The native layer name carries the decoder layer index.
        self.ffn_transport = RemoteFFNProxy(layer_idx=self.layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        if shared_experts_input is not None:
            raise NotImplementedError(
                "Attention remote MoE has no local shared experts",
            )
        return self.ffn_transport._send_and_receive(
            hidden_states,
            **self._routing_payload(hidden_states),
        )

    def _routing_payload(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Routing tensors sent with the hidden states; none when FFN gates."""
        return {}


class AFDExternalRoutingMoERunner(AFDRemoteMoERunner):
    """Native MoE runner that applies the gate held on the Attention role.

    As native ``MoERunner._forward_impl`` does when it holds a gate, the
    runner ignores its ``router_logits`` argument and computes the logits
    from the hidden states. It sends them with the hidden states, and the FFN
    role runs its external-router experts.
    """

    def _routing_payload(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert self.gate is not None
        router_logits, _ = self.gate(hidden_states)
        return {"router_logits": router_logits}


def build_attention_moe_runner(
    vllm_config: VllmConfig,
    *,
    gate: nn.Module | None,
    **moe_kwargs: Any,
) -> MoERunner:
    """Build an Attention-role MoE runner through native ``FusedMoEFactory``.

    ``moe_kwargs`` are the native model's ``FusedMoEFactory`` arguments. Pass
    the model's gate exactly when ``compute_gate_on_attention`` is set; the
    runner then applies it and sends the router logits. This function fixes
    the arguments that describe local experts: the runner registers like a
    native MoE layer but allocates no expert weights and runs no expert
    kernel.
    """
    afd_config = parse_afd_config(vllm_config, validate=False)
    placement = (current_platform.device_type, afd_config.connector)
    if placement != (ATTENTION_MOE_RUNNER_DEVICE, ATTENTION_MOE_RUNNER_CONNECTOR):
        raise ValueError(
            f"unsupported Attention remote MoE configuration: {placement!r}",
        )
    if (gate is not None) != afd_config.compute_gate_on_attention:
        raise ValueError(
            "Attention remote MoE takes the model's gate if and only if "
            "compute_gate_on_attention is set",
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_eplb:
        raise RuntimeError(
            "Remote MoE does not support Attention-local EPLB (enable_eplb)",
        )
    if parallel_config.eplb_config.num_redundant_experts != 0:
        raise RuntimeError(
            "Remote MoE does not support Attention-local redundant experts",
        )
    if vllm_config.aux_output_config.enable_return_routed_experts:
        raise RuntimeError(
            "Remote MoE does not support Attention-local routed_experts capture",
        )
    if "routed_scaling_factor" in moe_kwargs and not math.isfinite(
        moe_kwargs["routed_scaling_factor"],
    ):
        raise ValueError("routed_scaling_factor must be finite")
    # A parameter-free container: no local quantized weights, shared experts,
    # transforms or EPLB, and unit TP/DP/PCP sizes, so no expert sharding.
    # The FFN role applies routed_scaling_factor; the runner only records it.
    factory_kwargs: dict[str, Any] = {
        "gate": gate,
        "quant_config": None,
        "shared_experts": None,
        "shared_expert_gate": None,
        "routed_input_transform": None,
        "routed_output_transform": None,
        "n_shared_experts": None,
        "fuse_shared_experts": False,
        "apply_routed_scale_to_output": True,
        "enable_eplb": False,
        "num_redundant_experts": 0,
        "is_sequence_parallel": False,
        "tp_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "runner_cls": (
            AFDExternalRoutingMoERunner if gate is not None else AFDRemoteMoERunner
        ),
        "runner_args": None,
        "routed_experts_cls": AFDRemoteRoutedExperts,
        "routed_experts_args": None,
    }
    reserved = moe_kwargs.keys() & factory_kwargs.keys()
    if reserved:
        raise ValueError(
            "Attention remote MoE factory reserves these arguments: "
            + ", ".join(sorted(reserved)),
        )
    return fused_moe.FusedMoEFactory(**moe_kwargs, **factory_kwargs)


__all__ = [
    "AFDExternalRoutingMoERunner",
    "AFDRemoteMoEMethod",
    "AFDRemoteMoERunner",
    "AFDRemoteRoutedExperts",
    "RemoteFFNProxy",
    "build_attention_moe_runner",
]
