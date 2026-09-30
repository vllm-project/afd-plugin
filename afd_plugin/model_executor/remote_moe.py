# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native MoE construction and runners for Attention-to-FFN handoff."""

import math
from typing import Any

import torch
from vllm.config import VllmConfig
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
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


class AFDRemoteMoEMethod(FusedMoEMethodBase):
    """Keep the native lifecycle without local expert weights or kernels."""

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        pass

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        return None

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        return hidden_size, intermediate_size_per_partition


class AFDRemoteRoutedExperts(RoutedExperts):
    """Use the native expert descriptor without allocating expert parameters."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        return AFDRemoteMoEMethod(moe_config)


def build_attention_moe_runner(
    vllm_config: VllmConfig,
    **moe_kwargs: Any,
) -> MoERunner:
    """Construct a CUDA P2P remote runner through the live native factory."""
    afd_config = parse_afd_config(vllm_config, validate=False)
    key = (
        current_platform.device_type,
        afd_config.connector,
        afd_config.compute_gate_on_attention,
    )
    if key[:2] != ("cuda", "P2pNcclAFDConnector"):
        raise ValueError(f"unsupported Attention remote MoE configuration: {key!r}")
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_eplb:
        raise RuntimeError(
            "Remote MoE does not support Attention-local EPLB (enable_eplb)",
        )
    if parallel_config.eplb_config.num_redundant_experts != 0:
        raise RuntimeError(
            "Remote MoE does not support Attention-local redundant experts",
        )
    model_config = vllm_config.model_config
    if model_config is not None and model_config.enable_return_routed_experts:
        raise RuntimeError(
            "Remote MoE does not support Attention-local routed_experts capture",
        )
    if not math.isfinite(moe_kwargs.get("routed_scaling_factor", 1.0)):
        raise ValueError("routed_scaling_factor must be finite")
    runner_cls = (
        AFDExternalRoutingMoERunner
        if afd_config.compute_gate_on_attention
        else AFDRemoteMoERunner
    )
    # Unit dimensions describe a non-computing container, not FFN topology.
    factory_kwargs = dict(
        quant_config=None,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        n_shared_experts=None,
        apply_routed_scale_to_output=True,
        enable_eplb=False,
        num_redundant_experts=0,
        is_sequence_parallel=False,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        runner_cls=runner_cls,
        runner_args=None,
        routed_experts_cls=AFDRemoteRoutedExperts,
    )
    reserved = moe_kwargs.keys() & factory_kwargs.keys()
    if reserved:
        raise ValueError(
            "Attention remote MoE factory reserves these arguments: "
            + ", ".join(sorted(reserved))
        )
    return fused_moe.FusedMoE(**moe_kwargs, **factory_kwargs)


class AFDRemoteMoERunner(MoERunner):
    """Delegate synchronous MoE execution to the FFN role."""

    @property
    def is_internal_router(self) -> bool:
        return True

    def maybe_init_modular_kernel(self) -> None:
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise NotImplementedError(
                "experts-boundary input_ids transport is not implemented",
            )
        send_kwargs = (
            {} if self.is_internal_router else {"router_logits": router_logits}
        )
        return remote_ffn_forward(hidden_states, layer_idx=self.layer_id, **send_kwargs)


class AFDExternalRoutingMoERunner(AFDRemoteMoERunner):
    """Transfer model-computed router logits with the hidden states."""

    @property
    def is_internal_router(self) -> bool:
        return False


def remote_ffn_forward(
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    **send_kwargs: torch.Tensor,
) -> torch.Tensor:
    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("RemoteFFNProxy requires AFD forward metadata")
    forward_context = get_forward_context()
    stage_idx = int(
        getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
    )
    afd_metadata.stage_idx = stage_idx
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=layer_idx,
        stage_idx=stage_idx,
        seq_len=int(hidden_states.shape[0]),
    )
    context = AFDTransferContext(metadata=metadata)
    afd_metadata.connector.send_attn_output(
        hidden_states,
        context,
        **send_kwargs,
    )
    hidden_states = maybe_apply_dbo_yield(hidden_states, role="attention")
    return afd_metadata.connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=stage_idx,
    )
