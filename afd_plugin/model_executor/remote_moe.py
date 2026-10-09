# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native MoE construction and runners for Attention-to-FFN handoff."""

import importlib
import math
from abc import abstractmethod
from types import MappingProxyType
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

from afd_plugin.config import AFD_ASYNC_CONNECTOR, CAMP2P_CONNECTOR, parse_afd_config
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


_ATTENTION_MOE_RUNNERS = MappingProxyType(
    {
        ("cuda", "P2pNcclAFDConnector", False): (
            "afd_plugin.model_executor.remote_moe",
            "AFDRemoteMoERunner",
        ),
        ("cuda", "P2pNcclAFDConnector", True): (
            "afd_plugin.model_executor.remote_moe",
            "AFDExternalRoutingMoERunner",
        ),
        ("npu", CAMP2P_CONNECTOR, False): (
            "afd_plugin.model_executor.remote_moe",
            "AFDRemoteMoERunner",
        ),
        ("npu", AFD_ASYNC_CONNECTOR, True): (
            "afd_plugin.model_executor.npu.remote_moe",
            "AFDCAMAsyncMoERunner",
        ),
    }
)


def build_attention_moe_runner(
    vllm_config: VllmConfig,
    *,
    gate: torch.nn.Module | None = None,
    attention_shared_experts: torch.nn.Module | None = None,
    num_shared_experts: int | None = None,
    shared_output_divisor_fp16: float = 1.0,
    **moe_kwargs: Any,
) -> MoERunner:
    """Construct the selected remote runner through the live native factory."""
    afd_config = parse_afd_config(vllm_config, validate=False)
    device_type = current_platform.device_type
    key = (device_type, afd_config.connector, afd_config.compute_gate_on_attention)
    if key not in _ATTENTION_MOE_RUNNERS:
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
    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        if vllm_config.additional_config.get("mix_placement", False):
            raise RuntimeError(
                "Async CAM uses routed-only expert IDs "
                "and does not support mix_placement"
            )
        if gate is None:
            raise ValueError("CAMAsync remote MoE requires an Attention gate")
    elif attention_shared_experts is not None:
        raise ValueError("attention_shared_experts requires CAMAsync remote MoE")
    if attention_shared_experts is not None and (
        not math.isfinite(shared_output_divisor_fp16) or shared_output_divisor_fp16 <= 0
    ):
        raise ValueError(
            "shared_output_divisor_fp16 must be finite and positive "
            "when attention_shared_experts is provided"
        )
    if not math.isfinite(moe_kwargs.get("routed_scaling_factor", 1.0)):
        raise ValueError("routed_scaling_factor must be finite")
    if device_type == "npu":
        from afd_plugin.model_executor.npu.remote_moe import (
            validate_remote_moe_config,
        )

        validate_remote_moe_config()

    module_path, class_name = _ATTENTION_MOE_RUNNERS[key]
    runner_cls = vars(importlib.import_module(module_path))[class_name]
    runner_args = None
    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        runner_args = {
            "num_shared_experts": num_shared_experts,
            "attention_shared_experts": attention_shared_experts,
            "shared_output_divisor_fp16": shared_output_divisor_fp16,
        }
    else:
        # Synchronous routing belongs to the model's outer gate or the FFN role.
        gate = None

    # Unit dimensions describe a non-computing container, not FFN topology.
    factory_kwargs = dict(
        quant_config=None,
        gate=gate,
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
        runner_args=runner_args,
        routed_experts_cls=AFDRemoteRoutedExperts,
    )
    reserved = moe_kwargs.keys() & factory_kwargs.keys()
    if reserved:
        raise ValueError(
            "Attention remote MoE factory reserves these arguments: "
            + ", ".join(sorted(reserved))
        )
    # Resolve the live package factory so Ascend's platform patch is honored.
    return fused_moe.FusedMoE(**moe_kwargs, **factory_kwargs)


class AFDRemoteMoERunnerBase(MoERunner):
    """Native lifecycle for parameter-free remote experts."""

    @property
    def is_internal_router(self) -> bool:
        return True

    def maybe_init_modular_kernel(self) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class AFDRemoteMoERunner(AFDRemoteMoERunnerBase):
    """Delegate synchronous MoE execution to the FFN role."""

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
