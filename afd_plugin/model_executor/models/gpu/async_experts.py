# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native modular expert kernels over rows already dispatched by AFD."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from afd_plugin.connectors.metadata import AFDF2ATransferPayload

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

PRE_ROUTED_TOP_K = 1
LOCAL_MOE_ALL2ALL_BACKEND = "allgather_reducescatter"


class NativeMoEModel(Protocol):
    FusedMoEFactory: Callable[..., "MoERunner"]


class AsyncGpuRoutedExperts(RoutedExperts):
    """Keep native expert placement, with local-only kernel preparation."""

    afd_global_expert_ids: torch.Tensor

    # Native FusedMoEFactory's routed_experts_cls extension point supplies the
    # exact RoutedExperts signature. AFD already dispatched the rows; local DP1
    # preparation must be selected before quantization chooses its backend.
    # Upstream: vLLM 5df63a7203cdb9328d824a27b682e77c7b9928be,
    # model_executor/layers/fused_moe/routed_experts.py::RoutedExperts.__init__.
    # Delegation retains weight allocation/loading and native quantization.
    # Remove this adapter when native experts accept pre-dispatched rows.
    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str | None = "up_proj",
        is_fused_checkpoint_transposed: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        # ### PATCH START: local preparation, unchanged EP weight ownership.
        dispatch_top_k = moe_config.experts_per_token
        moe_config.experts_per_token = PRE_ROUTED_TOP_K
        moe_config.max_num_tokens *= dispatch_top_k
        moe_config.max_capture_size *= dispatch_top_k
        moe_config.moe_parallel_config = replace(
            moe_config.moe_parallel_config,
            dp_size=1,
            dp_rank=0,
            sp_size=1,
            all2all_backend=LOCAL_MOE_ALL2ALL_BACKEND,
        )
        # ### PATCH END: local preparation, unchanged EP weight ownership.
        super().__init__(
            layer_name,
            params_dtype,
            moe_config,
            quant_config,
            expert_map_manager,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            is_fused_checkpoint_transposed,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
            e_score_correction_bias,
            apply_router_weight_on_input,
        )
        self.register_buffer(
            "afd_global_expert_ids",
            torch.tensor(
                self.expert_map_manager.get_local_expert_ids(),
                dtype=torch.int32,
                device=next(self.parameters()).device,
            ),
            persistent=False,
        )
        if self.quant_method.is_monolithic:
            raise ValueError(
                "GpuAsyncAFDConnector requires a modular MoE backend that accepts "
                "Attention-side top-k routing; select a modular --moe-backend"
            )


@contextmanager
def use_local_expert_factory(native_model: NativeMoEModel) -> Iterator[None]:
    """Select the native factory's experts class only during FFN construction.

    The native DeepSeek constructors do not expose routed_experts_cls. Scope
    that supported factory argument to their synchronous construction and
    restore the module binding even if construction fails. Weight processing
    later uses each instance's already-local MoE configuration.
    """
    original = native_model.FusedMoEFactory
    native_model.FusedMoEFactory = partial(
        original, routed_experts_cls=AsyncGpuRoutedExperts
    )
    try:
        yield
    finally:
        native_model.FusedMoEFactory = original


def compute_local_experts(
    experts: AsyncGpuRoutedExperts,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    shared_experts: nn.Module | None = None,
    expand_x_shared: torch.Tensor | None = None,
    shared_scale: float = 1.0,
) -> AFDF2ATransferPayload:
    """Compute each pre-routed row once; the connector performs top-k combine."""
    num_rows = hidden_states.shape[0]
    if num_rows:
        expert_ids = torch.repeat_interleave(
            experts.afd_global_expert_ids,
            group_list.to(torch.int64),
            output_size=num_rows,
        ).unsqueeze(1)
        row_weights = torch.ones(
            (num_rows, 1), dtype=torch.float32, device=hidden_states.device
        )
        routed_output = experts.forward_modular(hidden_states, row_weights, expert_ids)
        if not isinstance(routed_output, torch.Tensor):
            raise RuntimeError("AFD local experts require finalized per-row output")
    else:
        routed_output = torch.empty_like(hidden_states)
    shared_output = None
    if (
        shared_experts is not None
        and expand_x_shared is not None
        and expand_x_shared.shape[0]
    ):
        shared_output = shared_experts(expand_x_shared)
        if shared_scale != 1.0:
            shared_output = shared_output * shared_scale
    return AFDF2ATransferPayload(routed_output, shared_output)
