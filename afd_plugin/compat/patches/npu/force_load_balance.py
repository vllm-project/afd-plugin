# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch vllm-ascend W8A8 MoE to support force load balance.

This module patches only the Ascend W8A8 FusedMoE path. When
``enable_force_load_balance`` is set in ``additional_config``, routed
``topk_ids`` are replaced with deterministic fake expert ids before
``build_fused_experts_input``. This keeps routed-token volume evenly balanced
across EP ranks for communication profiling.

Force load balance changes model outputs. It is a benchmark/profiling switch,
not a production correctness feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)

_FORCE_LB_DETERMINISTIC_SEED = 1024


@dataclass(frozen=True)
class ForceLoadBalanceConfig:
    """Force-load-balance parameters for one AscendFusedMoE layer.

    Args:
        n_routed_experts: Number of routed experts in the MoE layer.
        ep_size: Number of expert-parallel ranks.
        ep_rank: Source expert-parallel rank that builds the fake routing cycle.
        top_k: Number of routed experts selected for each token.
        topn_per_rank: Number of local experts per EP rank used by the fake
            routing cycle. A value of 0 means all routed experts participate.
    """

    n_routed_experts: int
    ep_size: int
    ep_rank: int
    top_k: int
    topn_per_rank: int


def _get_force_lb_max_tokens(vllm_config: VllmConfig) -> int:
    max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", None)
    if not isinstance(max_tokens, int):
        max_tokens = 128
    return max(max_tokens, 1)


def _validate_force_lb_config(config: ForceLoadBalanceConfig) -> None:
    assert config.ep_size > 0, "ep_size must be positive"
    assert 0 <= config.ep_rank < config.ep_size, (
        "ep_rank must be within the expert-parallel group"
    )
    assert config.n_routed_experts % config.ep_size == 0, (
        "force load balance requires n_routed_experts to be divisible by ep_size"
    )

    if config.topn_per_rank == 0:
        return

    assert config.topn_per_rank > 0, "force_load_balance_topn_per_rank must be >= 0"
    local_routed_experts = config.n_routed_experts // config.ep_size
    assert config.topn_per_rank <= local_routed_experts, (
        "force_load_balance_topn_per_rank exceeds routed experts on each FFN rank"
    )
    assert config.top_k <= config.topn_per_rank * config.ep_size, (
        "top_k must be <= force_load_balance_topn_per_rank * ep_size"
    )


def _build_expert_cycle(
    config: ForceLoadBalanceConfig,
    device: torch.device,
) -> torch.Tensor:
    local_routed_experts = config.n_routed_experts // config.ep_size
    if config.topn_per_rank > 0:
        per_rank_cycles = [
            torch.arange(
                rank * local_routed_experts,
                rank * local_routed_experts + config.topn_per_rank,
                device=device,
                dtype=torch.int32,
            )
            for rank in range(config.ep_size)
        ]
        expert_cycle = torch.cat(per_rank_cycles, dim=0)
    else:
        generator_device = torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(_FORCE_LB_DETERMINISTIC_SEED)
        expert_cycle = torch.randperm(
            config.n_routed_experts,
            generator=generator,
            device=generator_device,
            dtype=torch.int32,
        ).to(device=device, non_blocking=True)

    # Shift every expert by whole target-rank blocks so source EP ranks use
    # different phases without changing the deterministic cycle order.
    source_rank_expert_offset = config.ep_rank * local_routed_experts
    return (expert_cycle + source_rank_expert_offset) % config.n_routed_experts


def _build_topk_buffer(
    config: ForceLoadBalanceConfig,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    expert_cycle = _build_expert_cycle(config, device)
    total_needed = max_tokens * config.top_k
    repeat_times = (total_needed + expert_cycle.numel() - 1) // expert_cycle.numel()
    expanded = expert_cycle.repeat(repeat_times)[:total_needed]
    return expanded.reshape(max_tokens, config.top_k)


def _init_force_lb_buffer(
    method: AscendW8A8DynamicFusedMoEMethod,
    config: ForceLoadBalanceConfig,
    max_tokens: int,
    device: torch.device,
) -> None:
    _validate_force_lb_config(config)
    buffer = _build_topk_buffer(config, max_tokens, device)

    method.force_lb_fake_topk_buffer = buffer
    method.max_force_lb_tokens = max_tokens

    logger.info(
        "AFD force load balance buffer initialized: ep_size=%s top_k=%s"
        " topn_per_rank=%s shape=%s preview=%s",
        config.ep_size,
        config.top_k,
        config.topn_per_rank,
        tuple(buffer.shape),
        buffer[: min(8, max_tokens)].cpu().tolist(),
    )


def _get_force_lb_topk_ids(
    method: AscendW8A8DynamicFusedMoEMethod,
    config: ForceLoadBalanceConfig,
    batch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    buffer = method.force_lb_fake_topk_buffer
    if buffer is None:
        raise RuntimeError("force_lb_fake_topk_buffer is not initialized")

    if batch_tokens > buffer.size(0):
        new_max_tokens = max(batch_tokens, buffer.size(0) * 2)
        logger.warning(
            "Growing AFD force load balance buffer: old_tokens=%s new_tokens=%s",
            buffer.size(0),
            new_max_tokens,
        )
        _init_force_lb_buffer(method, config, new_max_tokens, device)
        buffer = method.force_lb_fake_topk_buffer
        assert buffer is not None

    if buffer.device != device:
        buffer = buffer.to(device, non_blocking=True)
        method.force_lb_fake_topk_buffer = buffer

    return buffer[:batch_tokens, : config.top_k]


# Upstream: vllm-ascend bd69bad88fc19e1aeeea585416d408df8bda8fef
# quantization/methods/w8a8/w8a8_dynamic.py::AscendW8A8DynamicFusedMoEMethod
# Patch reason: AFD needs configuration after the construction context ends.
# Patch functionality: capture the benchmark switch and lazy buffer capacity.
# Signature: matches upstream; no added parameters.
def __init__(self):
    vllm_config = get_current_vllm_config()
    ascend_config = get_ascend_config()
    self.dynamic_eplb = (
        False
        if vllm_config.use_v2_model_runner
        else ascend_config.eplb_config.dynamic_eplb
    )
    self.use_expert_weight_list = self.dynamic_eplb or (
        vllm_config.use_v2_model_runner is True
        and vllm_config.parallel_config.enable_eplb is True
    )
    self.in_dtype = vllm_config.model_config.dtype
    try:
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
    except AttributeError:
        logger.warning_once(
            "[vllm-ascend/W8A8_DYNAMIC] MC2 group metadata unavailable, "
            "falling back to empty moe_all_to_all_group_name."
        )
        self.moe_all_to_all_group_name = ""

    # ### PATCH START: capture AFD force-load-balance configuration
    additional_config = vllm_config.additional_config or {}
    self.enable_force_load_balance = bool(
        additional_config.get("enable_force_load_balance", False)
    )
    self.force_load_balance_topn_per_rank = int(
        additional_config.get("force_load_balance_topn_per_rank", 0)
    )
    self.max_force_lb_tokens = _get_force_lb_max_tokens(vllm_config)
    self.force_lb_fake_topk_buffer: torch.Tensor | None = None
    # ### PATCH END: capture AFD force-load-balance configuration


# Upstream: vllm-ascend bd69bad88fc19e1aeeea585416d408df8bda8fef
# quantization/methods/w8a8/w8a8_dynamic.py::AscendW8A8DynamicFusedMoEMethod
# Patch reason: AFD communication profiling needs deterministic expert IDs.
# Patch functionality: replace routed IDs while retaining native payload assembly
# and profile/EPLB precedence. Remove when native routing offers this policy.
# Signature: matches upstream; no added parameters.
def apply(
    self,
    layer: "AscendRoutedExperts",  # noqa: UP037
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: Any | None,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor:
    lora_context = getattr(layer, "_ascend_moe_lora_context", None)
    assert topk_ids is not None
    assert topk_weights is not None

    # ### PATCH START: deterministic AFD routing after native selection
    # Native profiling and forced EPLB selection take precedence over the
    # benchmark switch captured during construction.
    if (
        self.enable_force_load_balance
        and not _EXTRA_CTX.in_profile_run
        and not get_ascend_config().enable_force_eplb
    ):
        shared_topk_count = (layer.n_shared_experts or 0) if layer.mix_placement else 0
        routed_topk = topk_ids.shape[1] - shared_topk_count
        force_lb_config = ForceLoadBalanceConfig(
            n_routed_experts=layer.moe_config.num_logical_experts,
            ep_size=layer.moe_config.ep_size,
            ep_rank=layer.moe_config.ep_rank,
            top_k=routed_topk,
            topn_per_rank=self.force_load_balance_topn_per_rank,
        )
        if self.force_lb_fake_topk_buffer is None:
            _init_force_lb_buffer(
                self, force_lb_config, self.max_force_lb_tokens, topk_ids.device
            )
        fake_routed_topk_ids = _get_force_lb_topk_ids(
            self, force_lb_config, topk_ids.shape[0], topk_ids.device
        ).to(topk_ids.dtype)
        if shared_topk_count:
            topk_ids = torch.cat(
                [fake_routed_topk_ids, topk_ids[:, routed_topk:]], dim=1
            )
        else:
            topk_ids = fake_routed_topk_ids
    # ### PATCH END: deterministic AFD routing after native selection

    activation = getattr(layer, "activation", "silu")
    moe_comm_method = _EXTRA_CTX.moe_comm_method
    return moe_comm_method.fused_experts(
        fused_experts_input=build_fused_experts_input(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=layer,
            quant_type=self.quant_type,
            dynamic_eplb=self.use_expert_weight_list,
            expert_map=layer.ascend_expert_map,
            global_redundant_expert_num=layer.global_redundant_expert_num,
            mc2_mask=layer.ascend_mc2_mask,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            pertoken_scale=layer.ascend_pertoken_scale,
            activation=activation,
            lora_context=lora_context,
        ),
        quant_method=self,
    )


AscendW8A8DynamicFusedMoEMethod.__init__ = __init__
AscendW8A8DynamicFusedMoEMethod.apply = apply


__all__: list[str] = []
