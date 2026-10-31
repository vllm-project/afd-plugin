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

from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any

import torch
import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
from vllm.config import VllmConfig, get_current_vllm_config
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.methods.w8a8_dynamic import (
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

    fused_moe_module.logger.info(
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
        fused_moe_module.logger.warning(
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


_ORIGINAL_INIT_ATTR = "_afd_force_lb_original_init"
_ORIGINAL_APPLY_ATTR = "_afd_force_lb_original_apply"

if not hasattr(AscendW8A8DynamicFusedMoEMethod, _ORIGINAL_INIT_ATTR):
    setattr(
        AscendW8A8DynamicFusedMoEMethod,
        _ORIGINAL_INIT_ATTR,
        AscendW8A8DynamicFusedMoEMethod.__init__,
    )
if not hasattr(AscendW8A8DynamicFusedMoEMethod, _ORIGINAL_APPLY_ATTR):
    setattr(
        AscendW8A8DynamicFusedMoEMethod,
        _ORIGINAL_APPLY_ATTR,
        AscendW8A8DynamicFusedMoEMethod.apply,
    )

_ORIGINAL_INIT = getattr(AscendW8A8DynamicFusedMoEMethod, _ORIGINAL_INIT_ATTR)
_ORIGINAL_APPLY = getattr(AscendW8A8DynamicFusedMoEMethod, _ORIGINAL_APPLY_ATTR)


# ### PATCH START: capture AFD force-load-balance configuration
def __init__(self):
    """Run the version-specific upstream initializer, then capture AFD state."""

    _ORIGINAL_INIT(self)
    vllm_config = get_current_vllm_config()
    additional_config = vllm_config.additional_config or {}
    self.enable_force_load_balance = bool(
        additional_config.get("enable_force_load_balance", False)
    )
    self.force_load_balance_topn_per_rank = int(
        additional_config.get("force_load_balance_topn_per_rank", 0)
    )
    self.max_force_lb_tokens = _get_force_lb_max_tokens(vllm_config)
    self.force_lb_fake_topk_buffer: torch.Tensor | None = None


def _clone_apply_with_selector(selector: Callable[..., Any]) -> Callable[..., Any]:
    """Clone upstream ``apply`` with a per-call expert selector override."""

    apply_globals = dict(_ORIGINAL_APPLY.__globals__)
    apply_globals["select_experts"] = selector
    cloned = FunctionType(
        _ORIGINAL_APPLY.__code__,
        apply_globals,
        name=_ORIGINAL_APPLY.__name__,
        argdefs=_ORIGINAL_APPLY.__defaults__,
        closure=_ORIGINAL_APPLY.__closure__,
    )
    cloned.__kwdefaults__ = _ORIGINAL_APPLY.__kwdefaults__
    return cloned


# Patch reason: AFD profiling needs deterministic balanced routed expert ids.
# Patch functionality: delegates the complete MoE path to the installed
# vllm-ascend version and overrides only its local ``select_experts`` result.
# Signature: matches vllm-ascend 0.23 RFC and the frozen 0.26 baseline.
def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    is_prefill: bool = True,
    enable_force_load_balance: bool = False,
    log2phy: torch.Tensor | None = None,
    global_redundant_expert_num: int = 0,
    pertoken_scale: Any | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    mc2_mask: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> torch.Tensor:
    upstream_apply = _ORIGINAL_APPLY

    if self.enable_force_load_balance and not enable_force_load_balance:
        if getattr(self, "multistream_overlap_gate", False):
            raise RuntimeError(
                "AFD force load balance is incompatible with multistream_overlap_gate"
            )

        upstream_selector = upstream_apply.__globals__.get("select_experts")
        if upstream_selector is None:
            raise RuntimeError("vllm-ascend W8A8 apply has no select_experts hook")

        n_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=n_shared_experts,
        )
        force_lb_config = ForceLoadBalanceConfig(
            n_routed_experts=num_logical_experts,
            ep_size=int(layer.moe_config.ep_size),
            ep_rank=int(layer.moe_config.ep_rank),
            top_k=top_k,
            topn_per_rank=self.force_load_balance_topn_per_rank,
        )

        def select_experts_with_force_lb(*args, **kwargs):
            topk_weights, topk_ids = upstream_selector(*args, **kwargs)
            assert topk_ids is not None
            if self.force_lb_fake_topk_buffer is None:
                _init_force_lb_buffer(
                    self,
                    force_lb_config,
                    self.max_force_lb_tokens,
                    topk_ids.device,
                )
            fake_ids = _get_force_lb_topk_ids(
                self,
                force_lb_config,
                topk_ids.shape[0],
                topk_ids.device,
            ).to(topk_ids.dtype)
            if getattr(layer, "mix_placement", False):
                fake_ids = torch.cat([fake_ids, topk_ids[:, top_k:]], dim=1)
            return topk_weights, fake_ids

        upstream_apply = _clone_apply_with_selector(select_experts_with_force_lb)

    return upstream_apply(
        self,
        layer=layer,
        x=x,
        router_logits=router_logits,
        top_k=top_k,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_experts=num_experts,
        expert_map=expert_map,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        is_prefill=is_prefill,
        enable_force_load_balance=enable_force_load_balance,
        log2phy=log2phy,
        global_redundant_expert_num=global_redundant_expert_num,
        pertoken_scale=pertoken_scale,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        mc2_mask=mc2_mask,
        tid2eid=tid2eid,
    )


AscendW8A8DynamicFusedMoEMethod.__init__ = __init__
AscendW8A8DynamicFusedMoEMethod.apply = apply
# ### PATCH END: capture AFD force-load-balance configuration


__all__: list[str] = []
