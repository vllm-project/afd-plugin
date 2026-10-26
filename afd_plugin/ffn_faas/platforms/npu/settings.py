# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch settings for the standalone CAM FFN process."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from afd_plugin.config import AFD_ASYNC_CONNECTOR


@dataclass(frozen=True)
class FFNConfig:
    model: Path
    attention_ranks: int
    attention_tp: int
    ffn_ranks: int
    rank: int = 0
    local_rank: int = 0
    device: int = 0
    host: str = "127.0.0.1"
    port: int = 29600
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    full_graph: bool = False
    ffn_two_stream: bool = False
    ffn_compute_aic_num: int | None = None
    ffn_compute_aiv_num: int | None = None
    ffn_combine_aiv_num: int | None = None
    hccl_buffer_size: int | None = None
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            self.attention_ranks,
            self.attention_tp,
            self.ffn_ranks,
            self.max_num_batched_tokens,
            self.max_num_seqs,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "rank counts, TP and scheduler capacities must be positive"
            )
        if self.attention_ranks % self.attention_tp:
            raise ValueError("Attention ranks must be divisible by Attention TP")
        if self.max_num_batched_tokens % self.attention_tp:
            raise ValueError(
                "the CAM dispatch envelope must be divisible by Attention TP"
            )
        if not 0 <= self.rank < self.ffn_ranks or min(self.local_rank, self.device) < 0:
            raise ValueError("invalid F rank or local device")
        if not self.host or not 0 < self.port < 65536:
            raise ValueError("CAM rendezvous requires a host and a valid port")
        if not math.isfinite(self.shutdown_timeout) or self.shutdown_timeout <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        optional_positive = (
            self.ffn_compute_aic_num,
            self.ffn_compute_aiv_num,
            self.ffn_combine_aiv_num,
            self.hccl_buffer_size,
        )
        if any(value is not None and value <= 0 for value in optional_positive):
            raise ValueError("explicit CAM core and HCCL budgets must be positive")

    def additional_config(self) -> dict:
        extra = {
            "dynamicQuant": 1,
            "attn_ranks_per_dp": self.attention_tp,
            "ffn_two_stream": self.ffn_two_stream,
        }
        for name, value in (
            ("ffn_compute_aic_num", self.ffn_compute_aic_num),
            ("ffn_compute_aiv_num", self.ffn_compute_aiv_num),
            ("ffn_combine_aiv_num", self.ffn_combine_aiv_num),
            ("hccl_buffer_size", self.hccl_buffer_size),
        ):
            if value is not None:
                extra[name] = value
        return {
            "afd": {
                "connector": AFD_ASYNC_CONNECTOR,
                "role": "ffn",
                "async_dp": True,
                "host": self.host,
                "port": self.port,
                "num_attention_ranks": self.attention_ranks,
                "num_ffn_ranks": self.ffn_ranks,
                "compute_gate_on_attention": True,
                "connector_extra_config": extra,
            },
            "enable_prefill_mc2": False,
            "enable_force_eplb": False,
            "enable_cpu_binding": True,
            "ascend_fusion_config": {"fusion_ops_gmmswigluquant": True},
        }


def validate_static_experts(
    *,
    enable_eplb: bool,
    enable_elastic_ep: bool,
    redundant_experts: int,
    dynamic_eplb: bool = False,
) -> None:
    """Keep native expert state resident without enabling unsupported relocation."""
    if enable_eplb or enable_elastic_ep or redundant_experts != 0 or dynamic_eplb:
        raise ValueError(
            "standalone CAM requires static expert placement; EPLB, elastic EP "
            "and redundant experts are not supported"
        )


def validate_native_expert_map(
    *, num_experts: int, ffn_ranks: int, ffn_rank: int, expert_map: list[int] | None
) -> frozenset[int]:
    """Require the native sharding to match CAM's contiguous expert placement."""
    if ffn_ranks <= 0 or num_experts % ffn_ranks or not 0 <= ffn_rank < ffn_ranks:
        raise ValueError("CAM experts require a divisible, valid F topology")
    local_experts = num_experts // ffn_ranks
    if expert_map is None:
        if ffn_ranks != 1:
            raise ValueError("sharded native experts require an explicit map")
        # The native factory omits the map when every expert is local.
        return frozenset(range(num_experts))
    if len(expert_map) != num_experts:
        raise ValueError("native expert map has the wrong global expert count")
    owned = frozenset(expert for expert, local in enumerate(expert_map) if local >= 0)
    if sorted(expert_map[expert] for expert in owned) != list(range(local_experts)):
        raise ValueError("native expert map is not a local bijection")
    first = ffn_rank * local_experts
    end = first + local_experts
    if owned != frozenset(range(first, end)) or expert_map[first:end] != list(
        range(local_experts)
    ):
        raise ValueError("native placement differs from contiguous CAM routing")
    return owned
