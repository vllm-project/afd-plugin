# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch settings for the standalone GPU DeepEP M2N FFN process.

vLLM's native engine CLI and ``additional_config["afd"]`` own the engine, AFD
topology and connector options; ``GPUFFNConfig`` keeps only the launcher and
shutdown facts plus the options the FFN service itself fixes (payload format,
execution mode and lane count). Connector option values stay with
``DeepEPM2NConfig``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from afd_plugin.ffn_faas.platforms.launch import validate_ffn_rank_settings

# The only qualified payload: BF16 rows against BF16 expert weights.
BF16_PAYLOAD = "bf16"
SINGLE_LANE = 1
# DBO runs two micro-batches, one DeepEP lane each.
DBO_LANES = 2
FFN_GRAPH_MODE = "FULL_DECODE_ONLY"
EAGER_MODE = "eager"


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/settings.py::FFNConfig.
# Reuse: PARTIAL - the launch module owns rank and shutdown validation; GPU
#   also fixes the payload format, the execution mode and the lane count.
@dataclass(frozen=True)
class GPUFFNConfig:
    """Frozen settings of one F rank in a standalone DeepEP M2N deployment."""

    model: Path
    ffn_ranks: int
    rank: int = 0
    local_rank: int = 0
    graph: bool = False
    dbo: bool = False
    connector_extra_config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        """Reject launches no Attention peer could pair with, on the CPU.

        Checks this rank, the shutdown timeout, and the two connector options
        the service fixes: ``payload_format`` must be ``bf16`` and an explicit
        ``num_lanes`` must equal the lane count ``--enable-dbo`` implies (1
        without, 2 with). Every other connector option is validated by
        ``DeepEPM2NConfig.from_mapping`` once the backend runs.
        """
        validate_ffn_rank_settings(
            ffn_ranks=self.ffn_ranks,
            rank=self.rank,
            local_rank=self.local_rank,
            shutdown_timeout=self.shutdown_timeout,
        )
        payload_format = self.connector_extra_config.get("payload_format", BF16_PAYLOAD)
        if payload_format != BF16_PAYLOAD:
            raise ValueError(
                f"GPU standalone FFN supports only payload_format={BF16_PAYLOAD!r}, "
                f"got {payload_format!r}"
            )
        num_lanes = self.connector_extra_config.get("num_lanes")
        if num_lanes is not None and num_lanes != self.num_lanes:
            raise ValueError(
                f"connector num_lanes={num_lanes!r} disagrees with "
                f"{'--enable-dbo' if self.dbo else 'no --enable-dbo'} "
                f"({self.num_lanes} lane(s))"
            )

    @property
    def num_lanes(self) -> int:
        """DeepEP lanes this rank creates: two under DBO, otherwise one."""
        return DBO_LANES if self.dbo else SINGLE_LANE

    @property
    def graph_mode(self) -> str:
        """Status name of the FFN execution mode."""
        return FFN_GRAPH_MODE if self.graph else EAGER_MODE
