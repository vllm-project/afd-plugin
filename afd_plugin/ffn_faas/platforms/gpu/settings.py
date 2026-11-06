# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch settings for the standalone GPU DeepEP M2N FFN process.

``GPUFFNConfig`` produces the same AFD config a Worker-path FFN recipe passes
to ``vllm serve``, plus the ``EngineArgs`` of an F-only world launched by
``torchrun``. Connector option values stay with ``DeepEPM2NConfig``; only the
options the FFN service itself decides (payload format, lane count) are
checked here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig
from afd_plugin.ffn_faas.platforms.launch import validate_common_ffn_launch

# The only qualified payload: BF16 rows against BF16 expert weights.
BF16_PAYLOAD = "bf16"
SINGLE_LANE = 1
# DBO runs two micro-batches, one DeepEP lane each.
DBO_LANES = 2
FFN_GRAPH_MODE = "FULL_DECODE_ONLY"
EAGER_MODE = "eager"
EXTERNAL_LAUNCHER = "external_launcher"
DEFAULT_AFD_PORT = AFDConfig.port
DEFAULT_AFD_TIMEOUT_S = AFDConfig.afd_process_group_timeout_s


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/settings.py::FFNConfig.
# Reuse: PARTIAL - the launch module owns rank, rendezvous and shutdown
#   validation; NPU also owns the CAM dispatch envelope, Attention TP and core
#   budgets, which GPU adopts from the M2N handshake instead.
@dataclass(frozen=True)
class GPUFFNConfig:
    """Frozen settings of one F rank in a standalone DeepEP M2N deployment."""

    model: Path
    attention_ranks: int
    ffn_ranks: int
    rank: int = 0
    local_rank: int = 0
    host: str = AFDConfig.host
    port: int = DEFAULT_AFD_PORT
    graph: bool = False
    dbo: bool = False
    connector_extra_config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    afd_timeout_s: int = DEFAULT_AFD_TIMEOUT_S
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        """Reject launches no Attention peer could pair with, on the CPU.

        Checks rank counts, this rank, the rendezvous address, timeouts, and
        the two connector options the service fixes: ``payload_format`` must
        be ``bf16`` and an explicit ``num_lanes`` must equal the lane count
        ``--dbo`` implies (1 without, 2 with). Every other connector option is
        validated by ``DeepEPM2NConfig.from_mapping`` once the backend runs.
        """
        validate_common_ffn_launch(
            attention_ranks=self.attention_ranks,
            ffn_ranks=self.ffn_ranks,
            rank=self.rank,
            local_rank=self.local_rank,
            host=self.host,
            port=self.port,
            shutdown_timeout=self.shutdown_timeout,
        )
        if self.afd_timeout_s <= 0:
            raise ValueError("--afd-timeout must be positive")
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
                f"{'--dbo' if self.dbo else 'no --dbo'} ({self.num_lanes} lane(s))"
            )

    @property
    def num_lanes(self) -> int:
        """DeepEP lanes this rank creates: two under DBO, otherwise one."""
        return DBO_LANES if self.dbo else SINGLE_LANE

    @property
    def graph_mode(self) -> str:
        """Status name of the FFN execution mode."""
        return FFN_GRAPH_MODE if self.graph else EAGER_MODE

    def afd_additional_config(self) -> dict[str, Any]:
        """``additional_config`` with the same AFD shape a Worker-path FFN uses.

        Role ``ffn`` on ``DeepEPM2NAFDConnector`` with the Attention-side gate
        (required by M2N), the rendezvous address, the rank counts, the
        rendezvous timeout and the connector options passed through unchanged.
        """
        return {
            "afd": {
                "role": "ffn",
                "connector": DEEPEP_M2N_CONNECTOR,
                "host": self.host,
                "port": self.port,
                "num_attention_ranks": self.attention_ranks,
                "num_ffn_ranks": self.ffn_ranks,
                "compute_gate_on_attention": True,
                "afd_process_group_timeout_s": self.afd_timeout_s,
                "connector_extra_config": dict(self.connector_extra_config),
            }
        }

    # NPU parallel: afd_plugin/ffn_faas/platforms/npu/backend.py::
    #   CAMFFNBackend.prepare builds its EngineArgs inline.
    # Reuse: PARTIAL - external_launcher, DP = F ranks, TP = 1 and expert
    #   parallelism are identical; NPU adds Ascend quantization and the CAM
    #   dispatch envelope, GPU adds graph mode, DBO and the EP weight filter.
    #   REFACTOR-CANDIDATE(R4): one F-only EngineArgs builder with
    #   platform-specific extras.
    def engine_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``EngineArgs`` of this F-only rank.

        DP = F ranks and TP = 1 make vLLM's EP group the whole F world, with
        ``data_parallel_rank`` = this rank (``resolve_role_rank`` then gives
        the same M2N role rank as a Worker-path FFN). ``external_launcher``
        uses the ``torchrun`` world. Graph mode enables FULL_DECODE_ONLY,
        which turns on the FFN graph cache; F needs no capture sizes because
        graph keys come from Attention's control payloads. ``enable_dbo``
        gives vLLM two micro-batches, so the connector creates two lanes.
        Scheduler limits keep vLLM defaults: F adopts the dispatch capacity
        and bucket ladder from Attention in the M2N handshake. vLLM's
        ``all2all_backend`` and ``moe_backend`` are not set, so the handshake
        sees the idle default backend and ``select_routed_moe_backend`` can
        pin ``auto`` to Triton. The EP weight filter skips non-local expert
        tensors before they are read, as the Worker-path recipes do.
        """
        return {
            "model": str(self.model),
            "dtype": "bfloat16",
            "skip_tokenizer_init": True,
            "tensor_parallel_size": 1,
            "data_parallel_size": self.ffn_ranks,
            "data_parallel_rank": self.rank,
            "enable_expert_parallel": True,
            "enable_ep_weight_filter": True,
            "distributed_executor_backend": EXTERNAL_LAUNCHER,
            "enforce_eager": not self.graph,
            "compilation_config": (
                {"cudagraph_mode": FFN_GRAPH_MODE} if self.graph else {}
            ),
            "enable_dbo": self.dbo,
            "additional_config": self.afd_additional_config(),
        }


def parse_connector_extra_config(text: str) -> Mapping[str, Any]:
    """Parse ``--connector-extra-config`` and require a JSON object.

    Only the structure is checked here; values are validated by the backend's
    ``DeepEPM2NConfig.from_mapping``, which owns unknown keys and ranges.
    Returns a read-only mapping so the frozen settings stay immutable.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--connector-extra-config is not valid JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("--connector-extra-config must be a JSON object")
    return MappingProxyType(value)
