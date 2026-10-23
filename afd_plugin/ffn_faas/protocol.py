# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Session capabilities and the versioned native FFN FaaS contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum

ABI_VERSION = 1
MAX_GENERATION = 2**31 - 2
MAX_DIMENSION = 4096
MAX_CAPACITY = 4096
MAX_SLOTS = 8
DIMENSION_ALIGNMENT = 16
SIGNAL_LINE_BYTES = 64
DEFAULT_IDLE_POLLS = 16
DEFAULT_COMBINE_POLLS = 1024
MAX_SCOPED_EXCHANGES = 4
DEFAULT_SCOPED_COMBINE_POLLS = 1000000


class Mode(IntEnum):
    INITIALIZE = 0
    PUBLISH = 1
    EXECUTE = 2
    COMBINE = 3
    END_SOURCE = 4
    ABORT_SESSION = 5
    PEEK_RESULTS = 6
    EXCHANGE = 8
    SCOPED_EXCHANGE = 9
    PREPARE = 10
    FINISH = 11


class Result(IntEnum):
    OK = 0
    IDLE = 1
    BACKPRESSURE = 2
    DRAINED = 3
    INVALID = -1
    TIMED_OUT = -2
    PEER_FAILED = -3


@dataclass(frozen=True)
class SourceChunkPlan:
    """Bounded A-TP shape agreement, never an F execution schedule."""

    call: int
    first_sequence: int
    layer: int
    chunks: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> SourceChunkPlan:
        return cls(**json.loads(payload))

    def agree(self, peers: list[SourceChunkPlan]) -> int:
        for peer in [self, *peers]:
            if (peer.call, peer.first_sequence, peer.layer) != (
                self.call,
                self.first_sequence,
                self.layer,
            ):
                raise ValueError("Attention TP call/sequence/layer identities disagree")
            if peer.chunks <= 0:
                raise ValueError(
                    "each TP contribution requires at least a zero-work chunk"
                )
        chunks = max(plan.chunks for plan in [self, *peers])
        if (
            self.first_sequence <= 0
            or self.first_sequence + chunks - 1 > MAX_GENERATION
        ):
            raise ValueError("chunked work exhausts the session generation range")
        return chunks


@dataclass(frozen=True)
class SessionSpec:
    """Exactly matched capabilities; no host per-layer scheduling information."""

    model_identity: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    attention_ranks: int
    ffn_ranks: int
    attention_tp: int
    model_layers: int
    layer_indices: tuple[int, ...]
    capacity: int = 128
    slots: int = 2
    dtype: str = "bfloat16"
    activation: str = "silu"
    abi_version: int = ABI_VERSION
    weight_quantization: str = "none"
    swiglu_limit: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_identity:
            raise ValueError("model_identity must identify the fixed checkpoint")
        if self.abi_version != ABI_VERSION:
            raise ValueError("unsupported FFN FaaS ABI version")
        if self.dtype != "bfloat16" or self.activation != "silu":
            raise ValueError("FFN FaaS activation contract requires BF16 SwiGLU")
        if self.weight_quantization not in ("none", "w4a8_per_channel"):
            raise ValueError("unsupported FFN FaaS weight quantization")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("SwiGLU limit must be finite and nonnegative")
        if self.weight_quantization == "none" and self.swiglu_limit != 0:
            raise ValueError("the BF16 executor requires an unlimited SwiGLU")
        for size in (self.hidden_size, self.intermediate_size):
            if not 0 < size <= MAX_DIMENSION or size % DIMENSION_ALIGNMENT:
                raise ValueError("dimensions must be multiples of 16 in [16, 4096]")
        if self.attention_ranks <= 0 or self.ffn_ranks <= 0:
            raise ValueError("both roles require at least one rank")
        if self.attention_tp <= 0 or self.attention_ranks % self.attention_tp:
            raise ValueError("attention_tp must divide attention_ranks")
        if self.num_experts <= 0 or self.num_experts % self.ffn_ranks:
            raise ValueError("experts must divide evenly over FFN ranks")
        if not 0 < self.top_k <= min(self.num_experts, self.hidden_size):
            raise ValueError("invalid routed top-k")
        if not 0 < self.capacity <= MAX_CAPACITY or not 0 < self.slots <= MAX_SLOTS:
            raise ValueError("invalid bounded slot capacity")
        if self.model_layers <= 0 or not self.layer_indices:
            raise ValueError("at least one routed layer is required")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("layer_indices must be unique and sorted")
        if self.layer_indices[0] < 0 or self.layer_indices[-1] >= self.model_layers:
            raise ValueError("routed layer is outside the model")

    @property
    def world_size(self) -> int:
        return self.attention_ranks + self.ffn_ranks

    @property
    def source_count(self) -> int:
        return self.attention_ranks // self.attention_tp

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ffn_ranks

    @property
    def required_window_bytes(self) -> int:
        control = (1 + self.world_size + self.attention_ranks) * SIGNAL_LINE_BYTES
        inputs = 3 * SIGNAL_LINE_BYTES + self.capacity * (
            self.hidden_size * 2 + self.top_k * 4
        )
        results = 2 * SIGNAL_LINE_BYTES + (
            self.capacity * self.top_k * self.hidden_size * 2
        )
        inputs = self._align(inputs)
        results = self._align(results)
        return control + self.slots * max(
            self.attention_ranks * inputs, self.ffn_ranks * results
        )

    @staticmethod
    def _align(size: int) -> int:
        return (size + SIGNAL_LINE_BYTES - 1) // SIGNAL_LINE_BYTES * SIGNAL_LINE_BYTES

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def native_config(self, rank: int, poll_budget: int) -> list[int]:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the A/F world")
        if not 0 < poll_budget <= 100_000_000:
            raise ValueError("poll budget must be bounded and positive")
        return [
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
            self.attention_ranks,
            self.ffn_ranks,
            self.attention_tp,
            self.slots,
            self.capacity,
            rank,
            self.model_layers,
            len(self.layer_indices),
            poll_budget,
        ]

    def validate_work(self, sequence: int, layer: int, tokens: int) -> None:
        if not 0 < sequence <= MAX_GENERATION:
            raise ValueError(
                "work generation exhausted or invalid; restart the session"
            )
        if layer not in self.layer_indices:
            raise ValueError("work targets a layer absent from FFN weights")
        if not 0 <= tokens <= self.capacity:
            raise ValueError("work token count exceeds the negotiated capacity")
