# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only checkpoint description and loading adapter for ModelSlim V4 W4A8.

Keep checkpoint names intact. Device loading must delegate packed conversion,
scales and compensation to the pinned Ascend W4A8 loader; this description does
not allocate tensors or initialize native runtime state.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.ffn_faas.platforms.npu.model import RoutedWeightOwner

PACKED_WEIGHTS_PER_BYTE = 2
PROJECTIONS_PER_EXPERT = 3
SOURCE_PARAMETERS_PER_PROJECTION = 4
ENCODED_SCALE_BYTES = 8
FLOAT_PARAMETER_BYTES = 4
GATE_UP_PROJECTIONS = 2


@dataclass(frozen=True)
class V4CheckpointDescription:
    root: Path
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    model_layers: int
    hash_layers: int
    routed_scaling_factor: float
    swiglu_limit: float
    weight_map: Mapping[str, str]
    routed_weight_map: Mapping[str, str]

    @property
    def layer_indices(self) -> tuple[int, ...]:
        # The pinned V4 decoder constructs routed MoE on every main layer.
        # MTP checkpoint layers are outside this service's main-model contract.
        return tuple(range(self.model_layers))

    @classmethod
    def read(cls, model_path: str | Path) -> V4CheckpointDescription:
        root = Path(model_path).resolve(strict=True)
        names = (
            "config.json",
            "quant_model_weights.safetensors.index.json",
            "quant_model_description.json",
        )
        raw = {name: (root / name).read_bytes() for name in names}
        config, index, quant = (json.loads(raw[name]) for name in names)
        if config["architectures"] != ["DeepseekV4ForCausalLM"]:
            raise ValueError("V4 FFN weights require DeepseekV4ForCausalLM")
        if config["hidden_act"] != "silu" or config["scoring_func"] != "sqrtsoftplus":
            raise ValueError("V4 FFN adapter requires SiLU and sqrtsoftplus routing")
        if config.get("torch_dtype", "bfloat16") != "bfloat16":
            raise ValueError("V4 FFN activation contract requires BF16")
        if quant.get("group_size", 0) != 0:
            raise ValueError("V4 FFN requires per-channel W4A8_DYNAMIC weights")

        def positive(name: str) -> int:
            value = config[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"V4 checkpoint {name} must be a positive integer")
            return value

        layers = positive("num_hidden_layers")
        experts = positive("n_routed_experts")
        top_k = positive("num_experts_per_tok")
        hash_layers = config["num_hash_layers"]
        if (
            not isinstance(hash_layers, int)
            or isinstance(hash_layers, bool)
            or not 0 <= hash_layers <= layers
            or top_k > experts
        ):
            raise ValueError("invalid V4 Hash layer or routed top-k count")
        scale, limit = (
            float(config["routed_scaling_factor"]),
            float(config["swiglu_limit"]),
        )
        if (
            not math.isfinite(scale)
            or scale <= 0
            or not math.isfinite(limit)
            or limit < 0
        ):
            raise ValueError("invalid V4 routed scale or SwiGLU limit")
        mapping = index["weight_map"]
        if not isinstance(mapping, dict) or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in mapping.items()
        ):
            raise ValueError("invalid V4 safetensors weight index")
        for shard in set(mapping.values()):
            if not (root / shard).resolve(strict=True).is_relative_to(root):
                raise ValueError("V4 checkpoint shard is outside its root")
        routed = {}
        for name, shard in mapping.items():
            match = re.fullmatch(
                r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\."
                r"(weight|weight_scale|weight_offset|scale_bias)",
                name,
            )
            if match is None or int(match[1]) >= layers:
                continue
            if int(match[2]) >= experts:
                raise ValueError(f"V4 routed expert is outside the model: {name}")
            if quant.get(name) != "W4A8_DYNAMIC":
                raise ValueError(f"V4 routed tensor requires W4A8_DYNAMIC: {name}")
            routed[name] = shard
        # Fail before device allocation, including missing scales/compensation.
        # This checks source names; it does not reproduce native packed mapping.
        for layer in range(layers):
            for expert in range(experts):
                for projection in ("w1", "w2", "w3"):
                    for suffix in (
                        "weight",
                        "weight_scale",
                        "weight_offset",
                        "scale_bias",
                    ):
                        name = (
                            f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                        )
                        if name not in routed:
                            raise ValueError(
                                f"V4 routed checkpoint tensor is missing: {name}"
                            )
        return cls(
            root,
            positive("hidden_size"),
            positive("moe_intermediate_size"),
            experts,
            top_k,
            layers,
            hash_layers,
            scale,
            limit,
            MappingProxyType(mapping),
            MappingProxyType(routed),
        )


class V4ModelAdapter:
    """Interpret only the validated V4 Flash per-channel ModelSlim contract."""

    def __init__(self, checkpoint: V4CheckpointDescription) -> None:
        self.checkpoint = checkpoint
        self._sources: dict[int, list[tuple[int, str, str]]] = {
            layer: [] for layer in checkpoint.layer_indices
        }
        for name, shard in checkpoint.routed_weight_map.items():
            fields = name.split(".", 5)
            layer = int(fields[1])
            if layer in self._sources:
                self._sources[layer].append((int(fields[4]), name, shard))

    @property
    def root(self) -> Path:
        return self.checkpoint.root

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self.checkpoint.layer_indices

    @property
    def num_experts(self) -> int:
        return self.checkpoint.num_experts

    def estimated_parameter_bytes(self, local_experts: int) -> int:
        # ModelSlim retains offsets as well as the executor's six tensor views.
        # The resource records distinct actual storage sizes after native loading.
        checkpoint = self.checkpoint
        matrices = (
            PROJECTIONS_PER_EXPERT
            * checkpoint.hidden_size
            * checkpoint.intermediate_size
            // PACKED_WEIGHTS_PER_BYTE
        )
        channels = (
            GATE_UP_PROJECTIONS * checkpoint.intermediate_size + checkpoint.hidden_size
        )
        auxiliary = channels * (
            ENCODED_SCALE_BYTES + FLOAT_PARAMETER_BYTES + FLOAT_PARAMETER_BYTES
        )
        return len(checkpoint.layer_indices) * local_experts * (matrices + auxiliary)

    def local_sources(
        self, layer_index: int, owned: frozenset[int]
    ) -> list[tuple[str, str]]:
        selected = [
            (name, shard)
            for expert, name, shard in self._sources[layer_index]
            if expert in owned
        ]
        required = (
            len(owned) * PROJECTIONS_PER_EXPERT * SOURCE_PARAMETERS_PER_PROJECTION
        )
        if len(selected) != required:
            raise ValueError("V4 layer is missing local native source parameters")
        return selected

    def create_owner(self, config: VllmConfig) -> RoutedWeightOwner:
        from afd_plugin.ffn_faas.platforms.npu.v4_weights import V4RoutedWeightOwner

        return V4RoutedWeightOwner(self.checkpoint, config)
