# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only metadata for the original ModelSlim DeepSeek-V4 W4A8 checkpoint.

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
