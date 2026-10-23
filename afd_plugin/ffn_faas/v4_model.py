# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only metadata for the original ModelSlim DeepSeek-V4 W4A8 checkpoint.

Keep checkpoint names intact. Device loading must delegate packed conversion,
scales and compensation to the pinned Ascend W4A8 loader; this description does
not allocate tensors or initialize native runtime state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from afd_plugin.ffn_faas.protocol import SessionSpec


@dataclass(frozen=True)
class V4CheckpointDescription:
    root: Path
    identity: str
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
    metadata_sha256: Mapping[str, str]

    @property
    def layer_indices(self) -> tuple[int, ...]:
        # The pinned V4 decoder constructs routed MoE on every main layer.
        # MTP checkpoint layers are outside this service's main-model contract.
        return tuple(range(self.model_layers))

    def session_spec(
        self,
        *,
        attention_ranks: int,
        ffn_ranks: int,
        attention_tp: int,
        capacity: int = 128,
        slots: int = 2,
        layer_indices: tuple[int, ...] | None = None,
    ) -> SessionSpec:
        selected = self.layer_indices if layer_indices is None else layer_indices
        if not set(selected).issubset(self.layer_indices):
            raise ValueError("requested V4 weights include a non-main-model layer")
        return SessionSpec(
            self.identity,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
            attention_ranks,
            ffn_ranks,
            attention_tp,
            self.model_layers,
            selected,
            capacity=capacity,
            slots=slots,
            weight_quantization="w4a8_per_channel",
            swiglu_limit=self.swiglu_limit,
        )

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
        for shard in set(routed.values()):
            if not (root / shard).resolve(strict=True).is_relative_to(root):
                raise ValueError(f"V4 checkpoint shard is outside its root: {shard}")
        return cls(
            root,
            "sha256:"
            + hashlib.sha256(b"\0".join(raw[name] for name in names)).hexdigest(),
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
            MappingProxyType(
                {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in raw.items()
                }
            ),
        )
