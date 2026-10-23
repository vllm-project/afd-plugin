# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-only checkpoint planning and streaming BF16 expert loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.protocol import SessionSpec

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.npu import NativeWeightView


def is_v4_checkpoint(model_path: str | Path) -> bool:
    """Return whether config.json declares a DeepSeek-V4 architecture.

    Mirrors the feature-validation rule: any architecture containing
    ``DeepseekV4`` selects the V4 adapter, including the AFD-prefixed name.
    """
    config = json.loads((Path(model_path) / "config.json").read_bytes())
    architectures = config.get("architectures") or ()
    return any("DeepseekV4" in str(architecture) for architecture in architectures)


@dataclass(frozen=True)
class ExpertWeights:
    compact_layer: int
    local_expert: int
    gate: str
    up: str
    down: str


@dataclass(frozen=True)
class CheckpointDescription:
    root: Path
    identity: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    model_layers: int
    layer_indices: tuple[int, ...]
    weight_map: Mapping[str, str]

    @classmethod
    def read(cls, model_path: str | Path) -> CheckpointDescription:
        root = Path(model_path).resolve(strict=True)
        config_bytes = (root / "config.json").read_bytes()
        index_bytes = (root / "model.safetensors.index.json").read_bytes()
        config = json.loads(config_bytes)
        index = json.loads(index_bytes)
        if config["architectures"] != ["DeepseekV2ForCausalLM"]:
            raise ValueError("FFN FaaS v1 checkpoint adapter supports DeepSeek-V2 only")
        if config["hidden_act"] != "silu" or config.get("quantization_config"):
            raise ValueError("FFN FaaS v1 requires unquantized SwiGLU experts")
        if config.get("torch_dtype", "bfloat16") != "bfloat16":
            raise ValueError("FFN FaaS v1 requires a BF16 checkpoint")

        def positive(name: str) -> int:
            value = config[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"checkpoint {name} must be a positive integer")
            return value

        layers = positive("num_hidden_layers")
        first = config["first_k_dense_replace"]
        frequency = positive("moe_layer_freq")
        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not 0 <= first < layers
        ):
            raise ValueError("invalid first routed-expert layer")
        routed = tuple(
            layer for layer in range(first, layers) if layer % frequency == 0
        )
        if not routed:
            raise ValueError("checkpoint contains no routed expert layers")
        mapping = index["weight_map"]
        if not isinstance(mapping, dict) or not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in mapping.items()
        ):
            raise ValueError("invalid safetensors weight index")
        return cls(
            root,
            "sha256:" + hashlib.sha256(config_bytes + b"\0" + index_bytes).hexdigest(),
            positive("hidden_size"),
            positive("moe_intermediate_size"),
            positive("n_routed_experts"),
            positive("num_experts_per_tok"),
            layers,
            routed,
            MappingProxyType(mapping),
        )

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
            raise ValueError("requested weights include a non-routed model layer")
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
        )

    def plan(self, spec: SessionSpec, ffn_rank: int) -> Iterator[ExpertWeights]:
        if spec.model_identity != self.identity:
            raise ValueError("session refers to a different checkpoint")
        if not 0 <= ffn_rank < spec.ffn_ranks:
            raise ValueError("FFN rank is outside the expert layout")
        first = ffn_rank * spec.local_experts
        for compact, layer in enumerate(spec.layer_indices):
            for local in range(spec.local_experts):
                prefix = f"model.layers.{layer}.mlp.experts.{first + local}"
                names = tuple(
                    f"{prefix}.{projection}.weight"
                    for projection in ("gate_proj", "up_proj", "down_proj")
                )
                for name in names:
                    shard = self.root / self.weight_map[name]
                    if not shard.resolve(strict=True).is_relative_to(self.root):
                        raise ValueError(
                            f"checkpoint shard is outside its root: {shard}"
                        )
                yield ExpertWeights(compact, local, *names)


class FFNModel:
    """Own only this F rank's expert tensors; executor views are borrowed."""

    def __init__(
        self, checkpoint: CheckpointDescription, spec: SessionSpec, ffn_rank: int
    ) -> None:
        self.checkpoint = checkpoint
        self.spec = spec
        self.ffn_rank = ffn_rank
        # Validate every required file/key before allocating device weights.
        self.plan = tuple(checkpoint.plan(spec, ffn_rank))
        self._view: NativeWeightView | None = None

    @property
    def parameter_bytes(self) -> int:
        return (
            len(self.spec.layer_indices)
            * self.spec.local_experts
            * 3
            * self.spec.hidden_size
            * self.spec.intermediate_size
            * 2
        )

    @property
    def native_view(self) -> NativeWeightView:
        if self._view is None:
            raise RuntimeError("FFN weights have not been loaded")
        return self._view

    def load(
        self,
        device: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        # Device imports are confined to loading, not metadata/service imports.
        import torch
        import torch_npu  # noqa: F401
        from safetensors import safe_open

        from afd_plugin.ffn_faas.npu import NativeWeightView

        if self._view is not None:
            raise RuntimeError("FFN weights are already loaded")
        torch.npu.set_device(device)
        target = torch.device(f"npu:{device}")
        s = self.spec
        mapping = torch.full((s.model_layers,), -1, dtype=torch.int32)
        for compact, layer in enumerate(s.layer_indices):
            mapping[layer] = compact
        w13 = torch.empty(
            (
                len(s.layer_indices),
                s.local_experts,
                2 * s.intermediate_size,
                s.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=target,
        )
        w2 = torch.empty(
            (len(s.layer_indices), s.local_experts, s.hidden_size, s.intermediate_size),
            dtype=torch.bfloat16,
            device=target,
        )
        self._view = NativeWeightView(w13, w2, mapping.to(target))
        shards = sorted(
            {
                self.checkpoint.weight_map[name]
                for item in self.plan
                for name in (item.gate, item.up, item.down)
            }
        )
        try:
            with ExitStack() as stack:
                readers = {
                    name: stack.enter_context(
                        safe_open(
                            str(self.checkpoint.root / name),
                            framework="pt",
                            device="cpu",
                        )
                    )
                    for name in shards
                }
                for completed, item in enumerate(self.plan, 1):
                    destination = w13[item.compact_layer, item.local_expert]
                    views = (
                        (item.gate, destination[: s.intermediate_size]),
                        (item.up, destination[s.intermediate_size :]),
                        (item.down, w2[item.compact_layer, item.local_expert]),
                    )
                    for name, view in views:
                        tensor = readers[self.checkpoint.weight_map[name]].get_tensor(
                            name
                        )
                        if tensor.dtype != torch.bfloat16 or tensor.shape != view.shape:
                            raise ValueError(
                                f"unsupported expert dtype or shape: {name}"
                            )
                        view.copy_(tensor, non_blocking=False)
                    if progress is not None:
                        progress(completed, len(self.plan))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        # Runner calls this only after executor completion and peer quiescence.
        self._view = None
