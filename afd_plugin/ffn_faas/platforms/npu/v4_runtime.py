# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 checkpoint interpretation and native routed-only construction hooks.

The CAM model resource owns loading and lifetime. V4 source names, ModelSlim
parameter counts and the native loader receiver remain specific to this adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.ffn_faas.platforms.npu.model import RoutedWeightOwner

PACKED_WEIGHTS_PER_BYTE = 2
PROJECTIONS_PER_EXPERT = 3
SOURCE_PARAMETERS_PER_PROJECTION = 4
ENCODED_SCALE_BYTES = 8
FLOAT_PARAMETER_BYTES = 4
GATE_UP_PROJECTIONS = 2


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
