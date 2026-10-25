# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone V4 parameter lifetime using the pinned native routed loader.

Metadata construction imports no tensor/runtime libraries. The platform owner
provides a live F-only compute environment before loading and retains it through
execution. No CausalLM, EngineCore, upstream ModelRunner or forward is constructed.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from afd_plugin.ffn_faas.platforms.npu.v4_weights import V4RoutedWeightOwner
    from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights

PACKED_WEIGHTS_PER_BYTE = 2
PROJECTIONS_PER_EXPERT = 3
SOURCE_PARAMETERS_PER_PROJECTION = 4
ENCODED_SCALE_BYTES = 8
FLOAT_PARAMETER_BYTES = 4
GATE_UP_PROJECTIONS = 2


def validate_native_expert_map(
    *, num_experts: int, ffn_ranks: int, ffn_rank: int, expert_map: list[int] | None
) -> frozenset[int]:
    """Require the native sharding to match CAM's contiguous expert placement."""
    if ffn_ranks <= 0 or num_experts % ffn_ranks or not 0 <= ffn_rank < ffn_ranks:
        raise ValueError("V4 experts require a divisible, valid F topology")
    local_experts = num_experts // ffn_ranks
    if expert_map is None:
        if ffn_ranks != 1:
            raise ValueError("sharded native V4 experts require an explicit map")
        # The native factory omits the map when every expert is local.
        return frozenset(range(num_experts))
    if len(expert_map) != num_experts:
        raise ValueError("native V4 expert map has the wrong global expert count")
    owned = frozenset(expert for expert, local in enumerate(expert_map) if local >= 0)
    if sorted(expert_map[expert] for expert in owned) != list(range(local_experts)):
        raise ValueError("native V4 expert map is not a local bijection")
    first = ffn_rank * local_experts
    end = first + local_experts
    if owned != frozenset(range(first, end)) or expert_map[first:end] != list(
        range(local_experts)
    ):
        raise ValueError("native V4 placement differs from contiguous CAM routing")
    return owned


class V4FFNModel:
    """Own only routed W4A8 parameters; executors borrow the cached views."""

    def __init__(
        self,
        checkpoint: V4CheckpointDescription,
        ffn_ranks: int,
        ffn_rank: int,
    ) -> None:
        if ffn_ranks <= 0 or checkpoint.num_experts % ffn_ranks:
            raise ValueError("V4 routed experts must divide across F ranks")
        if not 0 <= ffn_rank < ffn_ranks:
            raise ValueError("V4 FFN rank is outside the F-only loading world")
        self.checkpoint, self.ffn_ranks, self.ffn_rank = checkpoint, ffn_ranks, ffn_rank
        self.local_experts = checkpoint.num_experts // ffn_ranks
        self._owner: V4RoutedWeightOwner | None = None
        self._views: tuple[W4A8LayerWeights, ...] | None = None
        self._loading = False
        self._loaded = False

        # Native ModelSlim retains offsets in addition to the six kernel views.
        # Before load this is an aligned-geometry estimate; afterward we record
        # the actual distinct parameter storage sizes, including retained views.
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
        self._parameter_bytes = (
            len(checkpoint.layer_indices) * self.local_experts * (matrices + auxiliary)
        )
        self._sources: dict[int, list[tuple[int, str, str]]] = {
            layer: [] for layer in checkpoint.layer_indices
        }
        for name, shard in checkpoint.routed_weight_map.items():
            fields = name.split(".", 5)
            layer = int(fields[1])
            if layer in self._sources:
                self._sources[layer].append((int(fields[4]), name, shard))

    @property
    def parameter_bytes(self) -> int:
        return self._parameter_bytes

    @property
    def native_view(self) -> tuple[W4A8LayerWeights, ...]:
        if not self._loaded or self._views is None:
            raise RuntimeError("V4 FFN weights have not been loaded")
        return self._views

    @property
    def native_owner(self) -> V4RoutedWeightOwner:
        if not self._loaded or self._owner is None:
            raise RuntimeError("native routed parameters have not finished loading")
        return self._owner

    def load(
        self,
        device: torch.device,
        vllm_config: VllmConfig,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if self._loading or self._owner is not None or self._loaded:
            raise RuntimeError("V4 FFN weights are already loading or loaded")
        self._loading = True
        try:
            self._load_native(device, vllm_config, progress)
        finally:
            self._loading = False

    def _load_native(
        self,
        device: torch.device,
        config: VllmConfig,
        progress: Callable[[int, int], None] | None,
    ) -> None:
        import torch
        from safetensors import safe_open
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import get_ep_group

        from afd_plugin.ffn_faas.platforms.npu.v4_weights import V4RoutedWeightOwner

        group = get_ep_group()
        if group.world_size != self.ffn_ranks or group.rank_in_group != self.ffn_rank:
            raise ValueError("native V4 expert group differs from the F world")
        target = device
        checkpoint = self.checkpoint
        total = len(checkpoint.layer_indices) * self.local_experts
        completed = 0
        with set_current_vllm_config(config):
            with torch.device(target):
                owner = V4RoutedWeightOwner(self.checkpoint, config)
            self._owner = owner
            for layer_index in checkpoint.layer_indices:
                with torch.device(target):
                    layer = owner.create_layer(layer_index)
                routed = layer.mlp.experts.routed_experts
                if routed.local_num_experts != self.local_experts:
                    raise ValueError(
                        "native V4 local expert count differs from the F topology"
                    )
                mapping = routed.expert_map
                owned = validate_native_expert_map(
                    num_experts=checkpoint.num_experts,
                    ffn_ranks=self.ffn_ranks,
                    ffn_rank=self.ffn_rank,
                    expert_map=mapping.cpu().tolist() if mapping is not None else None,
                )
                selected = [
                    (name, shard)
                    for expert, name, shard in self._sources[layer_index]
                    if expert in owned
                ]
                required = (
                    self.local_experts
                    * PROJECTIONS_PER_EXPERT
                    * SOURCE_PARAMETERS_PER_PROJECTION
                )
                if len(selected) != required:
                    raise ValueError(
                        "V4 layer is missing local native source parameters"
                    )
                with ExitStack() as stack:
                    readers = {
                        shard: stack.enter_context(
                            safe_open(
                                str(self.checkpoint.root / shard),
                                framework="pt",
                                device="cpu",
                            )
                        )
                        for shard in sorted({shard for _, shard in selected})
                    }
                    owner.load_weights(
                        (name, readers[shard].get_tensor(name))
                        for name, shard in selected
                    )
                    owner.finish_layer(layer_index)
                    # Startup-only fence: source mappings and packing temporaries
                    # can retire before the next layer, and progress means packed.
                    torch.npu.synchronize(device)
                completed += len(owned)
                if progress is not None:
                    progress(completed, total)
            views = owner.weight_views()
            if tuple(view.layer_idx for view in views) != checkpoint.layer_indices:
                raise ValueError("native V4 views do not match the main-model layers")
            self._views = views
            storage_sizes: dict[tuple[str, int], int] = {}
            for parameter in owner.parameters():
                storage = parameter.untyped_storage()
                key = (str(parameter.device), storage.data_ptr())
                storage_sizes[key] = storage.nbytes()
            self._parameter_bytes = sum(storage_sizes.values())

        self._loaded = True

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.checkpoint.layer_indices

    def get_async_cam_w4a8_layers(self) -> tuple[list[W4A8LayerWeights], str]:
        return list(self.native_view), ""

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        group_list: torch.Tensor | None,
        dynamic_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        raise RuntimeError("standalone V4 requires the native layered W4A8 executor")

    def close(self) -> None:
        """The backend retires execution and its compute environment first."""
        if self._loading:
            raise RuntimeError("cannot close V4 FFN weights during loading")
        self._views = None
        self._owner = None
        self._loaded = False
