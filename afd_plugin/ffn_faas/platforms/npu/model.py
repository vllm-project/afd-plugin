# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routed parameter lifetime for the existing Ascend CAM/W4A8 model seam.

Model adapters supply checkpoint names and native construction at load time.
The mature runtime borrows cached weight views, without per-step adaptation.
This module is CPU-importable; native dependencies are loaded inside load().
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from afd_plugin.ffn_faas.platforms.npu.settings import validate_native_expert_map

if TYPE_CHECKING:
    import torch
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights


class RoutedWeightOwner(Protocol):
    """Native routed modules retained until CAM and compute groups retire."""

    def create_layer(self, layer_index: int) -> nn.Module: ...
    def routed_experts(self, layer_index: int) -> RoutedExperts: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
    def finish_layer(self, layer_index: int) -> None: ...
    def weight_views(self) -> tuple[W4A8LayerWeights, ...]: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...


class RoutedModelAdapter(Protocol):
    """Only facts and loading hooks consumed by this CAM integration."""

    @property
    def root(self) -> Path: ...
    @property
    def layer_indices(self) -> tuple[int, ...]: ...
    @property
    def num_experts(self) -> int: ...
    def estimated_parameter_bytes(self, local_experts: int) -> int: ...
    def local_sources(
        self, layer_index: int, owned: frozenset[int]
    ) -> list[tuple[str, str]]: ...
    def create_owner(self, config: VllmConfig) -> RoutedWeightOwner: ...


class CAMRoutedModel:
    """Own loading and native resources; lend the existing runtime its views."""

    def __init__(
        self, adapter: RoutedModelAdapter, ffn_ranks: int, ffn_rank: int
    ) -> None:
        if ffn_ranks <= 0 or adapter.num_experts % ffn_ranks:
            raise ValueError("routed experts must divide across F ranks")
        if not 0 <= ffn_rank < ffn_ranks:
            raise ValueError("FFN rank is outside the F-only loading world")
        self.adapter, self.ffn_ranks, self.ffn_rank = adapter, ffn_ranks, ffn_rank
        self.local_experts = adapter.num_experts // ffn_ranks
        self._owner: RoutedWeightOwner | None = None
        self._views: tuple[W4A8LayerWeights, ...] | None = None
        self._loading = False
        self._loaded = False
        self._parameter_bytes = adapter.estimated_parameter_bytes(self.local_experts)

    @property
    def root(self) -> Path:
        return self.adapter.root

    @property
    def parameter_bytes(self) -> int:
        return self._parameter_bytes

    @property
    def native_view(self) -> tuple[W4A8LayerWeights, ...]:
        if not self._loaded or self._views is None:
            raise RuntimeError("FFN weights have not been loaded")
        return self._views

    @property
    def native_owner(self) -> RoutedWeightOwner:
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
            raise RuntimeError("FFN weights are already loading or loaded")
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

        group = get_ep_group()
        if group.world_size != self.ffn_ranks or group.rank_in_group != self.ffn_rank:
            raise ValueError("native expert group differs from the F world")
        adapter = self.adapter
        total = len(adapter.layer_indices) * self.local_experts
        completed = 0
        with set_current_vllm_config(config):
            with torch.device(device):
                owner = adapter.create_owner(config)
            # A partial layer/packing failure must leave its owner reachable.
            self._owner = owner
            for layer_index in adapter.layer_indices:
                with torch.device(device):
                    owner.create_layer(layer_index)
                routed = owner.routed_experts(layer_index)
                if routed.local_num_experts != self.local_experts:
                    raise ValueError(
                        "native local expert count differs from the F topology"
                    )
                mapping = routed.expert_map
                owned = validate_native_expert_map(
                    num_experts=adapter.num_experts,
                    ffn_ranks=self.ffn_ranks,
                    ffn_rank=self.ffn_rank,
                    expert_map=mapping.cpu().tolist() if mapping is not None else None,
                )
                selected = adapter.local_sources(layer_index, owned)
                with ExitStack() as stack:
                    readers = {
                        shard: stack.enter_context(
                            safe_open(
                                str(adapter.root / shard), framework="pt", device="cpu"
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
            if tuple(view.layer_idx for view in views) != adapter.layer_indices:
                raise ValueError("native views do not match the routed model layers")
            self._views = views
            storage_sizes: dict[tuple[str, int], int] = {}
            for parameter in owner.parameters():
                storage = parameter.untyped_storage()
                key = (str(parameter.device), storage.data_ptr())
                storage_sizes[key] = storage.nbytes()
            self._parameter_bytes = sum(storage_sizes.values())
        self._loaded = True

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.adapter.layer_indices

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
        raise RuntimeError("standalone CAM requires the native layered W4A8 executor")

    def close(self) -> None:
        """The backend retires execution and its compute environment first."""
        if self._loading:
            raise RuntimeError("cannot close FFN weights during loading")
        self._views = None
        self._owner = None
        self._loaded = False
