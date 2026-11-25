# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared experts-only FFN model shell and routed-experts protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.model_executor.models._weight_utils import layer_weight_path

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

_ConfigT = TypeVar("_ConfigT")


class RoutedExpertsModel(Protocol):
    """Contract for routed-experts FFN models read by the runner and loader."""

    def get_experts_layer_indices(self) -> tuple[int, ...]: ...

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts: ...

    def is_unused_checkpoint_weight(self, name: str) -> bool: ...


def default_routed_experts(block: nn.Module) -> RoutedExperts:
    """Return the routed-experts container from a standard vLLM MoE block."""
    return block.experts.routed_experts


def is_routed_block_weight(
    name: str,
    *,
    ffn_submodule: str = "mlp",
    allowed_children: tuple[str, ...] = ("experts", "gate"),
    layer_prefixes: tuple[str, ...] = ("model.layers.",),
) -> bool:
    """Return whether a checkpoint path targets routed experts or their gate."""
    if not name.startswith(layer_prefixes):
        return False
    layer_path = layer_weight_path(name)
    if layer_path is None:
        return False
    _layer_idx, stage, remainder = layer_path
    return (
        stage == ffn_submodule and bool(remainder) and remainder[0] in allowed_children
    )


@dataclass(frozen=True)
class FFNExpertsSpec(Generic[_ConfigT]):
    """Model-family specification for building an experts-only FFN shell."""

    ffn_submodule: str
    is_moe_layer: Callable[[_ConfigT, int], bool]
    build_sparse_block: Callable[[VllmConfig, int, str], nn.Module]
    is_routed_expert_weight: Callable[[str], bool]
    routed_experts: Callable[[nn.Module], RoutedExperts] = default_routed_experts
    hf_to_vllm_mapper: WeightsMapper | None = None
    validate: Callable[[VllmConfig, AFDConfig], None] | None = None


class _RoutedExpertsLayer(nn.Module):
    """One MoE decoder layer reduced to its sparse FFN block."""

    def __init__(self, ffn_submodule: str, block: nn.Module) -> None:
        super().__init__()
        setattr(self, ffn_submodule, block)


class _RoutedExpertsStack(nn.Module):
    """The ``model`` submodule holding one slot per decoder layer."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model_prefix: str,
        spec: FFNExpertsSpec[Any],
        num_layers: int,
        moe_layer_index_set: frozenset[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            if layer_idx in moe_layer_index_set:
                block_prefix = f"{model_prefix}.layers.{layer_idx}.{spec.ffn_submodule}"
                block = spec.build_sparse_block(vllm_config, layer_idx, block_prefix)
                layers.append(_RoutedExpertsLayer(spec.ffn_submodule, block))
            else:
                layers.append(nn.Module())
        self.layers = nn.ModuleList(layers)


class AFDFFNExpertsModel(nn.Module):
    """Shared FFN-only model shell that builds only MoE sparse blocks.

    Subclasses attach a class-level ``experts_spec`` (or callers pass
    ``spec=`` directly) describing the model family's layer schedule, sparse
    block constructor, and routed-expert checkpoint weight predicate. Non-MoE
    layers remain empty modules with zero parameters.
    """

    fall_back_to_pt_during_load = False
    experts_spec: ClassVar[FFNExpertsSpec[Any] | None] = None

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        spec: FFNExpertsSpec[Any] | None = None,
    ) -> None:
        resolved_spec = spec if spec is not None else type(self).experts_spec
        if resolved_spec is None:
            raise TypeError(
                f"{type(self).__name__} requires an FFNExpertsSpec via "
                "experts_spec or spec="
            )
        afd_config = parse_afd_config(vllm_config, validate=False)
        if resolved_spec.validate is not None:
            resolved_spec.validate(vllm_config, afd_config)

        text_config = vllm_config.model_config.hf_text_config
        num_layers = int(text_config.num_hidden_layers)
        moe_layer_indices = tuple(
            layer_idx
            for layer_idx in range(num_layers)
            if resolved_spec.is_moe_layer(text_config, layer_idx)
        )
        if not moe_layer_indices:
            raise RuntimeError(
                f"{type(self).__name__} found no MoE layers across "
                f"{num_layers} decoder layers"
            )

        super().__init__()
        self._spec = resolved_spec
        self.afd_config = afd_config
        self.quant_config = vllm_config.quant_config
        self._moe_layer_indices = moe_layer_indices
        self._moe_layer_index_set = frozenset(moe_layer_indices)
        model_prefix = maybe_prefix(prefix, "model")
        self.model = _RoutedExpertsStack(
            vllm_config=vllm_config,
            model_prefix=model_prefix,
            spec=resolved_spec,
            num_layers=num_layers,
            moe_layer_index_set=self._moe_layer_index_set,
        )

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        """Whether vLLM can skip this checkpoint tensor or shard."""
        layer_path = layer_weight_path(name)
        if layer_path is None or layer_path[0] not in self._moe_layer_index_set:
            return True
        return not self._spec.is_routed_expert_weight(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drop unused checkpoint weights and load the kept routed experts."""
        filtered_weights = (
            (name, tensor)
            for name, tensor in weights
            if not self.is_unused_checkpoint_weight(name)
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            filtered_weights,
            mapper=self._spec.hf_to_vllm_mapper,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Return the MoE decoder layer indices built by this shell."""
        return self._moe_layer_indices

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts:
        """Return the vLLM routed-experts container for one MoE layer."""
        if layer_idx not in self._moe_layer_index_set:
            raise ValueError(
                f"layer {layer_idx} is not an MoE layer in "
                f"{type(self).__name__} (MoE layers: {self._moe_layer_indices})"
            )
        block = getattr(self.model.layers[layer_idx], self._spec.ffn_submodule)
        return self._spec.routed_experts(block)


__all__ = [
    "AFDFFNExpertsModel",
    "FFNExpertsSpec",
    "RoutedExpertsModel",
    "default_routed_experts",
    "is_routed_block_weight",
]
