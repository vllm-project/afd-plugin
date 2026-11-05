# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-only Qwen3 MoE model: the routed experts and nothing else.

The Worker-path FFN loads ``AFDQwen3MoeForCausalLM`` with ``role="ffn"``. Its
decoder layers already skip Attention, but the CausalLM still builds the
embedding, final norm, LM head and logits processor (about 1.16 GiB per F
rank for Qwen3-30B-A3B, never used). This shell keeps the native module tree
only where the routed experts live, ``model.layers.<i>.mlp``, built from the
native ``Qwen3MoeSparseMoeBlock``, so vLLM's own loader, EP weight filter,
FusedMoE expert loading and ``process_weights_after_loading`` run unchanged.

The MoE forward never runs here: the M2N connector's executor reads the
experts' parameters as views (``local_experts_from_vllm``). The native gate is
built because the block constructs it and the strict loader expects its
weight; it is never executed (Attention routes).
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models import qwen3_moe as native
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.torch_utils import set_default_torch_dtype

from afd_plugin.model_executor.models.qwen3_moe import _weight_layer_path

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

ROUTED_STAGE = "mlp"
MODEL_PREFIX = "model"


def _is_routed_mlp_path(name: str) -> bool:
    """Whether a checkpoint tensor belongs to a decoder layer's MoE block.

    Keeps ``model.layers.<i>.mlp.*`` (routed experts and the unused gate) and
    drops embeddings, norms, Attention and the LM head, which this shell does
    not construct. Reuses the AFD Qwen3 wrapper's path parser so both models
    classify checkpoint names identically.
    """
    layer_path = _weight_layer_path(name)
    return layer_path is not None and layer_path[1] == ROUTED_STAGE


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/model.py::RoutedWeightOwner
#   via V4ModelAdapter.create_owner/create_layer (native routed modules built
#   layer by layer from the pinned Ascend loader).
# Reuse: NO - NPU constructs and loads W4A8 layered weights through its own
#   safetensors reader; GPU keeps vLLM's module tree and native loader.
class _Qwen3RoutedExpertsLayer(nn.Module):
    """One decoder layer reduced to its native MoE block at ``<prefix>.mlp``."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        """Construct the native ``Qwen3MoeSparseMoeBlock`` with its native prefix.

        The prefix keeps FusedMoE's ``layer_name`` and its
        ``static_forward_context`` key identical to the Worker-path model.
        """
        super().__init__()
        self.mlp = native.Qwen3MoeSparseMoeBlock(
            vllm_config=vllm_config, prefix=f"{prefix}.{ROUTED_STAGE}"
        )


class _Qwen3RoutedExpertsStack(nn.Module):
    """The ``model`` submodule: only ``layers``, each holding its MoE block."""

    # Upstream: vLLM v0.32.0, vllm/model_executor/models/qwen3_moe.py,
    # Qwen3MoeModel.hf_to_vllm_mapper; reused, not copied.
    hf_to_vllm_mapper = native.Qwen3MoeModel.hf_to_vllm_mapper

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        """Build one ``_Qwen3RoutedExpertsLayer`` per decoder layer."""
        super().__init__()
        num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
        self.layers = nn.ModuleList(
            _Qwen3RoutedExpertsLayer(vllm_config, f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_layers)
        )

    # Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
    # vllm/model_executor/models/qwen3_moe.py, Qwen3MoeModel.load_weights.
    # Copied unchanged: the same AutoWeightsLoader, ignored suffixes and
    # checkpoint mapper, so expert stacking and EP sharding stay in FusedMoE's
    # own loader. Re-copy on a vLLM upgrade.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``layers.*`` tensors through the native Qwen3 model loader path.

        ``AutoWeightsLoader`` walks the module tree and hands each MoE block's
        ``experts.*`` tensors to FusedMoE's ``load_weights``, which stacks
        per-expert gate/up/down projections into ``w13``/``w2`` and keeps
        only this rank's experts. Returns the loaded parameter names for the
        strict loaded-weights check.
        """
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".bias",
                "_bias",
                ".weight_scale",
                "_weight_scale",
                ".input_scale",
                "_input_scale",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3RoutedExpertsModel(nn.Module):
    """FFN-only Qwen3 MoE model consumed by ``GPUFFNRoutedExpertsModelRunner``.

    Implements ``RoutedExpertsModel``: ``get_experts_layer_indices`` and
    ``get_routed_experts``, the same accessors ``AFDQwen3MoeForCausalLM``
    exposes on the Worker path.
    """

    # Same as native Qwen3MoeForCausalLM: load safetensors only, never fall
    # back to PyTorch checkpoint files.
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig) -> None:
        """Build ``model.layers.<i>.mlp`` per layer, named as in the native model.

        Must run under ``set_current_vllm_config`` and after
        ``select_routed_moe_backend``: FusedMoE reads the current config and
        picks its weight layout at construction.
        """
        super().__init__()
        self.model = _Qwen3RoutedExpertsStack(vllm_config, MODEL_PREFIX)

    # Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
    # vllm/model_executor/models/qwen3_moe.py, Qwen3MoeForCausalLM.load_weights.
    # Copied with one marked difference, the routed-path filter.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drop non-MoE checkpoint tensors, then load like the native CausalLM.

        The native CausalLM delegates to ``AutoWeightsLoader(self)``, which
        recurses into ``model.load_weights``. Tensors this shell does not
        construct (embeddings, norms, Attention, LM head) are filtered first,
        as ``AFDQwen3MoeForCausalLM.load_weights`` filters by role, but
        stricter: that wrapper still keeps non-layer tensors for its
        embedding and LM head.
        """
        # ### PATCH START: keep only routed MoE checkpoint paths.
        weights = (
            (name, tensor) for name, tensor in weights if _is_routed_mlp_path(name)
        )
        # ### PATCH END: keep only routed MoE checkpoint paths.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Every decoder layer is an MoE layer for a supported checkpoint."""
        return tuple(range(len(self.model.layers)))

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts:
        """The vLLM module holding one MoE layer's local experts."""
        return self.model.layers[layer_idx].mlp.experts.routed_experts


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/model.py::CAMRoutedModel.load
#   (layer-by-layer safetensors load with per-layer progress, then storage
#   accounting).
# Reuse: PARTIAL - storage accounting is the same computation
#   (REFACTOR-CANDIDATE(R7)); loading differs: vLLM's loader exposes no
#   per-layer callback, so progress is reported only before and after.
# Upstream source: vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be),
#   vllm/model_executor/model_loader/base_loader.py, BaseModelLoader.load_model.
#   Same steps; the model is constructed here instead of by initialize_model,
#   and online quantization is not handled (BF16 only).
def load_qwen3_routed_experts(
    vllm_config: VllmConfig, device: torch.device
) -> Qwen3RoutedExpertsModel:
    """Construct the FFN-only shell on ``device`` and load its weights.

    Follows ``BaseModelLoader.load_model``: construct and load under the model
    dtype and target device, run vLLM's ``DefaultModelLoader.load_weights``
    (EP weight filter, then the strict loaded-weights check), release loading
    temporaries, run ``process_weights_after_loading`` (the Triton backend
    keeps the FusedMoE layout), and return the model in eval mode. The caller
    holds ``set_current_vllm_config``.
    """
    model_config = vllm_config.model_config
    loader = get_model_loader(vllm_config.load_config)
    with set_default_torch_dtype(model_config.dtype), torch.device(device):
        model = Qwen3RoutedExpertsModel(vllm_config=vllm_config)
        loader.load_weights(model, model_config)
    gc.collect()
    torch.accelerator.empty_cache()
    process_weights_after_loading(model, model_config, device)
    gc.collect()
    torch.accelerator.empty_cache()
    return model.eval()


# NPU parallel: CAMRoutedModel._load_native storage accounting.
# Reuse: YES - same computation; REFACTOR-CANDIDATE(R7).
def parameter_storage_bytes(model: nn.Module) -> int:
    """Bytes of the distinct storages behind ``model``'s parameters.

    Views that share storage (for example the connector's expert views) are
    counted once.
    """
    storage_sizes: dict[tuple[str, int], int] = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storage_sizes[(str(parameter.device), storage.data_ptr())] = storage.nbytes()
    return sum(storage_sizes.values())
