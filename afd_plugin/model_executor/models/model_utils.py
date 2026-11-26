# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared AFD model configuration helpers and the AFD runner hook contract."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Final, Literal

from afd_plugin import _MODEL_REGISTRATIONS, _QWEN3_5_MODEL_REGISTRATIONS
from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config

if TYPE_CHECKING:
    import torch
    from torch import nn
    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from afd_plugin.connectors.metadata import AFDExpertRoutingSpec


_ROUTED_EXPERTS_MODEL_ARCHITECTURES: Final[dict[str, str]] = {
    "Qwen3MoeForCausalLM": "AFDQwen3MoeRoutedExperts",
}


def has_afd_model_registration(model_config: ModelConfig) -> bool:
    """Return whether the architecture has an AFD registration mapping."""

    return any(
        model_arch.removeprefix("AFD") in _MODEL_REGISTRATIONS
        for model_arch in model_config.hf_config.architectures
    )


def get_afd_model_config(
    model_config: ModelConfig,
    *,
    device_type: Literal["cuda", "npu"],
    routed_experts_connector: bool = False,
) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _MODEL_REGISTRATIONS:
            if model_arch == "DSparkDraftModel" and device_type != "npu":
                raise ValueError("AFD DSpark currently supports only the NPU backend")
            if model_arch in _QWEN3_5_MODEL_REGISTRATIONS and device_type != "cuda":
                raise ValueError(
                    "AFD Qwen3.5/3.6 supports CUDA execution only; "
                    f"got device_type={device_type!r}",
                )
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            routed_architecture = (
                _ROUTED_EXPERTS_MODEL_ARCHITECTURES.get(model_arch)
                if routed_experts_connector and device_type == "cuda"
                else None
            )
            afd_model_config.hf_config.architectures = [
                routed_architecture or f"AFD{model_arch}"
            ]
            return afd_model_config
    return model_config


def configure_afd_dspark(vllm_config: VllmConfig) -> None:
    """Resolve the native DSpark config through the role-local AFD model."""
    speculative = vllm_config.speculative_config
    if speculative is not None and speculative.method == "dspark":
        if parse_afd_config(vllm_config).connector != AFD_ASYNC_CONNECTOR:
            raise ValueError("AFD DSpark requires CAMAsyncAFDConnector")
        target = vllm_config.model_config.hf_text_config
        draft = speculative.draft_model_config.hf_text_config
        schemas = (
            ("hidden_size", target.hidden_size, draft.hidden_size),
            (
                "num_experts_per_tok",
                target.num_experts_per_tok,
                draft.num_experts_per_tok,
            ),
            ("n_routed_experts", target.n_routed_experts, draft.n_routed_experts),
        )
        for field, target_value, draft_value in schemas:
            if target_value != draft_value:
                raise ValueError(
                    f"DSpark CAM transport {field} mismatch: "
                    f"target={target_value}, draft={draft_value}"
                )
        speculative.draft_model_config = get_afd_model_config(
            speculative.draft_model_config, device_type="npu"
        )


class AFDModelHooks:
    """Inner-model half of the AFD runner contract.

    Mixed into the AFD inner model, the class that owns ``self.layers`` and is
    reached through ``AFDCausalLMHooks.afd_inner_model()``. It holds the hooks
    the AFD workers call on ``ModelRunner.get_model()``:

    * ``compute_ffn_output`` runs one layer's FFN stage on the FFN role.
    * ``compute_experts_output`` runs routed experts from router logits that
      Attention sent.
    * ``get_experts_layer_indices`` lists the MoE layers.
    * ``get_experts_routing_spec`` returns the static router width and dtype
      used for graph capture.

    Models keep only the hooks whose logic is model specific, plus the optional
    class flags the runners read (``afd_requires_input_ids`` and
    ``afd_requires_attention_routing`` on the outer mixin). Adapters that serve a
    routed-experts connector use ``AFDRoutedExpertsModelHooks`` instead.
    """

    layers: nn.ModuleList

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **ffn_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one layer's FFN stage, forwarding transport tensors unchanged.

        Runners add ``input_ids`` (GPU V4) or ``topk_ids``/``topk_weights``
        (NPU Attention-side routing, plus ``activation_scale`` for MXFP8
        payloads) only when the model asks for them; the layer validates its
        own payload.
        """
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **ffn_inputs)

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Run one layer's routed experts from Attention-side router logits."""
        return self.layers[layer_idx].compute_experts_output(
            hidden_states,
            router_logits,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Return the MoE layer indices; every AFD model overrides this."""
        raise NotImplementedError("AFD model must declare its MoE layer inventory")

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        """Return the static router contract; models without a router override."""
        raise NotImplementedError("AFD model does not expose a native router contract")


class AFDRoutedExpertsModelHooks(AFDModelHooks):
    """Inner-model hooks for adapters that serve a routed-experts connector.

    Only adapters whose MoE layers keep vLLM's ``RoutedExperts`` at
    ``layer.mlp.experts.routed_experts`` opt in (DeepSeek V2 family and Qwen3
    MoE). Others do not expose the accessor, so a routed-experts runner fails
    at the attribute lookup instead of walking a layer that has no such module.
    """

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts:
        """Return the vLLM module holding one MoE layer's local experts.

        Connectors that route on Attention run these experts themselves. The
        FFN never applies ``routed_scaling_factor``: Attention's runner scales
        the combined output.
        """
        return self.layers[layer_idx].mlp.experts.routed_experts


class AFDCausalLMHooks:
    """Outer-model half of the AFD runner contract.

    Mixed into the class returned by ``ModelRunner.get_model()``. Every hook
    forwards to the inner AFD model from ``afd_inner_model()``, so subclasses
    override only the model-specific hooks (``get_experts_layer_indices``, a
    native ``get_experts_routing_spec``, or a layer-specific
    ``compute_experts_output``) and the two transport flags.
    """

    # Runners read these directly; a model that needs the transport to carry a
    # tensor overrides them.
    afd_requires_input_ids: bool = False
    afd_requires_attention_routing: bool = False

    model: AFDModelHooks

    def afd_inner_model(self) -> AFDModelHooks:
        """Return the inner AFD model; multimodal wrappers return the text one."""
        return self.model

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **ffn_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.afd_inner_model().compute_ffn_output(
            hidden_states,
            layer_idx,
            **ffn_inputs,
        )

    def compute_experts_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.afd_inner_model().compute_experts_output(
            hidden_states,
            layer_idx,
            router_logits,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.afd_inner_model().get_experts_layer_indices()

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        return self.afd_inner_model().get_experts_routing_spec(layer_idx)


class AFDRoutedExpertsCausalLMHooks(AFDCausalLMHooks):
    """Outer-model hooks for adapters that serve a routed-experts connector."""

    model: AFDRoutedExpertsModelHooks

    def afd_inner_model(self) -> AFDRoutedExpertsModelHooks:
        return self.model

    def get_routed_experts(self, layer_idx: int) -> RoutedExperts:
        return self.afd_inner_model().get_routed_experts(layer_idx)
