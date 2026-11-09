# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from afd_plugin import (
    _MINIMAX_MODEL_REGISTRATIONS,
    _MODEL_REGISTRATIONS,
    _QWEN3_5_MODEL_REGISTRATIONS,
)
from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


def has_afd_model_registration(model_config: ModelConfig) -> bool:
    """Return whether the model resolves to a registered AFD implementation."""

    return any(
        model_arch.removeprefix("AFD") in _MODEL_REGISTRATIONS
        for model_arch in model_config.hf_config.architectures
    )


def get_afd_model_config(
    model_config: ModelConfig,
    *,
    device_type: Literal["cuda", "npu"],
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
            if model_arch in _MINIMAX_MODEL_REGISTRATIONS and device_type != "npu":
                raise ValueError(
                    "AFD MiniMax-M3 supports Ascend NPU execution only; "
                    f"got device_type={device_type!r}",
                )
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
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
