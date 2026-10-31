# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from afd_plugin import _DEEPSEEK_MODEL_REGISTRATIONS

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


def get_afd_model_config(model_config: ModelConfig) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _DEEPSEEK_MODEL_REGISTRATIONS:
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
            return afd_model_config
    return model_config


def install_afd_speculative_model_config(vllm_config: VllmConfig) -> None:
    """Resolve the draft model to its AFD role wrapper in-place."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        return
    speculative_config.draft_model_config = get_afd_model_config(
        draft_model_config,
    )
