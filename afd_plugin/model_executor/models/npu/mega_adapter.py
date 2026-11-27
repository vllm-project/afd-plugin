# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model admission for the reusable Mega backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.connectors.npu.async_mega.mega_layers import MegaModelSpec


def get_mega_model_spec(config: VllmConfig) -> MegaModelSpec:
    # Extend model admission here when another adapter is validated. Connector
    # and FFN execution consume the same explicit dimensions/weights contract.
    if config.model_config.hf_config.model_type == "deepseek_v4":
        from .deepseek_v4_mega import flash_mega_model_spec

        return flash_mega_model_spec(config)
    raise ValueError("Mega currently supports the DeepSeek-V4-Flash model adapter")
