# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Weightless Attention-side experts for the pinned Ascend MoE factory."""

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

from afd_plugin.model_executor.ffn.remote import AFDRemoteRoutedExperts


class AFDAscendRemoteRoutedExperts(AFDRemoteRoutedExperts, AscendRoutedExperts):
    """Preserve AFD's weightless method through Ascend initialization."""

    def _replace_quant_method(self, quant_method: FusedMoEMethodBase) -> None:
        """Keep the remote method for post-load processing.

        At bd42eea0, AscendRoutedExperts initializes the upstream weight owner
        first, then installs its local BF16 method when quant_config is None.
        Here the method selected by _get_quant_method owns the weightless
        lifecycle, including the post-load hook.
        """
