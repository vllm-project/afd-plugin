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
        """Ascend 7dcdb690 installs a local BF16 method after weight creation.
        The remote method owns post-load processing for these weightless experts.
        """
