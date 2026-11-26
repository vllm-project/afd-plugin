# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 FFN post-load weight processing on Ascend."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4DecoderLayer,
    )


def prepare_attention_gate_w8a8_scales(
    layers: Iterable[torch.nn.Module],
) -> None:
    """Retain the GMM scale on its routed owner after native quant finalize."""
    # Model hooks select FFN decoders; ModuleList exposes their type as Module.
    for module in layers:
        layer = cast("AFDDeepseekV4DecoderLayer", module)
        if not layer.is_moe_layer:
            continue
        experts = cast("AscendMoERunner", layer.mlp.experts)
        if experts.quant_type != QuantType.W8A8:
            continue
        owner = cast("AscendRoutedExperts", experts.routed_experts)
        if owner.dynamic_eplb:
            continue
        # bd42eea's MegaMoE finalizer deletes this ordinary derived Tensor,
        # retaining w13_weight_scale on the same owner. This is a cache-lifetime
        # check, not runtime-version detection; preserve every existing value.
        if "w13_weight_scale_fp32" not in owner.__dict__:
            owner.w13_weight_scale_fp32 = cast(
                torch.Tensor, owner.w13_weight_scale
            ).data.to(torch.float32)
