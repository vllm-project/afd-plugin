# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Preserve A5 MXFP scale layout during Ascend AllToAllV routing.

Upstream source: ``vllm_ascend/ops/fused_moe/token_dispatcher.py`` at
commit ``11ee45653b199a097805b87011824a81ffa51b95``.

Remove this compatibility patch after vLLM-Ascend routes MXFP token and scale
payloads without collapsing the scale tensor consumed by grouped matmul.
"""

from __future__ import annotations

import inspect
import logging

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    TokenDispatcherWithAll2AllV,
)

logger = logging.getLogger(__name__)

_MXFP_ALLTOALLV_PATCH_ATTR = "_afd_plugin_mxfp_alltoallv_patch_state"
_TARGET_POSTPROCESS_PARAMETERS = (
    "self",
    "global_input_tokens",
    "dynamic_scale_after_all2all",
    "global_input_tokens_local_experts_indices",
    "with_quant",
    "dst_type",
    "scale_type",
)


def _permute_mxfp_byte_payload(
    payload: torch.Tensor,
    expert_indices: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute a logically quantized payload through its physical bytes."""

    if payload.ndim < 2:
        raise ValueError(
            "MXFP payload must have at least 2 dimensions, "
            f"got shape {tuple(payload.shape)}",
        )
    if payload.element_size() != 1:
        raise ValueError(
            f"MXFP payload must use 1-byte storage, got dtype {payload.dtype}",
        )
    if payload.shape[0] != expert_indices.numel():
        raise ValueError(
            "MXFP payload token count does not match expert indices: "
            f"{payload.shape[0]} != {expert_indices.numel()}",
        )

    trailing_shape = tuple(payload.shape[1:])
    payload_bytes = payload.reshape(payload.shape[0], -1).view(torch.int8)
    permuted_bytes, reverse_mapping = torch_npu.npu_moe_token_permute(
        payload_bytes,
        expert_indices,
    )
    permuted = permuted_bytes.view(output_dtype).reshape(
        permuted_bytes.shape[0],
        *trailing_shape,
    )
    return permuted, reverse_mapping


# Patch reason: vLLM-Ascend 11ee4565 sends a physical uint8 MXFP payload to
# torch-npu 2.10 MoeInitRoutingV3, whose output scale is 1D instead of the 3D
# layout required by the A5 grouped-matmul path.
# Patch functionality: retain the upstream AllToAllV behavior except that the
# affected MXFP branch permutes token and scale byte payloads separately and
# restores their original trailing dimensions.
# Signature: matches vLLM-Ascend 11ee4565; no added parameters.
def _dispatch_postprocess(
    self,
    global_input_tokens,
    dynamic_scale_after_all2all,
    global_input_tokens_local_experts_indices,
    with_quant,
    dst_type,
    scale_type,
):
    # Early return if no local experts or no tokens
    if self.num_local_experts <= 1:
        return global_input_tokens, dynamic_scale_after_all2all, None

    assert global_input_tokens_local_experts_indices is not None, (
        "global_input_tokens_local_experts_indices must be provided"
    )

    if with_quant:
        if scale_type == torch.float8_e8m0fnu:
            # ### PATCH START: preserve A5 MXFP scale layout
            if dynamic_scale_after_all2all is None:
                raise ValueError(
                    "MXFP scale is required for quantized AllToAllV dispatch",
                )
            global_input_tokens, reversed_global_input_permutation_mapping = (
                _permute_mxfp_byte_payload(
                    global_input_tokens,
                    global_input_tokens_local_experts_indices,
                    output_dtype=global_input_tokens.dtype,
                )
            )
            dynamic_scale_after_all2all, _ = _permute_mxfp_byte_payload(
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                output_dtype=torch.uint8,
            )
            # ### PATCH END: preserve A5 MXFP scale layout
            return (
                global_input_tokens,
                dynamic_scale_after_all2all,
                reversed_global_input_permutation_mapping,
            )
        dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
            dynamic_scale_after_all2all.unsqueeze(-1),
            global_input_tokens_local_experts_indices,
        )
        dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

    # Non-quantized case
    global_input_tokens, reversed_global_input_permutation_mapping = (
        torch_npu.npu_moe_token_permute(
            global_input_tokens,
            global_input_tokens_local_experts_indices,
        )
    )
    return (
        global_input_tokens,
        dynamic_scale_after_all2all,
        reversed_global_input_permutation_mapping,
    )


def apply_afd_mxfp_alltoallv_patch() -> bool:
    """Patch the affected vLLM-Ascend 11ee4565 dispatcher contract.

    Other dispatcher signatures are left unchanged. In particular, this keeps
    the compatibility module from overriding older 3da28f9 or newer fixed
    vLLM-Ascend implementations.
    """

    if hasattr(TokenDispatcherWithAll2AllV, _MXFP_ALLTOALLV_PATCH_ATTR):
        return True

    original_postprocess = TokenDispatcherWithAll2AllV._dispatch_postprocess
    parameter_names = tuple(inspect.signature(original_postprocess).parameters)
    if parameter_names != _TARGET_POSTPROCESS_PARAMETERS:
        logger.debug(
            "AFD A5 MXFP AllToAllV patch skipped for dispatcher signature %s",
            parameter_names,
        )
        return False

    TokenDispatcherWithAll2AllV._dispatch_postprocess = _dispatch_postprocess
    setattr(
        TokenDispatcherWithAll2AllV,
        _MXFP_ALLTOALLV_PATCH_ATTR,
        original_postprocess,
    )
    logger.info("AFD A5 MXFP AllToAllV compatibility patch applied")
    return True


__all__ = ["apply_afd_mxfp_alltoallv_patch"]
