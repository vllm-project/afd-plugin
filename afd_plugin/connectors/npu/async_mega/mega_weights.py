# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Convert output-major MXFP4 expert tensors to Mega kernel storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .mega_layers import LayerWeights

ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29
# UMDK's TensorListWrapper maps a byte container's C0=16 to FP4 C0=32.
PACKED_BYTE_NZ_C0_16 = 50


def _nz_bytes(dense: torch.Tensor) -> torch.Tensor:
    experts, rows, packed_k = dense.shape
    tile = 16
    return (
        dense.reshape(experts, rows // tile, tile, packed_k // tile, tile)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )


def _packed_weight(weight: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    import torch_npu

    if weight.dtype != torch.uint8 or tuple(weight.shape) != shape:
        raise ValueError(
            f"Expected packed FP4 weight {shape}, got {weight.dtype} {weight.shape}"
        )
    fmt = torch_npu.get_npu_format(weight)
    if fmt == PACKED_BYTE_NZ_C0_16:
        return weight
    if fmt != ACL_FORMAT_FRACTAL_NZ:
        raise ValueError(f"Unexpected native FP4 weight storage format {fmt}")
    dense = torch_npu.npu_format_cast(
        weight, ACL_FORMAT_ND, input_dtype=torch_npu.float4_e2m1fn_x2
    )
    converted = torch_npu.empty_with_format(
        shape,
        dtype=torch.uint8,
        device=weight.device,
        acl_format=PACKED_BYTE_NZ_C0_16,
    )
    if torch_npu.get_npu_format(converted) != PACKED_BYTE_NZ_C0_16:
        raise RuntimeError("Could not allocate Mega packed FP4 C0=16 byte storage")
    # Both projections must be tile-aligned. Copy exactly the physical bytes
    # [expert, K1, N1, N0=16, packed_K0=16], preserving the two FP4 nibbles.
    tiles = _nz_bytes(dense)
    if converted.untyped_storage().nbytes() != tiles.numel():
        raise RuntimeError("Unexpected padded size for aligned FP4 expert weights")
    torch.ops.npu.copy_memory_(converted, tiles)
    return converted


def convert_expert_weights(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
) -> LayerWeights:
    """Accept output-major packed FP4 weights and group-32 E8M0 scales.

    Model adapters own native parameter axes and parameter replacement. This
    function only validates tensor geometry and converts the kernel layout.
    """
    local = weight1.shape[0]
    for scale, n, k in (
        (scale1, 2 * intermediate, hidden),
        (scale2, hidden, intermediate),
    ):
        if scale.dtype != torch.uint8 or tuple(scale.shape) != (local, n, k // 64, 2):
            raise ValueError("Expected output-major packed group-32 E8M0 scales")
        if not scale.is_contiguous():
            raise ValueError("Mega requires contiguous output-major expert scales")
    w1 = _packed_weight(weight1, (local, 2 * intermediate, hidden // 2))
    w2 = _packed_weight(weight2, (local, hidden, intermediate // 2))
    return [w1], [w2], [scale1], [scale2]
