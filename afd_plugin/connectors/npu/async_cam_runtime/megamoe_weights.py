# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Adapt the post-load DeepSeek MXFP4 tensors once, before the FFN loop."""

import torch

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


def _packed_weight(
    parameter: torch.nn.Parameter, shape: tuple[int, ...]
) -> torch.Tensor:
    import torch_npu

    weight = parameter.detach().transpose(1, 2)
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
        raise RuntimeError("Could not allocate MegaMoE packed FP4 C0=16 byte storage")
    # Both Flash projections are tile-aligned. Copy exactly the physical bytes
    # [expert, K1, N1, N0=16, packed_K0=16], preserving the two FP4 nibbles.
    tiles = _nz_bytes(dense)
    if converted.untyped_storage().nbytes() != tiles.numel():
        raise RuntimeError("Unexpected padded size for aligned FP4 expert weights")
    torch.ops.npu.copy_memory_(converted, tiles)
    # Keep a single resident copy per layer and retain the native logical axes.
    parameter.data = converted.transpose(1, 2)
    return converted


def expert_weights(moe, rank: int, hidden: int, intermediate: int):
    from vllm_ascend.quantization.quant_type import QuantType

    experts = moe.experts
    if experts.quant_type != QuantType.W4A8MXFP or experts.routed_experts.dynamic_eplb:
        raise ValueError("MegaMoE routed experts require static W4A8 MXFP weights")
    local = moe.n_local_physical_experts
    if moe.ep_rank != rank or moe.physical_expert_start != rank * local:
        raise ValueError("FFN EP rank does not own the expected contiguous experts")
    owner = experts.routed_experts
    # The native loader transposes both weight and scale views after packing.
    w1 = _packed_weight(owner.w13_weight, (local, 2 * intermediate, hidden // 2))
    w2 = _packed_weight(owner.w2_weight, (local, hidden, intermediate // 2))
    s1 = owner.w13_weight_scale.detach().view(torch.uint8).transpose(1, 2)
    s2 = owner.w2_weight_scale.detach().view(torch.uint8).transpose(1, 2)
    for scale, n, k in ((s1, 2 * intermediate, hidden), (s2, hidden, intermediate)):
        if scale.dtype != torch.uint8 or tuple(scale.shape) != (local, n, k // 64, 2):
            raise ValueError(
                "Expected packed group-32 E8M0 scales from the native Flash loader"
            )
        if not scale.is_contiguous():
            raise ValueError("MegaMoE requires contiguous output-major expert scales")
    return [w1], [w2], [s1], [s2]
