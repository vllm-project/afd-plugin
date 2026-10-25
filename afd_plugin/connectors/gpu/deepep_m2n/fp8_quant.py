# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FP8 block-128 helpers for the DeepEP M2N expert path (SM90, fp32 scales).

``silu_mul_fp8_quant`` is the fused step between the two FP8 grouped GEMMs:
``fp8(silu(gate) * up * topk_weight)`` with one fp32 scale per row and
128-channel group, written in the MN-major, TMA-aligned layout DeepGEMM's SM90
kernel reads without a transpose. Folding the top-k weight in here makes the
second GEMM's output already weighted (it is linear), so no separate multiply
is needed before combine.

Adapted from FastAFD ``kernel/deepgemm_fused_quant.py``
(``_silu_mul_fp8_quant_psum_deep_gemm``, itself from vLLM), which only emits
SM100 UE8M0 scales. Rows are processed in parallel blocks instead of per
expert; rows past the last expert's end are skipped, and padding rows between
expert groups are quantized but never read by combine.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from afd_plugin.connectors.routed_experts import FP8_DTYPE

FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min
GROUP_SIZE = 128
# DeepGEMM's TMA needs the M stride of fp32 scales aligned to 16 bytes.
TMA_ALIGN_FP32_ELEMENTS = 4
EPS = 1e-10
BLOCK_M = 32


def tma_aligned_rows(num_rows: int) -> int:
    return -(-num_rows // TMA_ALIGN_FP32_ELEMENTS) * TMA_ALIGN_FP32_ELEMENTS


@triton.jit
def _silu_mul_fp8_quant_kernel(
    x_ptr,
    topk_weights_ptr,
    psum_ptr,
    y_q_ptr,
    y_s_ptr,
    num_rows,
    num_experts,
    stride_x_m,
    stride_yq_m,
    stride_ys_m,
    stride_ys_g,
    intermediate: tl.constexpr,
    group_size: tl.constexpr,
    apply_topk_weight: tl.constexpr,
    block: tl.constexpr,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
):
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    valid_rows = tl.minimum(tl.load(psum_ptr + num_experts - 1), num_rows)
    rows = (pid_m * block + tl.arange(0, block)).to(tl.int64)
    row_mask = rows < valid_rows
    cols = group * group_size + tl.arange(0, group_size)
    gate_ptrs = x_ptr + rows[:, None] * stride_x_m + cols[None, :]
    gate = tl.load(gate_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    up = tl.load(gate_ptrs + intermediate, mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )
    y = gate / (1.0 + tl.exp(-gate)) * up
    if apply_topk_weight:
        weights = tl.load(topk_weights_ptr + rows, mask=row_mask, other=0.0)
        y = y * weights.to(tl.float32)[:, None]
    scale = tl.maximum(tl.max(tl.abs(y), axis=1), eps) / fp8_max
    y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(
        y_q_ptr + rows[:, None] * stride_yq_m + cols[None, :],
        y_q,
        mask=row_mask[:, None],
    )
    tl.store(y_s_ptr + rows * stride_ys_m + group * stride_ys_g, scale, mask=row_mask)


def silu_mul_fp8_quant(
    gate_up: torch.Tensor,
    recv_count_psum: torch.Tensor,
    topk_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``gate_up`` ``[M, 2I]`` BF16 -> FP8 ``[M, I]`` and fp32 scales ``[M, I/128]``.

    The scales are MN-major with a TMA-aligned M stride (DeepGEMM SM90 SFA).
    """
    num_rows, intermediate_x2 = gate_up.shape
    intermediate = intermediate_x2 // 2
    if intermediate % GROUP_SIZE != 0:
        raise ValueError(f"intermediate size {intermediate} is not a multiple of 128")
    num_groups = intermediate // GROUP_SIZE
    y_q = torch.empty((num_rows, intermediate), dtype=FP8_DTYPE, device=gate_up.device)
    y_s = torch.empty_strided(
        (num_rows, num_groups),
        (1, tma_aligned_rows(num_rows)),
        dtype=torch.float32,
        device=gate_up.device,
    )
    if num_rows == 0:
        return y_q, y_s
    grid = (triton.cdiv(num_rows, BLOCK_M), num_groups)
    _silu_mul_fp8_quant_kernel[grid](
        gate_up,
        topk_weights if topk_weights is not None else gate_up,
        recv_count_psum,
        y_q,
        y_s,
        num_rows,
        recv_count_psum.shape[0],
        gate_up.stride(0),
        y_q.stride(0),
        y_s.stride(0),
        y_s.stride(1),
        intermediate=intermediate,
        group_size=GROUP_SIZE,
        apply_topk_weight=topk_weights is not None,
        block=BLOCK_M,
        eps=EPS,
        fp8_min=FP8_MIN,
        fp8_max=FP8_MAX,
    )
    return y_q, y_s


def quantize_fp8_block128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` weights -> FP8 and fp32 per-128x128-block scales.

    Same layout as vLLM's block-FP8 ``FusedMoE`` weights (``w13_weight`` /
    ``w13_weight_scale_inv``), so real checkpoints can be passed through as is.
    """
    num_experts, n, k = w.shape
    if n % GROUP_SIZE or k % GROUP_SIZE:
        raise ValueError(f"weight shape {tuple(w.shape)} is not 128-block aligned")
    blocks = w.float().view(
        num_experts, n // GROUP_SIZE, GROUP_SIZE, k // GROUP_SIZE, GROUP_SIZE
    )
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=EPS) / FP8_MAX
    w_q = (blocks / scale[:, :, None, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return w_q.view(num_experts, n, k), scale


def dequantize_fp8_block128(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    num_experts, n, k = w_q.shape
    blocks = w_q.float().view(
        num_experts, n // GROUP_SIZE, GROUP_SIZE, k // GROUP_SIZE, GROUP_SIZE
    )
    return (blocks * scale[:, :, None, :, None]).view(num_experts, n, k)


def dequantize_fp8_rows(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-row, per-128-channel FP8 activations -> fp32 (any scale strides)."""
    num_rows, hidden = x_q.shape
    groups = x_q.float().view(num_rows, hidden // GROUP_SIZE, GROUP_SIZE)
    return (groups * scale[:, :, None]).view(num_rows, hidden)
