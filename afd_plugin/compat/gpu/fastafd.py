# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bindings and weight layout for the FastAFD M2N MegaMoE kernels.

Compat helper for ``FastAFDAFDConnector``. The native kernels are the FastAFD
M2N MegaMoE family vendored under ``csrc/gpu/fastafd`` and built into
``afd_plugin._C_fastafd`` with ``AFD_BUILD_FASTAFD_OPS=1``. They JIT-compile on
first use from the headers installed under ``afd_plugin/_fastafd/include``.

The weight helpers are ported from FastAFD's ``megamoe_m2n_mega.py`` at
3c7161949310b6d59d6b4cf9bf997a4935c8113b (MIT, see
``csrc/gpu/fastafd/LICENSE.FastAFD``):
https://github.com/hao-ai-lab/FastAFD/tree/3c7161949310b6d59d6b4cf9bf997a4935c8113b
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from types import ModuleType

import torch

FASTAFD_KERNEL_ROOT = Path(__file__).resolve().parents[2] / "_fastafd"
# Scale granularity of the MegaMoE FP8 weights and activations (UE8M0 scales).
FASTAFD_SCALE_GROUP = 32
# Checkpoint FP8 block size the requantization accepts.
FASTAFD_CHECKPOINT_BLOCK = 128
# Gate/up rows are interleaved in groups of this size for the fused SwiGLU.
_GATE_UP_INTERLEAVE = 8
# UTCCP scale-factor tiles transpose [4, 32] row groups inside 128 rows.
_UTCCP_ROWS = 128
_JIT_CACHE_DIR_ENV = "DG_JIT_CACHE_DIR"


@cache
def fastafd_ops() -> ModuleType:
    """Import the native kernels and point their JIT compiler at the headers."""
    from torch.utils.cpp_extension import CUDA_HOME

    try:
        from afd_plugin import _C_fastafd
    except ImportError as exc:
        raise RuntimeError(
            "FastAFDAFDConnector needs the FastAFD kernels; reinstall "
            "afd-plugin with AFD_BUILD_FASTAFD_OPS=1",
        ) from exc
    if not (FASTAFD_KERNEL_ROOT / "include" / "deep_gemm").is_dir():
        raise RuntimeError(
            f"FastAFD kernel headers are missing under {FASTAFD_KERNEL_ROOT}; "
            "reinstall afd-plugin with AFD_BUILD_FASTAFD_OPS=1",
        )
    if CUDA_HOME is None:
        raise RuntimeError("FastAFD kernels JIT-compile with nvcc; set CUDA_HOME")
    os.environ.setdefault(
        _JIT_CACHE_DIR_ENV,
        str(Path.home() / ".cache" / "afd_plugin" / "fastafd_jit"),
    )
    _C_fastafd.init(str(FASTAFD_KERNEL_ROOT), CUDA_HOME)
    return _C_fastafd


def ceil_to_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Round positive float scales up to the next power of two (UE8M0)."""
    bits = scales.abs().float().view(torch.int)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exponent.clamp(1, 254) << 23).view(torch.float)


def requant_fp8_block_to_per32(
    weight: torch.Tensor,
    block_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize 128x128 block-scaled FP8 expert weights to per-32 UE8M0.

    ``weight`` is e4m3 ``[E, N, K]`` and ``block_scales`` is float
    ``[E, ceil(N/128), ceil(K/128)]``. Each block scale is rounded up to a
    power of two and the weights are rescaled by ``scale / rounded``, which is
    one e4m3 round trip and therefore lossy. Returns the new weights and the
    kernel-layout packed UE8M0 scales.
    """
    num_experts, n, k = weight.shape
    scales = block_scales.float()
    rounded = ceil_to_ue8m0(scales)
    ratio = torch.where(rounded > 0, scales / rounded, torch.zeros_like(scales))
    requantized = torch.empty_like(weight, memory_format=torch.contiguous_format)
    for expert in range(num_experts):
        expert_ratio = torch.repeat_interleave(
            ratio[expert], FASTAFD_CHECKPOINT_BLOCK, dim=0
        )[:n]
        expert_ratio = torch.repeat_interleave(
            expert_ratio, FASTAFD_CHECKPOINT_BLOCK, dim=1
        )[:, :k]
        requantized[expert].copy_(
            (weight[expert].float() * expert_ratio).to(torch.float8_e4m3fn)
        )
    group_scales = torch.repeat_interleave(rounded, FASTAFD_CHECKPOINT_BLOCK, dim=1)
    group_scales = torch.repeat_interleave(
        group_scales[:, :n],
        FASTAFD_CHECKPOINT_BLOCK // FASTAFD_SCALE_GROUP,
        dim=2,
    )[:, :, : k // FASTAFD_SCALE_GROUP].contiguous()
    packed_scales = fastafd_ops().transform_sf_into_required_layout(
        group_scales, n, k, (1, FASTAFD_SCALE_GROUP), num_experts
    )
    return requantized, packed_scales


def _interleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    num_experts, rows, *rest = tensor.shape
    half = rows // 2
    gate = tensor[:, :half].reshape(
        num_experts, half // _GATE_UP_INTERLEAVE, _GATE_UP_INTERLEAVE, *rest
    )
    up = tensor[:, half:].reshape(
        num_experts, half // _GATE_UP_INTERLEAVE, _GATE_UP_INTERLEAVE, *rest
    )
    interleaved = torch.empty_like(tensor)
    view = interleaved.reshape(
        num_experts, half // _GATE_UP_INTERLEAVE, 2, _GATE_UP_INTERLEAVE, *rest
    )
    view[:, :, 0].copy_(gate)
    view[:, :, 1].copy_(up)
    return interleaved


def _transpose_scales_for_utccp(scales: torch.Tensor) -> torch.Tensor:
    num_experts, rows, packed_k = scales.shape
    transposed = (
        scales.reshape(num_experts, rows // _UTCCP_ROWS, 4, 32, packed_k)
        .transpose(2, 3)
        .reshape(num_experts, rows, packed_k)
    )
    return torch.empty_like(scales).copy_(transposed)


def transform_weights_for_m2n(
    gate_up: tuple[torch.Tensor, torch.Tensor],
    down: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Kernel layout of requantized ``(weight, packed_scales)`` pairs.

    ``gate_up`` is ``[E, 2I, H]`` with the gate in the first ``I`` rows; its
    weights and scales are gate/up interleaved. Scales of both projections get
    the UTCCP transpose.
    """
    gate_up_weight = _interleave_gate_up(gate_up[0])
    gate_up_scales = _transpose_scales_for_utccp(_interleave_gate_up(gate_up[1]))
    return (gate_up_weight, gate_up_scales), (
        down[0],
        _transpose_scales_for_utccp(down[1]),
    )


def make_weight_descriptors(
    gate_up: tuple[torch.Tensor, torch.Tensor],
    down: tuple[torch.Tensor, torch.Tensor],
    *,
    num_experts_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> torch.Tensor:
    """Per-layer TMA descriptor blob, uint8 ``[4, 128]`` on the CPU.

    The descriptors embed the weight addresses, so the weights must stay alive
    and in place while the kernels use the blob.
    """
    return fastafd_ops().make_m2n_weight_descs(
        gate_up[0],
        gate_up[1],
        down[0],
        down[1],
        num_experts_per_rank,
        hidden_size,
        intermediate_size,
    )


__all__ = [
    "FASTAFD_CHECKPOINT_BLOCK",
    "FASTAFD_KERNEL_ROOT",
    "FASTAFD_SCALE_GROUP",
    "ceil_to_ue8m0",
    "fastafd_ops",
    "make_weight_descriptors",
    "requant_fp8_block_to_per32",
    "transform_weights_for_m2n",
]
