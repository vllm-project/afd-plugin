# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepGEMM grouped GEMMs the M2N experts use, from the ``deep_gemm`` package.

Uses the installed upstream DeepGEMM (https://github.com/deepseek-ai/DeepGEMM),
installed separately: its build needs torch, CUDA and git submodules. Its
kernels are JIT-compiled at runtime. Only the ``deepgemm`` expert backend
needs it.

Only the psum ("prefix-sum contiguous") layout is used: DeepEP's expanded
dispatch already groups received rows per local expert, padded to the
DeepGEMM M alignment, and its per-expert end offsets are the group boundaries.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch


@functools.cache
def load_deepgemm() -> ModuleType:
    """DeepGEMM's native module; importing ``deep_gemm`` initializes its JIT."""
    # Deferred: the fused_experts backend runs without DeepGEMM installed.
    import deep_gemm

    return deep_gemm._C


def contiguous_m_alignment() -> int:
    """Row alignment DeepGEMM requires between experts in the contiguous layout."""
    return int(load_deepgemm().get_theoretical_mk_alignment_for_contiguous_layout())


def set_contiguous_m_alignment(alignment: int) -> None:
    load_deepgemm().set_mk_alignment_for_contiguous_layout(alignment)


def set_num_sms(num_sms: int) -> None:
    """Size DeepGEMM's persistent grid (process-wide) to ``num_sms`` SMs."""
    load_deepgemm().set_num_sms(num_sms)


def m_grouped_bf16_gemm_nt_psum(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    psum_rows_per_expert: torch.Tensor,
    expected_m: int,
) -> None:
    """``out[rows of e] = a[rows of e] @ b[e].T`` for every local expert ``e``.

    ``a``: ``[M, K]`` BF16, rows grouped per expert. ``b``: ``[E, N, K]`` BF16.
    ``out``: ``[M, N]`` BF16. ``psum_rows_per_expert``: ``[E]`` int32 end offset
    of each expert's rows (DeepEP's expanded-dispatch prefix sum).
    """
    load_deepgemm().m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        out,
        psum_rows_per_expert,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )


def m_grouped_fp8_gemm_nt_psum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    psum_rows_per_expert: torch.Tensor,
    expected_m: int,
) -> None:
    """FP8 version of ``m_grouped_bf16_gemm_nt_psum`` (SM90 "1d2d" recipe).

    ``a``: ``[M, K]`` FP8 with ``a_scale`` ``[M, K/128]`` fp32 (one scale per row
    and 128 channels; MN-major TMA-aligned avoids a transpose). ``b``:
    ``[E, N, K]`` FP8 with ``b_scale`` ``[E, N/128, K/128]`` fp32 block scales
    (vLLM's block-FP8 layout). ``out``: ``[M, N]`` BF16.
    """
    load_deepgemm().m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale),
        (b, b_scale),
        out,
        psum_rows_per_expert,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )
