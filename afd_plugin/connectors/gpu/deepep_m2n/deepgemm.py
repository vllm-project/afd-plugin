# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepGEMM grouped GEMMs the M2N experts use.

The kernels come from the DeepGEMM copy bundled in the vLLM wheel,
``vllm.third_party.deep_gemm``, so the connector runs the DeepGEMM version the
pinned vLLM ships and a stock vLLM install needs no extra build. A separately
installed upstream ``deep_gemm`` package is not used here (vLLM's own kernels
may still prefer it).

The kernels are JIT-compiled at runtime. Only the ``deepgemm`` expert backend
needs DeepGEMM.

Only the psum ("prefix-sum contiguous") layout is used: DeepEP's expanded
dispatch already groups received rows per local expert, padded to the
DeepGEMM M alignment, and its per-expert end offsets are the group boundaries.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import torch

# The DeepGEMM copy vendored in the vLLM wheel.
DEEPGEMM_PACKAGE = "vllm.third_party.deep_gemm"


@functools.cache
def load_deepgemm() -> ModuleType:
    """The native ``_C`` module of the DeepGEMM bundled with vLLM.

    Importing the package initializes its JIT. The import is deferred to first
    use because the ``fused_experts`` backend runs without DeepGEMM.
    """
    try:
        return importlib.import_module(f"{DEEPGEMM_PACKAGE}._C")
    except ModuleNotFoundError as error:
        # Only a vLLM build without DeepGEMM gets the hint; any other missing
        # module means the bundled copy is broken and surfaces unchanged.
        missing = error.name
        if missing is None or not (
            missing == DEEPGEMM_PACKAGE or DEEPGEMM_PACKAGE.startswith(f"{missing}.")
        ):
            raise
        raise ModuleNotFoundError(
            "The DeepEP M2N 'deepgemm' expert backend needs the DeepGEMM bundled "
            f"with vLLM ({DEEPGEMM_PACKAGE}), which this vLLM build does not "
            "include. Use a vLLM CUDA build, or set expert_backend: fused_experts.",
            name=error.name,
        ) from error


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
