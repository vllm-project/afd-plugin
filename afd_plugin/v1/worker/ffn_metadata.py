# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe token metadata helpers for AFD FFN runtimes."""

from __future__ import annotations


def aggregate_ffn_token_counts(
    attention_counts: tuple[int, ...],
    *,
    attention_size: int,
    ffn_size: int,
    fallback: int = 1,
) -> tuple[int, ...]:
    """Aggregate consecutive Attention-rank counts for each FFN rank.

    The blocks are the ones the rank mapping builds: contiguous and differing
    in size by at most one, so ``3A2F`` groups ``{A0, A1}`` onto ``F0`` and
    ``{A2}`` onto ``F1``. For example, ``4A2F`` counts ``(0, 4, 5, 6)`` become
    ``(5, 11)`` because every zero-token Attention peer contributes one
    placeholder row. Missing peers use the same per-peer fallback, so empty
    counts become ``(2, 2)``.
    """

    fallback_count = max(1, int(fallback))
    fallback_counts = tuple(fallback_count for _ in range(max(0, ffn_size)))
    if ffn_size <= 0 or attention_size < ffn_size:
        return fallback_counts

    expanded_counts = attention_counts
    if (
        attention_counts
        and len(attention_counts) < attention_size
        and attention_size % len(attention_counts) == 0
    ):
        attention_ranks_per_count = attention_size // len(attention_counts)
        expanded_counts = tuple(
            attention_counts[rank // attention_ranks_per_count]
            for rank in range(attention_size)
        )

    # Block ``g`` spans Attention ranks ``[ceil(g*A/F), ceil((g+1)*A/F))``, the
    # same partition ``build_rank_mapping`` uses. The ``+ ffn_size - 1`` rounds
    # the division up; it is a plain ``//`` when ``ffn_size`` divides ``g*A``.
    return tuple(
        sum(
            max(1, int(expanded_counts[attention_rank]))
            if attention_rank < len(expanded_counts)
            else fallback_count
            for attention_rank in range(
                (ffn_rank * attention_size + ffn_size - 1) // ffn_size,
                ((ffn_rank + 1) * attention_size + ffn_size - 1) // ffn_size,
            )
        )
        for ffn_rank in range(ffn_size)
    )


def project_ffn_token_counts_to_dp(
    ffn_counts: tuple[int, ...],
    *,
    dp_size: int,
) -> tuple[int, ...]:
    """Project FFN role-rank counts to vLLM data-parallel rank counts.

    For example, two FFN TP ranks per DP rank project
    ``(8, 8, 13, 13)`` to ``(8, 13)``.
    """

    if len(ffn_counts) == dp_size or dp_size <= 0 or len(ffn_counts) % dp_size != 0:
        return ffn_counts

    ffn_ranks_per_dp = len(ffn_counts) // dp_size
    return tuple(ffn_counts[rank * ffn_ranks_per_dp] for rank in range(dp_size))


__all__ = ["aggregate_ffn_token_counts", "project_ffn_token_counts_to_dp"]
