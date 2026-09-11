# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA graph policy helpers for AFD runtimes.

This module intentionally avoids importing torch or vLLM at module import time.
It works with real vLLM config objects and with the small SimpleNamespace fakes
used by CPU-safe tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from vllm.config import VllmConfig

FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
_SUPPORTED_GRAPH_MODES = {FULL_DECODE_ONLY}


class AFDGraphRunMode(str, Enum):
    EAGER = "eager"
    WARMUP = "warmup"
    CAPTURE = "capture"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class AFDCUDAGraphPolicy:
    """Resolved AFD CUDA graph policy for one runtime role."""

    enabled: bool
    mode_name: str | None
    allow_attention_full_decode_only: bool
    enable_ffn_graph_cache: bool
    allow_cuda_graph_with_ubatching: bool = False


def validate_cuda_graph_mode(
    vllm_config: VllmConfig,
    *,
    role: str | None = None,
) -> AFDCUDAGraphPolicy:
    """Return the CUDA graph policy or raise for unsupported AFD modes."""

    enforce_eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
    mode_name = cudagraph_mode_name(vllm_config)
    graph_enabled = not enforce_eager

    if not graph_enabled:
        return AFDCUDAGraphPolicy(
            enabled=False,
            mode_name=mode_name,
            allow_attention_full_decode_only=False,
            enable_ffn_graph_cache=False,
        )

    if mode_name not in _SUPPORTED_GRAPH_MODES:
        role_suffix = f" for {role}" if role else ""
        raise RuntimeError(
            "AFD only supports CUDA graph mode "
            f"{FULL_DECODE_ONLY}{role_suffix}; got {mode_name!r}.",
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    use_ubatching = bool(getattr(parallel_config, "use_ubatching", False))
    num_ubatches = getattr(parallel_config, "num_ubatches", None)
    allow_ubatching = use_ubatching and int(num_ubatches or 0) == 2
    if use_ubatching and not allow_ubatching:
        raise RuntimeError(
            "AFD CUDA graph support currently supports ubatching only for "
            f"{FULL_DECODE_ONLY} with exactly two ubatches; "
            f"got num_ubatches={num_ubatches!r}.",
        )

    return AFDCUDAGraphPolicy(
        enabled=True,
        mode_name=mode_name,
        allow_attention_full_decode_only=role in (None, "attention"),
        enable_ffn_graph_cache=role in (None, "ffn"),
        allow_cuda_graph_with_ubatching=allow_ubatching,
    )


def cudagraph_mode_name(vllm_config: VllmConfig) -> str | None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None:
        return None

    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name

    text = str(mode)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or None


def make_ffn_graph_key(
    dp_metadata_list: Mapping[int, object],
    *,
    attention_size: int | None = None,
    ffn_size: int | None = None,
    fallback: int = 1,
) -> tuple[tuple[int, tuple]]:
    """Extract the AFD FFN graph hashable key from DP metadata."""

    key_parts: list[tuple[int, tuple]] = []
    values_tuple: tuple[Any, ...]
    # Narrowed here rather than inside _use_ffn_aggregated_key so the sizes
    # stay non-optional for the int() calls below.
    aggregated = (
        attention_size is not None
        and ffn_size is not None
        and _use_ffn_aggregated_key(attention_size, ffn_size)
    )
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = getattr(metadata, "num_tokens_across_dp_cpu", None)
        if values is None:
            if aggregated:
                assert ffn_size is not None
                values_tuple = tuple(
                    max(1, int(fallback)) for _ in range(int(ffn_size))
                )
            else:
                values_tuple = (repr(metadata),)
        else:
            values_tuple = _metadata_values_tuple(values)
            if aggregated:
                assert attention_size is not None and ffn_size is not None
                values_tuple = _aggregate_ffn_values_tuple(
                    values_tuple,
                    attention_size=int(attention_size),
                    ffn_size=int(ffn_size),
                    fallback=int(fallback),
                )
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)  # type: ignore[return-value]


def padded_ffn_graph_shape(
    *,
    num_tokens: int,
    topk: int,
    ffn_size: int,
    has_shared_experts: bool,
) -> tuple[int, int]:
    """Rows the captured shape has to hold.

    The grouped GEMM does not care how many rows are real -- it reads its
    grouping from a device-side count vector -- so a fixed row count can be
    captured once and every smaller item padded up to it. What the shape has to
    be is an upper bound for every item padded into it, which is the
    largest batch the sender can produce.

    A token contributes at most ``topk`` partials to any single FFN rank (the
    case where every one of its experts lives there). Shared-expert rows are
    split contiguously across the FFN ranks, so a rank holds at most
    ``ceil(num_tokens / ffn_size)`` of them, and none at all without shared
    experts.

    Returns:
        ``(max_routed_rows, max_shared_rows)``.
    """
    if num_tokens <= 0 or topk <= 0 or ffn_size <= 0:
        raise ValueError(
            "padded FFN graph shape needs positive num_tokens, topk and "
            f"ffn_size; got {num_tokens}, {topk}, {ffn_size}",
        )
    max_routed = num_tokens * topk
    max_shared = -(-num_tokens // ffn_size) if has_shared_experts else 0
    return max_routed, max_shared


# Every replay costs its captured row count, not the item's real one, so a
# single graph at the upper bound charges the worst case to every item. The
# bound assumes all of a token's topk partials land on one rank; with experts
# spread over the ranks a token sends about ``topk / ffn_size`` of them, so a
# real item occupies about ``max_routed / ffn_size`` rows -- call that the
# expected size -- and a DBO ubatch half of that again.
#
# The ladder is therefore built as multiples of the expected size rather than
# as fractions of the worst case. That distinction matters at the top of the
# common range: routing scatter puts about half the items just above the
# expected size, and a bucket boundary sitting exactly on it sends every one of
# them a full step up. The multiples below cluster tightly just above 1.0 (and
# above 0.5, where a DBO ubatch lands) so those items pay a few percent instead
# of a quarter. Each entry costs one captured graph per MoE layer, so the
# density is a memory trade.
# 0.5 and 0.55 cover a DBO ubatch, which is half an item; 1.0 and 1.1 cover a
# whole one. Each entry costs one captured graph per MoE layer, so the density
# is spent at those two clusters rather than spread evenly.
PADDED_FFN_GRAPH_EXPECTED_MULTIPLES: tuple[float, ...] = (
    0.25,
    0.5,
    0.55,
    0.7,
    1.0,
    1.1,
    1.25,
    1.5,
)

PADDED_FFN_GRAPH_FRACTIONS_ENV = "AFD_FFN_GRAPH_FRACTIONS"

# Bucket row counts are rounded up to this, so a bucket is always a whole
# number of GEMM tiles rather than a ragged tail.
PADDED_FFN_BUCKET_ALIGNMENT = 128

# How much padding a replay may carry before running the item eagerly instead.
# A replay pays for its whole bucket but saves the per-layer launch work; eager
# pays launches but computes only the real rows. Measured on DeepSeek-V4 2A2F,
# eager beat a replay padded to 1.18x by 6%, so a replay only pays for itself
# when it is close to the item's own size. Items landing exactly on a bucket
# still replay; the rest take the cheaper path.
MAX_REPLAY_PADDING_RATIO = 1.05


def resolve_padded_ffn_graph_fractions(
    environ: Mapping[str, str] | None = None,
) -> tuple[float, ...]:
    """Parse the multiples override, falling back to the default set."""
    import os

    source = os.environ if environ is None else environ
    raw = source.get(PADDED_FFN_GRAPH_FRACTIONS_ENV, "").strip()
    if not raw:
        return PADDED_FFN_GRAPH_EXPECTED_MULTIPLES
    return tuple(float(part) for part in raw.replace(",", " ").split())


def padded_ffn_graph_buckets(
    max_routed: int,
    *,
    ffn_size: int = 1,
    fractions: tuple[float, ...] | None = None,
) -> tuple[int, ...]:
    """Ascending distinct row counts to capture for one MoE layer.

    The multiples are of the expected item size, ``max_routed / ffn_size``, not
    of ``max_routed`` -- see the comment on the default set. Entries are rounded
    up to ``PADDED_FFN_BUCKET_ALIGNMENT`` so a bucket boundary stays a sane GEMM
    tile count, clamped to ``max_routed``, and deduplicated.

    The ladder need not reach ``max_routed``: an item above the largest bucket
    runs eager, the same fallback an item larger than the captured shape has
    always taken. That is only safe because ``capture_padded_ffn_graphs`` pins
    the shared MoE workspace at the ceiling before capturing -- see the note
    there; without it, the first oversized item grows the workspace and
    invalidates every captured graph.
    """
    if max_routed <= 0:
        raise ValueError(f"max_routed must be positive; got {max_routed}")
    if ffn_size <= 0:
        raise ValueError(f"ffn_size must be positive; got {ffn_size}")
    if fractions is None:
        fractions = resolve_padded_ffn_graph_fractions()
    expected = max_routed / ffn_size
    buckets = set()
    for multiple in fractions:
        if multiple <= 0:
            raise ValueError(
                f"padded FFN graph multiples must be positive; got {multiple}",
            )
        rows = int(expected * multiple)
        rows = -(-rows // PADDED_FFN_BUCKET_ALIGNMENT) * PADDED_FFN_BUCKET_ALIGNMENT
        buckets.add(min(max(rows, PADDED_FFN_BUCKET_ALIGNMENT), max_routed))
    return tuple(sorted(buckets))


def shared_rows_for_bucket(
    bucket: int,
    *,
    max_routed: int,
    max_shared: int,
) -> int:
    """Shared-expert rows captured alongside a bucket's routed rows.

    Both counts scale with the item's token count, so a bucket holding half the
    routed rows needs half the shared rows. Capturing every bucket at
    ``max_shared`` instead makes a half-sized item pay double on the shared
    expert -- measured as graphs losing to eager under DBO even once the routed
    rows were bucketed.
    """
    if max_shared <= 0:
        return 0
    rows = -(-max_shared * bucket // max_routed)
    return min(max(rows, 1), max_shared)


def select_padded_ffn_bucket(
    buckets: tuple[int, ...],
    routed_rows: int,
    shared_rows: int = 0,
    *,
    max_routed: int | None = None,
    max_shared: int = 0,
) -> int | None:
    """Smallest captured bucket holding both row counts, or ``None``.

    ``None`` means no bucket fits and the caller runs the item eagerly, which
    is what an item larger than the captured maximum has always done. The
    shared rows are checked too, because each bucket's graph captures only its
    own share of them.
    """
    for bucket in buckets:
        if routed_rows > bucket:
            continue
        if max_routed is not None and shared_rows > shared_rows_for_bucket(
            bucket, max_routed=max_routed, max_shared=max_shared
        ):
            continue
        return bucket
    return None


def pad_counts_to_shape(
    counts: Tensor,
    *,
    padded_rows: int,
    actual_rows: int,
) -> None:
    """Grow ``counts`` in place so its entries sum to ``padded_rows``.

    The padding lands on the last expert, which is where the padded rows are:
    real rows are grouped by expert in ascending order, so the tail of the row
    range belongs to the last expert either way. That keeps every real row's
    expert assignment untouched.

    The padded rows carry whatever the input buffer last held. Their output is
    sliced off and discarded, and a grouped GEMM is row-independent, so their
    content cannot reach a real row.
    """
    if actual_rows > padded_rows:
        raise ValueError(
            f"{actual_rows} rows do not fit the padded shape {padded_rows}",
        )
    counts[-1] += padded_rows - actual_rows


def graph_run_mode(
    *,
    is_warmup: bool,
    is_graph_capturing: bool,
    is_graph_replaying: bool,
    graph_enabled: bool,
    graph_exists: bool,
) -> AFDGraphRunMode:
    if is_warmup:
        return AFDGraphRunMode.WARMUP
    if is_graph_capturing:
        return AFDGraphRunMode.CAPTURE
    if is_graph_replaying and graph_enabled and graph_exists:
        return AFDGraphRunMode.REPLAY
    return AFDGraphRunMode.EAGER


def _metadata_values_tuple(values: Any) -> tuple[int, ...]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    elif hasattr(values, "item"):
        values = [values.item()]
    try:
        return tuple(int(value) for value in values)
    except TypeError:
        return (int(values),)


def _use_ffn_aggregated_key(
    attention_size: int | None,
    ffn_size: int | None,
) -> bool:
    return (
        attention_size is not None
        and ffn_size is not None
        and int(attention_size) >= int(ffn_size)
        and int(attention_size) % int(ffn_size) == 0
    )


def _aggregate_ffn_values_tuple(
    values: tuple[int, ...],
    *,
    attention_size: int,
    ffn_size: int,
    fallback: int,
) -> tuple[int, ...]:
    # Expand DP-level values to AFD-level when TP > 1.
    # With TP > 1, attention_size = num_attention_ranks includes TP workers
    # but values only has dp_size entries (from num_tokens_across_dp_cpu).
    # Each DP rank's count is replicated tp_size times because all TP workers
    # within the same DP rank process the same tokens.
    expanded = values
    if len(values) < attention_size and attention_size % len(values) == 0:
        tp_size = attention_size // len(values)
        expanded = tuple(values[i // tp_size] for i in range(attention_size))
    if len(expanded) < attention_size:
        return tuple(max(1, int(fallback)) for _ in range(ffn_size))
    group_size = attention_size // ffn_size
    return tuple(
        max(1, sum(expanded[idx * group_size : (idx + 1) * group_size]))
        for idx in range(ffn_size)
    )


__all__ = [
    "AFDCUDAGraphPolicy",
    "AFDGraphRunMode",
    "FULL_DECODE_ONLY",
    "cudagraph_mode_name",
    "graph_run_mode",
    "make_ffn_graph_key",
    "pad_counts_to_shape",
    "MAX_REPLAY_PADDING_RATIO",
    "PADDED_FFN_BUCKET_ALIGNMENT",
    "PADDED_FFN_GRAPH_EXPECTED_MULTIPLES",
    "PADDED_FFN_GRAPH_FRACTIONS_ENV",
    "padded_ffn_graph_buckets",
    "resolve_padded_ffn_graph_fractions",
    "padded_ffn_graph_shape",
    "select_padded_ffn_bucket",
    "shared_rows_for_bucket",
    "validate_cuda_graph_mode",
]
