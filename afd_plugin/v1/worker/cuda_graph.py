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
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
_SUPPORTED_GRAPH_MODES = {FULL_DECODE_ONLY}


class AFDGraphRunMode(StrEnum):
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
    uniform_padding: bool = False,
) -> tuple[tuple[int, tuple[object, ...]], ...]:
    """Extract the AFD FFN graph hashable key from DP metadata.

    ``uniform_padding`` selects the NPU CAMP2p row-count semantics: the
    Attention side pads every rank to the largest count, so each FFN rank
    receives ``uniform * group_size`` rows.  The default keeps the GPU
    contiguous-grouping sum.
    """

    key_parts: list[tuple[int, tuple[object, ...]]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = getattr(metadata, "num_tokens_across_dp_cpu", None)
        values_tuple: tuple[object, ...]
        if values is None:
            if (
                _use_ffn_aggregated_key(attention_size, ffn_size)
                and ffn_size is not None
            ):
                values_tuple = tuple(
                    max(1, int(fallback)) for _ in range(int(ffn_size))
                )
            else:
                values_tuple = (repr(metadata),)
        else:
            counts = _metadata_values_tuple(values)
            values_tuple = counts
            if (
                _use_ffn_aggregated_key(attention_size, ffn_size)
                and attention_size is not None
                and ffn_size is not None
            ):
                values_tuple = _aggregate_ffn_values_tuple(
                    counts,
                    attention_size=attention_size,
                    ffn_size=ffn_size,
                    fallback=int(fallback),
                    uniform_padding=uniform_padding,
                )
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)


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


def _metadata_values_tuple(values: object) -> tuple[int, ...]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    elif hasattr(values, "item"):
        values = [values.item()]
    try:
        return tuple(int(value) for value in values)  # type: ignore[attr-defined]
    except TypeError:
        return (int(values),)  # type: ignore[call-overload]


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
    uniform_padding: bool = False,
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
    if uniform_padding and group_size > 1:
        # N:1 mapping with uniform padding (NPU CAMP2p): the Attention side
        # pads every rank up to the largest count across all Attention
        # ranks, so every FFN rank receives ``uniform * group_size`` rows.
        # Must stay in sync with the NPU runner's token aggregation: a
        # replayed graph must have been captured for exactly the row count
        # the current step produces.
        uniform = max(1, max(expanded))
        return tuple(uniform * group_size for _ in range(ffn_size))
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
    "validate_cuda_graph_mode",
]
