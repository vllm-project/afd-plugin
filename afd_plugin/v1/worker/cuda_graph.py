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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    """Extract the AFD FFN graph key, including exact Attention peer shapes."""

    key_parts: list[tuple[int, tuple]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = getattr(metadata, "num_tokens_across_dp_cpu", None)
        if values is None:
            if _use_ffn_peer_layout_key(attention_size, ffn_size):
                values_tuple = tuple(
                    max(1, int(fallback)) for _ in range(int(attention_size))
                )
            else:
                values_tuple = (repr(metadata),)
        else:
            values_tuple = _metadata_values_tuple(values)
            if _use_ffn_peer_layout_key(attention_size, ffn_size):
                values_tuple = _expand_attention_values_tuple(
                    values_tuple,
                    attention_size=int(attention_size),
                    fallback=int(fallback),
                )
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)


def make_mtp_ffn_graph_key(
    dp_metadata_list: Mapping[int, object],
    *,
    attention_size: int,
    ffn_size: int,
    fallback: int,
) -> tuple[int, ...]:
    """Return the merged MTP token layout for every Attention peer.

    The target decoder may use one or two stages, but the upstream proposer
    merges their hidden states into one draft phase.  Retain the exact peer
    layout instead of only the per-FFN aggregate: two layouts with the same
    sum capture different HCCL slices and must not share a graph.
    """

    decoder_key = make_ffn_graph_key(
        dp_metadata_list,
        attention_size=attention_size,
        ffn_size=ffn_size,
        fallback=fallback,
    )
    peer_totals = [0] * int(attention_size)
    for _, stage_values in decoder_key:
        if len(stage_values) != int(attention_size):
            raise ValueError(
                "MTP graph key requires one token count per Attention peer: "
                f"{len(stage_values)} != {attention_size}",
            )
        for peer_index, value in enumerate(stage_values):
            peer_totals[peer_index] += int(value)
    return tuple(max(1, value) for value in peer_totals)


def graph_run_mode(
    *,
    is_warmup: bool,
    is_graph_capturing: bool,
    graph_enabled: bool,
    graph_exists: bool,
) -> AFDGraphRunMode:
    if is_warmup:
        return AFDGraphRunMode.WARMUP
    if is_graph_capturing:
        return AFDGraphRunMode.CAPTURE
    if graph_enabled and graph_exists:
        return AFDGraphRunMode.REPLAY
    return AFDGraphRunMode.EAGER


def _metadata_values_tuple(values: object) -> tuple[int, ...]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    elif hasattr(values, "item"):
        values = [values.item()]
    try:
        return tuple(int(value) for value in values)
    except TypeError:
        return (int(values),)


def _use_ffn_peer_layout_key(
    attention_size: int | None,
    ffn_size: int | None,
) -> bool:
    return (
        attention_size is not None
        and ffn_size is not None
        and min(int(attention_size), int(ffn_size)) > 0
        and max(int(attention_size), int(ffn_size))
        % min(int(attention_size), int(ffn_size))
        == 0
    )


def _expand_attention_values_tuple(
    values: tuple[int, ...],
    *,
    attention_size: int,
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
        return tuple(max(1, int(fallback)) for _ in range(attention_size))
    # Graph capture records one HCCL recv/send per Attention peer. Retaining only
    # the FFN aggregate would alias different peer slice shapes with equal sums.
    return tuple(max(1, int(value)) for value in expanded[:attention_size])


__all__ = [
    "AFDCUDAGraphPolicy",
    "AFDGraphRunMode",
    "FULL_DECODE_ONLY",
    "cudagraph_mode_name",
    "graph_run_mode",
    "make_ffn_graph_key",
    "make_mtp_ffn_graph_key",
    "validate_cuda_graph_mode",
]
