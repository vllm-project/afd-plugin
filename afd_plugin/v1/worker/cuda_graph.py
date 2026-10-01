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

from afd_plugin.a2e_layout import ffn_receive_rows

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
) -> tuple[tuple[int, tuple], ...]:
    """Extract the AFD FFN graph hashable key from DP metadata."""

    attention_ranks = 0 if attention_size is None else int(attention_size)
    ffn_ranks = 0 if ffn_size is None else int(ffn_size)
    aggregated = _use_ffn_aggregated_key(attention_size, ffn_size)
    key_parts: list[tuple[int, tuple]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = getattr(metadata, "num_tokens_across_dp_cpu", None)
        if values is None:
            if aggregated:
                values_tuple: tuple = tuple(
                    max(1, int(fallback)) for _ in range(ffn_ranks)
                )
            else:
                values_tuple = (repr(metadata),)
        else:
            values_tuple = _metadata_values_tuple(values)
            if aggregated:
                values_tuple = _aggregate_ffn_values_tuple(
                    values_tuple,
                    attention_size=attention_ranks,
                    ffn_size=ffn_ranks,
                    fallback=int(fallback),
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
    items: Any = values
    tolist = getattr(values, "tolist", None)
    item = getattr(values, "item", None)
    if callable(tolist):
        items = tolist()
    elif callable(item):
        items = [item()]
    try:
        return tuple(int(value) for value in items)
    except TypeError:
        return (int(items),)


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
    # Only the AFD NPU runners pass the role sizes, so this branch describes the
    # A2E tile layout: every FFN rank receives ``attention_size // ffn_size`` tiles
    # of the largest count in its (strided) Attention peer group. The key has to
    # match the rows the transfer actually delivers, because a key built from the
    # real counts would send an uneven step to eager execution with a tile A2E
    # cannot represent.
    return tuple(
        ffn_receive_rows(
            values,
            ffn_rank,
            attention_size=int(attention_size),
            ffn_size=int(ffn_size),
            fallback=int(fallback),
        )
        for ffn_rank in range(max(1, int(ffn_size)))
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
