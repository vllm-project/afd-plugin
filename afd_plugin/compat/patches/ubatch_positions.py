# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Propagate ``positions`` through vLLM's ubatch attention-metadata split.

``CommonAttentionMetadata.positions`` is per-token and optional; upstream's
``split_attn_metadata`` rebuilds each ubatch's metadata without it, so every
ubatch sees ``positions=None``. The DeepSeek-V4 C128A metadata builder asserts
``positions`` is present, which makes any DBO-split forward fail. Re-slice the
source positions onto each rebuilt metadata after the upstream split.
"""

from __future__ import annotations

from vllm.v1.worker import ubatch_utils as ubatch_utils_module

_upstream_split_attn_metadata = ubatch_utils_module.split_attn_metadata


# Patch reason: upstream split_attn_metadata drops CommonAttentionMetadata
# .positions, and the DeepSeek-V4 C128A metadata builder asserts on it, so
# every DBO-split forward of a V4 model dies in the attention metadata build.
# Patch functionality: after the upstream split, re-slice the source
# metadata's per-token positions onto each ubatch's metadata by that
# ubatch's token slice; None stays None.
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0, vllm/v1/worker/ubatch_utils.py
def split_attn_metadata(ubatch_slices, common_attn_metadata):
    results = _upstream_split_attn_metadata(ubatch_slices, common_attn_metadata)
    positions = common_attn_metadata.positions
    if positions is not None:
        for ubatch_slice, ubatch_metadata in zip(ubatch_slices, results, strict=False):
            ubatch_metadata.positions = positions[ubatch_slice.token_slice]
    return results


split_attn_metadata.__afd_positions_propagated = True  # type: ignore[attr-defined]


def apply_positions_propagation() -> None:
    """Install the position-propagating splitter into vLLM's module namespaces.

    The plugin loads before vLLM's worker stack is importable, so importing
    ``gpu_model_runner`` here would fail on a partial import; the source
    module is always patched (the runner binds the name when it is first
    imported, picking the patched function up), and the runner's namespace
    is only re-aliased when that module already exists in ``sys.modules``.
    Idempotent via a marker attribute on the installed function.
    """
    import sys

    if getattr(split_attn_metadata, "_afd_positions_propagated_installed", False):
        return
    split_attn_metadata._afd_positions_propagated_installed = True  # type: ignore[attr-defined]
    ubatch_utils_module.split_attn_metadata = split_attn_metadata
    runner = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    if runner is not None and hasattr(runner, "split_attn_metadata"):
        runner.split_attn_metadata = split_attn_metadata


apply_positions_propagation()

__all__ = ["apply_positions_propagation", "split_attn_metadata"]
