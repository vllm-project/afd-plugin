# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP forward-context coordination.

This module patches:
1. ``vllm.forward_context.set_forward_context``

Why:
    vLLM 0.28.0 constructs ``DPMetadata`` and coordinates token counts across
    MoE DP ranks whenever DP size is greater than one. AFD async-DP uses the
    connector data flow instead of vLLM's DP metadata control plane, so those
    all-reduce and metadata paths must be skipped for the AFD async connector.

How:
    This module is imported by ``register_afd()``. Importing it patches
    ``set_forward_context`` in-place. The patched function expands the pinned
    upstream implementation and only changes the DP metadata section for AFD
    async configs.

Future plan:
    Remove this patch when vLLM exposes a plugin-owned way to opt out of MoE
    DP metadata coordination per engine role.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeAlias

import vllm.forward_context as forward_context_module
from vllm.config import CUDAGraphMode

from afd_plugin.compat.vllm import TARGET_VLLM_VERSION
from afd_plugin.config import is_afd_async_dp

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.worker.ubatch_utils import UBatchSlices

    AttentionMetadataMapping: TypeAlias = (
        dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    )
_FORWARD_CONTEXT_IMPORT_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.kv_connector_model_runner_mixin",
    "vllm_ascend.ascend_forward_context",
)


# Patch reason: upstream set_forward_context always creates DPMetadata for MoE
# DP ranks, but AFD async-DP uses connector-driven coordination instead.
# Patch functionality: preserves the target upstream tag's forward-context
# setup and skips DPMetadata creation only for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    need_to_track_batchsize = (
        forward_context_module.track_batchsize and attn_metadata is not None
    )
    if need_to_track_batchsize:
        forward_context_module.forward_start_time = (
            forward_context_module.time.perf_counter()
        )

    dp_metadata: forward_context_module.DPMetadata | None = None
    # ### PATCH START: AFD async-DP forward context metadata
    # AFD async-DP coordinates batches through connector flow, so skip vLLM's
    # native DPMetadata construction only for AFD async configs.
    if not is_afd_async_dp(vllm_config) and (
        (
            vllm_config.parallel_config.data_parallel_size > 1
            or vllm_config.parallel_config.use_sequence_parallel_moe
        )
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if (
            num_tokens_across_dp is None
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = (
                forward_context_module.coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=vllm_config.parallel_config,
                    allow_microbatching=False,
                )
            )
            assert num_tokens_across_dp is not None
        elif num_tokens_across_dp is None:
            assert num_tokens is not None
            num_tokens_across_dp = forward_context_module.torch.tensor(
                [num_tokens],
                dtype=forward_context_module.torch.int32,
            )
        dp_metadata = forward_context_module.DPMetadata.make(
            vllm_config.parallel_config,
            num_tokens or 0,
            num_tokens_across_dp,
        )
    # ### PATCH END: AFD async-DP forward context metadata

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or forward_context_module.BatchDescriptor(
            num_tokens=num_tokens,
        )

    additional_kwargs = (
        forward_context_module.current_platform.set_additional_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            dp_metadata=dp_metadata,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
        )
    )

    forward_context = forward_context_module.create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding=is_padding,
    )

    try:
        with forward_context_module.override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            synchronize = forward_context_module.current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = forward_context_module.time.perf_counter()
            # time measurement is in milliseconds
            forward_context_module.batchsize_forward_time[batchsize].append(
                (now - forward_context_module.forward_start_time) * 1000,
            )
            if (
                now - forward_context_module.last_logging_time
                > forward_context_module.batchsize_logging_interval
            ):
                forward_context_module.last_logging_time = now
                forward_stats = []
                for bs, times in forward_context_module.batchsize_forward_time.items():
                    if len(times) <= 1:
                        continue
                    medium = forward_context_module.torch.quantile(
                        forward_context_module.torch.tensor(times),
                        q=0.5,
                    ).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    forward_context_module.logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_VERSION)


if _is_target_vllm_compatible():
    forward_context_module.set_forward_context = set_forward_context
    for module_name in _FORWARD_CONTEXT_IMPORT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.set_forward_context = set_forward_context
    forward_context_module.logger.debug(
        "AFD async-DP forward-context patch applied",
    )


__all__: list[str] = []
