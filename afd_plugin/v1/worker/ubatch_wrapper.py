# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned vLLM ubatch wrapper.

This runtime module depends on vLLM's native ubatching stack.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
    get_forward_context,
)
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper
from vllm.v1.worker.ubatching import make_ubatch_contexts

from afd_plugin.config import is_afd_active
from afd_plugin.connectors import AFDForwardContextMetadata
from afd_plugin.v1.worker.attention_metadata import build_ubatch_dp_metadata_list


class AFDUBatchWrapper(UBatchWrapper):
    """Thin AFD-aware subclass of vLLM's native ``UBatchWrapper``."""

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        super().__init__(runnable, vllm_config, runtime_mode, device)
        self._afd_metadata_installer: Callable[[ForwardContext], None] | None = None

    def configure_afd_context_provider(
        self,
        installer: Callable[[ForwardContext], None],
    ) -> None:
        """Save the same typed installer used by the forward-context provider."""

        self._afd_metadata_installer = installer

    # Patch reason: native SM partitioning conflicts with AFD connector work.
    # Patch functionality: disable native SM partitioning only for active AFD.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.28.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 2cf0a6915ce544dc493a0990f2ea38d81601128a
    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        # ### PATCH START: leave all SMs visible to AFD compute and communication.
        if is_afd_active(vllm_config):
            return nullcontext()
        # ### PATCH END: leave all SMs visible to AFD compute and communication.
        return UBatchWrapper._create_sm_control_context(vllm_config)

    # Patch reason: native ubatch contexts do not carry AFD transfer metadata.
    # Patch functionality: install per-ubatch AFD context and control-plane
    # metadata while preserving native capture, replay, and execution behavior.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.28.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 2cf0a6915ce544dc493a0990f2ea38d81601128a
    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        if ubatch_slices is None:
            return super().__call__(*args, **kwargs)

        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        # ### PATCH START: install AFD metadata before splitting ubatches.
        parent_additional_kwargs = dict(forward_context.additional_kwargs)
        if "afd_metadata" not in parent_additional_kwargs:
            self._install_missing_afd_metadata(forward_context)
            parent_additional_kwargs = dict(forward_context.additional_kwargs)

        num_tokens = sum(int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices)
        dp_metadata = build_ubatch_dp_metadata_list(
            self.vllm_config,
            ubatch_slices,
        )
        # ### PATCH END: install AFD metadata before splitting ubatches.

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=forward_context.attn_metadata,
                slot_mapping=forward_context.slot_mapping,
                input_ids=kwargs["input_ids"],
                positions=kwargs["positions"],
                inputs_embeds=kwargs["inputs_embeds"],
                intermediate_tensors=kwargs["intermediate_tensors"],
                compute_stream=torch.cuda.current_stream(),
                dp_metadata=dp_metadata,
                batch_descriptor=forward_context.batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)

        if (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            get_offloader().sync_prev_onload()
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=ubatch_slices,
            attn_metadata=forward_context.attn_metadata,
            slot_mapping=forward_context.slot_mapping,
            input_ids=kwargs["input_ids"],
            positions=kwargs["positions"],
            inputs_embeds=kwargs["inputs_embeds"],
            intermediate_tensors=kwargs["intermediate_tensors"],
            compute_stream=torch.cuda.current_stream(),
            dp_metadata=dp_metadata,
            batch_descriptor=forward_context.batch_descriptor,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        with self.sm_control:
            return self._run_ubatches(ubatch_metadata, self.runnable)

    def _install_missing_afd_metadata(
        self,
        forward_context: ForwardContext,
    ) -> None:
        installer = self._afd_metadata_installer
        if installer is None:
            self._afd_use_native_ubatch_metadata = True
            return

        installer(forward_context)

    # Patch reason: native per-ubatch contexts omit AFD transfer metadata.
    # Patch functionality: clone the parent AFD context into each native ubatch.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.28.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 2cf0a6915ce544dc493a0990f2ea38d81601128a
    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        # ### PATCH START: resolve and validate the parent AFD context.
        parent_forward_context = get_forward_context()
        parent_additional_kwargs = dict(parent_forward_context.additional_kwargs)
        afd_metadata = parent_additional_kwargs.get("afd_metadata")
        if afd_metadata is None:
            if getattr(self, "_afd_use_native_ubatch_metadata", False):
                try:
                    return UBatchWrapper._make_ubatch_metadata(
                        self,
                        ubatch_slices,
                        attn_metadata,
                        slot_mapping,
                        input_ids,
                        positions,
                        inputs_embeds,
                        intermediate_tensors,
                        compute_stream,
                        dp_metadata,
                        batch_descriptor,
                        cudagraph_runtime_mode,
                    )
                finally:
                    self._afd_use_native_ubatch_metadata = False
            raise RuntimeError(
                "AFDUBatchWrapper requires "
                "ForwardContext.additional_kwargs['afd_metadata']",
            )
        # ### PATCH END: resolve and validate the parent AFD context.

        forward_contexts = []
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for idx, _ubatch_slice in enumerate(ubatch_slices):
            # ### PATCH START: attach one AFD context to each native ubatch.
            ubatch_afd_metadata = build_ubatch_afd_metadata(
                afd_metadata,
                ubatch_slices,
                idx,
            )
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[idx] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[idx],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[idx] if has_slot_mapping else None,
                    additional_kwargs=build_ubatch_additional_kwargs(
                        parent_additional_kwargs,
                        ubatch_afd_metadata,
                    ),
                ),
            )
            # ### PATCH END: attach one AFD context to each native ubatch.

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for idx, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[idx],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                ),
            )

        return ubatch_metadata


def build_ubatch_afd_metadata(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: Any,
    ubatch_idx: int,
) -> AFDForwardContextMetadata:
    """Clone parent AFD metadata for one vLLM ubatch."""

    if ubatch_idx < 0 or ubatch_idx >= len(ubatch_slices):
        raise IndexError(f"ubatch_idx {ubatch_idx} out of range")

    ubatch_slice = ubatch_slices[ubatch_idx]
    clone = afd_metadata.clone()
    clone.stage_idx = ubatch_idx
    clone.num_stages = len(ubatch_slices)
    clone.tokens_start_loc = [int(ubatch_slice.token_slice.start)]
    clone.requests_start_loc = [int(ubatch_slice.request_slice.start)]
    clone.tokens_lens = [int(ubatch_slice.num_tokens)]
    clone.tokens_unpadded_lens = [
        _resolve_ubatch_unpadded_tokens(afd_metadata, ubatch_slice, ubatch_idx),
    ]
    return clone


def build_ubatch_additional_kwargs(
    parent_additional_kwargs: dict[str, Any],
    afd_metadata: AFDForwardContextMetadata,
) -> dict[str, Any]:
    child_kwargs = dict(parent_additional_kwargs)
    child_kwargs["afd_metadata"] = afd_metadata
    return child_kwargs


def _resolve_ubatch_unpadded_tokens(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slice: Any,
    ubatch_idx: int,
) -> int:
    unpadded_lens = afd_metadata.tokens_unpadded_lens
    if ubatch_idx < len(unpadded_lens):
        return int(unpadded_lens[ubatch_idx])
    return int(ubatch_slice.num_tokens)


__all__ = [
    "AFDUBatchWrapper",
    "build_ubatch_additional_kwargs",
    "build_ubatch_afd_metadata",
    "build_ubatch_dp_metadata_list",
]
