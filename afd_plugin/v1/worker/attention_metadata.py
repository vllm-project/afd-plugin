# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared AFD runner metadata and rank helpers."""

from __future__ import annotations

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import DPMetadata, ForwardContext
from vllm.v1.worker.ubatch_utils import UBatchSlices

from afd_plugin.connectors import (
    AFDControlPayload,
    AFDDPMetadata,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.base import AFDConnectorBase


class AFDMetadataProviderMixin:
    """AFD metadata/control plumbing shared by the V1 and V2 runners.

    This mixin owns only the connector-facing metadata sidecar and DP control
    payload. ``AFDAttentionModelRunner`` consumes it alongside the existing V1
    Attention/graph lifecycle, while ``AFDAttentionModelRunnerV2`` consumes it
    alongside native MRV2 execution. Native runner state and model ownership
    remain in each vLLM base class.
    """

    _afd_is_profile: bool = False

    #: Provided by the consuming runner; declared so the mixin type-checks.
    connector: AFDConnectorBase
    vllm_config: VllmConfig
    _is_warmup: bool
    _afd_pending_metadata: AFDForwardContextMetadata | None
    _afd_transaction_counter: int

    def build_afd_metadata(
        self,
        ubatch_slices: UBatchSlices | None,
        num_tokens_unpadded: int,
    ) -> AFDForwardContextMetadata:
        """Build the AFD sidecar for one transaction without sending control.

        Without multiple ubatch stages, ``num_tokens_unpadded`` is the token count
        supplied by the caller.  With multiple stages, native slice lengths are
        authoritative for both token fields.  The returned metadata references
        the runner's connector and owns a fresh transaction id; this method does
        not mutate a ``ForwardContext`` or perform control- or data-plane I/O.
        """

        if ubatch_slices and len(ubatch_slices) > 1:
            tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            requests_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            tokens_unpadded_lens = [int(ub.num_tokens) for ub in ubatch_slices]
            num_stages = len(ubatch_slices)
        else:
            tokens_start_loc = [0]
            requests_start_loc = [0]
            tokens_lens = [num_tokens_unpadded]
            tokens_unpadded_lens = [num_tokens_unpadded]
            num_stages = 1

        return AFDForwardContextMetadata(
            tokens_start_loc=tokens_start_loc,
            requests_start_loc=requests_start_loc,
            stage_idx=0,
            connector=self.connector,
            tokens_lens=tokens_lens,
            num_stages=num_stages,
            transaction_id=self._next_afd_transaction_id(),
            tokens_unpadded_lens=tokens_unpadded_lens,
        )

    def send_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
        ubatch_slices: UBatchSlices | None,
    ) -> None:
        """Publish DP control for the pending AFD transaction.

        ``dp_metadata`` contains the native rank-specific counts selected for the
        current execution; graph lifecycle callers may supply metadata built from
        a padded descriptor.  ``ubatch_slices`` instead selects per-stage control
        payloads.  This method updates connector control state and then sends one
        payload; it does not allocate a transaction or perform data-plane work.
        """

        if self.connector.control_plane is None:
            # Control-plane-less connectors (GpuAsyncAFDConnector) carry all
            # per-stage control on the data plane and drive the FFN from its
            # own receive loop; each Attention replica also advances on its
            # own (see _dp_batch_coordination_disabled), so there is nothing
            # to publish here.
            return

        if ubatch_slices and len(ubatch_slices) > 1:
            dp_metadata_list = {
                idx: metadata
                for idx, metadata in enumerate(
                    build_ubatch_dp_metadata_list(self.vllm_config, ubatch_slices),
                )
            }
        else:
            dp_metadata = self._ensure_dp_metadata(dp_metadata)
            dp_metadata_list = {0: dp_metadata}
        is_warmup = self._is_warmup
        # Keep the V1 object.__new__ test seam and older graph lifecycle
        # callers compatible with runners created before this mixin existed.
        is_graph_capturing = getattr(self, "_afd_is_graph_capturing", False)
        is_graph_replaying = getattr(self, "_afd_is_graph_replaying", False)
        payload = AFDControlPayload(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
            is_graph_replaying=is_graph_replaying,
            is_profile=self._afd_is_profile,
        )
        self.connector.control_plane.update_state_from_dp_metadata(payload)
        self.connector.control_plane.send_dp_metadata_list(payload)

    def _ensure_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
    ) -> DPMetadata | AFDDPMetadata:
        if dp_metadata is not None:
            return dp_metadata

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        if dp_size != 1:
            raise RuntimeError("AFD expected vLLM DPMetadata for attention DP > 1")

        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP metadata fallback")
        if len(self._afd_pending_metadata.tokens_lens) != 1:
            raise RuntimeError("AFD DP=1 fallback only supports one stage")

        num_tokens = int(self._afd_pending_metadata.tokens_lens[0])
        num_tokens_across_dp_cpu = torch.tensor(
            [num_tokens],
            dtype=torch.int32,
            device="cpu",
        )
        return AFDDPMetadata(
            num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
            max_tokens_across_dp_cpu=torch.max(num_tokens_across_dp_cpu),
        )

    def build_capture_dp_metadata(
        self,
        num_tokens: int,
    ) -> DPMetadata | AFDDPMetadata:
        """Build, but do not send, uniform padded graph DP metadata.

        ``num_tokens`` is the padded CUDA graph token count executed on every DP
        rank during capture or replay.  The returned metadata is suitable for
        :meth:`send_dp_metadata`; this method has no transaction, context, or
        connector side effects.
        """

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        num_tokens_across_dp_cpu = torch.full(
            (dp_size,),
            int(num_tokens),
            dtype=torch.int32,
            device="cpu",
        )
        if dp_size > 1:
            return DPMetadata.make(
                self.vllm_config.parallel_config,
                int(num_tokens),
                num_tokens_across_dp_cpu,
            )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return AFDDPMetadata(
            num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
            max_tokens_across_dp_cpu=max_tokens_across_dp_cpu,
        )

    def _afd_num_tokens_for_context(self, forward_context: ForwardContext) -> int:
        return _forward_context_num_tokens(forward_context, self.vllm_config)

    def install_afd_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        """Install transaction metadata and publish its matching control.

        On the ordinary eager path, the method creates a pending transaction from
        the current context, writes it to ``ForwardContext.additional_kwargs``,
        and sends the context's native DP control before the model/data-plane
        forward.  Graph lifecycles may instead pre-stage the transaction and
        control payload; their suppression scope makes this method attach the
        staged sidecar without a duplicate send.  A FULL graph context that was
        not pre-sent uses its padded descriptor count for DP control.  Ubatch
        child contexts reuse their parent transaction without another send, and
        native FULL replay uses its pre-replay hook because it creates no context.
        """

        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        existing_metadata: AFDForwardContextMetadata | None = (
            forward_context.additional_kwargs.get("afd_metadata")
        )
        if existing_metadata is not None and _is_ubatch_child_afd_context(
            forward_context,
            existing_metadata,
        ):
            return

        if self._afd_pending_metadata is None:
            self._afd_pending_metadata = self.build_afd_metadata(
                forward_context.ubatch_slices,
                self._afd_num_tokens_for_context(forward_context),
            )
        if self._afd_pending_metadata is not None:
            forward_context.additional_kwargs["afd_metadata"] = (
                self._afd_pending_metadata
            )
        if getattr(self, "_afd_suppress_metadata_send", False):
            return
        dp_metadata = forward_context.dp_metadata
        ubatch_slices = forward_context.ubatch_slices
        padded_graph_tokens = _full_cudagraph_padded_tokens(forward_context)
        if padded_graph_tokens is not None and not ubatch_slices:
            dp_metadata = self.build_capture_dp_metadata(padded_graph_tokens)
        self.send_dp_metadata(dp_metadata, ubatch_slices)

    def _next_afd_transaction_id(self) -> str:
        counter = self._afd_transaction_counter
        self._afd_transaction_counter = counter + 1
        return f"afd-{counter}"


def _resolve_world_ranks() -> tuple[int, int]:
    group = get_world_group()
    return int(group.rank), int(group.local_rank)


def _is_ubatch_child_afd_context(
    forward_context: ForwardContext,
    afd_metadata: AFDForwardContextMetadata,
) -> bool:
    if forward_context.ubatch_slices is not None:
        return False
    if int(afd_metadata.num_stages or 1) <= 1:
        return False
    return len(afd_metadata.tokens_lens or []) == 1


def _forward_context_num_tokens(
    forward_context: ForwardContext,
    vllm_config: VllmConfig,
) -> int:
    dp_metadata = forward_context.dp_metadata
    dp_rank = int(vllm_config.parallel_config.data_parallel_rank)
    if dp_metadata is not None:
        return max(1, int(dp_metadata.num_tokens_across_dp_cpu[dp_rank]))

    batch_descriptor = forward_context.batch_descriptor
    if batch_descriptor is None:
        raise RuntimeError("AFD requires a native BatchDescriptor")
    return max(1, int(batch_descriptor.num_tokens))


def _full_cudagraph_padded_tokens(
    forward_context: ForwardContext,
) -> int | None:
    mode = forward_context.cudagraph_runtime_mode
    if mode != CUDAGraphMode.FULL:
        return None
    batch_descriptor = forward_context.batch_descriptor
    if batch_descriptor is None:
        return None
    return max(1, int(batch_descriptor.num_tokens))


def build_ubatch_dp_metadata_list(
    vllm_config: VllmConfig,
    ubatch_slices: UBatchSlices,
) -> list[DPMetadata | AFDDPMetadata]:
    """Create DP metadata for each ubatch.

    For DP=1 we use the plugin-owned metadata object to stay independent of
    vLLM internals. For DP>1 we delegate to vLLM's native ``DPMetadata.make``.
    """

    parallel_config = vllm_config.parallel_config
    dp_size = int(parallel_config.data_parallel_size)
    if dp_size <= 1:
        return [
            AFDDPMetadata(
                num_tokens_across_dp_cpu=torch.tensor(
                    [ubatch_slice.num_tokens],
                    dtype=torch.int32,
                    device="cpu",
                ),
                max_tokens_across_dp_cpu=torch.tensor(
                    [ubatch_slice.num_tokens],
                    dtype=torch.int32,
                    device="cpu",
                ),
            )
            for ubatch_slice in ubatch_slices
        ]

    ubatch_dp_metadata = []
    for ubatch_slice in ubatch_slices:
        num_tokens_across_dp_cpu = torch.tensor(
            [ubatch_slice.num_tokens] * dp_size,
            device="cpu",
            dtype=torch.int32,
        )
        ubatch_dp_metadata.append(
            DPMetadata.make(
                parallel_config,
                ubatch_slice.num_tokens,
                num_tokens_across_dp_cpu,
            ),
        )
    return ubatch_dp_metadata


__all__ = [
    "AFDMetadataProviderMixin",
    "build_ubatch_dp_metadata_list",
]
