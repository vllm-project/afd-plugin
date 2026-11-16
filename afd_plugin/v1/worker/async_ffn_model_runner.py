# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector-driven GPU FFN execution with local bucketed expert graphs.

The wire protocol and bucket policy follow spike/AsyncGPUConnector at 788d17d.
Native ModelRunnerV2 remains the Attention execution owner. This runner owns
only local expert compute and never joins Attention DP coordination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import graph_capture

from afd_plugin.compat.profiler import step_afd_gpu_profiler
from afd_plugin.connectors import AFDA2FTransferPayload, AFDF2ATransferPayload
from afd_plugin.connectors.gpu.async_gpu import (
    ConnectorShutdown,
    GpuAsyncAFDConnector,
    GpuAsyncTransferState,
)
from afd_plugin.v1.worker.cuda_graph import (
    MAX_REPLAY_PADDING_RATIO,
    pad_counts_to_shape,
    padded_ffn_graph_buckets,
    padded_ffn_graph_layer_fits,
    padded_ffn_graph_shape,
    resolve_padded_ffn_graph_reserve_bytes,
    select_padded_ffn_bucket,
    shared_rows_for_bucket,
)
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index
from afd_plugin.v1.worker.ffn_model_runner import (
    GPUFFNModelRunner,
    _ffn_forward_context,
)


@dataclass(slots=True)
class _PaddedFFNGraph:
    graph: torch.cuda.CUDAGraph
    routed_out: torch.Tensor
    shared_out: torch.Tensor | None


class GPUAsyncFFNModelRunner(GPUFFNModelRunner):
    """Run each arrived layer independently of other Attention producers."""

    connector: GpuAsyncAFDConnector

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.session = self.connector.ffn_session()
        self._first_step = True
        self._padded_graphs: dict[tuple[int, int], _PaddedFFNGraph] = {}
        self._padded_buckets: tuple[int, ...] = ()
        self._padded_hidden: torch.Tensor | None = None
        self._padded_counts: torch.Tensor | None = None
        self._padded_shared: torch.Tensor | None = None
        self._padded_max_routed = 0
        self._padded_max_shared = 0

    @torch.inference_mode()
    def execute_connector_driven_step(self, shutdown_requested: bool = False) -> bool:
        if not self.session.keep_serving(shutdown_requested):
            return False
        if self._first_step:
            self.session.before_first_step()
            self._first_step = False
        step_afd_gpu_profiler(self.prof)
        try:
            self._run_one_transaction()
        except ConnectorShutdown:
            return False
        self.session.after_step()
        return True

    def drain(self) -> None:
        self.session.drain(self._run_one_transaction)

    def _run_one_transaction(self) -> None:
        try:
            payload = self.connector.recv_attn_output(
                routed_out=self._padded_hidden,
                timeout_ms=self.connector.extra_info.recv_poll_timeout_ms,
            )
        except TimeoutError:
            return
        states = payload.context.states
        if not isinstance(states, GpuAsyncTransferState):
            raise RuntimeError("Async GPU FFN requires GpuAsyncTransferState")
        with _ffn_forward_context(self.vllm_config) as context:
            context.dp_metadata = None
            context.additional_kwargs["afd_metadata"] = payload.context.metadata
            _set_moe_layer_index(context, payload.context.metadata.layer_idx)
            result = self._compute_work_item(payload, states)
            self.connector.send_ffn_output(
                result.routed_output,
                payload.context,
                shared_output=result.shared_output,
            )

    def capture_padded_ffn_graphs(self) -> int:
        """Capture the local experts once per MoE layer, at the maximum shape.

        The connector-driven path has no control plane, so nothing tells this
        rank the shape of the next work item -- which is why it ran eagerly.
        Padding removes the need to know: a grouped GEMM takes its grouping
        from a device-side count vector rather than from its row count, so the
        largest shape can be captured and every item padded up to it, with the
        padding charged to the last expert and its output sliced off. Only the
        counts differ between replays, and a replay re-reads them.

        The shape is the largest batch the sender can produce,
        ``max_num_batched_tokens``, so prefill and decode both replay it.

        One graph per layer, because a graph records the weight pointers and
        each layer has its own. Input buffers are shared across layers -- work
        items are served one at a time on one stream.

        Called once, before the connector joins its process group. That
        rendezvous is the only barrier between the roles, and the Attention
        rank profiles -- and so dispatches -- the moment it clears; capturing
        after joining would leave those dispatches landing in a slot nobody is
        polling yet.

        Returns:
            Bytes of device memory the graphs took.
        """
        if not self.use_cuda_graph:
            return 0
        if self._padded_graphs:
            # start_ffn_server_loop is callable more than once.
            return 0
        if self.model is None:
            raise RuntimeError("capture_padded_ffn_graphs needs a loaded model")

        connector = self.connector
        device = torch.device(self.device)
        # This runs on whichever thread called start_ffn_server_loop, which is
        # not the serving thread and has no device bound yet.
        torch.cuda.set_device(device)

        self._padded_max_routed, self._padded_max_shared = padded_ffn_graph_shape(
            num_tokens=int(self.vllm_config.scheduler_config.max_num_batched_tokens),
            topk=connector.topk,
            ffn_size=connector.ffn_size,
            has_shared_experts=connector.has_shared_experts,
        )
        self._padded_hidden = torch.zeros(
            (self._padded_max_routed, connector.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        self._padded_counts = torch.zeros(
            connector.expert_per_rank,
            dtype=torch.int32,
            device=device,
        )
        self._padded_shared = (
            torch.zeros(
                (self._padded_max_shared, connector.hidden_size),
                dtype=self.dtype,
                device=device,
            )
            if self._padded_max_shared
            else None
        )
        # A grouping that fills the captured shape. Any grouping records the
        # same kernels; a replay re-reads the counts.
        self._padded_buckets = padded_ffn_graph_buckets(
            self._padded_max_routed,
            ffn_size=connector.ffn_size,
        )

        inner = self.model.model
        # Same source as the forward loop above: the model reports which of its
        # layers own experts. Reading a per-layer flag instead only works for
        # the adapters that happen to define one -- DeepSeek-V4's layers do not.
        experts_layer_indices = frozenset(self.model.get_experts_layer_indices())
        moe_layers = [
            layer_idx
            for layer_idx in range(inner.start_layer, inner.end_layer)
            if layer_idx in experts_layer_indices
        ]
        if not moe_layers:
            raise RuntimeError("Async GPU FFN graph capture requires MoE layers")
        num_warmups = max(
            1,
            int(self.vllm_config.compilation_config.cudagraph_num_of_warmups),
        )

        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        # Pin the shared MoE workspace at its ceiling before capturing anything.
        # It grows by freeing the old buffer and allocating a bigger one, so any
        # growth after a capture leaves that graph reading freed memory -- one
        # run died with an illegal memory access when an oversized item took the
        # eager path and grew it. One eager call at the largest shape any path
        # can ask for settles it, and costs a single forward instead of the
        # captured graph per layer that reserving it through the ladder would.
        with _ffn_forward_context(self.vllm_config) as warmup_context:
            _set_moe_layer_index(warmup_context, moe_layers[0])
            self._fill_padded_counts(self._padded_max_routed)
            self._padded_ffn_compute(moe_layers[0], self._padded_max_routed)
        torch.cuda.synchronize()

        reserve_bytes = resolve_padded_ffn_graph_reserve_bytes()
        per_layer_bytes = 0
        captured_layers = 0
        set_cudagraph_capturing_enabled(True)
        try:
            with (
                _ffn_forward_context(self.vllm_config) as forward_context,
                graph_capture(device=self.device),
            ):
                for layer_idx in moe_layers:
                    free_bytes = torch.cuda.mem_get_info()[0]
                    if not padded_ffn_graph_layer_fits(
                        free_bytes=free_bytes,
                        per_layer_bytes=per_layer_bytes,
                        reserve_bytes=reserve_bytes,
                    ):
                        # Every layer from here runs eagerly: a work item whose
                        # (layer, bucket) has no graph already takes that path.
                        break
                    _set_moe_layer_index(forward_context, layer_idx)
                    # Largest bucket first, and that order is load-bearing:
                    # vLLM's WorkspaceManager grows the shared MoE scratch by
                    # freeing the old buffer and allocating a bigger one, which
                    # leaves every graph already captured against the old
                    # pointer dangling. Capturing ascending therefore made each
                    # small bucket's graph fault on its first replay. Starting
                    # at the largest sizes the workspace once, and every
                    # smaller capture reuses it.
                    for bucket in reversed(self._padded_buckets):
                        # A replay costs its captured row count, so each bucket
                        # gets its own graph and an item takes the smallest one
                        # that holds it.
                        self._fill_padded_counts(bucket)
                        # Warm first: the fused MoE picks a kernel on its first
                        # call, and that choice must settle before capture --
                        # autotuning synchronizes, which capture forbids.
                        for _ in range(num_warmups):
                            self._padded_ffn_compute(layer_idx, bucket)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=self._graph_memory_pool):
                            payload = self._padded_ffn_compute(layer_idx, bucket)
                        self._padded_graphs[(layer_idx, bucket)] = _PaddedFFNGraph(
                            graph=graph,
                            routed_out=payload.routed_output,
                            shared_out=payload.shared_output,
                        )
                    per_layer_bytes = max(
                        per_layer_bytes,
                        free_bytes - torch.cuda.mem_get_info()[0],
                    )
                    captured_layers += 1
        finally:
            set_cudagraph_capturing_enabled(False)

        graph_bytes = start_free_gpu_memory - torch.cuda.mem_get_info()[0]
        logging.getLogger(__name__).info(
            "AFD FFN padded graphs ready: graphs=%d layers=%d/%d buckets=%s "
            "shared_rows=%d experts=%d size=%.1fMiB",
            len(self._padded_graphs),
            captured_layers,
            len(moe_layers),
            list(self._padded_buckets),
            self._padded_max_shared,
            connector.expert_per_rank,
            graph_bytes / 2**20,
        )
        return int(graph_bytes)

    def _fill_padded_counts(self, bucket: int) -> None:
        """Spread ``bucket`` rows over every local expert for capture.

        Not any grouping will do, even though the row total is what sizes the
        grouped GEMM. The fused MoE pads each non-empty expert's rows up to a
        block, so the scratch it reserves grows with the number of experts that
        have rows -- and a capture is stuck with the scratch it reserved. All
        rows on one expert reserves the least, and a replay whose real routing
        touches every expert then runs off the end of it. Spreading rows evenly
        is both the worst case for that padding and what real routing looks
        like, so the capture reserves enough for any replay.
        """
        assert self._padded_counts is not None
        num_experts = int(self._padded_counts.numel())
        share, remainder = divmod(bucket, num_experts)
        self._padded_counts.fill_(share)
        if remainder:
            # One row each rather than all on the last expert, so a bucket
            # smaller than the expert count still touches as many experts as
            # it has rows instead of collapsing onto one.
            self._padded_counts[:remainder] += 1

    def _shared_rows_for(self, bucket: int) -> int:
        """This bucket's share of the shared-expert rows."""
        return shared_rows_for_bucket(
            bucket,
            max_routed=self._padded_max_routed,
            max_shared=self._padded_max_shared,
            ffn_size=self.connector.ffn_size,
        )

    def _padded_ffn_compute(
        self,
        layer_idx: int,
        bucket: int,
    ) -> AFDF2ATransferPayload:
        """Run one layer's experts over the first ``bucket`` padded rows."""
        assert self._padded_hidden is not None
        shared = self._padded_shared
        if shared is not None:
            shared = shared[: self._shared_rows_for(bucket)]
        return self.get_model().compute_ffn_output(
            hidden_states=self._padded_hidden[:bucket],
            layer_idx=layer_idx,
            group_list=self._padded_counts,
            expand_x_shared=shared,
        )

    def _compute_work_item(
        self,
        work_item: AFDA2FTransferPayload,
        states: GpuAsyncTransferState,
    ) -> AFDF2ATransferPayload:
        """Replay this layer's smallest fitting padded graph, or run it eagerly.

        A replay costs its captured row count rather than the item's real one,
        so taking the smallest bucket that holds the item is what keeps a small
        item cheap -- a DBO ubatch is a quarter of a full batch's rows, and
        charging it the full-batch shape is what made ubatching a loss.

        Eager is the fallback for an item that does not fit the captured shape
        and for the empty item a decode can produce when none of a token's
        experts landed here.
        """
        routed_rows = int(states.routed_tokens)
        shared_rows = int(states.shared_tokens)
        bucket = (
            select_padded_ffn_bucket(
                self._padded_buckets,
                routed_rows,
                shared_rows,
                max_routed=self._padded_max_routed,
                max_shared=self._padded_max_shared,
                ffn_size=self.connector.ffn_size,
            )
            if routed_rows > 0
            else None
        )
        if bucket is not None and bucket > routed_rows * MAX_REPLAY_PADDING_RATIO:
            # The padding this replay would carry costs more than the launches
            # eager pays. Take the cheaper of the two per item rather than
            # charging every item to the nearest bucket above it.
            bucket = None
        graph = (
            self._padded_graphs.get((work_item.context.metadata.layer_idx, bucket))
            if bucket is not None
            else None
        )
        fits = (
            graph is not None
            and 0 < routed_rows <= self._padded_max_routed
            and shared_rows <= self._padded_max_shared
        )
        if not fits:
            return self.get_model().compute_ffn_output(
                hidden_states=work_item.hidden_states,
                layer_idx=work_item.context.metadata.layer_idx,
                group_list=states.group_list,
                expand_x_shared=states.expand_x_shared,
            )

        assert graph is not None and bucket is not None
        assert self._padded_hidden is not None
        assert self._padded_counts is not None
        if not states.staged_routed:
            # The receive could not use the buffer -- no graph existed when the
            # item arrived, or it did not fit -- so the rows still need moving.
            self._padded_hidden[:routed_rows].copy_(work_item.hidden_states)
        self._padded_counts.copy_(states.group_list)
        pad_counts_to_shape(
            self._padded_counts,
            padded_rows=bucket,
            actual_rows=routed_rows,
        )
        if self._padded_shared is not None and shared_rows:
            self._padded_shared[:shared_rows].copy_(states.expand_x_shared)
        graph.graph.replay()
        # Views into the graph's own output buffers. The next replay overwrites
        # them, and the reply that consumes them is queued before it on this
        # stream, so the ordering holds without a copy.
        return AFDF2ATransferPayload(
            routed_output=graph.routed_out[:routed_rows],
            shared_output=(
                graph.shared_out[:shared_rows]
                if graph.shared_out is not None and shared_rows
                else None
            ),
        )

    def shutdown(self) -> None:
        # A live producer can still write any symmetric slot or queue work.
        self.session.validate_shutdown()
        for capture in self._padded_graphs.values():
            capture.graph.reset()
        self._padded_graphs.clear()
        self._padded_hidden = None
        self._padded_counts = None
        self._padded_shared = None
        super().shutdown()
