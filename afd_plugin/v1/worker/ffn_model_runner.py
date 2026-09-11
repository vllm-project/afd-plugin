# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for AFD GPU execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config import update_config as update_vllm_config
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import DPMetadata, get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.workspace import reset_workspace_manager

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.connectors.gpu.async_gpu import (
    ConnectorShutdown,
    GpuAsyncTransferState,
)
from afd_plugin.connectors.metadata import AFDF2ATransferPayload
from afd_plugin.v1.worker.attention_model_runner import (
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.cuda_graph import (
    MAX_REPLAY_PADDING_RATIO,
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
    pad_counts_to_shape,
    padded_ffn_graph_buckets,
    padded_ffn_graph_shape,
    select_padded_ffn_bucket,
    shared_rows_for_bucket,
    validate_cuda_graph_mode,
)
from afd_plugin.v1.worker.ffn_metadata import (
    aggregate_ffn_token_counts,
    project_ffn_token_counts_to_dp,
)

# Name the logger inside vLLM's tree so its handler picks the lines up; a bare
# afd_plugin.* logger propagates to a handler-less root and is dropped.
logger = init_logger(f"vllm.{__name__}")

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


# How often the padded-graph runner reports its padding overhead. One line per
# this many replays keeps a 43-layer step from writing a line per layer.
_PADDED_STATS_EVERY = 2000


@dataclass(slots=True)
class _PaddedFFNGraph:
    """One MoE layer's experts, captured at the padded shape."""

    graph: torch.cuda.CUDAGraph
    routed_out: torch.Tensor
    shared_out: torch.Tensor | None


class GPUFFNModelRunner(LoRAModelRunnerMixin):
    """FFN model runner for AFD GPU execution.

    FFN steps are driven by the connector rather than the vLLM scheduler, in one
    of two ways. Control-plane connectors receive broadcast DP metadata and then
    walk every layer in lockstep with the Attention side. Connectors without a
    control plane (``control_plane is None``) instead pull one work item at a
    time from their receive loop, learning the layer and token counts from the
    arriving payload; see ``execute_connector_driven_step``.
    """

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: VllmConfig, device: object) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.load_config = vllm_config.load_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.afd_config = self.parse_config(vllm_config)
        fail_if_unsupported_ubatching(vllm_config)
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        # A connector without a control plane drives FFN steps from its own
        # receive loop instead of from broadcast DP metadata.
        self.is_connector_driven = self.connector.control_plane is None
        # The connector-driven path captures its own padded graphs and never
        # touches vLLM's graph machinery, so vLLM's cudagraph_mode says nothing
        # about it -- running the policy gate here would reject modes that are
        # simply irrelevant. The only question is whether graphs are wanted.
        if self.is_connector_driven:
            self.afd_cudagraph_policy = None
        else:
            self.afd_cudagraph_policy = validate_cuda_graph_mode(
                vllm_config,
                role="ffn",
            )

        self.model: Any = None
        self.model_memory_usage = 0
        self.num_layers = int(self.model_config.hf_text_config.num_hidden_layers)
        self.use_cuda_graph = (
            not bool(self.model_config.enforce_eager)
            if self.is_connector_driven
            else bool(
                self.afd_cudagraph_policy is not None
                and self.afd_cudagraph_policy.enable_ffn_graph_cache
            )
        )
        self._cuda_graphs: dict[tuple, dict[str, Any]] = {}
        self._graph_memory_pool: Any | None = None
        # Connector-driven padded graphs, one per MoE layer. Empty unless the
        # async connector runs with graphs on; see capture_padded_ffn_graphs.
        self._padded_graphs: dict[tuple[int, int], _PaddedFFNGraph] = {}
        self._padded_buckets: tuple[int, ...] = ()
        # Rows a replay really carried vs rows it was charged, so a run reports
        # how much of the FFN's GPU time went to padding. Host-side ints; the
        # ladder is only worth tuning against a measured distribution.
        self._padded_rows_real = 0
        self._padded_rows_charged = 0
        self._padded_replays = 0
        self._padded_eager_items = 0
        self._padded_hidden: torch.Tensor | None = None
        self._padded_counts: torch.Tensor | None = None
        self._padded_shared: torch.Tensor | None = None
        self._padded_max_routed = 0
        self._padded_max_shared = 0
        self.prof = create_afd_gpu_profiler("ffn")

    @property
    def _control_plane(self) -> Any:
        """The control plane, on the paths that only run when there is one."""
        control_plane = self.connector.control_plane
        assert control_plane is not None, (
            "control-plane FFN path reached on a connector-driven runner",
        )
        return control_plane

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def get_model(self) -> Any:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load the vLLM model."""

        model_loader = get_model_loader(self.load_config)
        with DeviceMemoryProfiler() as profiler:
            if self.model is None:
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config,
                )
            else:
                model_loader.load_weights(
                    self.model,
                    model_config=self.model_config,
                )
        self.model_memory_usage = profiler.consumed_memory

    def profile_run(self) -> None:
        pass

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None

    def execute_model(
        self,
        scheduler_output: SchedulerOutput | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
    ) -> None:
        step_afd_gpu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner requires dp_metadata_list")
        graph_key = make_ffn_graph_key(dp_metadata_list)
        cuda_graph_info = self._cuda_graphs.get(graph_key)
        run_mode = graph_run_mode(
            is_warmup=is_warmup,
            is_graph_capturing=is_graph_capturing,
            is_graph_replaying=is_graph_replaying,
            graph_enabled=bool(self.use_cuda_graph),
            graph_exists=cuda_graph_info is not None,
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            assert cuda_graph_info is not None
            cuda_graph_info["graph"].replay()
            return None

        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        return None

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        update_connector_state: bool = True,
    ) -> torch.Tensor | None:
        if update_connector_state:
            self._control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                    is_warmup=is_warmup,
                ),
            )

        rank_ffn_output = None
        num_layers = max(int(self.num_layers or 0), 1)
        experts_layer_indices = frozenset(
            self.model.get_experts_layer_indices(),
        )
        layer_indices = (
            tuple(sorted(experts_layer_indices))
            if self.afd_config.compute_gate_on_attention
            else tuple(range(num_layers))
        )
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        ffn_dp_metadata_list = {
            stage_idx: self._make_ffn_dp_metadata(dp_metadata_list[stage_idx])
            for stage_idx in stage_ids
        }
        recv_input_ids = getattr(self.model, "afd_requires_input_ids", False)
        with _ffn_forward_context(self.vllm_config) as forward_context:
            for layer_idx in layer_indices:
                uses_remote_experts = layer_idx in experts_layer_indices
                routing_spec = (
                    self.model.get_experts_routing_spec(layer_idx)
                    if uses_remote_experts and self.afd_config.compute_gate_on_attention
                    else None
                )
                for stage_idx in stage_ids:
                    recv_kwargs: dict[str, Any] = {}
                    if routing_spec is not None:
                        recv_kwargs["routing_spec"] = routing_spec
                    if recv_input_ids:
                        recv_kwargs["recv_input_ids"] = True
                    payload = self.connector.recv_attn_output(
                        ubatch_idx=stage_idx,
                        **recv_kwargs,
                    )
                    hidden_states = payload.hidden_states
                    context = payload.context
                    metadata = context.metadata
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = ffn_dp_metadata_list[
                            metadata.stage_idx
                        ]
                        forward_context.additional_kwargs["afd_metadata"] = metadata
                        _set_moe_layer_index(forward_context, layer_idx)
                    if (
                        uses_remote_experts
                        and self.afd_config.compute_gate_on_attention
                    ):
                        router_logits = payload.router_logits
                        assert router_logits is not None
                        rank_ffn_output = self.model.compute_experts_output(
                            hidden_states,
                            layer_idx,
                            router_logits,
                        )
                    else:
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states,
                            layer_idx,
                            input_ids=payload.input_ids,
                        )
                    self.connector.send_ffn_output(rank_ffn_output, context)
        return rank_ffn_output

    def execute_connector_driven_step(self) -> None:
        """Drain whatever the connector has already received, then return.

        Returning on an idle poll rather than blocking forever is what lets the
        worker loop observe its shutdown event. The batch size below is only a
        drain granularity: successive work items may belong to different layers
        of different Attention replicas.
        """
        step_afd_gpu_profiler(self.prof)
        self._ffn_forward_connector_driven()

    # ==================================================================
    # Connector-driven padded CUDA graphs
    # ==================================================================

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
        if not self.use_cuda_graph or not self.is_connector_driven:
            return 0
        if self._padded_graphs:
            # start_ffn_server_loop is callable more than once.
            return 0
        if self.model is None:
            raise RuntimeError("capture_padded_ffn_graphs needs a loaded model")

        connector: Any = self.connector
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

        set_cudagraph_capturing_enabled(True)
        try:
            with (
                _ffn_forward_context(self.vllm_config) as forward_context,
                graph_capture(device=self.device),
            ):
                for layer_idx in moe_layers:
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
        finally:
            set_cudagraph_capturing_enabled(False)

        graph_bytes = start_free_gpu_memory - torch.cuda.mem_get_info()[0]
        logger.info(
            "AFD FFN padded graphs ready: graphs=%d layers=%d buckets=%s "
            "shared_rows=%d experts=%d size=%.1fMiB",
            len(self._padded_graphs),
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
        return self.model.compute_ffn_output(
            hidden_states=self._padded_hidden[:bucket],
            layer_idx=layer_idx,
            group_list=self._padded_counts,
            expand_x_shared=shared,
        )

    def _compute_work_item(
        self,
        work_item: Any,
        states: GpuAsyncTransferState,
    ) -> torch.Tensor | AFDF2ATransferPayload:
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
            self._padded_graphs.get((work_item.layer_idx, bucket))
            if bucket is not None
            else None
        )
        fits = (
            graph is not None
            and 0 < routed_rows <= self._padded_max_routed
            and shared_rows <= self._padded_max_shared
        )
        if not fits:
            self._padded_eager_items += 1
            return self.model.compute_ffn_output(
                hidden_states=work_item.hidden_states,
                layer_idx=work_item.layer_idx,
                group_list=states.group_list,
                expand_x_shared=states.expand_x_shared,
            )

        assert graph is not None and bucket is not None
        self._padded_rows_real += routed_rows
        self._padded_rows_charged += bucket
        self._padded_replays += 1
        if self._padded_replays % _PADDED_STATS_EVERY == 0:
            logger.info(
                "AFD FFN padded replays=%d eager=%d rows_real=%d rows_charged=%d "
                "padding_overhead=%.2fx",
                self._padded_replays,
                self._padded_eager_items,
                self._padded_rows_real,
                self._padded_rows_charged,
                self._padded_rows_charged / max(self._padded_rows_real, 1),
            )

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

    def _ffn_forward_connector_driven(
        self,
    ) -> torch.Tensor | AFDF2ATransferPayload | None:
        stage_idx = 0
        rank_ffn_output = None
        connector = self.connector
        max_items = max(1, int(self.num_layers))

        with _ffn_forward_context(self.vllm_config) as forward_context:
            for _ in range(max_items):
                try:
                    work_item = connector.recv_ffn_work_item(  # type: ignore[attr-defined]
                        stage_idx=stage_idx,
                        max_num_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,
                        # Hand the graph its input buffer so the arrival's
                        # gather lands there directly. The gather had to write
                        # somewhere either way; staging afterwards would be a
                        # second pass over the whole payload, per layer.
                        routed_out=self._padded_hidden,
                    )
                except TimeoutError:
                    # Nothing pending; hand control back so the worker loop can
                    # check for shutdown.
                    return rank_ffn_output
                except ConnectorShutdown:
                    raise

                states = work_item.context.states
                if not isinstance(states, GpuAsyncTransferState):
                    raise RuntimeError(
                        "async GPU FFN work item requires GpuAsyncTransferState",
                    )
                metadata = work_item.context.metadata
                forward_context.dp_metadata = None
                forward_context.additional_kwargs["afd_metadata"] = metadata
                _set_moe_layer_index(forward_context, work_item.layer_idx)

                rank_ffn_output = self._compute_work_item(work_item, states)
                rank_ffn_output = connector.send_ffn_work_item_output(  # type: ignore[attr-defined]
                    work_item,
                    rank_ffn_output,
                )
        return rank_ffn_output

    def _execute_eager_mode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            return self.model.compute_ffn_output(hidden_states, layer_idx)
        return self.model.compute_ffn_output(
            hidden_states,
            layer_idx,
            input_ids=input_ids,
        )

    def _make_ffn_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata,
    ) -> AFDDPMetadata:
        attention_counts = tuple(
            int(count) for count in dp_metadata.num_tokens_across_dp_cpu
        )
        ffn_counts = aggregate_ffn_token_counts(
            attention_counts,
            attention_size=int(self.connector.attn_size),
            ffn_size=int(self.connector.ffn_size),
        )
        dp_counts = project_ffn_token_counts_to_dp(
            ffn_counts,
            dp_size=int(self.vllm_config.parallel_config.data_parallel_size),
        )
        return AFDDPMetadata(num_tokens_across_dp_cpu=dp_counts)

    def update_config(self, overrides: dict[str, Any]) -> None:
        for config_name, config_overrides in overrides.items():
            config = getattr(self, config_name)
            updated_config = update_vllm_config(config, config_overrides)
            setattr(self, config_name, updated_config)

    def reload_weights(self) -> None:
        if self.model is None:
            raise RuntimeError("Cannot reload weights before model is loaded")
        self.load_model()

    def _dummy_run(
        self,
        cudagraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_attn_graph_capturing: bool,
    ) -> None:
        mode_name = getattr(cudagraph_runtime_mode, "name", str(cudagraph_runtime_mode))
        if mode_name.endswith(".FULL"):
            mode_name = "FULL"

        if mode_name == "FULL":
            if self._graph_memory_pool is None:
                self._graph_memory_pool = torch.cuda.graph_pool_handle()
            graph_key = make_ffn_graph_key(dp_metadata_list)
            cudagraph = torch.cuda.CUDAGraph()
            # DP metadata receive/update is a control-plane side effect and must
            # complete before CUDA graph capture starts.
            self._control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                ),
            )
            with torch.cuda.graph(cudagraph, pool=self._graph_memory_pool):
                output = self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    update_connector_state=False,
                )
            self._cuda_graphs[graph_key] = {
                "graph": cudagraph,
                "input_hidden_states": output,
                "output": output,
            }
        else:
            self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_cuda_graph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner.capture_model requires metadata")

        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device=self.device):
                if is_warmup:
                    self._control_plane.update_state_from_dp_metadata(
                        _make_dp_metadata_payload(
                            dp_metadata_list,
                            is_graph_capturing=False,
                            is_warmup=True,
                        ),
                    )
                    self._ffn_forward(
                        dp_metadata_list=dp_metadata_list,
                        is_graph_capturing=False,
                        is_warmup=True,
                        update_connector_state=False,
                    )
                else:
                    self._dummy_run(
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        return int(cuda_graph_size)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        raise RuntimeError("FFN runners do not sample tokens")

    def add_lora(self, lora_request: Any) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    @property
    def lora_config(self) -> None:
        return None

    @property
    def is_pooling_model(self) -> bool:
        return False

    def get_supported_tasks(self) -> tuple[Any, ...]:
        return ()

    # Patch reason: the FFN runner owns GPUModelRunner-equivalent CUDA state
    # without inheriting GPUModelRunner's shutdown implementation.
    # Patch functionality: mirror the pinned native GPU resource cleanup and
    # then close AFD-owned profiler and connector resources.
    # Signature: matches GPUModelRunner.shutdown; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def shutdown(self) -> None:
        # ### PATCH START: release native-equivalent and AFD-owned GPU state.
        stop_afd_gpu_profiler(self.prof)
        try:
            for graph_info in self._cuda_graphs.values():
                graph_info["graph"].reset()
            self._cuda_graphs.clear()
            self._graph_memory_pool = None
            self.vllm_config.compilation_config.static_forward_context.clear()
            self.model = None
            _ROPE_DICT.clear()
            reset_workspace_manager()
        finally:
            self.connector.close()
        # ### PATCH END: release native-equivalent and AFD-owned GPU state.


def _resolve_world_ranks() -> tuple[int, int]:
    group = get_world_group()
    return int(group.rank), int(group.local_rank)


@contextmanager
def _ffn_forward_context(vllm_config: VllmConfig):
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        yield get_forward_context()


def _set_moe_layer_index(forward_context: Any, layer_idx: int) -> None:
    all_moe_layers = forward_context.all_moe_layers
    if not all_moe_layers:
        return

    target = f".layers.{int(layer_idx)}."
    for idx, layer_name in enumerate(all_moe_layers):
        if target in f".{layer_name}.":
            forward_context.moe_layer_index = idx
            return


def _make_dp_metadata_payload(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    *,
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
) -> AFDControlPayload:
    return AFDControlPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=is_graph_capturing,
        is_warmup=is_warmup,
    )


__all__ = ["GPUFFNModelRunner"]
