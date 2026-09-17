# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for AFD GPU execution."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, PerLayerAttnMetadata
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDForwardContextMetadata,
)
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.v1.worker.attention_metadata import (
    AFDMetadataProviderMixin,
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.cuda_graph import validate_cuda_graph_mode
from afd_plugin.v1.worker.ubatch_wrapper import AFDUBatchWrapper

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


@contextmanager
def _dp_batch_coordination_disabled(disabled: bool):
    """Skip vLLM's cross-DP batch agreement for connector-driven runs.

    ``GPUModelRunner._determine_batch_execution_and_padding`` all-reduces the
    batch shape across the DP group whenever ``data_parallel_size > 1``. Async
    AFD deliberately lets each Attention replica advance on its own, so an idle
    replica never joins that collective and a busy one blocks in it forever --
    which is where a 2A2F run hangs before it reaches the first MoE layer.

    Returning the single-rank answer (``num_tokens_across_dp=None``) makes the
    upstream function skip its DP-padding branch entirely, exactly as it does
    for ``data_parallel_size == 1``.
    """
    if not disabled:
        yield
        return

    original = gpu_model_runner.coordinate_batch_across_dp

    def _single_rank_coordination(*_args: Any, cudagraph_mode: int, **_kwargs: Any):
        return False, None, cudagraph_mode

    gpu_model_runner.coordinate_batch_across_dp = _single_rank_coordination
    try:
        yield
    finally:
        gpu_model_runner.coordinate_batch_across_dp = original


class AFDAttentionModelRunner(AFDMetadataProviderMixin, GPUModelRunner):
    """Attention model runner that injects AFD metadata into forward context."""

    afd_expected_role = "attention"

    #: Declared, not assigned: the ubatch-wrapper install both reads and
    #: rebinds it, which leaves its type unresolvable from the base class.
    model: Any

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        self.afd_config = self.parse_config(self.vllm_config)
        fail_if_unsupported_ubatching(self.vllm_config)
        self.afd_cudagraph_policy = validate_cuda_graph_mode(
            self.vllm_config,
            role="attention",
        )
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            self.vllm_config,
            self.afd_config,
        )
        # The connector rendezvous is deferred to the end of ``load_model()``
        # so Attention and FFN weight loading overlap; see that method. The
        # async GPU connector drives FFN work from its own receive loop and so
        # has no control plane, which is why there is no assertion here.
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_is_graph_replaying = False
        self._afd_pending_metadata: AFDForwardContextMetadata | None = None
        self._afd_suppress_metadata_send = False
        self._afd_transaction_counter = 0
        self.prof = create_afd_gpu_profiler("attention")

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: upstream load_model has no AFD connector lifecycle.
    # Patch functionality: after native weight loading and any AFD ubatch
    # wrapper installation, collectively initialize the AFD connector so
    # Attention and FFN model loading overlap across roles. Connector
    # initialization is skipped when it already succeeded (the FFN side uses
    # the same guard), and the rendezvous completes before memory profiling or
    # the first model forward. The connector itself is idempotent.
    # Signature: matches upstream; no added parameters.
    def load_model(self, load_dummy_weights: bool = False) -> None:
        use_ubatching = bool(self.vllm_config.parallel_config.use_ubatching)
        with _use_afd_ubatch_wrapper_during_load(use_ubatching):
            super().load_model(load_dummy_weights)
        if use_ubatching:
            self._install_afd_ubatch_wrapper()
        # Wrapper installation is local and non-blocking; the connector
        # rendezvous is the blocking cross-role collective, so it is
        # deliberately last: it doubles as the "both roles finished loading
        # weights" barrier before memory profiling.
        if not self.connector.is_initialized:
            self.connector.init_afd_connector()

    def _install_afd_ubatch_wrapper(self) -> None:
        model: Any = self.model
        if not isinstance(model, AFDUBatchWrapper):
            if isinstance(model, UBatchWrapper):
                model = model.unwrap()
            model = AFDUBatchWrapper(
                model,
                self.vllm_config,
                CUDAGraphMode.NONE,
                self.device,
            )
            self.model = model
        model.configure_afd_context_provider(
            self.install_afd_metadata_on_forward_context,
        )

    # Patch reason: AFD stages connector metadata before native Attention
    # metadata construction. In addition, vLLM v0.26 caches Attention metadata
    # without the ubatch id, so update-capable backends can reuse ubatch 0
    # sequence metadata for later ubatches.
    # Patch functionality: stage the AFD metadata and disable block-table-only
    # metadata updates while native multi-ubatch metadata is built. Remove the
    # cache workaround after vLLM PR #48659 is included in the pinned release.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    # Delegation exception: the upstream function is large; this override wraps
    # the native implementation and scopes the workaround to its metadata-build
    # window.
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        # ### PATCH START: stage AFD metadata and avoid cross-ubatch cache reuse.
        self._afd_pending_metadata = self.build_afd_metadata(
            ubatch_slices,
            int(num_tokens),
        )
        disabled_metadata_builders: dict[int, AttentionMetadataBuilder] = {}
        try:
            if ubatch_slices is not None and len(ubatch_slices) > 1:
                for kv_cache_group in self.attn_groups:
                    for attention_group in kv_cache_group:
                        for ubatch_idx in range(len(ubatch_slices)):
                            builder = attention_group.get_metadata_builder(ubatch_idx)
                            if builder.supports_update_block_table:
                                disabled_metadata_builders[id(builder)] = builder
                                builder.supports_update_block_table = False
            return super()._build_attention_metadata(
                num_tokens,
                num_reqs,
                max_query_len,
                num_tokens_padded,
                num_reqs_padded,
                ubatch_slices,
                logits_indices,
                use_spec_decode,
                for_cudagraph_capture,
                num_scheduled_tokens,
                cascade_attn_prefix_lens,
                slot_mappings,
            )
        finally:
            for builder in disabled_metadata_builders.values():
                builder.supports_update_block_table = True
        # ### PATCH END: stage AFD metadata and avoid cross-ubatch cache reuse.

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        with _dp_batch_coordination_disabled(
            self.connector.control_plane is None,
        ):
            (
                cudagraph_mode,
                batch_descriptor,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = super()._determine_batch_execution_and_padding(
                num_tokens,
                num_reqs,
                num_scheduled_tokens_np,
                max_num_scheduled_tokens,
                use_cascade_attn,
                allow_microbatching,
                force_eager,
                force_uniform_decode,
                force_has_lora,
                force_num_active_loras,
                num_encoder_reqs,
            )
        self._afd_is_graph_replaying = (
            not bool(getattr(self, "_is_warmup", False))
            and not bool(getattr(self, "_afd_is_graph_capturing", False))
            and cudagraph_mode == CUDAGraphMode.FULL
        )

        args = (
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            allow_microbatching,
            force_eager,
            force_uniform_decode,
            force_has_lora,
            force_num_active_loras,
            num_encoder_reqs,
        )
        kwargs: dict[str, Any] = {}

        # determine if ubatch should be activated.
        # 1. For dp = 1, vLLM hardcodes `should_ubatch=False`.
        # This is the extra support for dp = 1
        if self.vllm_config.parallel_config.data_parallel_size == 1:
            should_ubatch = self._should_ubatch_single_rank(
                batch_descriptor,
                args,
                kwargs,
            )

        # 2. For dp > 1, vLLM's coordinated decision (_post_process_ubatch)
        # only aborts when the last ubatch is empty. This ensures the first
        # ubatch is not empty
        elif should_ubatch:
            values = _batch_execution_values(args, kwargs)
            num_ubatches = self.vllm_config.parallel_config.num_ubatches
            num_tokens = int(values.get("num_tokens", 0))
            should_ubatch = num_tokens >= max(num_ubatches, 1)

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _should_ubatch_single_rank(
        self,
        batch_descriptor: BatchDescriptor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        """Rank-local replica of vLLM's DP-coordinated ubatch decision.

        Mirrors ``coordinate_batch_across_dp`` + ``_post_process_ubatch`` for
        a single rank: preconditions, then thresholds, then the empty-ubatch
        guard. The guard matters because ``split_attn_metadata`` splits on the
        padded token count (``batch_descriptor.num_tokens``, padded up to the
        cudagraph capture size for a FULL_DECODE graph) while the attention
        metadata only covers the real requests; if the last split point lands
        at/past the real token count the trailing ubatch starts outside every
        real request and vLLM asserts "Token slice start outside of first
        request" (e.g. 2 real tokens padded to capture size 64, split at 32).
        """
        parallel_config = self.vllm_config.parallel_config
        if not bool(parallel_config.use_ubatching):
            return False
        values = _batch_execution_values(args, kwargs)
        if not bool(values.get("allow_microbatching", True)):
            return False
        num_tokens = values["num_tokens"]
        num_ubatches = parallel_config.num_ubatches
        # Not covered by is_last_ubatch_empty: with no cudagraph padding,
        # fewer tokens than ubatches empties the *first* ubatch instead.
        if num_tokens < max(num_ubatches, 1):
            return False
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=values["max_num_scheduled_tokens"],
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=values["num_reqs"],
            force_uniform_decode=values.get("force_uniform_decode"),
        )
        if not check_ubatch_thresholds(
            parallel_config,
            num_tokens,
            bool(uniform_decode),
        ):
            return False
        padded_tokens = batch_descriptor.num_tokens
        return not is_last_ubatch_empty(num_tokens, padded_tokens, num_ubatches)

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        self.install_afd_metadata_on_forward_context(forward_context)
        return super()._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        step_afd_gpu_profiler(self.prof)
        previous_is_graph_replaying = getattr(self, "_afd_is_graph_replaying", False)
        self._afd_is_graph_replaying = False
        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            self._afd_is_graph_replaying = previous_is_graph_replaying

    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vLLM's DP dummy batch through the AFD model path.

        vLLM uses ``execute_dummy_batch`` on idle DP ranks while another DP rank
        is serving a request. The native dummy path calls the model directly,
        bypassing ``_model_forward()``, so we provide AFD metadata lazily when
        the plugin-owned model reads the current forward context. Do not force
        native attention metadata here: profiling dummy runs can happen before
        vLLM initializes ``kv_cache_config``.
        """

        previous_metadata = self._afd_pending_metadata
        previous_is_graph_capturing = getattr(
            self,
            "_afd_is_graph_capturing",
            False,
        )
        self._afd_is_graph_capturing = is_graph_capturing
        try:
            with use_afd_metadata_provider(
                self.install_afd_metadata_on_forward_context,
            ):
                return super()._dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode,
                    force_attention,
                    uniform_decode,
                    allow_microbatching,
                    skip_eplb,
                    is_profile,
                    create_mixed_batch,
                    remove_lora,
                    is_graph_capturing,
                    num_active_loras,
                    profile_seq_lens,
                )
        finally:
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_pending_metadata = previous_metadata

    # Patch reason: native capture does not publish AFD warmup/capture metadata.
    # Patch functionality: preserve the upstream warmup/capture flow while
    # publishing replayable connector state before formal graph capture.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
        profiler: AbstractContextManager[Any] | None = None,
    ):
        """Mirror vLLM warmup/capture while marking AFD warmup metadata.

        The native implementation calls ``self._dummy_run`` for warmups and
        formal capture. We keep that flow intact and only set ``_is_warmup``
        around the warmup calls so FFN ranks can distinguish warmup metadata
        from graph-capture metadata.
        """

        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        # ### PATCH START: expose warmup state to the AFD control plane.
        previous_is_warmup = bool(self._is_warmup)
        try:
            self._is_warmup = True
            for _ in range(int(num_warmups)):
                self._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=desc.uniform,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    num_active_loras=desc.num_active_loras,
                    profile_seq_lens=profile_seq_lens,
                )
        finally:
            self._is_warmup = previous_is_warmup
        # ### PATCH END: expose warmup state to the AFD control plane.

        # ### PATCH START: publish static AFD state before graph capture.
        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_graph_capturing = self._afd_is_graph_capturing
        try:
            # DP metadata transfer is a control-plane side effect.  The original
            # AFD path sends it before formal CUDA graph capture so the capture
            # contains only replayable model/data-plane work.
            self._afd_is_graph_capturing = True
            if allow_microbatching:
                # The AFD-aware ubatch wrapper builds the exact padded ubatch
                # slices used by vLLM and sends per-stage DP metadata before it
                # enters torch.cuda.graph(...).  Avoid sending a single-stage
                # capture payload here.
                self._afd_pending_metadata = None
                self._afd_suppress_metadata_send = False
            else:
                self._afd_pending_metadata = self.build_afd_metadata(
                    None,
                    int(desc.num_tokens),
                )
                self.send_dp_metadata(
                    self.build_capture_dp_metadata(int(desc.num_tokens)),
                    None,
                )
                self._afd_suppress_metadata_send = True
            with (
                profiler,
                torch.profiler.record_function(
                    f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}"
                ),
            ):
                self._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=desc.uniform,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    num_active_loras=desc.num_active_loras,
                    is_graph_capturing=True,
                    profile_seq_lens=profile_seq_lens,
                )
        finally:
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_suppress_metadata_send = previous_suppress_send
            self._afd_pending_metadata = previous_metadata
        # ### PATCH END: publish static AFD state before graph capture.

    # Patch reason: AFD owns an additional profiler and connector lifecycle.
    # Patch functionality: preserve native GPUModelRunner cleanup, then close
    # AFD-owned resources even when native cleanup raises.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def shutdown(self) -> None:
        # ### PATCH START: extend native shutdown with AFD resource cleanup.
        stop_afd_gpu_profiler(self.prof)
        try:
            super().shutdown()
        finally:
            self.connector.close()
        # ### PATCH END: extend native shutdown with AFD resource cleanup.


def fail_if_unsupported_ubatching(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    num_ubatches = int(parallel_config.num_ubatches)
    if bool(vllm_config.parallel_config.use_ubatching) and num_ubatches != 2:
        raise RuntimeError(
            "AFD ubatching currently supports exactly two ubatches; "
            f"got num_ubatches={num_ubatches}",
        )


fail_if_ubatching_enabled = fail_if_unsupported_ubatching


def fail_if_cuda_graph_enabled(vllm_config: VllmConfig) -> None:
    validate_cuda_graph_mode(vllm_config)


def _batch_execution_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    names = [
        "num_tokens",
        "num_reqs",
        "num_scheduled_tokens_np",
        "max_num_scheduled_tokens",
        "use_cascade_attn",
        "allow_microbatching",
        "force_eager",
        "force_uniform_decode",
        "force_has_lora",
        "force_num_active_loras",
        "num_encoder_reqs",
    ]
    values = dict(zip(names, args, strict=False))
    values.update(kwargs)
    return values


@contextmanager
def _use_afd_ubatch_wrapper_during_load(enabled: bool):
    if not enabled:
        yield
        return

    original = gpu_model_runner.UBatchWrapper
    gpu_model_runner.UBatchWrapper = AFDUBatchWrapper
    try:
        yield
    finally:
        gpu_model_runner.UBatchWrapper = original


__all__ = [
    "AFDAttentionModelRunner",
    "fail_if_cuda_graph_enabled",
    "fail_if_ubatching_enabled",
    "fail_if_unsupported_ubatching",
]
