# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned Ascend ubatch wrapper.

Provides the Ascend DBO wrapper used by AFD NPU runtimes while keeping the
implementation plugin-owned.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    DPMetadata,
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    GraphParams,
    get_graph_params,
)
from vllm_ascend.utils import enable_sp

from afd_plugin.v1.worker.npu.forward_context import (
    create_ascend_forward_context,
)
from afd_plugin.v1.worker.npu.mla_graph import (
    merge_mla_graph_params,
    new_mla_graph_params,
    override_mla_graph_params,
)
from afd_plugin.v1.worker.npu.ubatching import (
    AscendUBatchContext,
    make_ubatch_contexts,
)

AFD_NPU_NUM_UBATCHES = 2
_READY_BARRIER_PARTIES = AFD_NPU_NUM_UBATCHES + 1
AscendLastRankOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
AscendModelOutput: TypeAlias = AscendLastRankOutput | IntermediateTensors


def _cat_ubatch_outputs(
    sorted_results: list[AscendLastRankOutput],
) -> AscendLastRankOutput:
    """Preserve the current Ascend model-output structure across ubatches.

    Historical source: vLLM v0.26.0 commit 568afb3a1,
    ``gpu_ubatch_wrapper._cat_ubatch_outputs``. Ascend auxiliary hidden states
    use ``tuple[Tensor, list[Tensor]]`` rather than upstream's tuple of tensors,
    so this plugin-owned wrapper concatenates that concrete nested contract.
    """
    assert sorted_results
    first_result = sorted_results[0]
    # ### PATCH START: Ascend auxiliary hidden-state output
    if isinstance(first_result, tuple):
        tuple_results = cast(
            list[tuple[torch.Tensor, list[torch.Tensor]]],
            sorted_results,
        )
        num_aux_outputs = len(first_result[1])
        assert all(len(result[1]) == num_aux_outputs for result in tuple_results)
        return (
            torch.cat([result[0] for result in tuple_results], dim=0),
            [
                torch.cat(
                    [result[1][index] for result in tuple_results],
                    dim=0,
                )
                for index in range(num_aux_outputs)
            ],
        )
    # ### PATCH END: Ascend auxiliary hidden-state output
    return torch.cat(cast(list[torch.Tensor], sorted_results), dim=0)


@dataclass
class AscendUbatchMetadata(UbatchMetadata):
    context: AscendUBatchContext
    input_ids: torch.Tensor | None


@dataclass
class AscendNPUGraphMetaData:
    aclgraph: torch.npu.NPUGraph
    ubatch_metadata: list[AscendUbatchMetadata]
    outputs: AscendModelOutput | None = None
    mla_graph_params: tuple[GraphParams, GraphParams] | None = None


@dataclass(frozen=True)
class AscendNPUGraphKey:
    stage_num_tokens: tuple[int, int]
    has_lora: bool
    num_active_loras: int


FullGraphParamsUpdater = Callable[
    [ForwardContext, int],
    None,
]


class AscendUBatchWrapper(UBatchWrapper):
    """Ascend microbatch wrapper used only by AFD NPU runtimes."""

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.device,
        *,
        mla_full_graph_enabled: bool = False,
        full_graph_params_updater: FullGraphParamsUpdater | None = None,
        enable_enpu: bool = False,
    ):
        assert not enable_enpu, "AscendUBatchWrapper does not support ENPU"
        self.runnable = runnable
        self.vllm_config = vllm_config
        assert self.vllm_config.parallel_config.num_ubatches == AFD_NPU_NUM_UBATCHES
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(_READY_BARRIER_PARTIES)
        self.cudagraphs: dict[AscendNPUGraphKey, AscendNPUGraphMetaData] = {}
        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=runtime_mode,
            )
        self.device = device
        self.mla_full_graph_enabled = mla_full_graph_enabled
        self.full_graph_params_updater = full_graph_params_updater

    @property
    def graph_pool(self):
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        self.cudagraphs.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.concrete_aclgraph_entries.clear()

    def __getattr__(self, key: str):
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not found in AscendUBatchWrapper runnable."
        )

    def unwrap(self) -> Callable:
        return self.runnable

    def owns_full_graph_update(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        uses_mla_ubatch_full_graph = (
            self.mla_full_graph_enabled
            and forward_context.ubatch_slices is not None
            and forward_context.cudagraph_runtime_mode is CUDAGraphMode.FULL
        )
        if not uses_mla_ubatch_full_graph:
            return False
        if forward_context.max_tokens_across_pcp not in (None, 0):
            raise RuntimeError(
                "MLA DBO FULL graph does not support PCP execution",
            )
        return True

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if ubatch_slices is None:
            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            assert self.cudagraph_wrapper is not None
            return self.cudagraph_wrapper(*args, **kwargs)

        mla_full_graph_active = self.owns_full_graph_update(forward_context)
        attn_metadata = forward_context.attn_metadata
        if len(ubatch_slices) != AFD_NPU_NUM_UBATCHES:
            raise RuntimeError(
                "Ascend FULL graph requires exactly two ubatches; "
                f"got {len(ubatch_slices)}",
            )
        stage_num_tokens = (
            ubatch_slices[0].num_tokens,
            ubatch_slices[1].num_tokens,
        )
        graph_key = AscendNPUGraphKey(
            stage_num_tokens,
            batch_descriptor.has_lora,
            batch_descriptor.num_active_loras,
        )
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.npu.current_stream()

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            if dp_size > 1:
                ubatch_num_tokens_across_dp = torch.tensor(
                    [ubatch_slice.num_tokens] * dp_size,
                    device="cpu",
                    dtype=torch.int32,
                )
                ubatch_dp_metadata.append(
                    DPMetadata.make(
                        self.vllm_config.parallel_config,
                        ubatch_slice.num_tokens,
                        ubatch_num_tokens_across_dp,
                    )
                )
            else:
                ubatch_dp_metadata.append(None)

        if (
            graph_key not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            mla_graph_params = (
                self._new_mla_capture_params(stage_num_tokens)
                if mla_full_graph_active
                else None
            )
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                torch.npu.Stream(device=torch.npu.current_device()),
                ubatch_dp_metadata,
                batch_descriptor,
                CUDAGraphMode.NONE,
                mla_graph_params=mla_graph_params,
            )
            return self._capture_ubatches(
                ubatch_metadata,
                self.runnable,
                graph_key=graph_key,
                mla_graph_params=mla_graph_params,
            )
        if (
            graph_key in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[graph_key]
            if mla_full_graph_active:
                self._replay_mla_graph(
                    cudagraph_metadata,
                    forward_context,
                    stage_num_tokens[0],
                )
            else:
                torch.npu.current_stream().synchronize()
                cudagraph_metadata.aclgraph.replay()
            forward_context.dbo_enabled = True
            assert cudagraph_metadata.outputs is not None
            return cudagraph_metadata.outputs

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            compute_stream,
            ubatch_dp_metadata,
            batch_descriptor,
            CUDAGraphMode.NONE,
        )
        return self._run_ubatches(ubatch_metadata, self.runnable)

    def _new_mla_capture_params(
        self,
        stage_num_tokens: tuple[int, int],
    ) -> tuple[GraphParams, GraphParams]:
        if stage_num_tokens[0] != stage_num_tokens[1]:
            raise RuntimeError(
                "MLA DBO FULL graph requires equal padded token counts; "
                f"got {stage_num_tokens}",
            )

        aggregate_num_tokens = sum(stage_num_tokens)
        graph_params = get_graph_params()
        if (
            graph_params is None
            or graph_params.workspaces.get(aggregate_num_tokens) is None
        ):
            raise RuntimeError(
                "MLA DBO FULL graph requires the single-batch FIA workspace "
                f"for {aggregate_num_tokens} tokens",
            )
        workspace = graph_params.workspaces[aggregate_num_tokens]
        child_num_tokens = stage_num_tokens[0]
        return (
            new_mla_graph_params(child_num_tokens, workspace),
            new_mla_graph_params(child_num_tokens, workspace),
        )

    def _replay_mla_graph(
        self,
        graph_metadata: AscendNPUGraphMetaData,
        forward_context: ForwardContext,
        num_tokens: int,
    ) -> None:
        if graph_metadata.mla_graph_params is None:
            raise RuntimeError(
                "MLA DBO FULL graph cache entry has no capture registry",
            )
        if self.full_graph_params_updater is None:
            raise RuntimeError(
                "MLA DBO FULL graph cache entry has no parameter updater",
            )

        merged_metadata, merged_params = merge_mla_graph_params(
            forward_context.attn_metadata,
            graph_metadata.mla_graph_params,
            num_tokens,
        )

        def update_params() -> None:
            with override_mla_graph_params(
                forward_context,
                merged_metadata,
                merged_params,
            ):
                assert self.full_graph_params_updater is not None
                self.full_graph_params_updater(
                    forward_context,
                    num_tokens,
                )

        torch.npu.current_stream().synchronize()
        graph_metadata.aclgraph.replay()
        update_params()

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
        *,
        mla_graph_params: tuple[GraphParams, GraphParams] | None = None,
    ) -> list[AscendUbatchMetadata]:
        cur_forward_context = get_forward_context()
        forward_contexts = []
        for i, _ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=attn_metadata[i]
                    if attn_metadata is not None
                    else None,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    skip_compiled=cur_forward_context.skip_compiled,
                    mla_graph_params=(
                        mla_graph_params[i] if mla_graph_params is not None else None
                    ),
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        metadata_list: list[AscendUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
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
            metadata_list.append(
                AscendUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                )
            )
        return metadata_list

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        sliced_positions = (
            positions[:, tokens_slice]
            if positions.ndim == 2
            else positions[tokens_slice]
        )
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )

        if intermediate_tensors is not None and enable_sp():
            tp_size = get_tensor_model_parallel_world_size()
            start = (tokens_slice.start + tp_size - 1) // tp_size
            if start != 0:
                stop = (
                    start
                    + (tokens_slice.stop - tokens_slice.start + tp_size - 1) // tp_size
                )
            else:
                stop = (tokens_slice.stop + tp_size - 1) // tp_size
            tokens_slice = slice(start, stop)
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )
        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def _merge_intermediate_tensors(self, intermediate_tensor_list):
        assert len(intermediate_tensor_list) == 2
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat(
                [
                    intermediate_tensor_list[0].tensors[key],
                    intermediate_tensor_list[1].tensors[key],
                ],
                dim=0,
            )
        return IntermediateTensors(result)

    def _merge_outputs(
        self,
        sorted_results: list[AscendModelOutput],
        ubatch_metadata: list[AscendUbatchMetadata],
    ) -> AscendModelOutput:
        if not get_pp_group().is_last_rank:
            return self._merge_intermediate_tensors(
                cast(list[IntermediateTensors], sorted_results),
            )

        last_rank_results = cast(list[AscendLastRankOutput], sorted_results)
        # Native models return gathered, trimmed SP output for each ubatch.
        return _cat_ubatch_outputs(last_rank_results)

    @torch.inference_mode()
    def _run_ubatch_thread(self, results, model, ubatch_metadata):
        with ubatch_metadata.context:
            model_output = model(
                input_ids=ubatch_metadata.input_ids,
                positions=ubatch_metadata.positions,
                intermediate_tensors=ubatch_metadata.intermediate_tensors,
                inputs_embeds=ubatch_metadata.inputs_embeds,
            )
        results.append((ubatch_metadata.context.id, model_output))

    def _run_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model,
    ) -> AscendModelOutput:
        results: list[tuple[int, AscendModelOutput]] = []
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for _, value in sorted(results)]
        get_forward_context().dbo_enabled = True
        return self._merge_outputs(sorted_results, ubatch_metadata)

    def _capture_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model,
        *,
        graph_key: AscendNPUGraphKey,
        mla_graph_params: tuple[GraphParams, GraphParams] | None,
    ) -> AscendModelOutput:
        results: list[tuple[int, AscendModelOutput]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()

            cudagraph_metadata = AscendNPUGraphMetaData(
                aclgraph=torch.npu.NPUGraph(),
                ubatch_metadata=ubatch_metadata,
                mla_graph_params=mla_graph_params,
            )
            with torch.npu.graph(
                cudagraph_metadata.aclgraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for _, value in sorted(results)]
                cudagraph_metadata.outputs = self._merge_outputs(
                    sorted_results,
                    ubatch_metadata,
                )
            self.cudagraphs[graph_key] = cudagraph_metadata
        get_forward_context().dbo_enabled = True
        assert cudagraph_metadata.outputs is not None
        return cudagraph_metadata.outputs


__all__ = [
    "AscendNPUGraphKey",
    "AscendNPUGraphMetaData",
    "AscendModelOutput",
    "AscendUBatchWrapper",
    "AscendUbatchMetadata",
]
