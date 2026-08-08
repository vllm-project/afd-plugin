# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for AFD execution."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    get_forward_context,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPMetadataBuilder,
)
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.utils import (
    embedding_tp_enable,
    enable_sp,
    lmhead_tp_enable,
    oproj_tp_enable,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.model_runner_v1 import (
    SEQ_LEN_WITH_MAX_PA_WORKSPACE,
    NPUModelRunner,
    PerLayerAttnMetadata,
)

from afd_plugin.compat.npu import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    AFDConfig,
    parse_afd_config,
)
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (
    AsyncMoeStage,
    plan_async_moe_stages,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _forward_context_num_tokens,
    _full_cudagraph_padded_tokens,
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import AscendUBatchWrapper
from afd_plugin.v1.worker.npu.ubatch_utils import (
    check_enable_ubatch,
    maybe_create_ubatch_slices,
    pad_out_ubatch_slices,
    split_async_moe_attn_metadata,
    split_attn_metadata,
)
from afd_plugin.v1.worker.ubatch_wrapper import build_ubatch_dp_metadata_list

logger = init_logger(__name__)

ASYNC_MOE_STAGE_METADATA_BUILDER_OFFSET = 1


class AFDNPUAttentionModelRunner(NPUModelRunner):
    """NPU model runner that injects AFD metadata into Ascend forward context."""

    afd_expected_role = "attention"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        afd_config = self.parse_config(vllm_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        fail_if_unsupported_npu_afd_features(
            vllm_config,
            afd_config=afd_config,
        )
        rank, _ = _resolve_world_ranks()
        local_rank = int(device.index)
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.afd_async_extra_info = AFDAsyncExtraInfo()
        if afd_config.connector == AFD_ASYNC_CONNECTOR:
            connector_extra_info = self.connector.extra_info
            if not isinstance(connector_extra_info, AFDAsyncExtraInfo):
                raise TypeError(
                    "CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
                    f"{type(connector_extra_info).__name__}",
                )
            self.afd_async_extra_info = connector_extra_info
        self.connector.init_afd_connector()
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_pending_metadata: AFDForwardContextMetadata | None = None
        self._afd_suppress_metadata_send = False
        self._afd_transaction_counter = 0
        self._afd_async_moe_ubatch_metadata = None
        self._afd_live_execution = False
        self.ubatch_slices = None
        self.prof = create_afd_npu_profiler("attention")

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: vLLM-Ascend calls the execution/padding hook without opting
    # into microbatching, and AFD must keep that hook's upstream default intact.
    # Patch functionality: scope an AFD live-execution flag around the delegated
    # upstream request so the hook can distinguish live requests from dummy runs.
    # Signature: matches upstream; no added parameters.
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        step_afd_npu_profiler(self.prof)
        # ### PATCH START: AFD live execution scope
        self._afd_live_execution = True
        try:
            result = super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            self._afd_live_execution = False
        # ### PATCH END: AFD live execution scope
        return result

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner._model_forward.
    # Patch reason: the upstream forward path does not install AFD stage metadata
    # or expose Ascend ubatch slices to the model wrapper.
    # Patch functionality: inject AFD forward-context state while retaining the
    # upstream model invocation, ENPU ordering, and FlashComm output handling.
    # Signature: matches upstream; no added parameters.
    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        forward_context = get_forward_context()
        # ### PATCH START: AFD forward-context metadata
        if self.ubatch_slices is not None:
            forward_context.ubatch_slices = self.ubatch_slices
        forward_context.dbo_enabled = False
        self._install_afd_metadata_on_forward_context(forward_context)
        self._install_async_moe_ubatch_metadata_on_forward_context(forward_context)
        # ### PATCH END: AFD forward-context metadata

        assert self.model is not None
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }
        run_model = partial(self.model, **model_inputs)
        wrapper_owns_full_graph_update = isinstance(
            self.model, AscendUBatchWrapper
        ) and self.model.owns_full_graph_update(forward_context)

        if self.enable_enpu and not wrapper_owns_full_graph_update:
            self._update_full_graph_params_if_needed(
                forward_context,
                num_tokens_padded,
            )
        hidden_states = run_model()
        if not self.enable_enpu and not wrapper_owns_full_graph_update:
            self._update_full_graph_params_if_needed(
                forward_context,
                num_tokens_padded,
            )

        # ### PATCH START: AFD defers FlashComm gather to the ubatch wrapper
        if (
            forward_context.flash_comm_v1_enabled
            and not forward_context.dbo_enabled
            and not isinstance(hidden_states, IntermediateTensors)
        ):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        # ### PATCH END: AFD defers FlashComm gather to the ubatch wrapper
        return hidden_states

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner._build_attention_metadata.
    # Patch reason: upstream accepts ubatch slices but does not construct separate
    # Ascend attention metadata for each NPU ubatch.
    # Patch functionality: normalize padded slices, build AFD control metadata,
    # and route only split batches through the plugin-owned metadata builder.
    # Signature: matches upstream; no added parameters.
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
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        # ### PATCH START: AFD NPU ubatch metadata routing
        ubatch_slices = _normalize_metadata_ubatch_slices(
            ubatch_slices,
            num_tokens_padded,
            num_reqs_padded,
        )
        if self.afd_async_extra_info.async_moe_ubatching:
            self.ubatch_slices = None
            return self._build_attention_metadata_with_async_moe_ubatches(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=max_query_len,
                num_tokens_padded=num_tokens_padded,
                num_reqs_padded=num_reqs_padded,
                ubatch_slices=ubatch_slices,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                for_cudagraph_capture=for_cudagraph_capture,
                num_scheduled_tokens=num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            )
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            num_tokens,
        )
        self.ubatch_slices = ubatch_slices
        if ubatch_slices is not None:
            return self._build_attention_metadata_with_ubatches(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=max_query_len,
                num_tokens_padded=num_tokens_padded,
                num_reqs_padded=num_reqs_padded,
                ubatch_slices=ubatch_slices,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                for_cudagraph_capture=for_cudagraph_capture,
                num_scheduled_tokens=num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            )
        result = super()._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        )
        # ### PATCH END: AFD NPU ubatch metadata routing
        return result

    def _build_attention_metadata_with_async_moe_ubatches(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None,
        num_reqs_padded: int | None,
        ubatch_slices: UBatchSlices | None,
        logits_indices: torch.Tensor | None,
        use_spec_decode: bool,
        for_cudagraph_capture: bool,
        num_scheduled_tokens: dict[str, int] | None,
        num_scheduled_tokens_np: np.ndarray | None,
        cascade_attn_prefix_lens: list[list[int]] | None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        full_metadata = super()._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        )
        self._afd_async_moe_ubatch_metadata = None
        self._afd_pending_metadata = self._build_afd_metadata(
            None,
            num_tokens,
        )

        if num_scheduled_tokens_np is None:
            return full_metadata

        num_tokens_padded = int(num_tokens_padded or num_tokens)
        num_reqs_padded = int(num_reqs_padded or len(num_scheduled_tokens_np))
        use_sequence_parallel = bool(enable_sp(self.vllm_config))
        stages = plan_async_moe_stages(
            num_scheduled_tokens_np,
            split=self.afd_async_extra_info.async_moe_split,
            use_sequence_parallel=use_sequence_parallel,
            tensor_parallel_size=get_tensor_model_parallel_world_size(),
        )
        if stages is None:
            return full_metadata
        stage_slices = [
            UBatchSlice(stage.request_slice, stage.token_slice) for stage in stages
        ]

        logger.debug(
            "AFD NPU async MoE ubatch split; num_reqs=%s num_tokens=%s "
            "num_scheduled_tokens=%s split=%s sequence_parallel=%s "
            "request_slices=%s token_slices=%s stage_input_tokens=%s "
            "stage_actual_tokens=%s",
            len(num_scheduled_tokens_np),
            num_tokens,
            num_scheduled_tokens_np.tolist(),
            self.afd_async_extra_info.async_moe_split,
            use_sequence_parallel,
            [(stage.request_slice.start, stage.request_slice.stop) for stage in stages],
            [(stage.token_slice.start, stage.token_slice.stop) for stage in stages],
            [int(stage.input_tokens) for stage in stages],
            [stage.actual_tokens for stage in stages],
        )

        stage_attn_metadata, _ = self._build_attention_metadata_with_ubatches(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=stage_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            metadata_builder_offset=ASYNC_MOE_STAGE_METADATA_BUILDER_OFFSET,
            async_moe_stages=stages,
        )
        self._afd_async_moe_ubatch_metadata = AsyncMoeUbatchMetadata(
            attn_metadata=stage_attn_metadata,
            stages=stages,
            use_sequence_parallel=use_sequence_parallel,
            parent_input_tokens=num_tokens_padded,
        )
        return full_metadata

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner._build_attention_metadata.
    # Patch reason: upstream builds one metadata object even when AFD schedules
    # NPU execution stages, while CAMAsync also distinguishes real tokens from
    # stage-local physical padding.
    # Patch functionality: copy the pinned upstream builders, reserve isolated
    # builder indices for CAMAsync, and select the CAMAsync padding-aware split
    # only when ``async_moe_stages`` is supplied. Native DBO keeps the original
    # builder indices and metadata splitter.
    # Signature: adds plugin-owned ``metadata_builder_offset`` and
    # ``async_moe_stages`` parameters to the copied upstream signature.
    def _build_attention_metadata_with_ubatches(
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
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        # ### PATCH START: Async CAM stage parameters
        metadata_builder_offset: int = 0,
        async_moe_stages: tuple[AsyncMoeStage, ...] | None = None,
        # ### PATCH END: Async CAM stage parameters
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """Build isolated per-stage Ascend attention metadata.

        ``metadata_builder_offset`` reserves builder zero for full-batch
        metadata when Async CAM builds its two stage plans.
        ``async_moe_stages`` selects the Async CAM-only physical-padding
        splitter. Native DBO passes both defaults and retains upstream
        behavior.
        """

        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        # ### PATCH START: AFD per-ubatch metadata containers
        assert ubatch_slices is not None
        attn_metadata: list[dict[str, Any]] = [
            dict() for _ in range(len(ubatch_slices))
        ]
        # ### PATCH END: AFD per-ubatch metadata containers
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs

        if for_cudagraph_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_dcp_metadata(block_table_tensor: torch.Tensor):
            if not self.use_dcp:
                return None, block_table_tensor

            fixed_decode_seq_lens_cpu = None
            if self.use_async_spec_decode:
                fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[
                    :num_reqs
                ].numpy()

            assert num_reqs_padded is not None
            return self.dcp_manager.generate_dcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
                fixed_decode_seq_lens_cpu,
            )

        def _get_block_table_and_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
                blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
                slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if (
                self.model_config.enable_return_routed_experts
                and kv_cache_gid == 0
                and self.routed_experts_initialized
            ):
                num_slots = slot_mapping.shape[0]
                self.routed_experts_slot_mapping_device[:num_slots].copy_(
                    slot_mapping,
                )
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        self.long_seq_metadata, block_table_gid_0 = _get_dcp_metadata(
            block_table_gid_0,
        )
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        is_prefilling[num_reqs:] = False
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        if self.use_async_spec_decode:
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            is_prefilling=is_prefilling,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions,
            positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            context_parallel_metadata=self.long_seq_metadata,
            group_len=self.group_len.gpu[:num_reqs_padded],
            group_key_idx=self.group_key_idx.gpu[:num_reqs_padded],
            group_key_cache_idx=self.group_key_cache_idx.gpu[:num_reqs_padded],
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices,
            )

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            prefill_ratio_to_sas_metadata: dict[Any, Any],
            decode_ratio_to_sas_metadata: dict[Any, Any],
            common_ratio_to_sas_metadata: dict[Any, Any],
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            # ### PATCH START: Async CAM builder offset
            builder = attn_group.get_metadata_builder(
                metadata_builder_offset + (ubid or 0),
            )
            # ### PATCH END: Async CAM builder offset
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if isinstance(
                builder,
                AscendDSAMetadataBuilder | AscendDSACPMetadataBuilder,
            ):
                if for_cudagraph_capture:
                    prefill_ratio_to_sas_metadata = {}
                    decode_ratio_to_sas_metadata = {}
                    common_ratio_to_sas_metadata = {}
                extra_attn_metadata_args = dict(
                    num_reqs_actual=num_reqs,
                    prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                    decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                    common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                    block_size=attn_group.kv_cache_spec.block_size,
                )

            if for_cudagraph_capture and not isinstance(
                builder,
                AscendDSAMetadataBuilder
                | AscendDSACPMetadataBuilder
                | AscendSFADCPMetadataBuilder,
            ):
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
                if (
                    cudagraph_mode.has_full_cudagraphs()
                    and isinstance(builder, GDNAttentionMetadataBuilder)
                    and attn_metadata_i.num_prefills == 0
                    and attn_metadata_i.num_decodes == 0
                    and attn_metadata_i.num_spec_decodes > 0
                ):
                    attn_metadata_i.spec_state_indices_tensor[
                        attn_metadata_i.num_spec_decodes :
                    ].fill_(0)
            if isinstance(builder, AscendDSAMetadataBuilder):
                prefill_ratio_to_sas_metadata = builder.prefill_ratio_to_sas_metadata
                decode_ratio_to_sas_metadata = builder.decode_ratio_to_sas_metadata
                common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata

            # ### PATCH START: AFD per-ubatch metadata assignment
            assert ubid is not None
            attn_metadata_dict = attn_metadata[ubid]
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i
            # ### PATCH END: AFD per-ubatch metadata assignment

        prefill_ratio_to_sas_metadata: dict[Any, Any] = {}
        decode_ratio_to_sas_metadata: dict[Any, Any] = {}
        common_ratio_to_sas_metadata: dict[Any, Any] = {}
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(
            self.kv_cache_config.kv_cache_groups,
        ):
            cm = copy.copy(cm_base)
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )
            if self._has_gdn:
                attn_group = self.attn_groups[kv_cache_gid][0]
                # ### PATCH START: Async CAM GDN builder offset
                builder = attn_group.get_metadata_builder(
                    metadata_builder_offset,
                )
                # ### PATCH END: Async CAM GDN builder offset
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[
                        : num_reqs_padded + 1
                    ]
                    cm.query_start_loc = self.gdn_query_start_loc.gpu[
                        : num_reqs_padded + 1
                    ]
            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = (
                    _get_block_table_and_slot_mapping(
                        kv_cache_gid,
                    )
                )
            if self.speculative_config and isinstance(
                self.drafter,
                AscendStep3p5MTPProposer | AscendDSparkProposer,
            ):
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid,
                    cm.block_table_tensor,
                    cm.slot_mapping,
                )
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(
                    self.drafter,
                    AscendEagleProposer
                    | AscendDraftModelProposer
                    | AscendDflashProposer
                    | AscendDSparkProposer,
                ):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                # ### PATCH START: AFD stage metadata split
                if async_moe_stages is None:
                    ubatch_common_metadata = split_attn_metadata(
                        ubatch_slices,
                        cm,
                        num_tokens_padded,
                    )
                else:
                    ubatch_common_metadata = split_async_moe_attn_metadata(
                        async_moe_stages,
                        cm,
                        num_tokens_padded,
                    )
                for ubid, ubatch_cm in enumerate(ubatch_common_metadata):
                    _build_attn_group_metadata(
                        kv_cache_gid,
                        attn_gid,
                        ubatch_cm,
                        prefill_ratio_to_sas_metadata,
                        decode_ratio_to_sas_metadata,
                        common_ratio_to_sas_metadata,
                        ubid,
                    )
                # ### PATCH END: AFD stage metadata split

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges
            # ### PATCH START: AFD multimodal metadata assignment
            for ub_metadata in attn_metadata:
                for metadata in ub_metadata.values():
                    metadata.mm_prefix_range = req_doc_ranges
            # ### PATCH END: AFD multimodal metadata assignment

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(
                    num_tokens,
                    num_reqs,
                )
            )
        return attn_metadata, spec_decode_common_attn_metadata

    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self._dummy_run_inference_mode(
                num_tokens,
                with_prefill=with_prefill,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention,
                uniform_decode=uniform_decode,
                is_profile=is_profile,
                create_mixed_batch=create_mixed_batch,
                allow_microbatching=allow_microbatching,
                skip_eplb=skip_eplb,
                remove_lora=remove_lora,
                is_graph_capturing=is_graph_capturing,
                num_active_loras=num_active_loras,
                profile_seq_lens=profile_seq_lens,
                profile_cpp=profile_cpp,
            )

    def _dummy_run_inference_mode(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self._afd_is_graph_capturing
        self._afd_is_graph_capturing = bool(is_graph_capturing)
        if not (
            bool(self.vllm_config.parallel_config.use_ubatching)
            and allow_microbatching
            and not is_profile
        ):
            try:
                return super()._dummy_run(
                    num_tokens,
                    with_prefill=with_prefill,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    force_attention=force_attention,
                    uniform_decode=uniform_decode,
                    is_profile=is_profile,
                    create_mixed_batch=create_mixed_batch,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=skip_eplb,
                    remove_lora=remove_lora,
                    is_graph_capturing=is_graph_capturing,
                    num_active_loras=num_active_loras,
                    profile_seq_lens=profile_seq_lens,
                    profile_cpp=profile_cpp,
                )
            finally:
                self._afd_is_graph_capturing = previous
                self._afd_pending_metadata = None
                self._afd_async_moe_ubatch_metadata = None

        try:
            return self._dummy_run_with_ubatches(
                num_tokens,
                with_prefill=with_prefill,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                force_attention=force_attention,
                uniform_decode=uniform_decode,
                is_profile=is_profile,
                create_mixed_batch=create_mixed_batch,
                allow_microbatching=allow_microbatching,
                skip_eplb=skip_eplb,
                remove_lora=remove_lora,
                is_graph_capturing=is_graph_capturing,
                num_active_loras=num_active_loras,
                profile_seq_lens=profile_seq_lens,
                profile_cpp=profile_cpp,
            )
        finally:
            self._afd_is_graph_capturing = previous
            self._afd_pending_metadata = None
            self._afd_async_moe_ubatch_metadata = None

    # Upstream source: vLLM v0.26.0 commit 568afb3a1,
    # GPUModelRunner._warmup_and_capture.
    # Patch reason: AFD needs both single-stage and two-stage Ascend graph keys,
    # because live decode may fall below the DBO threshold.
    # Patch functionality: run the pinned warmup/capture hook once for each AFD
    # execution shape while coordinating metadata with the FFN workers.
    # Signature: matches upstream; no added parameters.
    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
        profiler: AbstractContextManager[Any] | None = None,
    ):
        """Capture both single-stage and ubatched FFN graph keys.

        Native vLLM only captures the ubatched graph when microbatching is
        allowed for a decode capture size. Original AFD also captures the
        corresponding non-ubatched decode graph first, because live decode can
        still produce a single-stage key below the ubatch threshold.
        """

        # ### PATCH START: AFD dual graph capture
        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups

        if allow_microbatching:
            self._afd_warmup_and_capture_once(
                desc=desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                profile_seq_lens=profile_seq_lens,
                allow_microbatching=False,
                num_warmups=int(num_warmups),
                profiler=nullcontext(),
            )

        self._afd_warmup_and_capture_once(
            desc=desc,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            profile_seq_lens=profile_seq_lens,
            allow_microbatching=allow_microbatching,
            num_warmups=int(num_warmups),
            profiler=profiler,
        )
        # ### PATCH END: AFD dual graph capture

    def _afd_warmup_and_capture_once(
        self,
        *,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None,
        allow_microbatching: bool,
        num_warmups: int,
        profiler: AbstractContextManager[Any],
    ) -> None:
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        previous_is_warmup = bool(self._is_warmup)
        try:
            self._is_warmup = True
            for _ in range(num_warmups):
                self._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=desc.uniform,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    num_active_loras=desc.num_active_loras,
                )
        finally:
            self._is_warmup = previous_is_warmup

        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_graph_capturing = self._afd_is_graph_capturing
        try:
            self._afd_is_graph_capturing = True
            if allow_microbatching:
                self._afd_pending_metadata = None
                self._afd_suppress_metadata_send = False
            else:
                self._afd_pending_metadata = self._build_afd_metadata(
                    None,
                    int(desc.num_tokens),
                )
                if self.connector.control_plane is not None:
                    self._send_dp_metadata(
                        self._build_capture_dp_metadata(int(desc.num_tokens)),
                        None,
                    )
                self._afd_suppress_metadata_send = True

            with (
                profiler,
                torch.profiler.record_function(
                    f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}",
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

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner._dummy_run.
    # Patch reason: upstream's dummy path forces ubatch slices to None, so it
    # cannot warm or capture the AFD two-stage Ascend execution path.
    # Patch functionality: preserve the pinned upstream dummy setup while
    # constructing and forwarding the same two ubatches used by live requests.
    # Signature: matches upstream; no added parameters.
    def _dummy_run_with_ubatches(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            raise NotImplementedError(
                "create_mixed_batch is used for warmup deepgemm; "
                "AFD NPU does not support it",
            )
        if uniform_decode:
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        elif profile_cpp:
            num_reqs = 1
            num_scheduled_tokens_list = [num_tokens] * num_reqs
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        # ### PATCH START: AFD dummy ubatch decision
        (
            _cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            _,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile
            or (cudagraph_runtime_mode == CUDAGraphMode.NONE)
            or profile_cpp,
            force_uniform_decode=uniform_decode,
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
        )
        # ### PATCH END: AFD dummy ubatch decision
        if self.use_dcp:
            self.dcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
            )
            if self.speculative_config:
                self.dcp_manager.query_lens_full.cpu[:num_reqs] = torch.from_numpy(
                    num_scheduled_tokens,
                )
                self.dcp_manager.query_lens_full.copy_to_gpu()
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        if self.dynamic_eplb:
            self.update_eplb_heat_collection_status(num_tokens_padded)

        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        with self.synchronize_input_prep():
            if self._should_build_dummy_attn_metadata(
                force_attention,
                is_profile,
                cudagraph_runtime_mode,
            ):
                self.attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and self.speculative_config.method == "mtp":
                    if self.vllm_config.model_config.use_mla:
                        self.attn_state = AscendAttentionState.SpecDecoding
                    else:
                        self.attn_state = AscendAttentionState.ChunkedPrefill
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens
                else:
                    seq_lens = (
                        SEQ_LEN_WITH_MAX_PA_WORKSPACE
                        if is_graph_capturing
                        and using_paged_attention(num_tokens, self.vllm_config)
                        else max_query_len
                    )

                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(
                    self.optimistic_seq_lens_cpu,
                    non_blocking=True,
                )

                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens,
                    self.query_pos.np,
                )
                self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()
                if self._has_gdn:
                    self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = (
                        cum_num_tokens
                    )
                    self.gdn_query_start_loc.copy_to_gpu()

                if not profile_cpp:
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_runtime_mode,
                        batch_desc.num_reqs,
                    )

                self.input_batch.block_table.commit_block_table(num_reqs_padded)
                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                # ### PATCH START: AFD dummy ubatch slices
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.vllm_config,
                )
                self.ubatch_slices = ubatch_slices_padded if pad_attn else ubatch_slices
                # ### PATCH END: AFD dummy ubatch slices
                if self.use_compress:
                    self.positions.fill_(127)
                    self._dsa_positions_cpu_buf.fill_(127)
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_query_len,
                    # ### PATCH START: AFD dummy ubatch metadata input
                    ubatch_slices=self.ubatch_slices,
                    # ### PATCH END: AFD dummy ubatch metadata input
                    for_cudagraph_capture=is_graph_capturing,
                    num_scheduled_tokens_np=num_scheduled_tokens,
                )
                if not is_graph_capturing:
                    for kv_cache_gid in range(
                        len(self.kv_cache_config.kv_cache_groups),
                    ):
                        block_table = self.input_batch.block_table[kv_cache_gid]
                        block_table.slot_mapping.gpu.fill_(-1)
            # ### PATCH START: AFD attention-free dummy ubatch slices
            elif should_ubatch:
                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.vllm_config,
                )
                self.ubatch_slices = ubatch_slices_padded if pad_attn else ubatch_slices
            else:
                self.ubatch_slices = None
            # ### PATCH END: AFD attention-free dummy ubatch slices

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras=(
                self.lora_config.max_loras
                if self.lora_config is not None
                else num_active_loras
            ),
        ):
            assert num_tokens_padded <= self.max_num_tokens
            if (
                self.supports_mm_inputs
                and not self.model_config.is_encoder_decoder
                or self.enable_prompt_embeds
            ):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    tp_size = get_tensor_model_parallel_world_size()
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (
                            self.max_num_tokens + tp_size - 1
                        ) // tp_size
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=max_actual_tokens,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded,
                    None,
                    False,
                )

            need_dummy_logits = not is_profile and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return None
                if hasattr(self.drafter, "model") and hasattr(
                    self.drafter.model,
                    "compute_logits",
                ):
                    return self.drafter.model.compute_logits(
                        hidden_states[dummy_indices]
                    )
                return None

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=is_profile,
                num_actual_tokens=num_tokens_padded,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                model_instance=self.model,
                has_sinks=self._has_sinks,
                input_ids=input_ids,
                eplb_heat_collection_status=(
                    self.eplb_heat_collection_status if self.dynamic_eplb else False
                ),
            ):
                outputs = self._model_forward(
                    num_tokens_padded,
                    input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            dummy_compute_logits(hidden_states)

            if self.drafter and not profile_cpp:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile,
                )
            if is_profile and self.dynamic_eplb:
                self.eplb_updator.adaptor.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.forward_end(self.eplb_heat_collection_status)
            self._finalize_dump_data(dump=False)
            if self.use_compress and force_attention:
                self.positions.fill_(0)
                self._dsa_positions_cpu_buf.fill_(0)
            return hidden_states, hidden_states

    def _build_afd_metadata(
        self,
        ubatch_slices: UBatchSlices | None,
        num_tokens_unpadded: int,
    ) -> AFDForwardContextMetadata:
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

    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        if self._afd_pending_metadata is None:
            self._afd_pending_metadata = self._build_afd_metadata(
                forward_context.ubatch_slices,
                _forward_context_num_tokens(forward_context, self.vllm_config),
            )

        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs["afd_metadata"] = self._afd_pending_metadata
        if self.connector.control_plane is None:
            return
        if self._afd_suppress_metadata_send:
            return
        dp_metadata = forward_context.dp_metadata
        ubatch_slices = forward_context.ubatch_slices
        padded_graph_tokens = _full_cudagraph_padded_tokens(forward_context)
        if padded_graph_tokens is not None and not ubatch_slices:
            dp_metadata = self._build_capture_dp_metadata(padded_graph_tokens)
        self._send_dp_metadata(dp_metadata, ubatch_slices)

    def _install_async_moe_ubatch_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        if self._afd_async_moe_ubatch_metadata is None:
            return
        metadata = self._afd_async_moe_ubatch_metadata
        runtime_sequence_parallel = bool(forward_context.flash_comm_v1_enabled)
        if runtime_sequence_parallel and not metadata.use_sequence_parallel:
            raise RuntimeError(
                "Async CAM runtime enabled FlashComm1 for a stage plan that "
                "was not TP-aligned",
            )
        if not runtime_sequence_parallel and metadata.use_sequence_parallel:
            # vLLM-Ascend decides whether FlashComm1 is active for each model
            # forward. When it disables FlashComm1, Attention keeps a
            # replicated token dimension and stage-local TP padding must not
            # leak into the non-SP model inputs or attention metadata.
            metadata = AsyncMoeUbatchMetadata(
                attn_metadata=metadata.attn_metadata,
                stages=tuple(
                    AsyncMoeStage(
                        request_slice=stage.request_slice,
                        token_slice=stage.token_slice,
                        input_tokens=stage.actual_tokens,
                    )
                    for stage in metadata.stages
                ),
                parent_input_tokens=metadata.parent_input_tokens,
                use_sequence_parallel=False,
            )
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs[ASYNC_MOE_UBATCH_METADATA_KEY] = metadata

    def _send_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
        ubatch_slices: UBatchSlices | None,
    ) -> None:
        assert self.connector.control_plane is not None, (
            "_send_dp_metadata needs control plane driven connectors"
        )

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
        is_warmup = bool(self._is_warmup)
        is_graph_capturing = bool(self._afd_is_graph_capturing)
        payload = AFDControlPayload(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        self.connector.control_plane.update_state_from_dp_metadata(payload)
        logger.warning(
            "AFD NPU Attention send_dp_metadata decision; world_rank=%d "
            "key=%s is_graph_capturing=%s is_warmup=%s",
            self.connector.world_rank,
            _dp_metadata_debug_key(dp_metadata_list),
            is_graph_capturing,
            is_warmup,
        )
        self.connector.control_plane.send_dp_metadata_list(payload)

    def _ensure_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
    ) -> DPMetadata | AFDDPMetadata:
        if dp_metadata is not None:
            return dp_metadata

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        if dp_size != 1:
            raise RuntimeError("AFD NPU Attention expected DPMetadata for DP > 1")
        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP fallback")

        num_tokens = int(self._afd_pending_metadata.tokens_lens[0])
        return _make_uniform_dp_metadata(dp_size, num_tokens)

    def _build_capture_dp_metadata(self, num_tokens: int) -> DPMetadata | AFDDPMetadata:
        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        return _make_uniform_dp_metadata(dp_size, int(num_tokens))

    def load_model(self) -> None:
        super().load_model()
        if bool(self.vllm_config.parallel_config.use_ubatching):
            self._install_ascend_ubatch_wrapper()

    def _install_ascend_ubatch_wrapper(self) -> None:
        if isinstance(self.model, AscendUBatchWrapper):
            return
        model = self.model
        runtime_mode = CUDAGraphMode.NONE
        if isinstance(model, ACLGraphWrapper):
            model = model.unwrap()
            runtime_mode = CUDAGraphMode.FULL
        elif self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            runtime_mode = CUDAGraphMode.FULL
        self.model = AscendUBatchWrapper(
            model,
            self.vllm_config,
            runtime_mode,
            self.device,
            mla_full_graph_enabled=(
                self.vllm_config.model_config.use_mla and not self.use_sparse
            ),
            full_graph_params_updater=self._update_full_graph_params_if_needed,
            enable_enpu=self.enable_enpu,
        )

    def get_model(self) -> nn.Module:
        if isinstance(self.model, AscendUBatchWrapper):
            return self.model.unwrap()
        return super().get_model()

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_attn_backend(kv_cache_config)
        if (
            bool(
                self.vllm_config.parallel_config.use_ubatching,
            )
            or self.afd_async_extra_info.async_moe_ubatching
        ):
            num_metadata_builders = (
                ASYNC_MOE_STAGE_METADATA_BUILDER_OFFSET
                + int(self.afd_async_extra_info.async_moe_num_ubatches)
                if self.afd_async_extra_info.async_moe_ubatching
                else int(self.vllm_config.parallel_config.num_ubatches)
            )
            self._ensure_metadata_builders(num_metadata_builders)

    def _ensure_metadata_builders(self, num_metadata_builders: int) -> None:
        for attn_groups in self.attn_groups:
            for attn_group in attn_groups:
                if len(attn_group.metadata_builders) >= num_metadata_builders:
                    continue
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    num_metadata_builders=num_metadata_builders,
                )

    def _sync_afd_metadata_across_dp(
        self,
        num_tokens_unpadded: int,
        num_tokens_padded: int | None = None,
        uniform_decode: bool = False,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode | None = None,
        allow_dp_padding: bool = False,
    ) -> tuple[bool, int, torch.Tensor | None, CUDAGraphMode]:
        if cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.NONE
        if num_tokens_padded is None:
            num_tokens_padded = num_tokens_unpadded

        if self.dp_size == 1:
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            return should_ubatch, num_tokens_padded, None, cudagraph_mode

        if self.connector.control_plane is None:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            return (
                should_ubatch,
                num_tokens_padded,
                num_tokens_after_padding,
                cudagraph_mode,
            )

        parallel_config = self.vllm_config.parallel_config
        can_skip_dp_sync = should_skip_allreduce_across_dp_group(
            self.vllm_config,
            is_draft_model,
        )
        may_ubatch = bool(parallel_config.enable_dbo and parallel_config.use_ubatching)
        if can_skip_dp_sync and not may_ubatch:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            return (
                should_ubatch,
                num_tokens_padded,
                num_tokens_after_padding,
                cudagraph_mode,
            )
        packed_tensor = torch.zeros(3, self.dp_size, device="cpu", dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens_unpadded
        packed_tensor[1][self.dp_rank] = num_tokens_padded
        packed_tensor[2][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        num_tokens_unpadded_across_dp = packed_tensor[0, :]
        num_tokens_padded_across_dp = packed_tensor[1, :]
        max_tokens_across_dp = int(num_tokens_padded_across_dp.max().item())
        min_tokens_across_dp = int(num_tokens_unpadded_across_dp.min().item())
        synced_cudagraph_mode = CUDAGraphMode(int(packed_tensor[-1, :].min().item()))

        should_ubatch = check_enable_ubatch(
            min_tokens_across_dp,
            max_tokens_across_dp,
            uniform_decode=uniform_decode,
            vllm_config=self.vllm_config,
        )

        if allow_dp_padding or is_draft_model or should_ubatch:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
        else:
            num_tokens_after_padding = num_tokens_padded_across_dp.cpu()
        return (
            should_ubatch,
            max_tokens_across_dp,
            num_tokens_after_padding,
            synced_cudagraph_mode,
        )

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner._determine_batch_execution_and_padding.
    # Patch reason: upstream intentionally leaves NPU microbatching disabled and
    # uses its native DP synchronization, which cannot coordinate AFD stages.
    # Patch functionality: retain the upstream signature and execution/padding
    # logic while enabling microbatching only during AFD live execution and using
    # the AFD control-plane-aware DP synchronization path.
    # Signature: matches upstream; no added parameters or changed defaults.
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
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
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        is_all_decode = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
        uniform_decode = (
            (
                (is_all_decode if self.speculative_config else True)
                and (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        def dispatch_cudagraph(
            num_tokens_to_dispatch, disable_full=False, valid_modes=None
        ):
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens_to_dispatch,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                valid_modes=valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                num_active_loras=num_active_loras,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            use_cascade_attn or has_encoder_output,
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if enable_sp(self.vllm_config):
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be a multiple "
                "of tensor parallel size"
            )

        should_ubatch, num_tokens_across_dp = False, None
        # ### PATCH START: AFD DP metadata synchronization
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, _, num_tokens_across_dp, synced_cudagraph_mode = (
                self._sync_afd_metadata_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode,
                    allow_dp_padding=(cudagraph_mode != CUDAGraphMode.NONE)
                    or enable_sp(self.vllm_config)
                    or oproj_tp_enable()
                    or embedding_tp_enable(),
                )
            )
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
                )
                assert batch_descriptor.num_tokens == num_tokens_padded
        else:
            should_ubatch = check_enable_ubatch(
                num_tokens,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
        # ### PATCH END: AFD DP metadata synchronization
        # ### PATCH START: AFD live NPU microbatching
        if not (allow_microbatching or self._afd_live_execution):
            should_ubatch = False
        # ### PATCH END: AFD live NPU microbatching

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    # Upstream source: vllm-ascend commit 80d8c194f,
    # NPUModelRunner.sync_and_slice_intermediate_tensors.
    # Patch reason: upstream sizes PP intermediate tensors from the combined
    # token count, which is too small when SP rounds each AFD ubatch separately.
    # Patch functionality: compute the sum of per-ubatch SP slices and grow the
    # reusable intermediate buffer before copying or returning that slice.
    # Signature: matches upstream; no added parameters.
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None
        tp = self.vllm_config.parallel_config.tensor_parallel_size

        slice_len = (num_tokens + tp - 1) // tp if enable_sp() else num_tokens
        if self.ubatch_slices is not None:
            # ### PATCH START: AFD per-ubatch intermediate slice and buffer
            slice_len = (
                sum(
                    (ubatch_slice.num_tokens + tp - 1) // tp
                    for ubatch_slice in self.ubatch_slices
                )
                if enable_sp()
                else sum(ubatch_slice.num_tokens for ubatch_slice in self.ubatch_slices)
            )
            intermediate_tensor_size = next(
                iter(self.intermediate_tensors.tensors.values()),
            ).size(0)
            if intermediate_tensor_size < slice_len:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=slice_len,
                    dtype=self.dtype,
                    device=self.device,
                )
            # ### PATCH END: AFD per-ubatch intermediate slice and buffer

        if sync_self:
            assert intermediate_tensors is not None
            # ### PATCH START: AFD intermediate copy length
            copy_len = slice_len
            # ### PATCH END: AFD intermediate copy length
            for k, v in intermediate_tensors.items():
                if k not in self.intermediate_tensors.tensors:
                    base_tensor = self.intermediate_tensors["hidden_states"]
                    self.intermediate_tensors[k] = v.new_empty(
                        (base_tensor.shape[0], *v.shape[1:]),
                    )
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len],
                    non_blocking=True,
                )
        # ### PATCH START: AFD intermediate output slice
        result = IntermediateTensors(
            {k: v[:slice_len] for k, v in self.intermediate_tensors.items()},
        )
        # ### PATCH END: AFD intermediate output slice
        return result

    def shutdown(self) -> None:
        stop_afd_npu_profiler(self.prof)
        self.connector.close()
        super().shutdown()

    def _next_afd_transaction_id(self) -> str:
        counter = self._afd_transaction_counter
        self._afd_transaction_counter = counter + 1
        return f"afd-npu-{counter}"


def _make_uniform_dp_metadata(dp_size: int, num_tokens: int) -> AFDDPMetadata:
    num_tokens_across_dp_cpu = torch.full(
        (int(dp_size),),
        int(num_tokens),
        dtype=torch.int32,
        device="cpu",
    )
    return AFDDPMetadata(num_tokens_across_dp_cpu=num_tokens_across_dp_cpu)


def _dp_metadata_debug_key(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
) -> tuple[tuple[int, tuple]]:
    key_parts: list[tuple[int, tuple]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values_tuple = tuple(
            int(value) for value in metadata.num_tokens_across_dp_cpu.tolist()
        )
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)


def _normalize_metadata_ubatch_slices(
    ubatch_slices: UBatchSlices | None,
    num_tokens_padded: int | None,
    num_reqs_padded: int | None,
) -> UBatchSlices | None:
    if not ubatch_slices:
        return ubatch_slices
    if num_tokens_padded is None or num_reqs_padded is None:
        return ubatch_slices

    last_slice = ubatch_slices[-1]
    if int(last_slice.token_slice.stop) != int(num_tokens_padded) or int(
        last_slice.request_slice.stop
    ) == int(num_reqs_padded):
        return ubatch_slices

    return pad_out_ubatch_slices(
        ubatch_slices,
        int(num_tokens_padded),
        int(num_reqs_padded),
    )


__all__ = ["AFDNPUAttentionModelRunner"]
