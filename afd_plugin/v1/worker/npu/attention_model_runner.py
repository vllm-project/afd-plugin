# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for AFD execution."""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
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
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm_ascend.ascend_forward_context import (
    select_moe_comm_method,
    set_ascend_forward_context,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.kvcomp_attn.attention_utils import build_kvcomp_metadata
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import (
    enable_sp,
    lmhead_tp_enable,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from afd_plugin.async_moe import AsyncMoeStage, plan_async_moe_stages
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
from afd_plugin.model_executor.models import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _forward_context_num_tokens,
    _full_cudagraph_padded_tokens,
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import AscendUBatchWrapper
from afd_plugin.v1.worker.npu.pcp_debug import (
    clone_pcp_metadata,
    debug_pcp_common_metadata_summary,
    debug_pcp_manager_summary,
    debug_pcp_metadata_enabled,
    debug_pcp_metadata_summary,
    debug_slice_summary,
    debug_value_summary,
    restore_pcp_manager_state,
    snapshot_pcp_manager_state,
)
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

    def __init__(self, vllm_config: VllmConfig, device: object) -> None:
        afd_config = self.parse_config(vllm_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        fail_if_unsupported_npu_afd_features(
            vllm_config,
            afd_config=afd_config,
        )
        rank, local_rank = _resolve_world_ranks()
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
        self.ubatch_slices = None
        self.prof = create_afd_npu_profiler("attention")

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def execute_model(self, *args: Any, **kwargs: Any) -> Any:
        step_afd_npu_profiler(self.prof)
        return super().execute_model(*args, **kwargs)

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        forward_context = get_forward_context()
        if self.ubatch_slices is not None:
            forward_context.ubatch_slices = self.ubatch_slices
        try:
            forward_context.dbo_enabled = bool(forward_context.dbo_enabled)
        except AttributeError:
            forward_context.dbo_enabled = False
        self._install_afd_metadata_on_forward_context(forward_context)
        self._install_async_moe_ubatch_metadata_on_forward_context(forward_context)

        (
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            model_kwargs,
        ) = _model_forward_values(args, kwargs)

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
                positions,
            )
        hidden_states = run_model()
        if not self.enable_enpu and not wrapper_owns_full_graph_update:
            self._update_full_graph_params_if_needed(
                forward_context,
                num_tokens_padded,
                positions,
            )

        if (
            forward_context.flash_comm_v1_enabled
            and not forward_context.dbo_enabled
            and not isinstance(hidden_states, IntermediateTensors)
        ):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    def _build_attention_metadata(self, *args: Any, **kwargs: Any) -> Any:
        values = _attention_metadata_values(args, kwargs)
        ubatch_slices = _normalize_metadata_ubatch_slices(
            values.get("ubatch_slices"),
            values,
        )
        if ubatch_slices is not values.get("ubatch_slices"):
            args, kwargs = _replace_attention_metadata_ubatch_slices(
                args,
                kwargs,
                ubatch_slices,
            )
        if self.afd_async_extra_info.async_moe_ubatching:
            self.ubatch_slices = None
            return self._build_attention_metadata_with_async_moe_ubatches(
                args,
                kwargs,
                values,
            )
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            int(values.get("num_tokens", 0)),
        )
        self.ubatch_slices = ubatch_slices
        if ubatch_slices is not None:
            return self._build_attention_metadata_with_ubatches(*args, **kwargs)
        return super()._build_attention_metadata(*args, **kwargs)

    def _build_attention_metadata_with_async_moe_ubatches(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        values: dict[str, Any],
    ) -> Any:
        full_metadata = super()._build_attention_metadata(*args, **kwargs)
        self._afd_async_moe_ubatch_metadata = None
        self._afd_pending_metadata = self._build_afd_metadata(
            None,
            int(values.get("num_tokens", 0)),
        )

        num_scheduled_tokens_np = values.get("num_scheduled_tokens_np")
        if num_scheduled_tokens_np is None:
            return full_metadata

        num_tokens = int(values.get("num_tokens", 0))
        num_tokens_padded = int(values.get("num_tokens_padded") or num_tokens)
        use_sequence_parallel = bool(enable_sp(self.vllm_config))
        stages = plan_async_moe_stages(
            num_scheduled_tokens_np,
            split=self.afd_async_extra_info.async_moe_split,
            use_sequence_parallel=use_sequence_parallel,
            tensor_parallel_size=get_tensor_model_parallel_world_size(),
        )
        if stages is None:
            return full_metadata

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
            [
                (ubatch_slice.request_slice.start, ubatch_slice.request_slice.stop)
                for ubatch_slice in stages
            ],
            [
                (ubatch_slice.token_slice.start, ubatch_slice.token_slice.stop)
                for ubatch_slice in stages
            ],
            [int(stage.input_tokens) for stage in stages],
            [stage.actual_tokens for stage in stages],
        )

        stage_args, stage_kwargs = _replace_attention_metadata_ubatch_slices(
            args,
            kwargs,
            stages,
        )
        stage_attn_metadata, _ = self._build_attention_metadata_with_ubatches(
            *stage_args,
            metadata_builder_offset=ASYNC_MOE_STAGE_METADATA_BUILDER_OFFSET,
            async_moe_stages=stages,
            **stage_kwargs,
        )
        self._afd_pending_metadata = self._build_afd_metadata(
            stages,
            int(values.get("num_tokens", 0)),
        )
        self._afd_async_moe_ubatch_metadata = AsyncMoeUbatchMetadata(
            attn_metadata=stage_attn_metadata,
            stages=stages,
            use_sequence_parallel=use_sequence_parallel,
            parent_input_tokens=num_tokens_padded,
        )
        return full_metadata

    # Patch reason: CAMAsync needs independent metadata builders and must slice
    # real tokens separately from stage-local physical padding.
    # Patch functionality: reserve builder indices for CAMAsync stages and use
    # the plugin-owned stage splitter only when ``async_moe_stages`` is passed;
    # the default native DBO path remains byte-for-byte on the upstream splitter.
    # Signature: adds plugin-owned ``metadata_builder_offset`` and
    # ``async_moe_stages`` parameters to the pinned upstream signature.
    def _build_attention_metadata_with_ubatches(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: Any | None = None,
        logits_indices: Any | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        # ### PATCH START: isolate Async CAM stage metadata
        metadata_builder_offset: int = 0,
        async_moe_stages: tuple[AsyncMoeStage, ...] | None = None,
        # ### PATCH END: isolate Async CAM stage metadata
    ) -> tuple[Any, Any | None]:
        """Build isolated per-stage Ascend attention metadata.

        ``metadata_builder_offset`` reserves builder zero for full-batch
        metadata when Async CAM builds its two stage plans.
        ``async_moe_stages`` selects the Async CAM-only physical-padding
        splitter. Native DBO passes both defaults and retains upstream
        behavior.
        """

        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        assert ubatch_slices is not None
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata: Any = [dict() for _ in range(len(ubatch_slices))]

        if self._seq_lens_cpu_event_pending and self._seq_lens_cpu_event is not None:
            self._seq_lens_cpu_event.synchronize()
            self._seq_lens_cpu_event_pending = False

        if for_cudagraph_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_pcp_metadata(block_table_tensor):
            if not self.use_cp:
                return None, block_table_tensor
            return self.pcp_manager.generate_pcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
            )

        def _build_stage_local_pcp_metadata(
            common_attn_metadata: AscendCommonAttentionMetadata,
            ubatch_slice: UBatchSlice,
            ubid: int,
            kv_cache_gid: int,
            attn_gid: int,
        ) -> None:
            if not self.use_cp or int(self.pcp_size) <= 1:
                return
            if self.speculative_config is not None:
                raise RuntimeError(
                    "async_moe_ubatching with PCP does not support speculative "
                    "decode metadata yet",
                )
            if bool(getattr(self.pcp_manager, "pcp_use_hybrid_attn", False)):
                raise RuntimeError(
                    "async_moe_ubatching with PCP does not support hybrid "
                    "attention metadata yet",
                )

            full_num_scheduled_tokens = (
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs]
                .to("cpu")
                .numpy()
                .copy()
            )
            original_num_scheduled_tokens = full_num_scheduled_tokens[
                ubatch_slice.request_slice
            ].copy()
            original_token_start = int(
                full_num_scheduled_tokens[: ubatch_slice.request_slice.start].sum(),
            )
            original_token_stop = int(
                full_num_scheduled_tokens[: ubatch_slice.request_slice.stop].sum(),
            )
            original_common_summary = debug_pcp_common_metadata_summary(
                common_attn_metadata,
            )
            stage_num_reqs = ubatch_slice.request_slice.stop - (
                ubatch_slice.request_slice.start
            )
            manager_state = snapshot_pcp_manager_state(self.pcp_manager)
            try:
                self.pcp_manager.init_batch_info(
                    original_num_scheduled_tokens,
                    stage_num_reqs,
                )
                stage_pcp_tokens, _ = self.pcp_manager.update_tokens_for_pcp(
                    original_num_scheduled_tokens,
                    self.arange_np,
                )
                stage_query_lens = torch.from_numpy(stage_pcp_tokens).to(
                    self.query_lens.device,
                )
                if debug_pcp_metadata_enabled():
                    logger.warning(
                        "AFD PCP stage split input; kv_cache_gid=%s attn_gid=%s "
                        "ubid=%s request_slice=%s token_slice=%s "
                        "stage_num_reqs=%s original_num_scheduled_tokens=%s "
                        "stage_pcp_tokens=%s common=%s manager_before=%s",
                        kv_cache_gid,
                        attn_gid,
                        ubid,
                        debug_slice_summary(ubatch_slice.request_slice),
                        debug_slice_summary(ubatch_slice.token_slice),
                        stage_num_reqs,
                        debug_value_summary(original_num_scheduled_tokens),
                        debug_value_summary(stage_pcp_tokens),
                        original_common_summary,
                        debug_pcp_manager_summary(self.pcp_manager),
                    )
                pcp_metadata, block_table_tensor = (
                    self.pcp_manager.generate_pcp_metadata(
                        int(common_attn_metadata.num_actual_tokens),
                        stage_query_lens,
                        self.input_batch,
                        stage_pcp_tokens,
                        common_attn_metadata.block_table_tensor,
                        stage_num_reqs,
                        stage_num_reqs,
                    )
                )
                if original_token_stop > original_token_start:
                    raw_slot_mapping = self.input_batch.block_table[
                        kv_cache_gid
                    ].slot_mapping.gpu[original_token_start:original_token_stop]
                    stage_num_tokens = int(stage_pcp_tokens.sum())
                    common_attn_metadata.slot_mapping = (
                        self.pcp_manager.get_padded_slot_mapping(
                            stage_num_tokens,
                            stage_num_tokens,
                            raw_slot_mapping,
                            kv_cache_gid,
                        ).clone()
                    )
                common_attn_metadata.prefill_context_parallel_metadata = (
                    clone_pcp_metadata(pcp_metadata)
                )
                common_attn_metadata.block_table_tensor = block_table_tensor
                if debug_pcp_metadata_enabled():
                    logger.warning(
                        "AFD PCP stage split result; kv_cache_gid=%s attn_gid=%s "
                        "ubid=%s request_slice=%s token_slice=%s "
                        "pcp_metadata=%s block_table_tensor=%s common_after=%s "
                        "manager_after=%s",
                        kv_cache_gid,
                        attn_gid,
                        ubid,
                        debug_slice_summary(ubatch_slice.request_slice),
                        debug_slice_summary(ubatch_slice.token_slice),
                        debug_pcp_metadata_summary(pcp_metadata),
                        debug_value_summary(block_table_tensor),
                        debug_pcp_common_metadata_summary(common_attn_metadata),
                        debug_pcp_manager_summary(self.pcp_manager),
                    )
            finally:
                restore_pcp_manager_state(self.pcp_manager, manager_state)

        def _get_block_table_and_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if self.pcp_size > 1:
                total_num_pcp_pads = sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs])
                if self.pcp_manager.pcp_use_hybrid_attn:
                    num_scheduled_tokens_padded = (
                        self.pcp_manager.num_scheduled_tokens_padded
                    )
                    assert num_scheduled_tokens_padded is not None
                    maybe_pcp_full_tokens = (
                        sum(num_scheduled_tokens_padded) * self.pcp_size
                        - total_num_pcp_pads
                    )
                else:
                    maybe_pcp_full_tokens = (
                        num_tokens * self.pcp_size - total_num_pcp_pads
                    )
            else:
                maybe_pcp_full_tokens = num_tokens_padded
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
                slot_mapping = blk_table.slot_mapping.gpu[:maybe_pcp_full_tokens]
                maybe_num_reqs_padded = (
                    num_reqs_padded * self.decode_token_per_req
                    if self.use_cp
                    else num_reqs_padded
                )
                blk_table_tensor = blk_table.get_device_tensor()[:maybe_num_reqs_padded]
                if self.pcp_size == 1:
                    slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                    blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if self.pcp_size > 1:
                slot_mapping = self.pcp_manager.get_padded_slot_mapping(
                    num_tokens,
                    num_tokens_padded,
                    slot_mapping,
                    kv_cache_gid,
                )
            if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
                self.cpu_slot_mapping = slot_mapping.cpu().numpy()
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        self.long_seq_metadata, block_table_gid_0 = _get_pcp_metadata(
            block_table_gid_0,
        )
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        if self.use_async_spec_decode:
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            prefill_context_parallel_metadata=self.long_seq_metadata,
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices,
            )

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: AscendCommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            # ### PATCH START: isolate Async CAM stage metadata
            builder = attn_group.get_metadata_builder(
                metadata_builder_offset + (ubid or 0),
            )
            # ### PATCH END: isolate Async CAM stage metadata
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                patch_torch_npu_argsort()
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
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

            assert ubid is not None
            attn_metadata_dict = attn_metadata[ubid]
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

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
                # ### PATCH START: isolate Async CAM stage metadata
                builder = attn_group.get_metadata_builder(
                    metadata_builder_offset,
                )
                # ### PATCH END: isolate Async CAM stage metadata
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
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(
                    self.drafter,
                    AscendEagleProposer
                    | AscendDraftModelProposer
                    | AscendDflashProposer,
                ):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            if self.enable_hamming_sparse is True:
                build_kvcomp_metadata(self.kvcomp_meta_data, cm)
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                # ### PATCH START: isolate Async CAM stage metadata
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
                # ### PATCH END: isolate Async CAM stage metadata
                for ubid, ubatch_cm in enumerate(ubatch_common_metadata):
                    _build_stage_local_pcp_metadata(
                        ubatch_cm,
                        ubatch_slices[ubid],
                        ubid,
                        kv_cache_gid,
                        attn_gid,
                    )
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, ubatch_cm, ubid)

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
            for ub_metadata in attn_metadata:
                for metadata in ub_metadata.values():
                    metadata.mm_prefix_range = req_doc_ranges

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
        cudagraph_runtime_mode: Any | None = None,
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
        count_prof_step: bool = False,
    ) -> Any:
        if count_prof_step:
            step_afd_npu_profiler(self.prof)

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
        cudagraph_runtime_mode: Any | None = None,
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
    ) -> Any:
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

    def _warmup_and_capture(self, *args: Any, **kwargs: Any) -> Any:
        """Capture both single-stage and ubatched FFN graph keys.

        Native vLLM only captures the ubatched graph when microbatching is
        allowed for a decode capture size. Original AFD also captures the
        corresponding non-ubatched decode graph first, because live decode can
        still produce a single-stage key below the ubatch threshold.
        """

        names = [
            "desc",
            "cudagraph_runtime_mode",
            "profile_seq_lens",
            "allow_microbatching",
            "num_warmups",
        ]
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        desc = values.get("desc")
        cudagraph_runtime_mode = values.get("cudagraph_runtime_mode")
        if desc is None or cudagraph_runtime_mode is None:
            return super()._warmup_and_capture(*args, **kwargs)

        num_warmups = values.get("num_warmups")
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        allow_microbatching = bool(values.get("allow_microbatching", False))
        profile_seq_lens = values.get("profile_seq_lens")

        if allow_microbatching:
            self._afd_warmup_and_capture_once(
                desc=desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                profile_seq_lens=profile_seq_lens,
                allow_microbatching=False,
                num_warmups=int(num_warmups),
                cudagraph_mode_cls=CUDAGraphMode,
            )

        return self._afd_warmup_and_capture_once(
            desc=desc,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            profile_seq_lens=profile_seq_lens,
            allow_microbatching=allow_microbatching,
            num_warmups=int(num_warmups),
            cudagraph_mode_cls=CUDAGraphMode,
        )

    def _afd_warmup_and_capture_once(
        self,
        *,
        desc: Any,
        cudagraph_runtime_mode: Any,
        profile_seq_lens: int | None,
        allow_microbatching: bool,
        num_warmups: int,
        cudagraph_mode_cls: Any,
    ) -> Any:
        force_attention = cudagraph_runtime_mode == cudagraph_mode_cls.FULL

        previous_is_warmup = bool(self._is_warmup)
        try:
            self._is_warmup = True
            for _ in range(num_warmups):
                self._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=cudagraph_mode_cls.NONE,
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

            return self._dummy_run(
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

    def _dummy_run_with_ubatches(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: Any | None = None,
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
    ) -> Any:
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
        if self.use_cp:
            self.pcp_manager.init_batch_info(num_scheduled_tokens, num_reqs)
            if self.speculative_config:
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs] = torch.from_numpy(
                    num_scheduled_tokens,
                )
                self.pcp_manager.query_lens_pcp_full.copy_to_gpu()
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

        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata = None
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
                    6144
                    if is_graph_capturing
                    and using_paged_attention(num_tokens, self.vllm_config)
                    else max_query_len
                )

            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

            cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens,
                self.query_pos.np,
            )
            self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()
            if self._has_gdn:
                self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                self.gdn_query_start_loc.copy_to_gpu()

            if not profile_cpp:
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_padded,
                    num_reqs_padded,
                    num_reqs,
                    cudagraph_runtime_mode,
                    batch_desc.num_reqs,
                )

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens,
                num_tokens_padded,
                num_reqs_padded,
                self.vllm_config,
            )
            self.ubatch_slices = ubatch_slices_padded if pad_attn else ubatch_slices
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=self.ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                num_scheduled_tokens_np=num_scheduled_tokens,
            )
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

            if self.drafter:
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
                target = (
                    self.model.language_model
                    if hasattr(self.model, "language_model")
                    else self.model
                )
                target.clear_all_moe_loads()
            if self.dynamic_eplb:
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
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
        if bool(getattr(self, "_afd_suppress_metadata_send", False)):
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
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs[ASYNC_MOE_UBATCH_METADATA_KEY] = (
            self._afd_async_moe_ubatch_metadata
        )

    def _send_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
        ubatch_slices: Any,
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

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        result = super().load_model(*args, **kwargs)
        if bool(self.vllm_config.parallel_config.use_ubatching):
            self._install_ascend_ubatch_wrapper()
        return result

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

    def get_model(self) -> Any:
        if isinstance(self.model, AscendUBatchWrapper):
            return self.model.unwrap()
        return super().get_model()

    def initialize_attn_backend(self, *args: Any, **kwargs: Any) -> Any:
        result = super().initialize_attn_backend(*args, **kwargs)
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
        return result

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

    def _sync_metadata_across_dp(
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
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                self.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
            )
            return should_ubatch, num_tokens_padded, None, cudagraph_mode

        if self.connector.control_plane is None:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                self.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
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
        may_ubatch = bool(
            getattr(parallel_config, "enable_dbo", False)
            and getattr(parallel_config, "use_ubatching", False)
        )
        if can_skip_dp_sync and not may_ubatch:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * self.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                self.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
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

        moe_comm_type = select_moe_comm_method(
            max_tokens_across_dp,
            self.vllm_config,
            is_draft_model,
        )
        should_ubatch = check_enable_ubatch(
            min_tokens_across_dp,
            max_tokens_across_dp,
            uniform_decode=uniform_decode,
            vllm_config=self.vllm_config,
            moe_comm_type=moe_comm_type,
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
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, _, num_tokens_across_dp, synced_cudagraph_mode = (
                self._sync_metadata_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode,
                    allow_dp_padding=(cudagraph_mode != CUDAGraphMode.NONE)
                    or enable_sp(self.vllm_config),
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
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                self.vllm_config,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
                moe_comm_type=moe_comm_type,
            )
        if not allow_microbatching:
            should_ubatch = False

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

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: Any | None,
        sync_self: bool,
    ) -> Any:
        assert self.intermediate_tensors is not None
        tp = self.vllm_config.parallel_config.tensor_parallel_size

        if self.ubatch_slices is None:
            slice_len = (num_tokens + tp - 1) // tp if enable_sp() else num_tokens
        else:
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

        if sync_self:
            assert intermediate_tensors is not None
            copy_len = slice_len
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len],
                    non_blocking=True,
                )
        return IntermediateTensors(
            {k: v[:slice_len] for k, v in self.intermediate_tensors.items()},
        )

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
        values = metadata.num_tokens_across_dp_cpu
        tolist = getattr(values, "tolist", None)
        if callable(tolist):
            values = tolist()
        elif hasattr(values, "item"):
            values = [values.item()]
        try:
            values_tuple = tuple(int(value) for value in values)
        except TypeError:
            values_tuple = (int(values),)
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)


def _attention_metadata_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    values = dict(zip(_ATTENTION_METADATA_ARG_NAMES, args, strict=False))
    values.update(kwargs)
    return values


def _normalize_metadata_ubatch_slices(
    ubatch_slices: Any,
    values: dict[str, Any],
) -> Any:
    if not ubatch_slices:
        return ubatch_slices
    num_tokens_padded = values.get("num_tokens_padded")
    num_reqs_padded = values.get("num_reqs_padded")
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


def _replace_attention_metadata_ubatch_slices(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ubatch_slices: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    ubatch_index = _ATTENTION_METADATA_ARG_NAMES.index("ubatch_slices")
    if len(args) > ubatch_index:
        new_args = list(args)
        new_args[ubatch_index] = ubatch_slices
        return tuple(new_args), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["ubatch_slices"] = ubatch_slices
    return args, new_kwargs


def _model_forward_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    names = [
        "num_tokens_padded",
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
    ]
    values = dict(zip(names, args, strict=False))
    model_kwargs = dict(kwargs)
    for name in names:
        if name in model_kwargs:
            values[name] = model_kwargs.pop(name)
    return (
        values["num_tokens_padded"],
        values.get("input_ids"),
        values.get("positions"),
        values.get("intermediate_tensors"),
        values.get("inputs_embeds"),
        model_kwargs,
    )


_ATTENTION_METADATA_ARG_NAMES = [
    "num_tokens",
    "num_reqs",
    "max_query_len",
    "num_tokens_padded",
    "num_reqs_padded",
    "ubatch_slices",
    "logits_indices",
    "use_spec_decode",
    "for_cudagraph_capture",
    "num_scheduled_tokens",
    "num_scheduled_tokens_np",
    "cascade_attn_prefix_lens",
]

__all__ = ["AFDNPUAttentionModelRunner"]
