# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA AFD wrapper for the NVIDIA DeepSeek-V4 implementation.

The split is placed immediately around each decoder FFN. Attention retains all
mHC residual-stream state; only the normalized two-dimensional FFN activation
and token-aligned input IDs cross the Attention-to-FFN boundary. The returned
FFN activation is the sole FFN-to-Attention tensor.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.models.deepseek_v4.nvidia import model as native
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import WorkspaceManager, current_workspace_manager

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from afd_plugin.v1.worker.attention_model_runner_v2 import AFDAttentionModelRunnerV2

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_BOTH_ROLES = frozenset(("attention", "ffn"))
_FLASHMLA_PROFILE_TOKENS = 1
# V4 CuTe compressor positions/slot pointers require eight int64 elements.
_V4_INPUT_ALIGNMENT_ELEMENTS = 8


class _AFDV4UBatchInputs:
    """Stable aligned prefixes shared by every layer in each V4 stage."""

    def __init__(
        self,
        input_buffers: InputBuffers,
        native_slot_mappings: torch.Tensor,
        num_ubatches: int,
    ) -> None:
        capacity = input_buffers.max_num_tokens
        num_kv_groups = native_slot_mappings.shape[0]
        pitch = (
            (capacity + _V4_INPUT_ALIGNMENT_ELEMENTS - 1)
            // _V4_INPUT_ALIGNMENT_ELEMENTS
            * _V4_INPUT_ALIGNMENT_ELEMENTS
        )
        # Native dummy and live inputs share these buffers, and stage 0 starts
        # at token 0. Choose slot borrowing once before graph capture.
        self.positions = (input_buffers.positions,) + tuple(
            input_buffers.positions.new_empty((capacity,))
            for _ in range(1, num_ubatches)
        )
        borrow_first_slots = (
            num_kv_groups <= 1
            or native_slot_mappings.stride(0) % _V4_INPUT_ALIGNMENT_ELEMENTS == 0
        )
        self.slot_mappings = tuple(
            None
            if stage == 0 and borrow_first_slots
            else input_buffers.positions.new_empty((num_kv_groups, pitch))
            for stage in range(num_ubatches)
        )


@contextmanager
def _use_v4_ubatch_preparation(
    runner: "AFDAttentionModelRunnerV2",
) -> Iterator[None]:
    """Prepare CUDA V4 stage inputs and indexer plans for native ubatching."""
    ubatch_runner = runner.ubatch_runner
    assert ubatch_runner is not None
    if runner._afd_v4_ubatch_inputs is None:
        runner._afd_v4_ubatch_inputs = _AFDV4UBatchInputs(
            runner.input_buffers,
            runner.block_tables.slot_mappings,
            ubatch_runner.num_ubatches,
        )
    inputs = runner._afd_v4_ubatch_inputs
    model_state = runner.model_state
    state = vars(model_state)
    had_prepare = "prepare_attn" in state
    previous_prepare = state.get("prepare_attn")
    # V4 uses the pinned DefaultModelState, whose concrete signature includes
    # the optional model-specific metadata argument absent on its base class.
    original_prepare_attn = cast(DefaultModelState, model_state).prepare_attn
    ubatch_state = vars(ubatch_runner)
    had_ubatch_prepare = "prepare" in ubatch_state
    previous_ubatch_prepare = ubatch_state.get("prepare")
    original_ubatch_prepare = ubatch_runner.prepare
    had_begin = "begin_capturable_run" in ubatch_state
    previous_begin = ubatch_state.get("begin_capturable_run")
    original_begin = ubatch_runner.begin_capturable_run

    # Patch reason: native ubatch views can start at a non-64-byte boundary.
    # Patch functionality: retain native stage-zero inputs where their fixed
    # layout is aligned; stage other inputs before native metadata builders.
    # Stable addresses are refreshed before FULL replay as well as eager/capture.
    # Signature: matches DefaultModelState.prepare_attn at vLLM 5df63a7,
    # vllm/v1/worker/gpu/model_states/default.py. Delegation retains native
    # metadata construction; remove when native stage buffers meet V4's ABI.
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        # ### PATCH START: stable aligned V4 stage metadata inputs.
        if ubatch_idx:
            positions = inputs.positions[ubatch_idx][: input_batch.positions.shape[0]]
            positions.copy_(input_batch.positions)
            input_batch = replace(input_batch, positions=positions)
        slots = inputs.slot_mappings[ubatch_idx]
        if slots is None:
            slots = slot_mappings
        else:
            slots = slots[:, : slot_mappings.shape[1]]
            slots.copy_(slot_mappings)
        return original_prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slots,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        # ### PATCH END: stable aligned V4 stage metadata inputs.

    # Patch reason: native model-input slicing recreates the unaligned view.
    # Patch functionality: bind the model call to the same persistent positions
    # prepared for this stage; native threads, streams and finish stay intact.
    # Signature: matches UBatchRunner.begin_capturable_run at vLLM 5df63a7,
    # vllm/v1/worker/gpu/ubatch_utils.py. The callable accepts the native open
    # model-input dictionary. Remove with the aligned V4 preparation adapter.
    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        # ### PATCH START: bind V4 model positions without changing scheduling.
        def aligned_model(**model_kwargs: Any) -> Any:
            stage = dbo_current_ubatch_id()
            model_kwargs["positions"] = inputs.positions[stage][
                : ubatch_state.slices[stage].num_tokens
            ]
            return model(**model_kwargs)

        return original_begin(
            aligned_model, model_inputs, ubatch_state, for_capture=for_capture
        )
        # ### PATCH END: bind V4 model positions without changing scheduling.

    # Patch reason: native preparation uses physical-SM indexer plans and
    # rebuilds context slots from potentially unaligned microbatch views.
    # Patch functionality: lease the UBatchRunner compute budget to the native
    # indexer builders, restore it even on failure, and bind aligned live/capture
    # slots. Native slicing, metadata construction and FULL replay stay native.
    # Signature: matches UBatchRunner.prepare in vllm/v1/worker/gpu/ubatch_utils.py
    # at vLLM 5df63a7203cdb9328d824a27b682e77c7b9928be; indexer.py at the same pin
    # owns the physical-capacity buffers and planner refill. Delegation avoids
    # copying the full native preparation. Remove when native propagates its SM
    # budget and provides stable aligned V4 stage inputs.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: V4 stage input and indexer preparation.
        indexer_budgets: list[tuple[DeepseekV32IndexerMetadataBuilder, int]] = []
        try:
            for groups in self.attn_groups:
                for group in groups:
                    for stage in range(self.num_ubatches):
                        builder = group.get_metadata_builder(stage)
                        if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                            indexer_budgets.append((builder, builder.num_sms))
                            builder.num_sms = self.sm_control.compute_sms
            prepared = original_ubatch_prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
        finally:
            # Reverse order also restores correctly if groups share a builder.
            for builder, num_sms in reversed(indexer_budgets):
                builder.num_sms = num_sms
        # FULL replay uses the context captured by the graph, not this fresh one.
        if cg_mode != CUDAGraphMode.FULL or for_capture:
            for stage, context in enumerate(prepared.forward_contexts):
                slots = inputs.slot_mappings[stage]
                if slots is not None:
                    context.slot_mapping = build_slot_mappings_by_layer(
                        slots[:, : prepared.slices[stage].num_tokens],
                        runner.kv_cache_config,
                    )
        return prepared
        # ### PATCH END: V4 stage input and indexer preparation.

    model_state.prepare_attn = MethodType(prepare_attn, model_state)
    ubatch_runner.begin_capturable_run = MethodType(begin_capturable_run, ubatch_runner)
    ubatch_runner.prepare = MethodType(prepare, ubatch_runner)
    try:
        yield
    finally:
        if had_ubatch_prepare:
            ubatch_runner.prepare = previous_ubatch_prepare
        else:
            del ubatch_runner.prepare
        if had_begin:
            ubatch_runner.begin_capturable_run = previous_begin
        else:
            del ubatch_runner.begin_capturable_run
        if had_prepare:
            model_state.prepare_attn = previous_prepare
        else:
            del model_state.prepare_attn


@contextmanager
def _use_flashmla_workspace_slot(
    manager: WorkspaceManager, slot: int
) -> Iterator[None]:
    """Select one real workspace owner only during startup reservation."""
    state = vars(manager)
    had_override = "_get_workspace_id" in state
    previous_override = state.get("_get_workspace_id")

    # Patch reason: native pre-KV profiling has no UBatch context and no public
    # workspace-slot selector. A real UBatch context would start handoff threads.
    # Patch functionality: route only this manager's allocation-only reservation
    # to a configured slot, then restore its exact instance/class lookup.
    # Signature: matches WorkspaceManager._get_workspace_id(self) -> int in
    # vllm/v1/worker/workspace.py, 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Remove this pin-local exception when native offers per-owner reservation.
    def _get_workspace_id(self) -> int:
        return slot

    # ### PATCH START: select one startup workspace slot without DBO threads.
    manager._get_workspace_id = MethodType(_get_workspace_id, manager)
    try:
        yield
    finally:
        if had_override:
            manager._get_workspace_id = previous_override
        else:
            del manager._get_workspace_id
    # ### PATCH END: select one startup workspace slot without DBO threads.


def _reserve_flashmla_ubatch_workspace(model: nn.Module, config: VllmConfig) -> None:
    """Reserve native V4 prefill scratch before the normal profile peak."""
    attention_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, DeepseekV4FlashMLAAttention)
    ]
    if not attention_layers:
        return
    manager = current_workspace_manager()
    # Native forward_mqa's NoMeta branch (nvidia/flashmla.py:105-131, vLLM
    # 5df63a7203cdb9328d824a27b682e77c7b9928be) reserves its configured bound,
    # zeros only this small output, and returns before KV/kernels/Indexer/FFN.
    # Visit all real layers: the first SWA-only layer does not bound C4 demand.
    # Known manager geometry is the pin-local exception above, not a new policy.
    with set_forward_context(None, config):
        for layer in attention_layers:
            q = layer.attn_sink.new_empty(
                (_FLASHMLA_PROFILE_TOKENS, layer.padded_heads, layer.head_dim),
                dtype=torch.bfloat16,
            )
            output = torch.empty_like(q)
            kv = q.new_empty((_FLASHMLA_PROFILE_TOKENS, layer.head_dim))
            positions = torch.empty(
                (_FLASHMLA_PROFILE_TOKENS,), dtype=torch.int64, device=q.device
            )
            for slot in range(manager._num_ubatches * manager._num_lanes):
                with _use_flashmla_workspace_slot(manager, slot):
                    layer.forward_mqa(q, kv, positions, output)


def _weight_layer_path(name: str) -> tuple[int, str] | None:
    """Extract the decoder layer index and first layer-local path component."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2]
    return None


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify a native DeepSeek-V4 checkpoint path by execution owner."""
    if name in {
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
    }:
        return _ATTENTION_ROLE
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        return _BOTH_ROLES
    _, stage = layer_path
    if stage == "ffn":
        return _FFN_ROLE
    return _ATTENTION_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only role-owned paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class RemoteDeepseekV4FFN(nn.Module):
    """Parameter-free FFN proxy carrying V4 hash-router token identifiers."""

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 remote FFN requires input_ids")
        if input_ids.ndim != 1 or input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "DeepSeek-V4 input_ids must be one-dimensional and token-aligned",
            )

        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("RemoteDeepseekV4FFN requires AFD forward metadata")
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=self.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            input_ids=input_ids,
        )
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        return afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )


class AFDDeepseekV4DecoderLayer(native.DeepseekV4DecoderLayer):
    """DeepSeek-V4 decoder layer with an FFN-boundary synchronous split."""

    # Patch reason: native DeepSeek-V4 always constructs Attention and FFN.
    # Patch functionality: allocate only the stage owned by the active AFD role
    # and pin sequence parallelism off (the native derivation would enable SP
    # for AFD DP>1 layouts, which the synchronous AFD boundary does not
    # support).
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        # ### PATCH START: construct a role-local decoder stage.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        layer_idx = int(prefix.rsplit(".", maxsplit=1)[-1])
        # Native derives SP from the EP/TP/DP layout; AFD splits roles across
        # dedicated workers and requires full-token boundary payloads.
        self.use_sequence_parallel = False
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # ### PATCH START: replace the remote stage with a parameter-free proxy.
        if afd_config.role == "attention":
            self.attn = native._select_dsv4_attn_cls(vllm_config)(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.ffn = RemoteDeepseekV4FFN(layer_idx=layer_idx)
        elif afd_config.role == "ffn":
            self.attn = native.PPMissingLayer()
            self.ffn = native.DeepseekV4MoE(
                vllm_config,
                prefix=f"{prefix}.ffn",
                use_sequence_parallel=self.use_sequence_parallel,
                num_hash_layers=config.num_hash_layers,
            )
        else:
            raise ValueError(f"unsupported AFD role {afd_config.role!r}")
        # ### PATCH END

        # ### PATCH START: mHC state and normalization are Attention-owned.
        if afd_config.role == "ffn":
            return
        # ### PATCH END
        self.attn_norm = native.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = native.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        if vllm_config.kernel_config.enable_jit_warmup:
            # MHC pre/post kernels execute only on the Attention role; the FFN
            # role returns above without mHC state, so it skips the warmup.
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                _HC_PRENORM_GEMM_TILELANG_KERNEL,
                _MHC_FUSED_TILELANG_KERNEL,
                _MHC_POST_TILELANG_KERNEL,
                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
            )
            from vllm.utils.deep_gemm import is_deep_gemm_supported

            include_pre_gemm_splits = is_deep_gemm_supported()
            _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
                vllm_config,
                hidden_size=self.hidden_size,
                hc_mult=self.hc_mult,
                use_norm_weight=True,
                include_pre_gemm_splits=include_pre_gemm_splits,
                include_broadcast_splits=(
                    native.get_pp_group().is_first_rank and layer_idx == 0
                ),
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=self.hc_post_alpha,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                norm_eps=(
                    self.attn_norm.variance_epsilon,
                    self.ffn_norm.variance_epsilon,
                ),
                broadcast_norm_eps=self.attn_norm.variance_epsilon,
            )
            if not include_pre_gemm_splits:
                _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.hc_mult,
                    n_out=self.hc_mult * (2 + self.hc_mult),
                )
            _MHC_POST_TILELANG_KERNEL.register_warmup(
                hidden_size=self.hidden_size,
                hc_mult=self.hc_mult,
            )
            _MHC_FUSED_TILELANG_KERNEL.register_warmup(
                vllm_config,
                hidden_size=self.hidden_size,
                hc_mult=self.hc_mult,
            )

    # Patch reason: native forward directly invokes its locally allocated FFN.
    # Patch functionality: preserve native mHC state locally while the proxy
    # transfers only the two-dimensional FFN activation and input IDs. The
    # v0.32.0 native sequence-parallel gather/scatter branches around the
    # attention call are omitted because AFD pins use_sequence_parallel off.
    # Padding-only AFD ubatches use zero attention output while retaining all
    # mHC and remote FFN work. Native FlashMLA's zero-token decode metadata has
    # no scheduler tile, so entering its forward consumer would assert.
    # Removal plan: drop this branch when native V4 attention supports zero
    # actual tokens in a nonempty padded ubatch. Capture retains full attention
    # because native dummy inputs publish a positive real extent for each stage.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        mega_gate_metadata: native.MegaGateRoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ### PATCH START: prohibit accidental execution on the FFN worker.
        if isinstance(self.attn, native.PPMissingLayer):
            raise RuntimeError("DeepSeek-V4 decoder forward is Attention-owned")
        # ### PATCH END
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            if x.dim() == 2:
                assert self.hc_attn_fn_broadcast is not None
                residual, post_mix, res_mix, x = native.mhc_pre_broadcast_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                    fn_broadcast=self.hc_attn_fn_broadcast,
                )
            else:
                residual = x
                post_mix, res_mix, x = native.mhc_pre_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                )
        else:
            residual, post_mix, res_mix, x = native.mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        # ### PATCH START: preserve padded-stage protocol with zero attention.
        afd_metadata = get_afd_metadata_from_forward_context()
        if (
            afd_metadata is not None
            and afd_metadata.num_stages > 1
            and afd_metadata.tokens_unpadded_lens == [0]
        ):
            x = torch.zeros_like(x)
        else:
            x = self.attn(positions, x, None)
        # ### PATCH END: preserve padded-stage protocol with zero attention.
        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = native.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )

        # ### PATCH START: this call enters the synchronous remote FFN proxy.
        # MegaMoE is excluded, so only activations and token IDs cross the wire.
        x = self.ffn(x, input_ids)
        # ### PATCH END
        return x, residual, post_mix, res_mix

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute the complete native V4 MoE, including its native router."""
        if not isinstance(self.ffn, native.DeepseekV4MoE):
            raise RuntimeError("DeepSeek-V4 FFN compute is FFN-role only")
        if input_ids is None:
            raise RuntimeError("DeepSeek-V4 FFN compute requires input_ids")
        return self.ffn(hidden_states, input_ids)


class AFDDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware DeepSeek-V4 model retaining mHC exclusively on Attention."""

    # Patch reason: native DeepSeek-V4 allocates every decoder stage and stream.
    # Patch functionality: build role-aware layers and Attention-only resources
    # and pin sequence parallelism off (the native derivation would enable SP
    # for AFD DP>1 layouts, which the synchronous AFD boundary does not
    # support).
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.32.0, vllm/models/deepseek_v4/nvidia/model.py
    # Commit: 5df63a7203cdb9328d824a27b682e77c7b9928be
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: validate the deliberately narrow first release.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        if native.current_platform.device_type != "cuda":
            raise RuntimeError("AFD DeepSeek-V4 supports CUDA only")
        if self.afd_config.connector != "P2pNcclAFDConnector":
            raise RuntimeError(
                "AFD DeepSeek-V4 requires the synchronous P2pNcclAFDConnector",
            )
        if self.afd_config.compute_gate_on_attention:
            raise RuntimeError(
                "AFD DeepSeek-V4 does not support compute_gate_on_attention",
            )
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("AFD DeepSeek-V4 does not support PP")
        if parallel_config.use_sequence_parallel_moe:
            raise RuntimeError("AFD DeepSeek-V4 does not support SP MoE")
        if parallel_config.enable_eplb:
            raise RuntimeError("AFD DeepSeek-V4 does not support EPLB")
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_native_mega_moe = (
            vllm_config.kernel_config.moe_backend in native.NATIVE_MEGA_MOE_BACKENDS
        )
        # ### PATCH START: MegaMoE has unproven role-local finalization semantics.
        if vllm_config.kernel_config.moe_backend in native.MEGA_MOE_BACKENDS:
            raise RuntimeError("AFD DeepSeek-V4 does not support MegaMoE")
        # Native derives SP from the EP/TP/DP layout; AFD splits roles across
        # dedicated workers and requires full-token boundary payloads.
        self.use_sequence_parallel = False
        # ### PATCH END
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # ### PATCH START: CUDA streams and sparse-index buffers are Attention-owned.
        if self.afd_config.role == "attention":
            aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
            )
        else:
            aux_stream_list = None
            self.topk_indices_buffer = None
        # ### PATCH END

        if native.get_pp_group().is_first_rank or native.spec_decode_needs_target_embed(
            vllm_config
        ):
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        # ### PATCH START: use the pinned role-aware layer constructor.
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        # ### PATCH END
        self.has_local_hash_moe = self.start_layer < min(
            self.end_layer, config.num_hash_layers
        )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        # ### PATCH START: final mHC state is constructed only on Attention.
        if self.afd_config.role == "attention":
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if (
            self.afd_config.role == "attention"
            and native.get_pp_group().is_last_rank
            and needs_mtp_hidden_states
        ):
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None
        if (
            self.afd_config.role == "attention"
            and vllm_config.kernel_config.enable_jit_warmup
            and native.get_pp_group().is_last_rank
        ):
            # The hc-head kernel runs only on the Attention role; the FFN role
            # never finalizes the mHC stream, so it skips the warmup.
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                _HC_HEAD_FUSED_TILELANG_KERNEL,
            )

            _HC_HEAD_FUSED_TILELANG_KERNEL.register_warmup(
                hidden_size=config.hidden_size,
                hc_mult=self.hc_mult,
                rms_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
            )
        # ### PATCH END

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(
            hidden_states,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(int(self.config.num_hidden_layers)))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return native expert mappings only where real experts are owned."""
        if self.afd_config.role == "attention":
            return []
        return super().get_expert_mapping()

    def finalize_mega_moe_weights(self) -> None:
        """MegaMoE is rejected before allocation, so no finalizer is needed."""

    def finalize_mhc_broadcast_weights(self) -> None:
        """Finalize only the Attention-owned first-layer broadcast matrix."""
        if self.afd_config.role == "ffn":
            return
        if (
            not native.get_pp_group().is_first_rank
            or self.start_layer >= self.end_layer
        ):
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, AFDDeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


class AFDDeepseekV4ForCausalLM(native.DeepseekV4ForCausalLM):
    """DeepSeek-V4 causal LM exposing the GPU FFN-runner model contract."""

    model_cls = AFDDeepseekV4Model
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(
            hidden_states,
            layer_idx,
            input_ids=input_ids,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(weights, role=self.afd_role),
        )


__all__ = ["AFDDeepseekV4ForCausalLM"]
