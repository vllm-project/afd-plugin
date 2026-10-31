# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for AFD features supported by the Ascend runtime."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    CAMP2P_CONNECTOR,
    AFDConfig,
    is_afd_async_dp,
    parse_afd_config,
)
from afd_plugin.envs import async_cam_layered_gmm_enabled
from afd_plugin.validation import cam_receive_capacity, requested_ffn_mode

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.connectors.base import ConnectorExtraInfo


def fail_if_unsupported_npu_afd_features(
    vllm_config: VllmConfig,
    *,
    afd_config: AFDConfig | None = None,
) -> None:
    """Fail fast for NPU AFD settings that are not currently supported."""

    afd_config = afd_config or parse_afd_config(vllm_config)
    layered_gmm_requested = async_cam_layered_gmm_enabled()
    if layered_gmm_requested and (
        afd_config.role != "ffn" or afd_config.connector != AFD_ASYNC_CONNECTOR
    ):
        raise RuntimeError(
            "AFD_ASYNC_CAM_LAYERED_GMM requires the async CAM FFN role; "
            f"got role={afd_config.role!r}, connector={afd_config.connector!r}"
        )
    is_dsv4 = _is_dsv4_target(vllm_config)
    if layered_gmm_requested and not is_dsv4:
        raise RuntimeError("AFD_ASYNC_CAM_LAYERED_GMM supports only DeepSeek V4")
    from afd_plugin.connectors.factory import AFDConnectorFactory

    extra_info = AFDConnectorFactory.parse_connector_extra_info(
        afd_config.connector,
        vllm_config,
    )

    if is_dsv4:
        _fail_if_unsupported_dsv4_connector(afd_config)

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        _fail_if_unsupported_npu_afd_async_features(
            vllm_config,
            afd_config,
            extra_info,
        )
        if is_dsv4:
            _fail_if_unsupported_dsv4_async_features(afd_config, extra_info)
        return

    if afd_config.compute_gate_on_attention:
        # The sync Attention-side gate prototype is deliberately narrow: one
        # connector, the target model and BF16 transport. Widening it without the
        # corresponding transport work would produce wrong numerics instead of an
        # error, so anything outside that shape stays rejected.
        _fail_if_unsupported_sync_attention_gate(
            vllm_config,
            afd_config,
            extra_info,
            is_dsv4=is_dsv4,
        )
    if afd_config.connector == CAMP2P_CONNECTOR:
        from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo

        if not isinstance(extra_info, CAMP2PExtraInfo):
            raise TypeError(
                "CAMP2pAFDConnector requires CAMP2PExtraInfo, got "
                f"{type(extra_info).__name__}",
            )
        extra_info.validate_supported()

    uses_ubatching = bool(vllm_config.parallel_config.use_ubatching)
    if uses_ubatching and int(vllm_config.parallel_config.num_ubatches) != 2:
        raise RuntimeError(
            "AFD NPU runtime supports exactly two ubatches when DBO is enabled",
        )
    model_config = vllm_config.model_config
    # Mirror the pinned NPUModelRunner's attention backend selection: a sparse
    # (SFA) or compressor (DSA) configuration selects a backend whose
    # graph-params update is a no-op, so only plain MLA takes the MLA DBO full
    # graph path that owns the merged registry.
    hf_text_config = model_config.hf_text_config
    uses_sparse_mla = hasattr(hf_text_config, "index_topk")
    uses_mla_compressor = hasattr(hf_text_config, "compress_ratios")
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    uses_mla_dbo_full_graph = (
        uses_ubatching
        and model_config.use_mla
        and not uses_sparse_mla
        and not uses_mla_compressor
        and cudagraph_mode.has_full_cudagraphs()
    )
    if uses_mla_dbo_full_graph and vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD NPU MLA DBO FULL graph does not support speculative decoding",
        )
    if uses_mla_dbo_full_graph and cudagraph_mode.name != "FULL_DECODE_ONLY":
        raise RuntimeError(
            "AFD NPU MLA DBO graph execution requires FULL_DECODE_ONLY",
        )


def _is_dsv4_target(vllm_config: VllmConfig) -> bool:
    """Return whether the target model is DeepSeek V4."""
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        return False
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        return True
    architectures = getattr(hf_config, "architectures", ()) or ()
    return any("DeepseekV4" in str(architecture) for architecture in architectures)


def _fail_if_unsupported_dsv4_async_features(
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    """Keep the initial DSV4 Async CAM target path deliberately narrow."""
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "DSV4 CAMAsyncAFDConnector requires compute_gate_on_attention=true",
        )
    if not isinstance(extra_info, AFDAsyncExtraInfo):
        raise TypeError(
            "DSV4 CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
            f"{type(extra_info).__name__}",
        )
    if extra_info.dynamic_quant != 1:
        raise RuntimeError(
            "DSV4 Flash-INT8 CAMAsyncAFDConnector requires dynamicQuant=1",
        )


def _fail_if_unsupported_dsv4_connector(afd_config: AFDConfig) -> None:
    """Reject DSV4 on connectors that cannot carry token ids.

    A DSV4 Hash layer routes by token identity, so whichever role owns the gate
    needs the tokens' ids. CAM async reads them from its dispatch metadata;
    CAMP2P moves them over the A2E ids channel with the gate left on FFN.
    """

    if afd_config.connector not in (AFD_ASYNC_CONNECTOR, CAMP2P_CONNECTOR):
        raise RuntimeError(
            "DSV4 NPU AFD supports only CAMAsyncAFDConnector and "
            f"CAMP2pAFDConnector; got {afd_config.connector!r}",
        )


def _fail_if_unsupported_sync_attention_gate(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: object,
    *,
    is_dsv4: bool,
) -> None:
    """Restrict the sync Attention-side gate prototype to what it supports today.

    DBO and ubatching are rejected rather than merely unused: the sync shell does not
    call the DBO yield, so enabling DBO here would interleave microbatches against a
    transport that never yields. Not calling the yield is not the same as forbidding
    the feature.
    """

    parallel = vllm_config.parallel_config
    # The routing travels on each rank's own activation, and only the TP=1 shape has
    # been verified end to end; a wider TP would split the activation without a
    # matching routing split on the Attention side.
    if int(parallel.tensor_parallel_size) != 1:
        raise RuntimeError(
            "Attention-side gate is only supported at tensor-parallel size 1; got "
            f"TP={int(parallel.tensor_parallel_size)}",
        )
    if bool(parallel.enable_dbo) or bool(parallel.use_ubatching):
        raise RuntimeError(
            "Attention-side gate with the synchronous connector does not support "
            "DBO or ubatching yet",
        )
    if afd_config.connector != CAMP2P_CONNECTOR:
        raise RuntimeError(
            "Attention-side gate is only supported with "
            f"{CAMP2P_CONNECTOR}, got {afd_config.connector!r}",
        )
    if not is_dsv4:
        raise RuntimeError(
            "Attention-side gate is only supported for the DeepSeek V4 target",
        )
    from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo

    if not isinstance(extra_info, CAMP2PExtraInfo):
        raise TypeError(
            "CAMP2pAFDConnector requires CAMP2PExtraInfo, got "
            f"{type(extra_info).__name__}",
        )


def _fail_if_unsupported_npu_afd_async_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not isinstance(extra_info, AFDAsyncExtraInfo):
        raise TypeError(
            "CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
            f"{type(extra_info).__name__}",
        )

    if vllm_config.additional_config.get("mix_placement", False):
        raise RuntimeError(
            "Async CAM uses routed-only expert IDs and does not support mix_placement"
        )

    parallel_config = vllm_config.parallel_config
    if extra_info.ffn_two_stream:
        if not afd_config.is_ffn_server:
            raise RuntimeError("ffn_two_stream may only be enabled on the FFN role")
        if (
            not vllm_config.use_v2_model_runner
            or not async_cam_layered_gmm_enabled()
            or not afd_config.compute_gate_on_attention
            or extra_info.dynamic_quant != 1
            or parallel_config.tensor_parallel_size != 1
            or parallel_config.use_sequence_parallel_moe
            or os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower()
            in ("1", "true")
        ):
            raise RuntimeError(
                "FFN two-stream requires MRV2 layered W4A8, dynamicQuant=1, "
                "Attention-side gate, F TP=1 and SP disabled"
            )
    if not is_afd_async_dp(vllm_config):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires additional_config['afd'] "
            "with async=true and connector='CAMAsyncAFDConnector'",
        )
    if not bool(vllm_config.model_config.enforce_eager):
        if afd_config.is_attention_server:
            raise RuntimeError(
                "CAMAsyncAFDConnector supports only eager Attention execution"
            )
        requested_ffn_mode(vllm_config)
        if not vllm_config.use_v2_model_runner:
            raise RuntimeError("Async FFN FULL requires MRV2")
        if parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("Async FFN FULL requires F TP=1")
    if bool(parallel_config.enable_dbo) or bool(parallel_config.use_ubatching):
        if not afd_config.is_attention_server:
            raise RuntimeError("Async CAM DBO/ubatching requires the Attention role")
        if parallel_config.num_ubatches != 2:
            raise RuntimeError("Async CAM DBO requires exactly two ubatches")
        if not afd_config.compute_gate_on_attention:
            raise RuntimeError("Async CAM DBO requires compute_gate_on_attention=true")
        if (
            parallel_config.pipeline_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
            or vllm_config.speculative_config is not None
        ):
            raise RuntimeError(
                "Async CAM DBO does not support PP, CP or speculative decoding"
            )
        if not _is_dsv4_target(vllm_config) and (
            vllm_config.model_config.hf_text_config.model_type
            not in {"deepseek_v2", "deepseek_v3"}
            or not vllm_config.model_config.use_mla
        ):
            raise RuntimeError(
                "Async CAM DBO requires DeepSeek-V2/V3 MLA or DeepSeek-V4"
            )
    speculative_config = vllm_config.speculative_config
    # Native draft MoE still uses Attention-side cross-DP collectives.
    if (
        afd_config.is_attention_server
        and speculative_config is not None
        and (speculative_config.use_eagle() or speculative_config.uses_draft_model())
    ):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support native draft models or MTP",
        )

    factor_value = os.environ.get("BATCH_SIZE_FACTOR", "1")
    try:
        # Python accepts underscores and Unicode digits that std::stof does not.
        if not factor_value.isascii() or "_" in factor_value:
            raise ValueError("Not a native floating-point literal")
        batch_size_factor = float(factor_value)
    except ValueError as exc:
        raise RuntimeError(
            "BATCH_SIZE_FACTOR must be a decimal number in (0, 1]",
        ) from exc
    if not 0 < batch_size_factor <= 1:
        raise RuntimeError("BATCH_SIZE_FACTOR must be a decimal number in (0, 1]")
    receive_capacity = cam_receive_capacity()
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # whole-expert chunks need a full TP group's batch; smaller
    # buffers require native sub-expert chunking.
    if receive_capacity < max_num_batched_tokens:
        raise RuntimeError(
            "CAMAsyncAFDConnector requires BATCH_SIZE_FACTOR to allocate at least "
            f"max_num_batched_tokens={max_num_batched_tokens} "
            f"rows; got {receive_capacity}",
        )
    if vllm_config.use_v2_model_runner and afd_config.is_attention_server:
        # CAM's per-rank window uses integer division, while replicated TP
        # input layout rounds up; reject capacities whose profile batch overflows.
        if (
            vllm_config.scheduler_config.max_num_batched_tokens
            % extra_info.attn_ranks_per_dp
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM requires max_num_batched_tokens "
                "divisible by attn_ranks_per_dp",
            )
        model_config = vllm_config.model_config
        if not _is_dsv4_target(vllm_config) and (
            model_config.hf_text_config.model_type not in {"deepseek_v2", "deepseek_v3"}
            or not model_config.use_mla
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM requires DeepSeek-V2/V3 MLA "
                "or DeepSeek-V4",
            )
        additional_config = vllm_config.additional_config
        if additional_config.get("enable_shared_expert_dp", False) or any(
            int(size) > 0
            for size in additional_config.get("finegrained_tp_config", {}).values()
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM does not support finegrained TP "
                "or shared-expert DP",
            )
    if extra_info.dynamic_quant not in (0, 1):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )
    _validate_cam_world_topology(vllm_config, afd_config, extra_info)


def _validate_cam_world_topology(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    """Require each role's local layout to fill the one CAM world."""
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not isinstance(extra_info, AFDAsyncExtraInfo):
        return
    parallel_config = vllm_config.parallel_config
    attn_ranks_per_dp = int(extra_info.attn_ranks_per_dp)
    if afd_config.role == "attention":
        if int(parallel_config.tensor_parallel_size) != attn_ranks_per_dp:
            raise RuntimeError(
                "CAMAsyncAFDConnector Attention tensor_parallel_size must equal "
                "attn_ranks_per_dp",
            )
        local_world_size = int(parallel_config.data_parallel_size) * attn_ranks_per_dp
        expected_world_size = afd_config.num_attention_ranks
    else:
        local_world_size = int(parallel_config.data_parallel_size) * int(
            parallel_config.tensor_parallel_size
        )
        expected_world_size = afd_config.num_ffn_ranks
    if local_world_size != expected_world_size:
        raise RuntimeError(
            "CAMAsyncAFDConnector "
            f"{afd_config.role} DPxTP world size must equal its configured "
            f"role size, got {local_world_size} and {expected_world_size}",
        )


__all__ = ["fail_if_unsupported_npu_afd_features"]
