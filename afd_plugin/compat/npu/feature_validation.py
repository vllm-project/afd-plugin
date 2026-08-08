# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for AFD features supported by the Ascend runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    AFDConfig,
    is_afd_async_dp,
    parse_afd_config,
)

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
    from afd_plugin.connectors.factory import AFDConnectorFactory

    extra_info = AFDConnectorFactory.parse_connector_extra_info(
        afd_config.connector,
        vllm_config,
    )

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        _fail_if_unsupported_npu_afd_async_features(
            vllm_config,
            afd_config,
            extra_info,
        )
        return

    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )
    if afd_config.connector == "CAMP2pAFDConnector":
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
    # Match the pinned NPUModelRunner's sparse-attention backend selection.
    uses_sparse_mla = hasattr(
        model_config.hf_text_config,
        "index_topk",
    )
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    uses_mla_dbo_full_graph = (
        uses_ubatching
        and model_config.use_mla
        and not uses_sparse_mla
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

    parallel_config = vllm_config.parallel_config
    if not is_afd_async_dp(vllm_config):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires additional_config['afd'] "
            "with async=true and connector='CAMAsyncAFDConnector'",
        )
    if not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "CAMAsyncAFDConnector supports only eager Attention/FFN execution",
        )
    if bool(parallel_config.enable_dbo) or bool(parallel_config.use_ubatching):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support vLLM native ubatching/DBO",
        )
    if extra_info.async_moe_ubatching:
        _fail_if_unsupported_npu_async_moe_ubatching_features(
            vllm_config,
            afd_config,
            num_ubatches=extra_info.async_moe_num_ubatches,
            split=extra_info.async_moe_split,
            attn_ranks_per_dp=extra_info.attn_ranks_per_dp,
        )
    if extra_info.dynamic_quant not in (0, 1):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )


def _fail_if_unsupported_npu_async_moe_ubatching_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    num_ubatches: int,
    split: str,
    attn_ranks_per_dp: int,
) -> None:
    from afd_plugin.connectors.npu.async_cam import (
        ASYNC_MOE_NUM_STAGES,
        ASYNC_MOE_REQUEST_SPLIT,
        ASYNC_MOE_TOKEN_SPLIT,
    )

    parallel_config = vllm_config.parallel_config
    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "async_moe_ubatching requires compute_gate_on_attention=true",
        )
    if num_ubatches != ASYNC_MOE_NUM_STAGES:
        raise RuntimeError(
            "async_moe_ubatching currently supports exactly two stages; "
            f"got async_moe_num_ubatches={num_ubatches}",
        )
    if split not in (ASYNC_MOE_REQUEST_SPLIT, ASYNC_MOE_TOKEN_SPLIT):
        raise RuntimeError(
            "async_moe_split must be 'request' or 'token'; "
            f"got async_moe_split={split!r}",
        )
    # Attention owns stage planning and SP layout conversion. FFN workers
    # consume CAM work items and may use an independent TP/EP topology.
    if afd_config.is_ffn_server:
        return
    if (
        int(parallel_config.prefill_context_parallel_size) > 1
        or int(parallel_config.decode_context_parallel_size) > 1
    ):
        raise RuntimeError(
            "async_moe_ubatching does not support context parallelism",
        )
    if attn_ranks_per_dp != int(parallel_config.tensor_parallel_size):
        raise RuntimeError(
            "async_moe_ubatching requires Attention attn_ranks_per_dp to "
            "equal tensor_parallel_size",
        )
    if (
        split == ASYNC_MOE_TOKEN_SPLIT
        and int(parallel_config.tensor_parallel_size) <= 1
    ):
        raise RuntimeError(
            "async_moe_split='token' requires Attention DP+TP/SP with "
            "tensor_parallel_size > 1",
        )


__all__ = ["fail_if_unsupported_npu_afd_features"]
