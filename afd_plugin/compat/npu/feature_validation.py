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


MAX_DSV4_AFD_SPECULATIVE_TOKENS = 3


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

    if _is_deepseek_v4(vllm_config):
        _fail_if_unsupported_deepseek_v4_features(vllm_config, afd_config)

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        _fail_if_unsupported_npu_afd_async_features(
            vllm_config,
            afd_config,
            extra_info,
        )
        return

    if (
        afd_config.compute_gate_on_attention
        and afd_config.connector != "WindowAFDConnector"
    ):
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
    if bool(parallel_config.use_ubatching):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support vLLM native ubatching/DBO",
        )
    if extra_info.async_moe_ubatching:
        _fail_if_unsupported_npu_async_moe_ubatching_features(
            vllm_config,
            afd_config,
            num_ubatches=extra_info.async_moe_num_ubatches,
            split=extra_info.async_moe_split,
        )
    if extra_info.dynamic_quant not in (0, 1):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )


def _is_deepseek_v4(vllm_config: VllmConfig) -> bool:
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    return any(
        architecture in {"DeepseekV4ForCausalLM", "AFDDeepseekV4ForCausalLM"}
        for architecture in architectures
    )


def _fail_if_unsupported_deepseek_v4_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Keep DSV4 AFD inside its validated eager and graph feature boxes."""
    from vllm_ascend.utils import is_dspark_config

    parallel_config = vllm_config.parallel_config
    supported_connectors = {
        "CAMP2pAFDConnector",
        "P2pHcclAFDConnector",
        "WindowAFDConnector",
    }
    if afd_config.connector not in supported_connectors:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only CAMP2pAFDConnector, "
            "P2pHcclAFDConnector, or WindowAFDConnector"
        )
    if (
        afd_config.connector == "CAMP2pAFDConnector"
        and afd_config.num_attention_ranks != afd_config.num_ffn_ranks
    ):
        raise RuntimeError(
            "DeepSeek-V4 CAMP2pAFDConnector requires equal Attention and FFN ranks"
        )
    tensor_parallel_size = int(parallel_config.tensor_parallel_size)
    if tensor_parallel_size not in (1, 2):
        raise RuntimeError("DeepSeek-V4 AFD supports only tensor_parallel_size=1 or 2")
    if tensor_parallel_size == 2:
        if afd_config.connector != "P2pHcclAFDConnector":
            raise RuntimeError("DeepSeek-V4 AFD TP2 supports only P2pHcclAFDConnector")
        if afd_config.num_attention_ranks != afd_config.num_ffn_ranks:
            raise RuntimeError(
                "DeepSeek-V4 AFD TP2 currently requires equal Attention and FFN ranks"
            )
        role_ranks = (
            afd_config.num_attention_ranks
            if afd_config.role == "attention"
            else afd_config.num_ffn_ranks
        )
        expected_role_ranks = (
            int(parallel_config.data_parallel_size) * tensor_parallel_size
        )
        if role_ranks != expected_role_ranks:
            raise RuntimeError(
                "DeepSeek-V4 AFD TP2 requires role ranks to equal DP x TP: "
                f"role={afd_config.role}, ranks={role_ranks}, "
                f"DP={int(parallel_config.data_parallel_size)}, "
                f"TP={tensor_parallel_size}"
            )
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("DeepSeek-V4 AFD supports only pipeline_parallel_size=1")
    if parallel_config.prefill_context_parallel_size != 1:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only prefill context parallel size 1"
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only decode context parallel size 1"
        )
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError("DeepSeek-V4 AFD does not support sequence-parallel MoE")
    if (
        afd_config.compute_gate_on_attention
        and afd_config.connector != "WindowAFDConnector"
    ):
        raise RuntimeError("DeepSeek-V4 AFD requires FFN-side gate computation")
    if afd_config.connector == "WindowAFDConnector":
        from vllm_ascend.ascend_config import get_ascend_config

        if not afd_config.compute_gate_on_attention:
            raise RuntimeError(
                "DeepSeek-V4 Window AFD requires compute_gate_on_attention=true"
            )
        role_ranks = (
            afd_config.num_attention_ranks
            if afd_config.role == "attention"
            else afd_config.num_ffn_ranks
        )
        if role_ranks != int(parallel_config.data_parallel_size):
            raise RuntimeError(
                "DeepSeek-V4 Window AFD requires the current role rank count "
                "to equal data_parallel_size: "
                f"role={afd_config.role}, ranks={role_ranks}, "
                f"DP={int(parallel_config.data_parallel_size)}"
            )
        if parallel_config.enable_eplb:
            raise RuntimeError("DeepSeek-V4 Window AFD does not support EPLB")
        if bool(getattr(get_ascend_config(), "mix_placement", False)):
            raise RuntimeError(
                "DeepSeek-V4 Window AFD with a dedicated shared-expert rank "
                "requires mix_placement=false"
            )
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None:
        uses_dspark = is_dspark_config(vllm_config)
        if uses_dspark and afd_config.role != "attention":
            raise RuntimeError(
                "DeepSeek-V4 AFD DSpark speculative config belongs only to "
                "the Attention service"
            )
        if not uses_dspark and afd_config.connector != "P2pHcclAFDConnector":
            raise RuntimeError("DeepSeek-V4 AFD MTP supports only P2pHcclAFDConnector")
        if getattr(speculative_config, "method", None) != "mtp":
            raise RuntimeError("DeepSeek-V4 AFD supports only MTP speculative method")
        draft_enforce_eager = bool(getattr(speculative_config, "enforce_eager", False))
        target_enforce_eager = bool(vllm_config.model_config.enforce_eager)
        if uses_dspark:
            draft_hf_config = speculative_config.draft_model_config.hf_config
            dspark_block_size = int(getattr(draft_hf_config, "dspark_block_size", 0))
            num_speculative_tokens = int(
                getattr(speculative_config, "num_speculative_tokens", 0)
            )
            if num_speculative_tokens != dspark_block_size:
                raise RuntimeError(
                    "DeepSeek-V4 AFD DSpark requires num_speculative_tokens "
                    "to match dspark_block_size: "
                    f"{num_speculative_tokens} != {dspark_block_size}"
                )
            if target_enforce_eager and not draft_enforce_eager:
                raise RuntimeError(
                    "DeepSeek-V4 AFD DSpark eager execution requires draft "
                    "enforce_eager=true"
                )
        else:
            num_speculative_tokens = int(
                getattr(speculative_config, "num_speculative_tokens", 0)
            )
            if not 1 <= num_speculative_tokens <= MAX_DSV4_AFD_SPECULATIVE_TOKENS:
                raise RuntimeError(
                    "DeepSeek-V4 AFD MTP requires num_speculative_tokens in "
                    f"[1, {MAX_DSV4_AFD_SPECULATIVE_TOKENS}]"
                )
            if target_enforce_eager and not draft_enforce_eager:
                raise RuntimeError(
                    "DeepSeek-V4 AFD MTP eager execution requires draft "
                    "enforce_eager=true"
                )
            if (
                tensor_parallel_size == 2
                and not target_enforce_eager
                and not draft_enforce_eager
                and parallel_config.use_ubatching
            ):
                raise RuntimeError(
                    "DeepSeek-V4 AFD TP2 full-draft MTP Graph U2 is not validated; "
                    "use the TP2 eager/U1 baseline or TP1 for this combined mode"
                )
            num_mtp_layers = int(
                getattr(
                    vllm_config.model_config.hf_config,
                    "num_nextn_predict_layers",
                    1,
                )
            )
            if num_mtp_layers != 1:
                raise RuntimeError("DeepSeek-V4 AFD MTP supports exactly one MTP layer")
    if not vllm_config.model_config.enforce_eager:
        cudagraph_mode = getattr(
            getattr(vllm_config, "compilation_config", None),
            "cudagraph_mode",
            None,
        )
        mode_name = getattr(cudagraph_mode, "name", None)
        if not isinstance(mode_name, str):
            mode_name = str(cudagraph_mode).rsplit(".", 1)[-1]
        if mode_name != "FULL_DECODE_ONLY":
            raise RuntimeError(
                "DeepSeek-V4 AFD graph execution supports only FULL_DECODE_ONLY"
            )
        if (
            parallel_config.use_ubatching
            and afd_config.connector not in {
                "P2pHcclAFDConnector",
                "WindowAFDConnector",
            }
        ):
            raise RuntimeError(
                "DeepSeek-V4 AFD graph U2 supports only P2pHcclAFDConnector "
                "or WindowAFDConnector"
            )
    _fail_if_unsupported_deepseek_v4_pd(vllm_config, afd_config)


def _fail_if_unsupported_deepseek_v4_pd(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Validate the Mooncake PD + AFD topology and connector contract."""
    kv_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_config is None:
        return

    if afd_config.role != "attention":
        raise RuntimeError(
            "DeepSeek-V4 AFD Mooncake PD attaches KV transfer only to Attention"
        )
    if afd_config.connector not in {
        "P2pHcclAFDConnector",
        "WindowAFDConnector",
    }:
        raise RuntimeError(
            "DeepSeek-V4 AFD Mooncake PD requires P2pHcclAFDConnector "
            "or WindowAFDConnector"
        )
    if getattr(kv_config, "kv_connector", None) != "MooncakeHybridConnector":
        raise RuntimeError("DeepSeek-V4 AFD PD supports only MooncakeHybridConnector")
    if getattr(kv_config, "kv_role", None) != "kv_consumer":
        raise RuntimeError(
            "DeepSeek-V4 AFD Decode Attention must use kv_role=kv_consumer"
        )
    parallel_config = vllm_config.parallel_config
    if int(parallel_config.tensor_parallel_size) not in (1, 2):
        raise RuntimeError("DeepSeek-V4 AFD Mooncake PD supports only TP1 or TP2")
    if int(getattr(kv_config, "kv_parallel_size", 1)) != 1:
        raise RuntimeError("DeepSeek-V4 AFD Mooncake PD requires kv_parallel_size=1")

    engine_id = getattr(kv_config, "engine_id", None)
    if not isinstance(engine_id, str) or not engine_id.strip():
        raise RuntimeError("DeepSeek-V4 AFD Mooncake PD requires a non-empty engine_id")
    kv_port = getattr(kv_config, "kv_port", None)
    if (
        not isinstance(kv_port, int)
        or isinstance(kv_port, bool)
        or not 1 <= kv_port <= 65535
    ):
        raise RuntimeError("DeepSeek-V4 AFD Mooncake PD requires kv_port in 1..65535")

    extra_config = getattr(kv_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, dict):
        raise RuntimeError(
            "DeepSeek-V4 AFD Mooncake PD requires kv_connector_extra_config"
        )
    prefill = _validate_mooncake_parallel_config(extra_config, "prefill")
    decode = _validate_mooncake_parallel_config(extra_config, "decode")

    if int(decode.get("pp_size", 1)) != 1:
        raise RuntimeError("DeepSeek-V4 AFD Mooncake PD requires decode pp_size=1")
    if prefill["tp_size"] < decode["tp_size"]:
        raise RuntimeError(
            "DeepSeek-V4 AFD Mooncake PD requires prefill TP >= decode TP"
        )
    expected_decode_dp = int(parallel_config.data_parallel_size)
    expected_decode_tp = int(parallel_config.tensor_parallel_size)
    if (
        decode["dp_size"] != expected_decode_dp
        or decode["tp_size"] != expected_decode_tp
    ):
        raise RuntimeError(
            "DeepSeek-V4 AFD Mooncake decode topology must match Attention DP/TP: "
            f"Mooncake DP={decode['dp_size']}, TP={decode['tp_size']}; "
            f"Attention DP={expected_decode_dp}, TP={expected_decode_tp}"
        )


def _validate_mooncake_parallel_config(
    extra_config: dict[str, object],
    role: str,
) -> dict[str, int]:
    topology = extra_config.get(role)
    if not isinstance(topology, dict):
        raise RuntimeError(
            f"DeepSeek-V4 AFD Mooncake PD requires {role} DP/TP topology"
        )
    for field in ("dp_size", "tp_size"):
        value = topology.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RuntimeError(
                f"DeepSeek-V4 AFD Mooncake {role}.{field} must be a positive integer"
            )
    pp_size = topology.get("pp_size", 1)
    if not isinstance(pp_size, int) or isinstance(pp_size, bool) or pp_size < 1:
        raise RuntimeError(
            f"DeepSeek-V4 AFD Mooncake {role}.pp_size must be a positive integer"
        )
    return {
        "dp_size": int(topology["dp_size"]),
        "tp_size": int(topology["tp_size"]),
        "pp_size": int(pp_size),
    }


def _fail_if_unsupported_npu_async_moe_ubatching_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    num_ubatches: int,
    split: str,
) -> None:
    from afd_plugin.connectors.npu.async_cam import ASYNC_MOE_REQUEST_SPLIT

    parallel_config = vllm_config.parallel_config
    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "async_moe_ubatching requires compute_gate_on_attention=true",
        )
    if num_ubatches != 2:
        raise RuntimeError(
            "async_moe_ubatching currently supports exactly two stages; "
            f"got async_moe_num_ubatches={num_ubatches}",
        )
    if split != ASYNC_MOE_REQUEST_SPLIT:
        raise RuntimeError(
            "async_moe_ubatching currently supports only request-boundary split; "
            f"got async_moe_split={split!r}",
        )
    if int(parallel_config.decode_context_parallel_size) > 1:
        raise RuntimeError(
            "async_moe_ubatching does not support decode context parallel metadata yet",
        )


__all__ = ["fail_if_unsupported_npu_afd_features"]
