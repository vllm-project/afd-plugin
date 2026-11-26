# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe validation helpers for AFD runtime wiring."""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING, Final

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    CAMP2P_CONNECTOR,
    DEEPEP_M2N_CONNECTORS,
    AFDConfig,
    parse_afd_config,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDCUDAGraphPolicy,
    cudagraph_mode_name,
    validate_cuda_graph_mode,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDAttentionWorker"
FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDFFNWorker"
ATTENTION_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.AFDAttentionModelRunnerV2"
)
FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.GPUFFNModelRunner"
ROUTED_FFN_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.GPUFFNRoutedExpertsModelRunner"
)
NPU_ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
NPU_FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUFFNWorker"
NPU_ATTENTION_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.npu.AFDNPUAttentionModelRunnerV2"
)
NPU_FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUFFNModelRunner"
VLLM_GPU_WORKER_FQCN: Final[str] = "vllm.v1.worker.gpu_worker.Worker"
VLLM_ASCEND_NPU_WORKER_FQCN: Final[str] = "vllm_ascend.worker.worker.NPUWorker"
VLLM_ASCEND_310P_WORKER_FQCN: Final[str] = "vllm_ascend._310p.worker_310p.NPUWorker310"
VLLM_ASCEND_XLITE_WORKER_FQCN: Final[str] = "vllm_ascend.xlite.xlite_worker.XliteWorker"
MRV2_NUM_UBATCHES: Final[int] = 2
GPU_MRV2_CONNECTORS: Final[tuple[str, ...]] = (
    "P2pNcclAFDConnector",
    *DEEPEP_M2N_CONNECTORS,
)
# Models whose Attention side computes the MoE gate and top-k and sends them
# with the hidden states, as a routed-experts connector (DeepEP M2N) needs.
ATTENTION_ROUTING_MODEL_ARCHITECTURES: Final[tuple[str, ...]] = (
    "Qwen3MoeForCausalLM",
    "Qwen3MoeRoutedExperts",
    "DeepseekV2ForCausalLM",
)


CAM_MAX_RECEIVE_TOKENS = 262144
CAM_MAX_SOURCE_ROWS = (1 << 16) - 1


def cam_receive_capacity() -> int:
    value = os.environ.get("BATCH_SIZE_FACTOR", "1")
    try:
        # Python accepts underscores and Unicode digits that std::stof does not.
        if not value.isascii() or "_" in value:
            raise ValueError("Not a native floating-point literal")
        factor = float(value)
    except ValueError as exc:
        raise RuntimeError(
            "BATCH_SIZE_FACTOR must be a decimal number in (0, 1]"
        ) from exc
    if not 0 < factor <= 1:
        raise RuntimeError("BATCH_SIZE_FACTOR must be a decimal number in (0, 1]")
    # Match the native std::stof + float multiplication, including rounding.
    factor = struct.unpack("f", struct.pack("f", factor))[0]
    capacity = int(CAM_MAX_RECEIVE_TOKENS * factor)
    if capacity < 1:
        raise RuntimeError("BATCH_SIZE_FACTOR must allocate at least one receive row")
    return capacity


def validate_no_pp_or_cp(vllm_config: VllmConfig) -> None:
    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise RuntimeError("AFD ModelRunnerV2 does not support PP or CP")


def validate_afd_world_size(vllm_config: VllmConfig, afd_config: AFDConfig) -> None:
    parallel = vllm_config.parallel_config
    configured = (
        afd_config.num_attention_ranks
        if afd_config.is_attention_server
        else afd_config.num_ffn_ranks
    )
    distributed = parallel.data_parallel_size * parallel.tensor_parallel_size
    if configured != distributed:
        raise RuntimeError(
            f"AFD ModelRunnerV2 {afd_config.role} ranks must match DP * TP: "
            f"configured={configured}, distributed={distributed}",
        )


def validate_async_cam_attention_graph(vllm_config: VllmConfig) -> None:
    """Validate AFD dependencies for native Async CAM Attention graphs."""
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    # Ascend exposes the MRV2 selection through its VllmConfig platform patch.
    if (
        not vllm_config.use_v2_model_runner
        or model.hf_text_config.model_type != "deepseek_v4"
    ):
        raise RuntimeError("Async CAM Attention graph requires MRV2 DeepSeek-V4")
    if parallel.enable_dbo or parallel.use_ubatching:
        raise RuntimeError("Async CAM Attention graph does not support DBO/ubatching")
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise RuntimeError("Async CAM Attention graph does not support PP or CP")


def validate_gpu_model_runner_v2_config(
    vllm_config: VllmConfig,
    *,
    expected_role: str,
    device_type: str,
) -> AFDCUDAGraphPolicy:
    """Validate the shared GPU ModelRunnerV2 deployment constraints.

    For the Attention role these checks guard native ModelRunnerV2 execution.
    For the FFN role they validate only the paired deployment and the
    ``Worker.init_device`` construction seam; ``GPUFFNModelRunner`` remains
    connector-driven and does not implement native ModelRunnerV2 execution.

    Model Runner V1 no longer needs a check here: vLLM 0.34 removed it, and its
    ``EngineArgs.create_engine_config`` rejects a leftover
    ``VLLM_USE_V2_MODEL_RUNNER`` that is not ``1`` before either AFD entry runs.
    """

    afd_config = parse_afd_config(vllm_config, expected_role=expected_role)
    if device_type != "cuda" or afd_config.connector not in GPU_MRV2_CONNECTORS:
        raise RuntimeError(
            "AFD ModelRunnerV2 requires CUDA and one of the synchronous "
            f"connectors {GPU_MRV2_CONNECTORS}",
        )

    parallel = vllm_config.parallel_config
    # DeepEP M2N: Attention may run TP > 1, each TP rank dispatching its own
    # slice of the tokens. FFN experts are not TP-sharded; that and the other
    # deployment checks (expert placement, all2all backend, weight format) run
    # in the connector handshake so that every rank fails together. It routes
    # on Attention, so it requires the gate there, and a model whose Attention
    # side sends the top-k. Workers validate before the connector joins the AFD
    # world, so both roles fail before any collective instead of one role
    # timing out.
    if afd_config.connector in DEEPEP_M2N_CONNECTORS:
        if expected_role == "ffn" and parallel.tensor_parallel_size != 1:
            raise RuntimeError(
                f"{afd_config.connector} requires FFN tensor_parallel_size=1"
            )
        if not afd_config.compute_gate_on_attention:
            raise RuntimeError(
                f"{afd_config.connector} requires compute_gate_on_attention=true"
            )
        # Accept the native architecture and its role-local AFD alias.
        architectures = vllm_config.model_config.hf_config.architectures
        if not any(
            architecture.removeprefix("AFD") in ATTENTION_ROUTING_MODEL_ARCHITECTURES
            for architecture in architectures
        ):
            raise RuntimeError(
                f"{afd_config.connector} supports only "
                f"{ATTENTION_ROUTING_MODEL_ARCHITECTURES}, got {architectures}"
            )
    validate_no_pp_or_cp(vllm_config)
    validate_afd_world_size(vllm_config, afd_config)
    if (
        not parallel.enable_expert_parallel
        or parallel.enable_elastic_ep
        or parallel.enable_eplb
        or parallel.use_sequence_parallel_moe
    ):
        raise RuntimeError("AFD ModelRunnerV2 requires static expert parallelism")
    if (
        parallel.enable_dbo or parallel.use_ubatching
    ) and parallel.num_ubatches != MRV2_NUM_UBATCHES:
        raise RuntimeError(
            "AFD ModelRunnerV2 DBO/ubatching requires exactly two microbatches",
        )

    # Keep importing this CPU-safe validation module independent of the vLLM
    # model-wrapper package; the validator itself runs only after vLLM config
    # construction.
    from afd_plugin.model_executor.models.model_utils import (
        has_afd_model_registration,
    )

    if not has_afd_model_registration(vllm_config.model_config):
        raise RuntimeError("AFD ModelRunnerV2 requires a registered AFD model")

    return validate_cuda_graph_mode(vllm_config, role=expected_role)


def validate_npu_model_runner_v2_config(
    vllm_config: VllmConfig,
    *,
    expected_role: str,
    device_type: str,
) -> None:
    """Validate the supported Ascend NPU ModelRunnerV2 deployment."""

    # Check the MRV2 selection supplied by the Ascend platform patch.
    if not vllm_config.use_v2_model_runner:
        raise RuntimeError("AFD NPU requires ModelRunnerV2")
    afd_config = parse_afd_config(vllm_config, expected_role=expected_role)
    if device_type != "npu":
        raise RuntimeError("AFD NPU ModelRunnerV2 requires NPU")
    if afd_config.connector not in (AFD_ASYNC_CONNECTOR, CAMP2P_CONNECTOR):
        raise RuntimeError(
            "AFD NPU ModelRunnerV2 requires either synchronous CAMP2pAFDConnector "
            "or Async CAM",
        )
    # The sync Attention-side gate limits (connector, target, TP, DBO/ubatching)
    # live in feature_validation and run from both workers before this point.

    parallel = vllm_config.parallel_config
    validate_no_pp_or_cp(vllm_config)
    validate_afd_world_size(vllm_config, afd_config)
    if (
        not parallel.enable_expert_parallel
        or parallel.enable_elastic_ep
        or parallel.enable_eplb
    ):
        raise RuntimeError("AFD ModelRunnerV2 requires static expert parallelism")
    if (
        afd_config.connector == AFD_ASYNC_CONNECTOR
        and afd_config.is_attention_server
        and not vllm_config.model_config.enforce_eager
    ):
        validate_async_cam_attention_graph(vllm_config)
    uses_dbo = parallel.enable_dbo or parallel.use_ubatching
    if uses_dbo:
        if parallel.num_ubatches != MRV2_NUM_UBATCHES:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 DBO requires exactly two ubatches"
            )
        if vllm_config.speculative_config is not None:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 DBO does not support speculative decoding"
            )

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        if not afd_config.async_dp:
            raise RuntimeError("Async CAM requires async DP (async=true)")
        if not afd_config.compute_gate_on_attention:
            raise RuntimeError("Async CAM requires compute_gate_on_attention=true")
        if uses_dbo and not afd_config.is_attention_server:
            raise RuntimeError("Async CAM DBO/ubatching requires the Attention role")
        if (
            not vllm_config.model_config.enforce_eager
            and not afd_config.is_attention_server
        ):
            if vllm_config.compilation_config.cudagraph_mode.name != "FULL":
                raise ValueError(
                    "Async FFN graph requires explicit cudagraph_mode=FULL"
                )
            if parallel.tensor_parallel_size != 1:
                raise RuntimeError("Async FFN FULL requires F TP=1")
    else:
        if (
            uses_dbo
            and expected_role == "attention"
            and parallel.data_parallel_size <= 1
        ):
            raise RuntimeError("AFD NPU ModelRunnerV2 DBO requires Attention DP > 1")

    if not vllm_config.model_config.enforce_eager:
        graph_mode = cudagraph_mode_name(vllm_config)
        if graph_mode not in {"FULL", "FULL_DECODE_ONLY"}:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 supports ACL graph modes FULL and "
                f"FULL_DECODE_ONLY; got {graph_mode!r}.",
            )

    from afd_plugin.model_executor.models.model_utils import (
        has_afd_model_registration,
    )

    if not has_afd_model_registration(vllm_config.model_config):
        raise RuntimeError("AFD ModelRunnerV2 requires a registered AFD model")


def normalize_qualname(value: str) -> str:
    return value.replace(":", ".")


def expected_worker_qualname(role: str) -> str:
    if role == "attention":
        return ATTENTION_WORKER_FQCN
    if role == "ffn":
        return FFN_WORKER_FQCN
    raise ValueError(f"unknown AFD role {role!r}")


def expected_npu_worker_qualname(role: str) -> str:
    if role == "attention":
        return NPU_ATTENTION_WORKER_FQCN
    if role == "ffn":
        return NPU_FFN_WORKER_FQCN
    raise ValueError(f"unknown AFD role {role!r}")


def afd_worker_qualname_for_platform_default(
    role: str,
    platform_worker_qualname: str,
    *,
    is_cuda: bool,
    device_type: str,
) -> str:
    """Select an AFD worker from the platform's normalized default worker."""

    normalized_platform_worker = normalize_qualname(platform_worker_qualname)
    if is_cuda and normalized_platform_worker == VLLM_GPU_WORKER_FQCN:
        return expected_worker_qualname(role)
    if (
        device_type == "npu"
        and normalized_platform_worker == VLLM_ASCEND_NPU_WORKER_FQCN
    ):
        return expected_npu_worker_qualname(role)
    if device_type == "npu" and normalized_platform_worker in {
        VLLM_ASCEND_310P_WORKER_FQCN,
        VLLM_ASCEND_XLITE_WORKER_FQCN,
    }:
        raise ValueError(
            "AFD automatic worker selection supports only the standard Ascend "
            "A2/A3 NPUWorker; the current Ascend platform selected "
            f"{platform_worker_qualname!r}",
        )
    raise ValueError(
        "AFD automatic worker selection does not support the current platform: "
        f"device_type={device_type!r}, worker={platform_worker_qualname!r}",
    )


def assert_compatible_afd_stack(
    vllm_config: VllmConfig,
    *,
    caller: str,
    expected_role: str | None = None,
    expected_worker_qualname_override: str | None = None,
) -> AFDConfig:
    """Validate AFD config and worker class wiring.

    This helper intentionally uses duck typing so unit tests and local CPU
    development do not need to construct a real vLLM ``VllmConfig``.
    """

    def _ctx() -> str:
        return f" (context: {caller!r})"

    config = parse_afd_config(vllm_config, expected_role=expected_role)

    parallel_config = vllm_config.parallel_config
    async_expected_worker = (
        expected_npu_worker_qualname(config.role)
        if config.connector == "CAMAsyncAFDConnector"
        else None
    )

    worker_cls_raw = parallel_config.worker_cls
    if not isinstance(worker_cls_raw, str):
        raise ValueError(
            "parallel_config.worker_cls must be a qualname string "
            f"(got type {type(worker_cls_raw).__name__}){_ctx()}",
        )
    if worker_cls_raw.strip() == "auto":
        raise ValueError(
            "parallel_config.worker_cls remained 'auto' after AFD config "
            "normalization; ensure the AFD general plugin is loaded before "
            f"VllmConfig is created{_ctx()}",
        )

    expected_qualname = (
        async_expected_worker
        or expected_worker_qualname_override
        or expected_worker_qualname(config.role)
    )
    worker_fqcn = normalize_qualname(worker_cls_raw.strip())
    expected_fqcn = normalize_qualname(expected_qualname)
    if worker_fqcn != expected_fqcn:
        prefix = (
            "CAMAsyncAFDConnector requires Ascend NPU worker class: "
            if async_expected_worker is not None
            else "invalid worker class for AFD runtime stack: "
        )
        raise ValueError(
            prefix + f"got={worker_fqcn!r} expected={expected_qualname!r}; "
            "remove --worker-cls to let AFD select it automatically, or pass "
            f"--worker-cls {expected_qualname}{_ctx()}",
        )

    return config


__all__ = [
    "ATTENTION_MODEL_RUNNER_FQCN",
    "ATTENTION_ROUTING_MODEL_ARCHITECTURES",
    "ATTENTION_WORKER_FQCN",
    "FFN_MODEL_RUNNER_FQCN",
    "FFN_WORKER_FQCN",
    "GPU_MRV2_CONNECTORS",
    "NPU_ATTENTION_MODEL_RUNNER_FQCN",
    "NPU_ATTENTION_WORKER_FQCN",
    "NPU_FFN_MODEL_RUNNER_FQCN",
    "NPU_FFN_WORKER_FQCN",
    "ROUTED_FFN_MODEL_RUNNER_FQCN",
    "assert_compatible_afd_stack",
    "expected_npu_worker_qualname",
    "expected_worker_qualname",
    "normalize_qualname",
    "validate_gpu_model_runner_v2_config",
    "validate_npu_model_runner_v2_config",
    "validate_afd_world_size",
    "validate_no_pp_or_cp",
]
