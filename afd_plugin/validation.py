# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe validation helpers for AFD runtime wiring."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Final

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.v1.worker.cuda_graph import (
    cudagraph_mode_name,
    validate_cuda_graph_mode,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDAttentionWorker"
FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDFFNWorker"
ATTENTION_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDAttentionModelRunner"
FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.GPUFFNModelRunner"
UBATCH_WRAPPER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDUBatchWrapper"
NPU_ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
NPU_FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUFFNWorker"
NPU_ATTENTION_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.npu.AFDNPUAttentionModelRunner"
)
NPU_FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.npu.AFDNPUFFNModelRunner"
VLLM_GPU_WORKER_FQCN: Final[str] = "vllm.v1.worker.gpu_worker.Worker"
VLLM_ASCEND_NPU_WORKER_FQCN: Final[str] = "vllm_ascend.worker.worker.NPUWorker"
VLLM_ASCEND_310P_WORKER_FQCN: Final[str] = "vllm_ascend._310p.worker_310p.NPUWorker310"
VLLM_ASCEND_XLITE_WORKER_FQCN: Final[str] = "vllm_ascend.xlite.xlite_worker.XliteWorker"
GPU_MRV2_NUM_UBATCHES: Final[int] = 2


def validate_gpu_model_runner_v2_config(
    vllm_config: VllmConfig,
    *,
    expected_role: str,
    device_type: str,
) -> None:
    """Validate the shared GPU ModelRunnerV2 deployment constraints.

    For the Attention role these checks guard native ModelRunnerV2 execution.
    For the FFN role they validate only the paired deployment and the
    ``Worker.init_device`` construction seam; ``GPUFFNModelRunner`` remains
    connector-driven and does not implement native ModelRunnerV2 execution.
    """

    afd_config = parse_afd_config(vllm_config, expected_role=expected_role)
    if (
        device_type != "cuda"
        or afd_config.connector != "P2pNcclAFDConnector"
        or afd_config.compute_gate_on_attention
    ):
        raise RuntimeError(
            "AFD ModelRunnerV2 requires CUDA, synchronous P2pNcclAFDConnector, and "
            "compute_gate_on_attention=false",
        )

    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise RuntimeError("AFD ModelRunnerV2 does not support PP or CP")

    configured_ranks = (
        afd_config.num_attention_ranks
        if expected_role == "attention"
        else afd_config.num_ffn_ranks
    )
    distributed_ranks = parallel.data_parallel_size * parallel.tensor_parallel_size
    if configured_ranks != distributed_ranks:
        raise RuntimeError(
            f"AFD ModelRunnerV2 {expected_role} ranks must match DP * TP: "
            f"configured={configured_ranks}, distributed={distributed_ranks}",
        )
    if (
        not parallel.enable_expert_parallel
        or parallel.enable_elastic_ep
        or parallel.enable_eplb
        or parallel.use_sequence_parallel_moe
        or vllm_config.compilation_config.pass_config.enable_sp
    ):
        raise RuntimeError("AFD ModelRunnerV2 requires static expert parallelism")
    if parallel.enable_dbo or parallel.use_ubatching:
        if parallel.num_ubatches != GPU_MRV2_NUM_UBATCHES:
            raise RuntimeError(
                "AFD ModelRunnerV2 DBO/ubatching requires exactly two microbatches",
            )
        # Native maybe_build_ubatch_runner requires DP > 1. FFN does not
        # construct it: its stage layout comes from Attention control messages.
        if expected_role == "attention" and parallel.data_parallel_size <= 1:
            raise RuntimeError(
                "AFD ModelRunnerV2 DBO/ubatching requires Attention DP > 1"
            )

    # Keep importing this CPU-safe validation module independent of the vLLM
    # model-wrapper package; the validator itself runs only after vLLM config
    # construction.
    from afd_plugin.model_executor.models.model_utils import (
        has_afd_model_registration,
    )

    if not has_afd_model_registration(vllm_config.model_config):
        raise RuntimeError("AFD ModelRunnerV2 requires a registered AFD model")

    validate_cuda_graph_mode(vllm_config, role=expected_role)


def validate_npu_model_runner_v2_config(
    vllm_config: VllmConfig,
    *,
    expected_role: str,
    device_type: str,
) -> None:
    """Validate the supported Ascend NPU ModelRunnerV2 deployment."""

    afd_config = parse_afd_config(vllm_config, expected_role=expected_role)
    if (
        device_type != "npu"
        or afd_config.connector != "CAMP2pAFDConnector"
        or afd_config.compute_gate_on_attention
    ):
        raise RuntimeError(
            "AFD NPU ModelRunnerV2 requires NPU, synchronous "
            "CAMP2pAFDConnector, and compute_gate_on_attention=false",
        )

    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise RuntimeError("AFD ModelRunnerV2 does not support PP or CP")

    configured_ranks = (
        afd_config.num_attention_ranks
        if expected_role == "attention"
        else afd_config.num_ffn_ranks
    )
    distributed_ranks = parallel.data_parallel_size * parallel.tensor_parallel_size
    if configured_ranks != distributed_ranks:
        raise RuntimeError(
            f"AFD ModelRunnerV2 {expected_role} ranks must match DP * TP: "
            f"configured={configured_ranks}, distributed={distributed_ranks}",
        )
    if (
        not parallel.enable_expert_parallel
        or parallel.enable_elastic_ep
        or parallel.enable_eplb
        or parallel.use_sequence_parallel_moe
        or vllm_config.compilation_config.pass_config.enable_sp
    ):
        raise RuntimeError("AFD ModelRunnerV2 requires static expert parallelism")
    if parallel.enable_dbo or parallel.use_ubatching:
        raise RuntimeError("AFD NPU ModelRunnerV2 does not support DBO or ubatching")

    from afd_plugin.model_executor.models.model_utils import (
        has_afd_model_registration,
    )

    if not has_afd_model_registration(vllm_config.model_config):
        raise RuntimeError("AFD ModelRunnerV2 requires a registered AFD model")

    if not vllm_config.model_config.enforce_eager:
        graph_mode = cudagraph_mode_name(vllm_config)
        if graph_mode not in {"FULL", "FULL_DECODE_ONLY"}:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 supports ACL graph modes FULL and "
                f"FULL_DECODE_ONLY; got {graph_mode!r}.",
            )


def normalize_qualname(value: str) -> str:
    return value.replace(":", ".")


def resolve_class_from_qualname(qualname: str, *, role: str = "class") -> type[Any]:
    """Resolve a dotted or colon-separated class path."""

    normalized = normalize_qualname(qualname.strip())
    if not normalized or "." not in normalized:
        raise ValueError(
            f"{role} must be a dotted qualname, got {qualname!r}",
        )
    module_name, obj_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{role} resolved to {type(obj).__name__}, expected a class",
        )
    return obj


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
    "ATTENTION_WORKER_FQCN",
    "FFN_MODEL_RUNNER_FQCN",
    "FFN_WORKER_FQCN",
    "NPU_ATTENTION_MODEL_RUNNER_FQCN",
    "NPU_ATTENTION_WORKER_FQCN",
    "NPU_FFN_MODEL_RUNNER_FQCN",
    "NPU_FFN_WORKER_FQCN",
    "UBATCH_WRAPPER_FQCN",
    "assert_compatible_afd_stack",
    "expected_npu_worker_qualname",
    "expected_worker_qualname",
    "normalize_qualname",
    "resolve_class_from_qualname",
    "validate_gpu_model_runner_v2_config",
    "validate_npu_model_runner_v2_config",
]
