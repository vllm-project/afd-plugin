# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Configuration shared by the API process and role workers (CPU safe)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from afd_plugin.config import AFDConfig, parse_afd_config, parse_optional_afd_config
from afd_plugin.v1.worker.cuda_graph import validate_cuda_graph_mode

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig

Role = Literal["attention", "ffn"]
ATTENTION_WORKERS = (
    "afd_plugin.v1.worker.AFDAttentionWorker",
    "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker",
)


def is_elastic_attention(config: VllmConfig) -> bool:
    # Duck-typed config doubles (unit tests) may omit additional_config; such
    # a config is never elastic. parse_optional_afd_config stays strict.
    if getattr(config, "additional_config", None) is None:
        return False
    afd = parse_optional_afd_config(config)
    return bool(
        afd is not None
        and afd.role == "attention"
        and config.parallel_config.enable_elastic_ep
    )


def is_elastic_attention_worker(parallel: ParallelConfig) -> bool:
    """Early ParallelConfig validation has worker_cls, but no additional_config."""
    worker = parallel.worker_cls
    return (
        parallel.enable_elastic_ep
        and isinstance(worker, str)
        and worker.strip().replace(":", ".") in ATTENTION_WORKERS
    )


@dataclass(frozen=True)
class ElasticTopology:
    attention_dp: int
    ffn_dp: int
    attention_tp: int = 1
    ffn_tp: int = 1

    def validate(self) -> None:
        if self.attention_dp < 2 or self.ffn_dp < 1:
            raise ValueError("Elastic AFD requires A DP >= 2 and F DP >= 1")
        # TP > 1 requires a separate static+EEP hardware qualification.
        if self.attention_tp != 1 or self.ffn_tp != 1:
            raise ValueError("This elastic AFD implementation currently requires TP=1")
        attention_ranks = self.attention_dp * self.attention_tp
        ffn_ranks = self.ffn_dp * self.ffn_tp
        if attention_ranks < ffn_ranks:
            raise ValueError("AFD requires attention ranks >= FFN ranks")
        # The CAMP2P data plane wires a fixed N:1 fanout: each FFN rank owns
        # attention_ranks // ffn_ranks shared-memory windows (connector.ratio)
        # and expects that many uniform row counts per step
        # (_num_tokens_for_ffn_rank). A non-divisible mapping (e.g. 3A x 2F)
        # leaves sources unwindowed and hangs the a2e kernels with an AI-core
        # vector timeout, so reject it at the API boundary instead.
        if attention_ranks % ffn_ranks:
            raise ValueError(
                "AFD elastic requires attention ranks divisible by FFN ranks "
                f"for the CAMP2P N:1 fanout (got {attention_ranks}A x "
                f"{ffn_ranks}F); scale attention to a multiple of the FFN "
                "rank count"
            )

    def resize(self, role: Role, size: int) -> ElasticTopology:
        if role not in ("attention", "ffn"):
            raise ValueError("role must be 'attention' or 'ffn'")
        if type(size) is not int or size <= 0:
            raise ValueError("new_data_parallel_size must be a positive integer")
        target = replace(
            self, **{"attention_dp" if role == "attention" else "ffn_dp": size}
        )
        target.validate()
        return target

    def afd_config(self, base: AFDConfig, *, port: int, role: Role) -> AFDConfig:
        return replace(
            base,
            role=role,
            port=port,
            num_attention_ranks=self.attention_dp * self.attention_tp,
            num_ffn_ranks=self.ffn_dp * self.ffn_tp,
        )


def validate_elastic_config(config: VllmConfig) -> ElasticTopology:
    afd = parse_afd_config(config, expected_role="attention")
    parallel = config.parallel_config
    raw = config.additional_config.get("afd_elastic", {})
    if not isinstance(raw, dict) or set(raw) - {"ffn_tensor_parallel_size"}:
        raise ValueError("afd_elastic accepts only ffn_tensor_parallel_size")
    ffn_tp = raw.get("ffn_tensor_parallel_size", 1)
    if type(ffn_tp) is not int or ffn_tp <= 0:
        raise ValueError("ffn_tensor_parallel_size must be a positive integer")
    if afd.num_ffn_ranks % ffn_tp:
        raise ValueError("num_ffn_ranks must be divisible by FFN TP")
    if parallel.elastic_ep_max_dp_size is None:
        raise ValueError(
            "Elastic AFD requires --elastic-ep-max-dp-size (native two-phase "
            "scaling uses it as the A DP bound, e.g. 2x the initial size)."
        )
    topology = ElasticTopology(
        parallel.data_parallel_size,
        afd.num_ffn_ranks // ffn_tp,
        parallel.tensor_parallel_size,
        ffn_tp,
    )
    topology.validate()
    if not config.use_v2_model_runner:
        raise ValueError(
            "Elastic AFD requires the vLLM ModelRunnerV2 stack "
            "(VLLM_USE_V2_MODEL_RUNNER=1, the default since vLLM 0.30)."
        )
    if not config.model_config.enforce_eager:
        validate_cuda_graph_mode(config, role="attention")
        # Upstream GPUModelRunner.load_model may return before installing its
        # CUDAGraphWrapper in STOCK_TORCH_COMPILE mode. F must not capture
        # communication while A executes an unwrapped forward.
        if config.compilation_config.mode.name == "STOCK_TORCH_COMPILE":
            raise ValueError(
                "Elastic AFD CUDA graphs do not support STOCK_TORCH_COMPILE; "
                "use compilation mode NONE (0) or VLLM_COMPILE (3)"
            )
    if afd.num_attention_ranks != topology.attention_dp * topology.attention_tp:
        raise ValueError("num_attention_ranks must equal A DP * A TP")
    if not is_elastic_attention_worker(parallel):
        raise ValueError("Elastic AFD requires an explicit AFD attention --worker-cls")
    if parallel.enable_eplb or not parallel.enable_expert_parallel:
        raise ValueError(
            "Elastic AFD requires expert parallel enabled and EPLB disabled"
        )
    if (
        parallel.data_parallel_backend != "ray"
        or parallel.distributed_executor_backend != "uni"
        or parallel.data_parallel_external_lb
        or parallel.data_parallel_hybrid_lb
        or parallel._api_process_count != 1
    ):
        raise ValueError(
            "Elastic AFD requires Ray DP, uni executor, and one internal LB"
        )
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or afd.async_dp
    ):
        raise ValueError("Elastic AFD requires PP=PCP=DCP=1 and synchronous DP")
    if parallel.use_ubatching and not parallel.enable_dbo:
        raise ValueError("Elastic AFD ubatching requires --enable-dbo")
    if not config.cache_config.kv_cache_memory_bytes:
        raise ValueError("Elastic AFD requires an explicit --kv-cache-memory-bytes")
    if (
        config.speculative_config is not None
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.model_config.enable_sleep_mode
    ):
        raise ValueError(
            "Elastic AFD does not support speculation, LoRA, KV transfer, sleep"
        )
    if afd.connector not in ("P2pNcclAFDConnector", "CAMP2pAFDConnector"):
        raise ValueError(
            "Elastic AFD currently requires P2pNcclAFDConnector (CUDA) or "
            "CAMP2pAFDConnector (Ascend NPU; the stateless HCCL EEP group "
            "qualification is provided by vllm-ascend feat/eep-device-group)."
        )
    return topology


def set_afd_config(config: VllmConfig, afd: AFDConfig) -> None:
    # Preserve connector-specific configuration, which is not part of AFDConfig.
    config.additional_config["afd"] = {
        **config.additional_config["afd"],
        "role": afd.role,
        "host": afd.host,
        "port": afd.port,
        "num_attention_ranks": afd.num_attention_ranks,
        "num_ffn_ranks": afd.num_ffn_ranks,
    }
