# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepEP M2N AFD connector: expert-routed Attention <-> FFN transport.

Port of FastAFD's DeepEP M2N path (``3c71619``) to the plugin connector
contract. Routing runs on the Attention side: each Attention rank sends only
the routed copies of its tokens, straight to the FFN ranks that own the chosen
experts; FFN ranks run their local experts on what they receive and send the
weighted results back, where combine sums each token's top-k contributions.

Topology:
    One AFD world ordered ``[F0, F1, ..., A0, A1, ...]`` (as the P2P
    connector). DeepEP sees it as a symmetric EP group in which FFN rank ``g``
    owns real experts ``[g * E/F, (g + 1) * E/F)`` and Attention ranks own
    dummy expert slots (see ``layout.py``). Any Attention:FFN ratio works;
    ``num_experts`` must be divisible by the FFN rank count.

Data path (per MoE layer and stage, identical order on every rank):
    Attention: ``send_attn_output(h, ctx, topk_ids=, topk_weights=)``
        dispatches; ``recv_ffn_output(ref, ubatch_idx)`` combines and returns
        ``[tokens, hidden]``.
    FFN: ``recv_attn_output(ubatch_idx)`` dispatches with no tokens and
        returns the received rows; ``run_experts(payload, weights)`` runs the
        local experts (the MoE FFN); ``send_ffn_output(out, ctx)`` combines.
    Both calls are collectives over all ranks: every rank must issue them for
    every MoE layer and stage, including ranks with no tokens.

Attention tensor parallelism:
    Every Attention TP rank is its own union rank. TP peers hold the same
    tokens, so each dispatches only its slice: the model pads the DP rank's
    ``N`` tokens to a multiple of the TP size, TP rank ``r`` dispatches rows
    ``[r * ceil(N/TP), (r + 1) * ceil(N/TP))`` and the TP group all-gathers the
    combined rows. Attention reports its TP size in the handshake; the FFN,
    which runs without TP, adopts it to plan the per-rank token counts.

Control plane:
    The existing ``AFDControlPayload`` is enough. Every rank derives the same
    per-stage plan from it (dispatch bucket, host sync, graph mode), so no
    per-layer control traffic exists. Attention rank 0 sends each step's
    payload to every FFN rank over the AFD world's NCCL group.

Transport:
    Dispatch and combine run on afdEP's DeepEP-derived ``ElasticBuffer``
    (https://github.com/ms-llmd/afdEP). By default it runs afdEP's AFD role
    path: Attention ranks only send in dispatch and only receive in combine,
    FFN ranks the reverse. It removes Masked DeepEP's entry barriers, union
    count exchange and all-rank exits, and sizes buffers and kernels by role.
    ``role_aware: false`` keeps afdEP's Masked DeepEP protocol for comparison.

Scope: BF16 or FP8 (block-128, fp32 scales) payload; eager, ``torch.compile``
(dispatch/combine are custom ops) and CUDA graphs; up to two lanes (DBO);
single node (multi-node needs RDMA pods).
"""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from afdep.deepep import DispatchHandle, ElasticBuffer, ReceiveResult
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_enabled

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDControlPlane, ConnectorExtraInfo
from afd_plugin.connectors.gpu.deepep_m2n.experts import (
    EXPERT_BACKENDS,
    DeepEPM2NExperts,
    ExpertBackend,
)
from afd_plugin.connectors.gpu.deepep_m2n.layout import DeepEPM2NLayout
from afd_plugin.connectors.gpu.deepep_m2n.network import (
    NetworkConfig,
    NetworkPlan,
    RankTopology,
    apply_rank_environment,
    discover_local_topology,
    plan_network,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.connectors.routed_experts import (
    FP8_DTYPE,
    AFDRoutedExpertsConnector,
    FFNLocalExperts,
    LocalExpertWeights,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)

PROTOCOL_VERSION = "deepep_m2n/1"
BF16_PAYLOAD = "bf16"
# FP8 e4m3 rows with one fp32 scale per row and 128 channels (SM90 recipe).
FP8_PAYLOAD = "fp8_block128_fp32"
PAYLOAD_FORMATS = (BF16_PAYLOAD, FP8_PAYLOAD)
FP8_GROUP_SIZE = 128
# Expert weight formats the FFN must load for each payload format.
BF16_WEIGHTS = "bf16"
FP8_BLOCK128_WEIGHTS = "fp8_block128"
PAYLOAD_WEIGHT_FORMATS = {
    BF16_PAYLOAD: BF16_WEIGHTS,
    FP8_PAYLOAD: FP8_BLOCK128_WEIGHTS,
}
# vLLM's all2all backend on the FFN. The connector already moved the rows, so
# vLLM's own MoE all-to-all never runs; this backend allocates no buffers and
# keeps vLLM's DeepEP out of the process.
IDLE_ALL2ALL_BACKEND = "allgather_reducescatter"
CONTROL_BACKENDS = ("nccl", "gloo")
# DeepEP V2 needs Hopper (SM90) or newer.
MIN_SM_MAJOR = 9
PROCESS_GROUP_NAME = "afd_deepep_m2n"
# CPU (gloo) group for the handshake and DeepEP's communicator bootstrap,
# created before any NCCL communicator so the network plan can still set NCCL_*.
BOOTSTRAP_GROUP_NAME = "afd_deepep_m2n_bootstrap"
# FastAFD passes 120 s for both the CPU wait and the GPU spin timeouts.
DEFAULT_TIMEOUT_S = 120
# Only Attention rank 0 sends the per-step control payload.
CONTROL_SENDER_ATTN_RANK = 0
# Lanes = in-flight micro-batches (DBO uses two).
MAX_LANES = 2
PER_BUFFER_COMM_STREAM_ENV = "AFD_DEEPEP_PER_BUFFER_COMM_STREAM"


@dataclass(frozen=True)
class DeepEPM2NConfig(ConnectorExtraInfo):
    """``connector_extra_config`` for ``DeepEPM2NAFDConnector``.

    Attributes:
        capacity_tokens_per_rank: Max tokens one Attention rank dispatches per
            call (with Attention TP, its slice); sizes the DeepEP buffers.
            ``None`` uses the larger of vLLM's ``max_num_batched_tokens`` and
            its largest CUDA-graph capture size, divided by the Attention TP
            size (``resolve_capacity_tokens_per_rank``).
        num_lanes: In-flight micro-batches (one DeepEP buffer each). ``None``
            uses vLLM's micro-batch count: two under DBO, otherwise one
            (``resolve_num_lanes``).
        payload_format: Dispatch wire format.
        expert_backend: ``deepgemm`` (fast) or ``fused_experts`` (reference).
        network: Fabric settings (GIN, NIC choice, hybrid mode, traffic
            class, extra NCCL env); see ``network.NetworkConfig`` and its
            presets.
        bucket_granularity: Dispatch bucket is rounded up to this multiple,
            then up to the next rung of the bucket ladder (``bucket_ladder``)
            or, above the ladder, to the capacity, so every step uses one of a
            few DeepEP kernel sets compiled at startup.
        num_sms: SMs DeepEP kernels use; ``0`` = a fixed fraction of the
            device's SMs (24 on H100).
        num_qps: RDMA queue pairs per DeepEP buffer; ``0`` = DeepEP's default.
        expert_num_sms: SMs the FFN's DeepGEMM expert kernels use; ``0`` = all.
            DeepGEMM splits its tiles statically over a persistent grid, so a
            transport kernel of the other DBO lane holding even one SM delays
            a share of every overlapping GEMM. Size the grid to the SMs that
            transport leaves free while it waits. Must be even (DeepGEMM
            launches 2-CTA clusters).
        cpu_timeout_s / gpu_timeout_s: DeepEP host-wait and kernel-spin
            timeouts.
        cpu_sync_min_tokens: Eager steps whose bucket is below this skip the
            host sync that reads receive counts; they allocate the worst-case
            receive size instead. ``0`` always syncs in eager mode. Attention
            ranks keep a zero buffer for the largest bucket that skips the
            sync (the top rung of ``bucket_ladder``), so raising this costs
            memory.
        control_backend: Backend of the connector's own process group
            (handshake, DeepEP communicator bootstrap, per-step control
            payload). ``gloo`` keeps it off the GPUs and the RDMA NICs, so
            DeepEP's communicator is the only RDMA user.
        role_aware: Run afdEP's AFD role path (the default). ``False`` runs its
            Masked DeepEP protocol. Every rank must use the same value; afdEP
            rejects a mismatch when the buffers are created.
    """

    capacity_tokens_per_rank: int | None = None
    num_lanes: int | None = None
    payload_format: str = "bf16"
    expert_backend: ExpertBackend = "deepgemm"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    bucket_granularity: int = 1
    num_sms: int = 0
    num_qps: int = 0
    expert_num_sms: int = 0
    cpu_timeout_s: int = DEFAULT_TIMEOUT_S
    gpu_timeout_s: int = DEFAULT_TIMEOUT_S
    cpu_sync_min_tokens: int = 0
    control_backend: str = "nccl"
    role_aware: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> DeepEPM2NConfig:
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown DeepEPM2NAFDConnector options: {unknown}")
        raw["network"] = NetworkConfig.from_mapping(raw.get("network"))
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if (
            self.capacity_tokens_per_rank is not None
            and self.capacity_tokens_per_rank <= 0
        ):
            raise ValueError("capacity_tokens_per_rank must be positive")
        if self.num_lanes is not None and not 1 <= self.num_lanes <= MAX_LANES:
            raise ValueError(f"num_lanes must be in [1, {MAX_LANES}]")
        if self.payload_format not in PAYLOAD_FORMATS:
            raise ValueError(f"payload_format must be one of {PAYLOAD_FORMATS}")
        if self.expert_backend not in EXPERT_BACKENDS:
            raise ValueError(f"expert_backend must be one of {EXPERT_BACKENDS}")
        if self.bucket_granularity <= 0:
            raise ValueError("bucket_granularity must be positive")
        if self.num_sms < 0:
            raise ValueError("num_sms must be non-negative")
        if self.expert_num_sms < 0 or self.expert_num_sms % 2:
            raise ValueError("expert_num_sms must be a non-negative even number")
        if self.cpu_sync_min_tokens < 0:
            raise ValueError("cpu_sync_min_tokens must be non-negative")
        if self.num_qps < 0:
            raise ValueError("num_qps must be non-negative")
        if self.cpu_timeout_s <= 0 or self.gpu_timeout_s <= 0:
            raise ValueError("cpu_timeout_s and gpu_timeout_s must be positive")
        if self.control_backend not in CONTROL_BACKENDS:
            raise ValueError(f"control_backend must be one of {CONTROL_BACKENDS}")
        if type(self.role_aware) is not bool:
            raise ValueError("role_aware must be a boolean")

    def to_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "network": self.network.to_mapping()}


# Attention derives the capacity from its own vLLM config and the FFN adopts
# it in the handshake (the FFN does not know the Attention TP size). The lane
# count is derived on both roles and must match, so Attention and FFN must
# share the DBO setting.
def resolve_capacity_tokens_per_rank(
    config: DeepEPM2NConfig, vllm_config: VllmConfig
) -> int:
    """Largest dispatch one Attention rank may issue: its TP slice of a prefill
    chunk or of the largest decode graph, whichever is bigger."""
    if config.capacity_tokens_per_rank is not None:
        return config.capacity_tokens_per_rank
    max_tokens = max(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.compilation_config.max_cudagraph_capture_size,
    )
    return -(-max_tokens // vllm_config.parallel_config.tensor_parallel_size)


def resolve_num_lanes(config: DeepEPM2NConfig, vllm_config: VllmConfig) -> int:
    """One lane per vLLM micro-batch, so a DBO deployment gets two."""
    if config.num_lanes is not None:
        return config.num_lanes
    parallel_config = vllm_config.parallel_config
    num_lanes = parallel_config.num_ubatches if parallel_config.use_ubatching else 1
    if num_lanes > MAX_LANES:
        raise ValueError(
            f"vLLM runs {num_lanes} micro-batches; DeepEP M2N supports at most "
            f"{MAX_LANES} lanes"
        )
    return num_lanes


@dataclass(frozen=True)
class StagePlan:
    """Per-stage values every rank derives from the same control payload."""

    tokens_per_attn_rank: tuple[int, ...]
    bucket: int
    do_cpu_sync: bool
    graph_mode: bool


def tokens_per_attention_rank(
    dp_metadata: AFDDPMetadata, attn_size: int, attn_tp_size: int
) -> tuple[int, ...]:
    """Token count every Attention rank dispatches for one stage.

    DP metadata lists one count per DP rank. The TP peers of a DP rank hold
    the same tokens and each dispatches a ``ceil(count / attn_tp_size)`` slice
    (the last slices include padding rows that route nowhere).
    """
    counts = [int(count) for count in dp_metadata.num_tokens_across_dp_cpu.flatten()]
    if len(counts) * attn_tp_size != attn_size:
        raise ValueError(
            f"DP metadata has {len(counts)} token counts; with Attention TP "
            f"{attn_tp_size} that does not cover {attn_size} Attention ranks"
        )
    return tuple(
        -(-count // attn_tp_size) for count in counts for _ in range(attn_tp_size)
    )


def plan_stage(
    dp_metadata: AFDDPMetadata,
    *,
    attn_size: int,
    attn_tp_size: int,
    capacity_tokens_per_rank: int,
    bucket_granularity: int,
    graph_mode: bool,
    bucket_ladder: Sequence[int],
    cpu_sync_min_tokens: int = 0,
) -> StagePlan:
    tokens = tokens_per_attention_rank(dp_metadata, attn_size, attn_tp_size)
    max_tokens = max(1, max(tokens))
    bucket = -(-max_tokens // bucket_granularity) * bucket_granularity
    if bucket > capacity_tokens_per_rank:
        raise ValueError(
            f"dispatch bucket {bucket} exceeds capacity_tokens_per_rank="
            f"{capacity_tokens_per_rank}"
        )
    # Graph capture/replay cannot block on the host. Eager steps read the real
    # receive counts so DeepEP allocates by actual rather than worst-case size,
    # unless the bucket is small enough that skipping the host round trip wins.
    do_cpu_sync = not graph_mode and bucket >= cpu_sync_min_tokens
    # DeepEP compiles its dispatch and combine kernels per bucket (it is a
    # template argument), so a bucket per token count runs nvcc on the request
    # path. Buckets therefore come from a fixed set, compiled at startup
    # (``precompile_buckets``). Above the ladder only synced eager steps
    # remain, which size receives by the real counts: they all use the
    # capacity. Smaller buckets may skip the sync (graph mode, or below
    # ``cpu_sync_min_tokens``) and allocate the worst case for their bucket,
    # so they round up only to the next rung: graph steps already sit on one
    # (vLLM pads them to a capture size). The rule depends only on the token
    # counts, because every rank must pass DeepEP the same bucket even when a
    # rank plans the step in another mode (an FFN eager fallback).
    if not bucket_ladder or bucket > bucket_ladder[-1]:
        bucket = capacity_tokens_per_rank
    else:
        bucket = next(rung for rung in bucket_ladder if rung >= bucket)
    return StagePlan(
        tokens_per_attn_rank=tokens,
        bucket=bucket,
        do_cpu_sync=do_cpu_sync,
        graph_mode=graph_mode,
    )


def no_sync_bucket_limit(
    *,
    capacity_tokens_per_rank: int,
    max_cudagraph_capture_size: int,
    attn_tp_size: int,
    bucket_granularity: int,
    cpu_sync_min_tokens: int,
) -> int:
    """Largest bucket a stage can dispatch without the host sync.

    ``plan_stage`` skips the sync in graph mode, where a DP rank's tokens are
    at most the largest CUDA-graph capture size and each Attention TP rank
    dispatches its slice, and in eager mode below ``cpu_sync_min_tokens``.
    Every other step syncs, and an Attention rank then receives no rows.
    """
    graph_tokens = -(-max_cudagraph_capture_size // attn_tp_size)
    graph_bucket = -(-graph_tokens // bucket_granularity) * bucket_granularity
    return min(capacity_tokens_per_rank, max(graph_bucket, cpu_sync_min_tokens - 1))


def bucket_ladder(
    *,
    capacity_tokens_per_rank: int,
    cudagraph_capture_sizes: Sequence[int],
    attn_tp_size: int,
    bucket_granularity: int,
    cpu_sync_min_tokens: int,
) -> tuple[int, ...]:
    """Sorted buckets that steps at or below the no-sync limit round up to.

    The rungs are the per-rank CUDA-graph buckets (each capture size's
    Attention TP slice, rounded to ``bucket_granularity``) plus the no-sync
    limit as the top rung. Empty when no step can skip the host sync (no
    graphs and ``cpu_sync_min_tokens`` 0): every eager step then uses the
    capacity.
    """
    limit = no_sync_bucket_limit(
        capacity_tokens_per_rank=capacity_tokens_per_rank,
        max_cudagraph_capture_size=max(cudagraph_capture_sizes, default=0),
        attn_tp_size=attn_tp_size,
        bucket_granularity=bucket_granularity,
        cpu_sync_min_tokens=cpu_sync_min_tokens,
    )
    if limit < 1:
        return ()
    rungs = {limit}
    for size in cudagraph_capture_sizes:
        tokens = -(-size // attn_tp_size)
        rung = -(-tokens // bucket_granularity) * bucket_granularity
        if 1 <= rung <= limit:
            rungs.add(rung)
    return tuple(sorted(rungs))


def precompile_buckets(
    ladder: Sequence[int], capacity_tokens_per_rank: int, cpu_sync_min_tokens: int
) -> tuple[tuple[int, bool], ...]:
    """``(bucket, do_cpu_sync)`` pairs ``plan_stage`` can give an eager step.

    A synced step lands on a rung at or above ``cpu_sync_min_tokens`` or on the
    capacity; a step below the threshold skips the sync and rounds up to the
    first rung that covers it. Graph steps also skip the sync, but CUDA-graph
    capture compiles their kernels already.
    """
    buckets = [(rung, True) for rung in ladder if rung >= cpu_sync_min_tokens]
    buckets.append((capacity_tokens_per_rank, True))
    if cpu_sync_min_tokens > 0 and ladder:
        covering = next(
            (rung for rung in ladder if rung >= cpu_sync_min_tokens - 1), ladder[-1]
        )
        buckets += [(rung, False) for rung in ladder if rung <= covering]
    return tuple(dict.fromkeys(buckets))


def worst_case_receive_rows(
    *,
    num_ranks: int,
    bucket: int,
    top_k: int,
    experts_per_rank: int,
    expert_alignment: int,
) -> int:
    """Rows DeepEP allocates for one rank's receive when it skips the host sync.

    Mirrors DeepEP V2's worst case for expanded receives (``ElasticBuffer``
    non-cached dispatch without CPU sync): every rank's bucket times the routes
    one token can send to this rank, plus alignment padding for each local
    expert.
    """
    rows = num_ranks * bucket * min(top_k, experts_per_rank)
    rows += (expert_alignment - 1) * experts_per_rank
    return -(-rows // expert_alignment) * expert_alignment


@dataclass(frozen=True)
class HandshakeResult:
    expert_alignment: int
    network_plan: NetworkPlan
    # Reported by Attention; FFN ranks adopt them.
    attn_tp_size: int
    capacity_tokens_per_rank: int
    bucket_ladder: tuple[int, ...]


def expert_weight_format(quant_config: QuantizationConfig | None) -> str:
    """Expert weight format vLLM will load, named like ``payload_format``."""
    if quant_config is None:
        return BF16_WEIGHTS
    if isinstance(quant_config, Fp8Config) and quant_config.weight_block_size == [
        FP8_GROUP_SIZE,
        FP8_GROUP_SIZE,
    ]:
        return FP8_BLOCK128_WEIGHTS
    return f"unsupported ({quant_config.get_name()})"


def ffn_deployment_info(vllm_config: VllmConfig) -> dict[str, Any]:
    """FFN-side vLLM settings that decide which experts each rank loads."""
    parallel_config = vllm_config.parallel_config
    return {
        "expert_parallel": parallel_config.enable_expert_parallel,
        "ep_size": parallel_config.data_parallel_size
        * parallel_config.tensor_parallel_size,
        "tensor_parallel_size": parallel_config.tensor_parallel_size,
        "expert_placement_strategy": parallel_config.expert_placement_strategy,
        "eplb": parallel_config.enable_eplb,
        "all2all_backend": parallel_config.all2all_backend,
        "weight_format": expert_weight_format(vllm_config.quant_config),
    }


def local_handshake_info(
    *,
    role: str,
    role_rank: int,
    world_rank: int,
    config: DeepEPM2NConfig,
    hidden_size: int,
    dtype: torch.dtype,
    num_experts: int,
    top_k: int,
    num_lanes: int,
    capacity_tokens_per_rank: int | None,
    bucket_ladder: Sequence[int] | None,
    attn_tp_size: int | None,
    compute_gate_on_attention: bool,
    expert_alignment: int | None,
    ffn_deployment: dict[str, Any] | None,
    sm: tuple[int, int],
    topology: RankTopology,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "role": role,
        "role_rank": role_rank,
        "world_rank": world_rank,
        "sm": list(sm),
        "payload_format": config.payload_format,
        "hidden_size": hidden_size,
        "dtype": str(dtype),
        "num_experts": num_experts,
        "top_k": top_k,
        "num_lanes": num_lanes,
        "capacity_tokens_per_rank": capacity_tokens_per_rank,
        "bucket_ladder": None if bucket_ladder is None else list(bucket_ladder),
        "bucket_granularity": config.bucket_granularity,
        "cpu_sync_min_tokens": config.cpu_sync_min_tokens,
        "attn_tp_size": attn_tp_size,
        "compute_gate_on_attention": compute_gate_on_attention,
        "network": str(config.network.to_mapping()),
        "expert_alignment": expert_alignment,
        "ffn_deployment": ffn_deployment,
        "topology": topology.to_mapping(),
    }


_MUST_MATCH = (
    "protocol",
    "payload_format",
    "hidden_size",
    "dtype",
    "num_experts",
    "top_k",
    "num_lanes",
    "compute_gate_on_attention",
    "network",
    "bucket_granularity",
    "cpu_sync_min_tokens",
)


def _check_ffn_deployment(
    world_rank: int,
    deployment: dict[str, Any] | None,
    layout: DeepEPM2NLayout,
    payload_format: str,
) -> None:
    """vLLM must place experts on FFN ranks exactly as DeepEP routes them.

    Otherwise rows reach a rank that holds other experts and the results are
    silently wrong. Every rank runs this, so Attention fails with the FFN.
    """
    if deployment is None:
        raise RuntimeError(f"FFN world rank {world_rank} sent no deployment info")
    expected = {
        "expert_parallel": True,
        "ep_size": layout.ffn_size,
        "tensor_parallel_size": 1,
        "expert_placement_strategy": "linear",
        "eplb": False,
        "all2all_backend": IDLE_ALL2ALL_BACKEND,
        "weight_format": PAYLOAD_WEIGHT_FORMATS[payload_format],
    }
    mismatches = {
        key: deployment[key]
        for key, value in expected.items()
        if deployment[key] != value
    }
    if mismatches:
        required = {key: expected[key] for key in mismatches}
        raise RuntimeError(
            f"FFN world rank {world_rank} is deployed with {mismatches}; "
            f"DeepEP M2N with payload_format={payload_format!r} needs {required}"
        )


def validate_handshake(
    infos: list[dict[str, Any]], layout: DeepEPM2NLayout, network: NetworkConfig
) -> HandshakeResult:
    """Fail fast, before any buffer allocation, if ranks disagree."""
    if len(infos) != layout.union_size:
        raise RuntimeError(
            f"handshake has {len(infos)} ranks, expected {layout.union_size}"
        )
    for world_rank, info in enumerate(infos):
        expected_role = "ffn" if world_rank < layout.ffn_size else "attention"
        if info["world_rank"] != world_rank or info["role"] != expected_role:
            raise RuntimeError(
                f"world rank {world_rank} reported role={info['role']!r} "
                f"world_rank={info['world_rank']}, expected {expected_role!r}"
            )
        if info["sm"][0] < MIN_SM_MAJOR:
            raise RuntimeError(
                f"world rank {world_rank} has SM{info['sm'][0]}{info['sm'][1]}; "
                f"DeepEP M2N needs SM{MIN_SM_MAJOR}0 or newer"
            )
    for key in _MUST_MATCH:
        values = {str(info[key]) for info in infos}
        if len(values) != 1:
            raise RuntimeError(f"ranks disagree on {key}: {sorted(values)}")
    if not infos[0]["compute_gate_on_attention"]:
        raise RuntimeError(
            "DeepEP M2N routes tokens on Attention: set "
            "compute_gate_on_attention=true on both roles"
        )
    for world_rank, info in enumerate(infos[: layout.ffn_size]):
        _check_ffn_deployment(
            world_rank, info["ffn_deployment"], layout, info["payload_format"]
        )
    alignments = {info["expert_alignment"] for info in infos[: layout.ffn_size]}
    if len(alignments) != 1 or None in alignments:
        raise RuntimeError(f"FFN ranks disagree on expert_alignment: {alignments}")
    attn_infos = infos[layout.ffn_size :]
    tp_sizes = {info["attn_tp_size"] for info in attn_infos}
    if len(tp_sizes) != 1 or None in tp_sizes:
        raise RuntimeError(f"Attention ranks disagree on their TP size: {tp_sizes}")
    attn_tp_size = int(tp_sizes.pop())
    if attn_tp_size <= 0 or layout.attn_size % attn_tp_size != 0:
        raise RuntimeError(
            f"Attention TP size {attn_tp_size} does not divide the "
            f"{layout.attn_size} Attention ranks"
        )
    # Attention resolves the capacity (it knows its TP size); an FFN rank
    # reports one only when it is configured explicitly, and it must agree.
    capacities = {info["capacity_tokens_per_rank"] for info in attn_infos}
    capacities |= {
        info["capacity_tokens_per_rank"]
        for info in infos[: layout.ffn_size]
        if info["capacity_tokens_per_rank"] is not None
    }
    if len(capacities) != 1 or None in capacities:
        raise RuntimeError(
            "ranks disagree on capacity_tokens_per_rank: "
            f"{sorted(map(str, capacities))}"
        )
    # Attention resolves it from its CUDA-graph sizes; plan_stage needs the
    # same ladder on every rank to pick the same bucket.
    ladders = {
        None if info["bucket_ladder"] is None else tuple(info["bucket_ladder"])
        for info in attn_infos
    }
    if len(ladders) != 1 or None in ladders:
        raise RuntimeError(
            f"Attention ranks disagree on bucket_ladder: {sorted(map(str, ladders))}"
        )
    # Every rank evaluates the same plan on the same list.
    plan = plan_network(
        [RankTopology.from_mapping(info["topology"]) for info in infos], network
    )
    return HandshakeResult(
        expert_alignment=int(alignments.pop()),
        network_plan=plan,
        attn_tp_size=attn_tp_size,
        capacity_tokens_per_rank=int(capacities.pop()),
        bucket_ladder=ladders.pop() or (),
    )


def _describe_tensor(tensor: torch.Tensor | None) -> list[Any] | None:
    return None if tensor is None else [str(tensor.dtype), list(tensor.shape)]


def local_model_info(
    moe_layer_indices: Sequence[int] | None,
    local_experts: Mapping[int, FFNLocalExperts] | None,
) -> dict[str, Any] | None:
    """What this rank loaded, in a picklable form; ``None`` if not reported."""
    if moe_layer_indices is None:
        return None
    experts = None
    if local_experts is not None:
        experts = {
            layer_idx: {
                "ids": list(layer_experts.expert_ids),
                "w13": _describe_tensor(layer_experts.weights.w13),
                "w2": _describe_tensor(layer_experts.weights.w2),
                "w13_scale": _describe_tensor(layer_experts.weights.w13_scale),
                "w2_scale": _describe_tensor(layer_experts.weights.w2_scale),
            }
            for layer_idx, layer_experts in local_experts.items()
        }
    return {"moe_layers": list(moe_layer_indices), "experts": experts}


def _expected_expert_tensors(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    payload_format: str,
) -> dict[str, list[Any] | None]:
    """dtype and shape of every ``LocalExpertWeights`` tensor."""
    if payload_format == BF16_PAYLOAD:
        weight_dtype = str(torch.bfloat16)
        w13_scale = w2_scale = None
    else:
        weight_dtype = str(FP8_DTYPE)
        hidden_blocks = -(-hidden_size // FP8_GROUP_SIZE)
        intermediate_blocks = -(-intermediate_size // FP8_GROUP_SIZE)
        w13_scale = [
            str(torch.float32),
            [num_experts, -(-2 * intermediate_size // FP8_GROUP_SIZE), hidden_blocks],
        ]
        w2_scale = [
            str(torch.float32),
            [num_experts, hidden_blocks, intermediate_blocks],
        ]
    return {
        "w13": [weight_dtype, [num_experts, 2 * intermediate_size, hidden_size]],
        "w2": [weight_dtype, [num_experts, hidden_size, intermediate_size]],
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
    }


def validate_model_infos(
    infos: list[dict[str, Any] | None],
    layout: DeepEPM2NLayout,
    *,
    hidden_size: int,
    payload_format: str,
) -> None:
    """After weights load, check what every rank actually holds.

    All ranks must run the same MoE layers (dispatch and combine are
    collectives issued once per MoE layer), and every FFN rank must hold
    exactly the experts DeepEP routes to it, in the payload's format. Ranks
    that report nothing (``None``) are skipped; an FFN rank must report its
    experts once any rank reports.
    """
    reported = [
        (world_rank, info) for world_rank, info in enumerate(infos) if info is not None
    ]
    if not reported:
        return
    layer_lists = {str(info["moe_layers"]) for _, info in reported}
    if len(layer_lists) != 1:
        raise RuntimeError(f"ranks disagree on MoE layers: {sorted(layer_lists)}")
    moe_layers = reported[0][1]["moe_layers"]
    for world_rank in range(layout.ffn_size):
        info = infos[world_rank]
        if info is None or info["experts"] is None:
            raise RuntimeError(f"FFN world rank {world_rank} reported no local experts")
        experts = info["experts"]
        if sorted(experts) != sorted(moe_layers):
            raise RuntimeError(
                f"FFN world rank {world_rank} holds experts for layers "
                f"{sorted(experts)}, expected MoE layers {moe_layers}"
            )
        expected_ids = list(layout.expert_range(world_rank))
        for layer_idx, described in experts.items():
            if described["ids"] != expected_ids:
                raise RuntimeError(
                    f"FFN world rank {world_rank} layer {layer_idx} holds experts "
                    f"{described['ids']}, but DeepEP M2N routes experts "
                    f"[{expected_ids[0]}, {expected_ids[-1] + 1}) to it: use "
                    "linear expert placement with EP size = num_ffn_ranks"
                )
            intermediate_size = described["w2"][1][2]
            expected = _expected_expert_tensors(
                layout.experts_per_rank, hidden_size, intermediate_size, payload_format
            )
            wrong = {
                name: described[name]
                for name, value in expected.items()
                if described[name] != value
            }
            if wrong:
                required = {name: expected[name] for name in wrong}
                raise RuntimeError(
                    f"FFN world rank {world_rank} layer {layer_idx} expert tensors "
                    f"{wrong} do not match payload_format={payload_format!r}, "
                    f"expected {required}"
                )


@dataclass(slots=True)
class DeepEPM2NTransferState(AFDTransferState):
    """FFN-side state from ``recv_attn_output`` needed by experts and combine."""

    handle: DispatchHandle
    recv_topk_weights: torch.Tensor
    recv_count_psum: torch.Tensor
    recv_counts_host: list[int]
    recv_scale: torch.Tensor | None
    lane: int
    # Recorded on the lane stream after dispatch; experts wait on it.
    ready_event: torch.cuda.Event
    # Recorded on the caller's stream after this lane's experts; the combine
    # waits on it instead of on everything the caller queued.
    experts_done: torch.cuda.Event | None = None


# Custom ops cannot take Python objects, so they find their connector by id
# (same pattern as the P2P connector's communicator registry).
_CONNECTORS: dict[int, DeepEPM2NAFDConnector] = {}
_CONNECTOR_IDS = itertools.count()
_DISPATCH_OP = "afd_deepep_m2n_dispatch"
_COMBINE_OP = "afd_deepep_m2n_combine"


def _live_stage_idx(traced_stage_idx: int) -> int:
    """The stage of the calling micro-batch, read when the op body runs.

    ``torch.compile`` freezes the ``stage_idx`` op argument at trace time, and
    vLLM's DBO micro-batch threads share one compiled graph, so every thread
    would pass the same stage. Inside a DBO thread the vLLM micro-batch id is
    the stage (one AFD stage per micro-batch); otherwise, as in direct eager
    calls, the argument is right.
    """
    return dbo_current_ubatch_id() if dbo_enabled() else traced_stage_idx


def _dispatch_op_impl(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    connector_id: int,
    stage_idx: int,
) -> None:
    _CONNECTORS[connector_id].dispatch_attention(
        hidden_states, topk_ids, topk_weights, _live_stage_idx(stage_idx)
    )


def _dispatch_op_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    connector_id: int,
    stage_idx: int,
) -> None:
    return None


def _combine_op_impl(out: torch.Tensor, connector_id: int, stage_idx: int) -> None:
    _CONNECTORS[connector_id].combine_attention(out, _live_stage_idx(stage_idx))


def _combine_op_fake(out: torch.Tensor, connector_id: int, stage_idx: int) -> None:
    return None


def _register_custom_ops() -> None:
    """Register the Attention-side ops once per process.

    Wrapping dispatch and combine in opaque custom ops keeps them intact under
    ``torch.compile``. ``hidden_states`` / ``out`` are declared mutated so the
    side-effecting ops are never dropped or reordered.
    """
    if not hasattr(torch.ops.vllm, _DISPATCH_OP):
        direct_register_custom_op(
            op_name=_DISPATCH_OP,
            op_func=_dispatch_op_impl,
            mutates_args=["hidden_states"],
            fake_impl=_dispatch_op_fake,
        )
    if not hasattr(torch.ops.vllm, _COMBINE_OP):
        direct_register_custom_op(
            op_name=_COMBINE_OP,
            op_func=_combine_op_impl,
            mutates_args=["out"],
            fake_impl=_combine_op_fake,
        )


def _keep_alive_on(stream: torch.cuda.Stream, *tensors: torch.Tensor | None) -> None:
    """Tell the caching allocator ``tensors`` are used on ``stream``.

    Also while capturing a CUDA graph: within one capture the allocator reuses
    a freed graph-pool block for the next allocation on the freeing stream,
    which would overwrite data a lane still reads on replay (a micro-batch's
    hidden states under its dispatch, combine output under the copy). PyTorch
    tracks stream uses recorded during capture and keeps such blocks out of
    reuse until the capture ends.
    """
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


class DeepEPM2NAFDConnector(AFDRoutedExpertsConnector):
    """Expert-routed Attention <-> FFN connector over afdEP's DeepEP V2.

    Each stage (micro-batch) has a lane: its own DeepEP buffer, its own DeepEP
    communication stream, and a lane stream on which the connector issues that
    lane's dispatch and combine. Lane work joins the caller's stream only
    where results are consumed (``recv_ffn_output``, ``run_experts``,
    ``synchronize_lanes``), so two lanes can overlap communication with
    compute (DBO order ``D0 D1 C0 C1``). With one lane everything runs on the
    caller's stream, which avoids the fork/join cost in eager mode.

    CUDA graphs: capture with ``is_graph_capturing``/``is_graph_replaying``
    payloads (no host sync, fixed worst-case shapes) after one warm-up step at
    the same bucket (it JIT-compiles the kernels and caches the Attention-side
    zero buffer). FFN ranks call ``synchronize_lanes`` before the capture ends.

    FFN memory: a lane's received rows and its expert output live in that
    lane's caching-allocator pool, without ``record_stream`` on the caller's
    stream. The caller's stream uses them only in that lane's experts, which
    the lane's combine waits for, and the lane's next allocation is used only
    by work issued after that combine. So the lane can reuse a freed block at
    once, also within one CUDA-graph capture: a capture holds about two
    layers' rows per lane, not every layer's.

    Lifecycle: ``bootstrap`` (before vLLM's distributed init: handshake and
    NCCL environment), then ``init_afd_connector`` (after weights load:
    post-load checks, NCCL control group and DeepEP buffers).
    """

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> DeepEPM2NConfig:
        return DeepEPM2NConfig.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        if not isinstance(self.extra_info, DeepEPM2NConfig):
            raise TypeError("DeepEPM2NAFDConnector expects DeepEPM2NConfig extra info")
        self.config = self.extra_info
        text_config = vllm_config.model_config.hf_text_config
        self.hidden_size: int = text_config.hidden_size
        self.top_k: int = text_config.num_experts_per_tok
        self.dtype: torch.dtype = vllm_config.model_config.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError(f"DeepEP M2N needs a bfloat16 model, got {self.dtype}")
        self.layout = DeepEPM2NLayout(
            ffn_size=afd_config.num_ffn_ranks,
            attn_size=afd_config.num_attention_ranks,
            num_experts=text_config.num_experts,
        )
        self.attn_size = self.layout.attn_size
        self.ffn_size = self.layout.ffn_size
        self.is_ffn = afd_config.role == "ffn"
        self.is_fp8_payload = self.config.payload_format == FP8_PAYLOAD
        self.payload_dtype = FP8_DTYPE if self.is_fp8_payload else self.dtype
        if self.is_fp8_payload and self.hidden_size % FP8_GROUP_SIZE != 0:
            raise ValueError(f"FP8 payload needs hidden_size % {FP8_GROUP_SIZE} == 0")
        self.world_rank = self.layout.world_rank(afd_config.role, role_rank)
        # The FFN runs without TP and learns the Attention TP size and the
        # resulting capacity in the handshake (``bootstrap``).
        self.attn_tp_size: int | None = (
            None if self.is_ffn else vllm_config.parallel_config.tensor_parallel_size
        )
        self.capacity_tokens_per_rank: int | None = (
            self.config.capacity_tokens_per_rank
            if self.is_ffn
            else resolve_capacity_tokens_per_rank(self.config, vllm_config)
        )
        # Buckets below the capacity (``plan_stage``); Attention derives them
        # from its CUDA-graph sizes, the FFN adopts them in the handshake.
        self.bucket_ladder: tuple[int, ...] | None = None
        if not self.is_ffn:
            assert self.attn_tp_size is not None
            assert self.capacity_tokens_per_rank is not None
            self.bucket_ladder = bucket_ladder(
                capacity_tokens_per_rank=self.capacity_tokens_per_rank,
                cudagraph_capture_sizes=(
                    vllm_config.compilation_config.cudagraph_capture_sizes or ()
                ),
                attn_tp_size=self.attn_tp_size,
                bucket_granularity=self.config.bucket_granularity,
                cpu_sync_min_tokens=self.config.cpu_sync_min_tokens,
            )
        self.num_lanes = resolve_num_lanes(self.config, vllm_config)
        self.device = torch.device("cuda", local_rank)
        # Where control payloads are staged: gloo moves CPU tensors.
        self.control_device = (
            torch.device("cpu")
            if self.config.control_backend == "gloo"
            else self.device
        )
        self._rendezvous = f"tcp://{afd_config.host}:{afd_config.port}"
        self._timeout = timedelta(seconds=afd_config.afd_process_group_timeout_s)
        self.group: dist.ProcessGroup | None = None
        self.bootstrap_group: dist.ProcessGroup | None = None
        self.network_plan: NetworkPlan | None = None
        self.lanes: list[ElasticBuffer] = []
        self.lane_streams: list[torch.cuda.Stream] = []
        self.experts: DeepEPM2NExperts | None = None
        self.expert_alignment = 1
        self.stage_plans: dict[int, StagePlan] = {}
        # Attention side: dispatch handles waiting for their combine, by stage.
        self._pending_combines: dict[int, DispatchHandle] = {}
        # FFN side: combine completion per lane, joined by synchronize_lanes.
        self._lane_done: dict[int, torch.cuda.Event] = {}
        # FFN side: lanes join the caller's stream once per step, at an event
        # recorded before any experts of the step (see _join_step).
        self._step_fork: torch.cuda.Event | None = None
        self._step_fork_capturing = False
        self._lanes_joined: set[int] = set()
        # Attention side: the zero rows combine takes from Attention ranks,
        # sized for the largest receive (see _allocate_combine_zeros).
        self._combine_zeros: torch.Tensor | None = None
        self._op_id = next(_CONNECTOR_IDS)
        self._initialized = False
        self.control_plane = DeepEPM2NAFDControlPlane(self)

    # ==============================
    # Lifecycle
    # ==============================

    def bootstrap(self) -> None:
        """Collective: join the AFD world, check compatibility, set NCCL env.

        Workers call this before vLLM's distributed init, because NCCL reads
        its environment (``NCCL_IB_HCA``, the GIN switch) once per process,
        when the first communicator is created. Uses gloo only, so it creates
        no NCCL communicator itself. ``init_afd_connector`` calls it when the
        caller has not.
        """
        if self.bootstrap_group is not None:
            return
        torch.cuda.set_device(self.device)
        self.bootstrap_group = init_afd_process_group(
            backend="gloo",
            init_method=self._rendezvous,
            world_size=self.layout.union_size,
            rank=self.world_rank,
            group_name=BOOTSTRAP_GROUP_NAME,
            timeout=self._timeout,
        )
        local_alignment = None
        if self.is_ffn:
            local_alignment = (
                DeepEPM2NExperts.deepgemm_alignment()
                if self.config.expert_backend == "deepgemm"
                else 1
            )
        info = local_handshake_info(
            role=self.afd_config.role,
            role_rank=self.role_rank,
            world_rank=self.world_rank,
            config=self.config,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            num_experts=self.layout.num_experts,
            top_k=self.top_k,
            num_lanes=self.num_lanes,
            capacity_tokens_per_rank=self.capacity_tokens_per_rank,
            bucket_ladder=self.bucket_ladder,
            attn_tp_size=self.attn_tp_size,
            compute_gate_on_attention=self.afd_config.compute_gate_on_attention,
            expert_alignment=local_alignment,
            ffn_deployment=(
                ffn_deployment_info(self.vllm_config) if self.is_ffn else None
            ),
            sm=torch.cuda.get_device_capability(self.device),
            topology=discover_local_topology(
                self.world_rank, self.device, self.local_rank
            ),
        )
        infos: list[dict[str, Any] | None] = [None] * self.layout.union_size
        dist.all_gather_object(infos, info, group=self.bootstrap_group)
        result = validate_handshake(
            [info for info in infos if info is not None],
            self.layout,
            self.config.network,
        )
        self.expert_alignment = result.expert_alignment
        self.network_plan = result.network_plan
        self.attn_tp_size = result.attn_tp_size
        self.capacity_tokens_per_rank = result.capacity_tokens_per_rank
        self.bucket_ladder = result.bucket_ladder
        if self.world_rank == 0:
            logger.info("DeepEP M2N network plan: %s", self.network_plan.summary())
        # Before any NCCL communicator in this process exists, when the worker
        # bootstraps ahead of vLLM's distributed init.
        apply_rank_environment(self.network_plan, self.config.network, self.world_rank)
        # DeepEP reads this when each buffer is created: with several lanes
        # each buffer needs its own communication stream.
        if self.num_lanes > 1:
            os.environ[PER_BUFFER_COMM_STREAM_ENV] = "1"

    def init_afd_connector(
        self,
        *,
        moe_layer_indices: Sequence[int] | None = None,
        local_experts: Mapping[int, FFNLocalExperts] | None = None,
    ) -> None:
        """Collective: check what every rank loaded, then create buffers.

        Args:
            moe_layer_indices: MoE layers this rank's model runs, in order.
                ``None`` skips the post-load checks for this rank.
            local_experts: FFN only: this rank's experts per MoE layer, as
                vLLM loaded them.
        """
        if self._initialized:
            return
        if local_experts is not None and not self.is_ffn:
            raise ValueError("only FFN ranks hold local experts")
        self.bootstrap()
        torch.cuda.set_device(self.device)
        _register_custom_ops()
        assert self.bootstrap_group is not None and self.network_plan is not None
        assert self.capacity_tokens_per_rank is not None
        model_infos: list[dict[str, Any] | None] = [None] * self.layout.union_size
        dist.all_gather_object(
            model_infos,
            local_model_info(moe_layer_indices, local_experts),
            group=self.bootstrap_group,
        )
        validate_model_infos(
            model_infos,
            self.layout,
            hidden_size=self.hidden_size,
            payload_format=self.config.payload_format,
        )
        self.group = (
            self.bootstrap_group
            if self.config.control_backend == "gloo"
            else init_afd_process_group(
                backend="nccl",
                init_method=self._rendezvous,
                world_size=self.layout.union_size,
                rank=self.world_rank,
                group_name=PROCESS_GROUP_NAME,
                timeout=self._timeout,
            )
        )
        try:
            for _ in range(self.num_lanes):
                self.lanes.append(
                    self._new_buffer(
                        group=self.bootstrap_group,
                        capacity_tokens_per_rank=self.capacity_tokens_per_rank,
                        hidden_size=self.hidden_size,
                        num_experts=self.layout.deepep_num_experts,
                        top_k=self.top_k,
                        allow_hybrid_mode=self.network_plan.hybrid,
                        use_fp8_dispatch=self.is_fp8_payload,
                        num_sms=self.config.num_sms,
                        num_qps=self.config.num_qps,
                        rdma_traffic_class=self.network_plan.rdma_traffic_class,
                        cpu_timeout_s=self.config.cpu_timeout_s,
                        gpu_timeout_s=self.config.gpu_timeout_s,
                    )
                )
                # One lane runs on the caller's stream: no fork/join cost.
                if self.num_lanes > 1:
                    self.lane_streams.append(torch.cuda.Stream(self.device))
            if self.is_ffn:
                self.experts = DeepEPM2NExperts(
                    self.config.expert_backend,
                    self.expert_alignment,
                    self.config.expert_num_sms,
                )
            else:
                self._allocate_combine_zeros()
            self._precompile_dispatch_kernels()
        except Exception:
            self.close()
            raise
        _CONNECTORS[self._op_id] = self
        self._initialized = True

    def _new_buffer(self, **kwargs: Any) -> ElasticBuffer:
        """One lane's afdEP transport."""
        # The DeepEP M2N layout puts FFN ranks first, as afdEP's role path expects.
        return ElasticBuffer(
            **kwargs, role_aware=self.config.role_aware, ffn_ranks=self.ffn_size
        )

    def close(self) -> None:
        """Destroy DeepEP buffers; safe after partial init and on repeat calls."""
        _CONNECTORS.pop(self._op_id, None)
        for lane in self.lanes:
            lane.destroy()
        self.lanes = []
        self.lane_streams = []
        self._pending_combines.clear()
        self._lane_done.clear()
        self._reset_step_fork()
        self._combine_zeros = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def domain(self) -> tuple[int, int]:
        """DeepEP ``(scale-out ranks, scale-up ranks)``: nodes and GPUs per node
        in hybrid mode, ``(1, world)`` when everything is in one NVLink domain."""
        self._check_initialized()
        return self.lanes[0].num_scaleout_ranks, self.lanes[0].num_scaleup_ranks

    # ==============================
    # Attention side
    # ==============================

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        *,
        topk_ids: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch each token to the FFN ranks owning its top-k experts.

        Args:
            hidden_states: ``[tokens, hidden]`` BF16 Attention output.
            context: Transfer context; ``metadata.stage_idx`` selects the lane
                (inside a vLLM DBO micro-batch thread, its micro-batch id does).
            topk_ids: ``[tokens, top_k]`` real expert ids. Required: routing
                runs on the Attention side with this connector.
            topk_weights: ``[tokens, top_k]`` routing weights. Required.
        """
        self._check_initialized()
        if topk_ids is None or topk_weights is None:
            raise ValueError("DeepEPM2NAFDConnector needs topk_ids and topk_weights")
        if not context.metadata.validate_tensor_shape(tuple(hidden_states.shape)):
            raise ValueError(
                f"hidden_states shape {tuple(hidden_states.shape)} does not match "
                f"{context.metadata.total_tokens} tokens"
            )
        torch.ops.vllm.afd_deepep_m2n_dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            self._op_id,
            context.metadata.stage_idx,
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Combine into ``ref_tensor``: ``sum_k weight_k * expert_k(token)``."""
        self._check_initialized()
        torch.ops.vllm.afd_deepep_m2n_combine(ref_tensor, self._op_id, ubatch_idx)
        return ref_tensor

    def dispatch_attention(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        stage_idx: int,
    ) -> None:
        """Eager body of the dispatch custom op (runs on the stage's lane)."""
        self._check_initialized()
        if stage_idx in self._pending_combines:
            raise RuntimeError(f"stage {stage_idx} dispatched twice without combine")
        lane_stream = self._lane_stream(stage_idx)
        self._fork_to(lane_stream)
        with torch.cuda.stream(lane_stream):
            x, x_scale = hidden_states, None
            if self.is_fp8_payload:
                x, x_scale = per_token_group_quant_fp8(
                    hidden_states, FP8_GROUP_SIZE, use_ue8m0=False
                )
            result = self._dispatch(stage_idx, x, x_scale, topk_ids, topk_weights)
        _keep_alive_on(lane_stream, hidden_states, topk_ids, topk_weights)
        self._pending_combines[stage_idx] = result.handle

    def combine_attention(self, out: torch.Tensor, stage_idx: int) -> None:
        """Eager body of the combine custom op: lane combine, then copy to out."""
        self._check_initialized()
        handle = self._pending_combines.pop(stage_idx)
        lane_stream = self._lane_stream(stage_idx)
        with torch.cuda.stream(lane_stream):
            # Attention ranks hold no experts; they contribute zero rows.
            combined = self._lanes(stage_idx).combine(
                self._combine_zero_rows(handle.recv_shape), handle
            )
        if combined.shape != out.shape:
            raise RuntimeError(
                f"combined shape {tuple(combined.shape)} != out {tuple(out.shape)}"
            )
        current = torch.cuda.current_stream(self.device)
        if lane_stream != current:
            current.wait_stream(lane_stream)
            _keep_alive_on(current, combined)
        out.copy_(combined)

    # ==============================
    # FFN side
    # ==============================

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Dispatch with no tokens; return the rows routed to local experts.

        The dispatch runs on the stage's lane; ``run_experts`` waits for it, so
        a later lane's dispatch can overlap this lane's experts. The lane does
        not wait for the caller's stream: its inputs are empty, and this
        lane's previous combine, which waits for the experts that read the
        lane's last rows, is ahead of it on the lane (see _join_step). The
        same order lets the rows stay in the lane's allocator pool (see the
        class docstring).
        """
        self._check_initialized()
        lane_stream = self._lane_stream(ubatch_idx)
        self._join_step(ubatch_idx, lane_stream)
        with torch.cuda.stream(lane_stream):
            empty = torch.empty(
                (0, self.hidden_size), dtype=self.payload_dtype, device=self.device
            )
            empty_scale = (
                torch.empty(
                    (0, self.hidden_size // FP8_GROUP_SIZE),
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.is_fp8_payload
                else None
            )
            no_routes = torch.empty(
                (0, self.top_k), dtype=torch.int64, device=self.device
            )
            result = self._dispatch(
                ubatch_idx, empty, empty_scale, no_routes, no_routes.float()
            )
            ready_event = torch.cuda.Event()
            ready_event.record(lane_stream)
        state = DeepEPM2NTransferState(
            handle=result.handle,
            recv_topk_weights=result.recv_topk_weights,
            recv_count_psum=result.recv_count_psum,
            recv_counts_host=result.recv_counts_host,
            recv_scale=result.recv_scale,
            lane=ubatch_idx,
            ready_event=ready_event,
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=ubatch_idx,
            seq_lens=[result.handle.recv_shape[0]],
        )
        return AFDA2FTransferPayload(
            hidden_states=result.recv_x,
            context=AFDTransferContext(metadata=metadata, states=state),
        )

    def run_experts(
        self, payload: AFDA2FTransferPayload, weights: LocalExpertWeights
    ) -> torch.Tensor:
        """MoE FFN on received rows: top-k weighted local-expert outputs."""
        self._check_initialized()
        if self.experts is None:
            raise RuntimeError("run_experts is only available on FFN ranks")
        state = self._transfer_state(payload.context)
        current = torch.cuda.current_stream(self.device)
        current.wait_event(state.ready_event)
        # In the lane's pool, like the rows (see the class docstring).
        with torch.cuda.stream(self._lane_stream(state.lane)):
            out = torch.empty(
                payload.hidden_states.shape, dtype=torch.bfloat16, device=self.device
            )
        out = self.experts(
            payload.hidden_states,
            state.recv_scale,
            state.recv_topk_weights,
            state.recv_count_psum,
            weights,
            out=out,
        )
        if self.lane_streams:
            experts_done = torch.cuda.Event()
            experts_done.record(current)
            state.experts_done = experts_done
        return out

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Combine on the lane: send each weighted row back to its token's rank.

        The lane waits only for this lane's experts (``run_experts``), not for
        other lanes' experts queued after them on the caller's stream; under
        DBO that wait would hold the other micro-batch's Attention.
        """
        self._check_initialized()
        state = self._transfer_state(context)
        lane_stream = self._lane_stream(state.lane)
        if state.experts_done is not None and lane_stream != torch.cuda.current_stream(
            self.device
        ):
            lane_stream.wait_event(state.experts_done)
            self._lanes_joined.add(state.lane)
        else:
            # Output not produced by run_experts: order after everything.
            self._fork_to(lane_stream)
        _keep_alive_on(lane_stream, ffn_output)
        with torch.cuda.stream(lane_stream):
            self._lanes(state.lane).combine(ffn_output, state.handle)
            done = torch.cuda.Event()
            done.record(lane_stream)
        self._lane_done[state.lane] = done

    def synchronize_lanes(self) -> None:
        """Make the caller's stream wait for every lane's outstanding combine.

        FFN ranks call this once per step (and before ending a CUDA-graph
        capture, which requires all forked streams to rejoin).
        """
        current = torch.cuda.current_stream(self.device)
        for done in self._lane_done.values():
            current.wait_event(done)
        self._lane_done.clear()
        self._reset_step_fork()

    # ==============================
    # Internals
    # ==============================

    def _dispatch(
        self,
        stage_idx: int,
        x: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> ReceiveResult:
        plan = self.stage_plans.get(stage_idx)
        if plan is None:
            raise RuntimeError(
                f"no plan for stage {stage_idx}; call "
                "control_plane.update_state_from_dp_metadata first"
            )
        return self._lanes(stage_idx).dispatch(
            x,
            x_scale,
            topk_ids,
            topk_weights,
            bucket=plan.bucket,
            expert_alignment=self.expert_alignment,
            do_cpu_sync=plan.do_cpu_sync,
        )

    def _lanes(self, stage_idx: int) -> ElasticBuffer:
        if not 0 <= stage_idx < len(self.lanes):
            raise RuntimeError(f"stage {stage_idx} has no lane ({len(self.lanes)})")
        return self.lanes[stage_idx]

    def _lane_stream(self, stage_idx: int) -> torch.cuda.Stream:
        self._lanes(stage_idx)
        if not self.lane_streams:
            return torch.cuda.current_stream(self.device)
        return self.lane_streams[stage_idx]

    def _fork_to(self, lane_stream: torch.cuda.Stream) -> None:
        """Order ``lane_stream`` after the caller's work (no-op for one lane)."""
        current = torch.cuda.current_stream(self.device)
        if lane_stream != current:
            lane_stream.wait_stream(current)

    def _join_step(self, lane: int, lane_stream: torch.cuda.Stream) -> None:
        """Order ``lane_stream`` after the caller's work from before this step.

        The first FFN lane operation of a step records one event on the
        caller's stream, before any of the step's experts; each lane waits on
        it once. This orders the lane after the previous step and, while
        capturing, joins the lane to the capture. Later operations on the lane
        are ordered by the lane itself. A change of capture mode starts over.
        """
        current = torch.cuda.current_stream(self.device)
        if lane_stream == current:
            return
        capturing = torch.cuda.is_current_stream_capturing()
        if self._step_fork is not None and capturing != self._step_fork_capturing:
            self._reset_step_fork()
        if lane in self._lanes_joined:
            return
        if self._step_fork is None:
            self._step_fork = torch.cuda.Event()
            self._step_fork.record(current)
            self._step_fork_capturing = capturing
        lane_stream.wait_event(self._step_fork)
        self._lanes_joined.add(lane)

    def _reset_step_fork(self) -> None:
        self._step_fork = None
        self._step_fork_capturing = False
        self._lanes_joined.clear()

    def _precompile_dispatch_kernels(self) -> None:
        """Compile every eager dispatch/combine variant before serving.

        Collective: every rank runs one empty dispatch and combine per lane
        for each ``precompile_buckets`` pair, in the same order, so DeepEP's
        JIT builds them now instead of on the first step of each size. Graph
        buckets compile during CUDA-graph capture.
        """
        assert self.bucket_ladder is not None
        assert self.capacity_tokens_per_rank is not None
        buckets = precompile_buckets(
            self.bucket_ladder,
            self.capacity_tokens_per_rank,
            self.config.cpu_sync_min_tokens,
        )
        x = torch.empty(
            (0, self.hidden_size), dtype=self.payload_dtype, device=self.device
        )
        x_scale = (
            torch.empty(
                (0, self.hidden_size // FP8_GROUP_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            if self.is_fp8_payload
            else None
        )
        topk_ids = torch.empty((0, self.top_k), dtype=torch.int64, device=self.device)
        topk_weights = torch.empty(
            (0, self.top_k), dtype=torch.float32, device=self.device
        )
        start = time.perf_counter()
        for lane in self.lanes:
            for bucket, do_cpu_sync in buckets:
                result = lane.dispatch(
                    x,
                    x_scale,
                    topk_ids,
                    topk_weights,
                    bucket=bucket,
                    expert_alignment=self.expert_alignment,
                    do_cpu_sync=do_cpu_sync,
                )
                recv_shape = result.handle.recv_shape
                expert_output = (
                    torch.zeros(recv_shape, dtype=self.dtype, device=self.device)
                    if self.is_ffn
                    else self._combine_zero_rows(recv_shape)
                )
                lane.combine(expert_output, result.handle)
        torch.cuda.synchronize(self.device)
        logger.info(
            "DeepEP M2N precompiled %d dispatch buckets x %d lanes in %.1f s: %s",
            len(buckets),
            len(self.lanes),
            time.perf_counter() - start,
            buckets,
        )

    def _allocate_combine_zeros(self) -> None:
        """One zero buffer for every combine on this Attention rank.

        Combine takes one zero row per row this rank received. A synced
        dispatch gives an Attention rank no rows; a dispatch that skips the
        sync gets DeepEP's worst case for its bucket, so the buffer covers the
        largest such bucket. Both lanes share it (combine only reads it), and
        captured graphs keep one stable address.
        """
        assert self.bucket_ladder is not None
        rows = worst_case_receive_rows(
            num_ranks=self.layout.union_size,
            bucket=self.bucket_ladder[-1] if self.bucket_ladder else 0,
            top_k=self.top_k,
            experts_per_rank=self.layout.experts_per_rank,
            expert_alignment=self.expert_alignment,
        )
        self._combine_zeros = torch.zeros(
            (rows, self.hidden_size), dtype=self.dtype, device=self.device
        )

    def _combine_zero_rows(self, shape: tuple[int, int]) -> torch.Tensor:
        """The first ``shape[0]`` rows of the zero buffer (a contiguous view)."""
        zeros = self._combine_zeros
        assert zeros is not None
        rows = shape[0]
        if rows > zeros.shape[0] or shape[1] != zeros.shape[1]:
            raise RuntimeError(
                f"combine needs zero rows of shape {shape} but the buffer holds "
                f"{tuple(zeros.shape)}: a dispatch skipped the host sync with a "
                "bucket above the ladder's top rung (no_sync_bucket_limit)"
            )
        return zeros[:rows]

    @staticmethod
    def _transfer_state(context: AFDTransferContext) -> DeepEPM2NTransferState:
        if not isinstance(context.states, DeepEPM2NTransferState):
            raise RuntimeError("context does not come from DeepEPM2N recv_attn_output")
        return context.states

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("DeepEPM2NAFDConnector is not initialized")


class DeepEPM2NAFDControlPlane(AFDControlPlane):
    """Per-step plan derivation plus payload delivery to every FFN rank."""

    def __init__(self, connector: DeepEPM2NAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if len(payload.dp_metadata_list) > connector.num_lanes:
            raise ValueError(
                f"{len(payload.dp_metadata_list)} stages exceed "
                f"num_lanes={connector.num_lanes}"
            )
        graph_mode = payload.is_graph_capturing or payload.is_graph_replaying
        if (
            connector.attn_tp_size is None
            or connector.capacity_tokens_per_rank is None
            or connector.bucket_ladder is None
        ):
            raise RuntimeError("DeepEPM2N control plane used before bootstrap")
        connector.stage_plans = {
            stage_idx: plan_stage(
                dp_metadata,
                attn_size=connector.attn_size,
                attn_tp_size=connector.attn_tp_size,
                capacity_tokens_per_rank=connector.capacity_tokens_per_rank,
                bucket_granularity=connector.config.bucket_granularity,
                graph_mode=graph_mode,
                bucket_ladder=connector.bucket_ladder,
                cpu_sync_min_tokens=connector.config.cpu_sync_min_tokens,
            )
            for stage_idx, dp_metadata in payload.dp_metadata_list.items()
        }

    def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if connector.is_ffn or connector.role_rank != CONTROL_SENDER_ATTN_RANK:
            return
        if connector.group is None:
            raise RuntimeError("DeepEPM2N control plane is not initialized")
        send_control_payload(
            payload,
            dst=list(range(connector.ffn_size)),
            group=connector.group,
            device=connector.control_device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.group is None:
            raise RuntimeError("DeepEPM2N control plane is not initialized")
        return recv_control_payload(
            src=connector.layout.world_rank("attention", CONTROL_SENDER_ATTN_RANK),
            group=connector.group,
            device=connector.control_device,
        )
