# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routed-only AFD asynchronous operator harness; imports no device packages eagerly.

The pure Python planner is also used by ``--dry-run``. Native execution requires
one torchrun process per A3/A5 device and a matching built AFD extension.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup

MAX_SEQUENCE_LENGTH = 262144
UINT16_MAX = 65535
MAX_LAYER_INDEX = (1 << 32) - 1
ALIGNMENT = 32
# Compatibility defaults for the existing A3 harness; A5 requires explicit inputs.
MAX_AIV_COUNT = 48
MAX_A5_RANKS = 64
MIB = 1024 * 1024
MAX_WINDOW_MB = 32768
CONSERVATIVE_UB_BYTES = 192 * 1024
IDS_CACHE_BYTES = 64 * 1024
INT8_BYTES = 1
INT16_BYTES = 2
INT32_BYTES = 4
INT64_BYTES = 8
BATCH_INFO_FIELDS = 5
QUANT_REDUCTION_FLOATS = 8
OP_NAMES = ("dispatch_send", "dispatch_recv", "combine_send", "combine_recv")
PROTOCOL = "cam_async_routed_only_compact_v2"
VALUE_PERIOD = 257
VALUE_DIVISOR = 128.0
EXPERT_MULTIPLIER_DIVISOR = 32.0
EXPERT_BIAS_DIVISOR = 64.0
INPUT_INDEX_MULTIPLIER = 13
INPUT_RANK_MULTIPLIER = 29
INPUT_LAYER_MULTIPLIER = 17
SCALE_VARIATION_PERIOD = 3
SCALE_RANK_MULTIPLIER = 11
QUANT_ANCHOR_NUMERATOR = 127.0
QUANT_ANCHOR_DIVISOR = 64.0
WEIGHT_VARIATION_PERIOD = 3
WEIGHT_VARIATION_DIVISOR = 4.0
# Versioned dynamic-quantization contract, from QuantProcess() in
# csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel: the kernel scales
# each row by the FP32 multiplier 127/amax, rounds half-to-even, and stores
# ``1.0f / (127.0f / amax)`` as the dequantization scale. The stored value is
# the reciprocal of an already-rounded quotient, so it is not bit-identical to
# ``amax/127`` in FP32 for most inputs; the golden reproduces the kernel's form
# rather than the algebraically equal one. `_quantize_contract()` is the single
# place where the CPU golden encodes these rules, so a contract change has to be
# made there and nowhere else.
QUANT_MULTIPLIER_NUMERATOR = 127.0
QUANT_CLAMP_LOW = -127
QUANT_CLAMP_HIGH = 127


def f32(value: float) -> float:
    """Round a Python float to the nearest IEEE-754 single, as FP32 arithmetic does."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quant_dynamic_scale(amax: float) -> float:
    """Per-row multiplier of the ``dispatch_send`` quantization contract.

    The kernel divides the constant ``127.0`` by the row maximum in FP32, which
    this reproduces by rounding each operand and the quotient to single
    precision. Written in plain Python so the arithmetic can be pinned by a CPU
    test without a tensor library; ``_quantize_contract`` is the tensor form of
    the same rule.
    """
    return f32(f32(QUANT_MULTIPLIER_NUMERATOR) / f32(amax))


def quant_stored_scale(amax: float) -> float:
    """Dequantization scale the contract stores for a row with this ``amax``.

    The kernel keeps the reciprocal of the already-rounded multiplier, so this
    is ``1.0f / (127.0f / amax)`` and not ``amax / 127.0f``. The two agree only
    when the division is exact; for a general row they differ in the last bit.
    """
    return f32(f32(1.0) / quant_dynamic_scale(amax))


def quantize_row(values: list[float]) -> list[int]:
    """Quantize one row exactly as the kernel does, in plain Python.

    ``CAST_RINT`` is round-half-to-even, which is what :func:`round` does.
    Round the multiply to FP32 before converting it to an integer, matching
    the kernel's separate ``Muls`` and ``Cast`` operations. The multiplier
    bounds every element by ``127``, so no clamp is needed; the result is still
    clamped to match the tensor form, which guards a hand-written input.

    This is the scalar companion to ``_quantize_contract``: a CPU test can pin
    the contract through it without a tensor library.
    """
    if not values:
        raise ValueError("quantization needs at least one element")
    amax = max(abs(f32(value)) for value in values)
    if amax == 0.0 or not math.isfinite(amax):
        raise ValueError(
            "dispatch quantization contract is undefined for a row whose amax "
            "is zero or non-finite"
        )
    multiplier = quant_dynamic_scale(amax)
    return [
        min(max(round(f32(f32(value) * multiplier)), QUANT_CLAMP_LOW), QUANT_CLAMP_HIGH)
        for value in values
    ]


@dataclass(frozen=True)
class HarnessConfig:
    attn_ranks: int = 2
    moe_ranks: int = 2
    tp_size: int = 2
    experts_per_rank: int = 4
    top_k: int = 2
    hidden_size: int = 256
    lengths: tuple[int, ...] = (7, 5)
    dtype: str = "fp16"
    dynamic_quant: int = 0
    routing: str = "balanced"
    capacity: int = 16
    window_mb: int = 64
    timeout_seconds: float = 120.0
    max_seq_len: int = 14
    soc_version: str = "910c"
    aiv_count: int | None = None
    ub_budget_bytes: int | None = None

    @property
    def world_size(self) -> int:
        return self.attn_ranks + self.moe_ranks

    @property
    def total_experts(self) -> int:
        return self.moe_ranks * self.experts_per_rank


@dataclass(frozen=True)
class ChunkPlan:
    tp_base: int
    start: int
    end: int
    total_rows: int
    counts: tuple[int, ...]
    prefixes: tuple[int, ...]
    table: tuple[int, ...]
    # Payload order: local expert, source TP rank, source token.
    rows: tuple[tuple[int, int, int], ...]

    def metadata(self, layer_id: int) -> tuple[int, ...]:
        return (
            self.total_rows,
            self.tp_base,
            layer_id,
            self.start,
            self.end,
            *self.prefixes,
            *self.table,
        )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--soc-version", choices=("910c", "950"), default="910c")
    parser.add_argument(
        "--aiv-count",
        type=int,
        help="Actual AIV count used by native tiling; required for 950. "
        f"910c defaults to {MAX_AIV_COUNT}. "
        "This planning input does not control kernel launch.",
    )
    parser.add_argument(
        "--ub-budget-bytes",
        type=int,
        help="Checked usable per-core UB budget in bytes; required for 950. "
        f"910c defaults to {CONSERVATIVE_UB_BYTES}. "
        "CPU planning does not verify device resources.",
    )
    parser.add_argument("--attn-ranks", type=int, default=2)
    parser.add_argument("--moe-ranks", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--lengths",
        default=None,
        help="Comma-separated positive batch sizes, one per Attention rank; "
        "one value broadcasts. Default alternates 7,5.",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--dynamic-quant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--routing", choices=("balanced", "sparse"), default="balanced")
    parser.add_argument(
        "--capacity",
        type=int,
        default=16,
        help="Power-of-two dispatch receive rows; every TP-aggregated expert must fit.",
    )
    parser.add_argument("--window-mb", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Window batch bound across TP (0 computes max(lengths)*tp_size).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the CPU-only plan; no torch/NPU imports or execution.",
    )


def config_from_args(args: argparse.Namespace) -> HarnessConfig:
    lengths = (
        tuple(int(value) for value in args.lengths.split(","))
        if args.lengths is not None
        else tuple(7 if rank % 2 == 0 else 5 for rank in range(args.attn_ranks))
    )
    if len(lengths) == 1:
        lengths *= args.attn_ranks
    config = HarnessConfig(
        attn_ranks=args.attn_ranks,
        moe_ranks=args.moe_ranks,
        tp_size=args.tp_size,
        experts_per_rank=args.experts_per_rank,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        lengths=lengths,
        dtype=args.dtype,
        dynamic_quant=args.dynamic_quant,
        routing=args.routing,
        capacity=args.capacity,
        window_mb=args.window_mb,
        timeout_seconds=args.timeout_seconds,
        max_seq_len=args.max_seq_len or max(lengths, default=0) * args.tp_size,
        soc_version=args.soc_version,
        aiv_count=args.aiv_count,
        ub_budget_bytes=args.ub_budget_bytes,
    )
    validate_config(config)
    return config


def route_ids(config: HarnessConfig, rank: int, token: int) -> tuple[int, ...]:
    # Sparse leaves the remaining experts/owners empty, including nonzero ranks
    # when K <= R. All K expert IDs within a token remain distinct.
    pool = config.total_experts if config.routing == "balanced" else config.top_k
    offset = (rank + token * config.top_k) % pool
    return tuple((offset + slot) % pool for slot in range(config.top_k))


def chunk_plans(config: HarnessConfig, owner: int) -> tuple[ChunkPlan, ...]:
    """Independent route expansion and greedy whole-expert receive partition."""
    plans = []
    expert_base = owner * config.experts_per_rank
    histograms = []
    buckets = []
    # Expand once per source, not once per expert: the reference-sized corpus
    # has hundreds of experts and otherwise takes quadratic planning work.
    for rank in range(config.attn_ranks):
        histogram = [0] * config.total_experts
        expert_tokens: list[list[int]] = [[] for _ in range(config.experts_per_rank)]
        for token in range(config.lengths[rank]):
            for expert in route_ids(config, rank, token):
                histogram[expert] += 1
                if expert_base <= expert < expert_base + config.experts_per_rank:
                    expert_tokens[expert - expert_base].append(token)
        histograms.append(histogram)
        buckets.append(expert_tokens)
    for tp_base in range(0, config.attn_ranks, config.tp_size):
        ranks = range(tp_base, tp_base + config.tp_size)
        table = tuple(
            histograms[rank][expert_base + expert]
            for rank in ranks
            for expert in range(config.experts_per_rank)
        )
        totals = tuple(
            sum(histograms[rank][expert_base + expert] for rank in ranks)
            for expert in range(config.experts_per_rank)
        )
        if max(totals) > config.capacity:
            raise ValueError(
                f"MoE {owner}, TP base {tp_base}: one expert requires {max(totals)} "
                f"rows, exceeding capacity {config.capacity}; whole experts must fit"
            )
        start = 0
        while start < config.experts_per_rank:
            end = start
            count = totals[start]
            while end + 1 < config.experts_per_rank:
                if count + totals[end + 1] > config.capacity:
                    break
                end += 1
                count += totals[end]
            rows = tuple(
                (expert_base + expert, rank, token)
                for expert in range(start, end + 1)
                for rank in ranks
                for token in buckets[rank][expert]
            )
            plans.append(
                ChunkPlan(
                    tp_base,
                    start,
                    end,
                    sum(totals),
                    tuple(
                        value if start <= expert <= end else 0
                        for expert, value in enumerate(totals)
                    ),
                    tuple(sum(histograms[rank][:expert_base]) for rank in ranks),
                    table,
                    rows,
                )
            )
            start = end + 1
    return tuple(plans)


def window_requirements(config: HarnessConfig) -> dict[str, int]:
    def align(size: int) -> int:
        return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    local_capacity = config.max_seq_len // config.tp_size
    attn = (
        align(4 * (config.moe_ranks + config.total_experts) * MAX_AIV_COUNT)
        + align(4 * config.moe_ranks)
        + align(2 * config.hidden_size * local_capacity * config.top_k)
        + ALIGNMENT
    )
    payload_bytes = (
        config.hidden_size + ALIGNMENT
        if config.dynamic_quant
        else 2 * config.hidden_size
    )
    moe = (
        config.attn_ranks
        * (
            align(8 * 5)
            + 2 * align(4 * config.experts_per_rank)
            + align(payload_bytes * local_capacity)
            + align(2 * local_capacity * config.top_k)
        )
        + ALIGNMENT
    )
    return {
        "attention_min_bytes": attn,
        "moe_min_bytes": moe,
        "minimum_bytes": max(attn, moe),
    }


def hardware_limits(config: HarnessConfig) -> tuple[int, int]:
    """Resolve declared resources without inferring A5 hardware from a CPU host."""
    if config.soc_version not in ("910c", "950"):
        raise ValueError("soc-version must be 910c or 950")
    if config.soc_version == "950" and (
        config.aiv_count is None or config.ub_budget_bytes is None
    ):
        raise ValueError("950 requires explicit --aiv-count and --ub-budget-bytes")
    aiv_count = MAX_AIV_COUNT if config.aiv_count is None else config.aiv_count
    ub_budget = (
        CONSERVATIVE_UB_BYTES
        if config.ub_budget_bytes is None
        else config.ub_budget_bytes
    )
    if aiv_count <= 0 or ub_budget <= 0:
        raise ValueError("aiv-count and ub-budget-bytes must be positive")
    if aiv_count > MAX_AIV_COUNT:
        raise ValueError(
            f"aiv-count must not exceed the native {MAX_AIV_COUNT}-AIV launch cap"
        )
    return aiv_count, ub_budget


def unified_buffer_requirements(config: HarnessConfig) -> dict[str, int]:
    """Conservative peak bytes from the four baseline ``op_kernel`` Init layouts.

    PreProcess scratch aliases later buffers, so count its peak separately.
    Receive addresses are copied for a whole per-source AIV slice without a
    scratch-capacity check in the kernel. Bound that slice by both the receive
    capacity and source assignments. The A3 compatibility default uses 48 AIVs
    and 192 KiB UB; A5 requires explicitly checked resource inputs. This planner
    does not query or qualify hardware from a CPU-only process.
    """

    def align(size: int) -> int:
        return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    owners = config.moe_ranks
    aiv_count, _ = hardware_limits(config)
    experts = config.total_experts
    local_experts = config.experts_per_rank
    tp = config.tp_size
    hidden = config.hidden_size
    metadata_bytes = align(INT64_BYTES * (BATCH_INFO_FIELDS + tp + local_experts * tp))
    statistics_bytes = align(INT32_BYTES * (owners + experts) * aiv_count)
    send_prefix = (
        IDS_CACHE_BYTES
        + align(INT32_BYTES * experts)
        + align(INT8_BYTES * owners)
        + align(INT16_BYTES * owners)
        + align(INT16_BYTES * experts)
    )
    send_buffers = (
        send_prefix
        + align(INT32_BYTES * owners)
        + align(INT32_BYTES * experts)
        + align(INT64_BYTES * BATCH_INFO_FIELDS)
        + align(INT16_BYTES * hidden)
        # One aligned uint16 token-address batch per global expert, followed
        # by its count and destination-offset uint32 arrays.
        + align(ALIGNMENT * experts)
        + align(INT32_BYTES * experts + INT32_BYTES * experts)
    )
    if config.dynamic_quant:
        send_buffers += (
            align(INT32_BYTES * hidden)
            + align(INT32_BYTES * hidden)
            + align(INT32_BYTES * QUANT_REDUCTION_FLOATS)
        )
    recv_payload_bytes = (
        hidden + ALIGNMENT if config.dynamic_quant else INT16_BYTES * hidden
    )
    recv_buffers = (
        align(INT64_BYTES * BATCH_INFO_FIELDS)
        + metadata_bytes
        + align(INT64_BYTES * local_experts)
        + align(INT32_BYTES * local_experts)
        + align(INT32_BYTES * local_experts)
        + align(recv_payload_bytes)
        + align(INT32_BYTES * local_experts + INT32_BYTES * local_experts)
    )
    source_chunk_rows = min(config.capacity, max(config.lengths) * config.top_k)
    rows_per_aiv = (source_chunk_rows + aiv_count - 1) // aiv_count
    recv_scratch = align(INT16_BYTES * rows_per_aiv)
    combine_send_buffers = (
        metadata_bytes
        + align(INT32_BYTES * local_experts * tp)
        + align(INT32_BYTES * tp)
        # The token buffer and completion prefixes reuse this region. At
        # least one full token must fit, otherwise the send loop cannot advance.
        + max(align(INT16_BYTES * hidden), align(INT32_BYTES * owners))
    )
    combine_recv_prefix = IDS_CACHE_BYTES + align(INT16_BYTES * experts)
    combine_recv_buffers = (
        combine_recv_prefix
        + align(INT32_BYTES * experts)
        + align(INT16_BYTES * experts)
        + align(INT32_BYTES * hidden)
        + align(INT32_BYTES * hidden)
        + align(INT32_BYTES * hidden)
        + align(INT16_BYTES * hidden)
        + align(INT32_BYTES * owners)
    )
    return {
        "dispatch_send": max(send_prefix + statistics_bytes, send_buffers),
        "dispatch_recv": recv_buffers + recv_scratch,
        "combine_send": combine_send_buffers,
        "combine_recv": max(
            combine_recv_prefix + statistics_bytes, combine_recv_buffers
        ),
    }


def validate_config(config: HarnessConfig) -> None:
    aiv_count, ub_budget = hardware_limits(config)
    positive = (
        config.attn_ranks,
        config.moe_ranks,
        config.tp_size,
        config.experts_per_rank,
        config.hidden_size,
        config.top_k,
    )
    if any(value <= 0 for value in positive):
        raise ValueError(
            "Ranks, TP, expert count, top-k and hidden size must be positive"
        )
    if config.attn_ranks % config.tp_size:
        raise ValueError("tp-size must divide attn-ranks")
    if config.soc_version == "950" and config.world_size > MAX_A5_RANKS:
        raise ValueError(f"950 world-size must not exceed {MAX_A5_RANKS}")
    if config.moe_ranks > MAX_AIV_COUNT:
        raise ValueError(
            "moe-ranks exceeds the harness's supported 48-AIV owner layout"
        )
    if len(config.lengths) != config.attn_ranks or any(
        length <= 0 for length in config.lengths
    ):
        raise ValueError(
            "lengths must contain one positive batch size per Attention rank"
        )
    if config.top_k > config.total_experts:
        raise ValueError("top-k exceeds the global routed expert count")
    if config.dtype not in ("fp16", "bf16") or config.dynamic_quant not in (0, 1):
        raise ValueError("Only fp16/bf16 and dynamic-quant 0/1 are supported")
    if config.routing not in ("balanced", "sparse"):
        raise ValueError("routing must be balanced or sparse")
    if config.hidden_size % ALIGNMENT:
        raise ValueError(
            "hidden-size must be divisible by 32 for the raw DataCopy path"
        )
    if not config.tp_size <= config.max_seq_len <= MAX_SEQUENCE_LENGTH:
        raise ValueError("max-seq-len must be in [tp-size, 262144]")
    if max(config.lengths) > config.max_seq_len // config.tp_size:
        raise ValueError("A rank's batch size exceeds max-seq-len // tp-size")
    if max(config.lengths) > UINT16_MAX:
        raise ValueError("A source batch exceeds uint16 token/counter capacity (65535)")
    if not 1 <= config.capacity <= MAX_SEQUENCE_LENGTH or config.capacity & (
        config.capacity - 1
    ):
        raise ValueError("capacity must be a power of two in [1, 262144]")
    if not 1 <= config.window_mb <= MAX_WINDOW_MB:
        raise ValueError("window-mb must be in [1, 32768]")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive and finite")
    if config.window_mb * MIB < window_requirements(config)["minimum_bytes"]:
        raise ValueError(
            "HCCL window is smaller than the kernel's static byte requirement"
        )
    for operation, required in unified_buffer_requirements(config).items():
        if required > ub_budget:
            raise ValueError(
                f"{operation} requires at least {required} UB bytes under the "
                f"{aiv_count}-AIV plan, exceeding the {ub_budget}-byte harness limit"
            )
    # Routing has no duplicate expert IDs, so each source/expert count and each
    # source/owner's unique payload count is bounded by the checked batch size.
    for owner in range(config.moe_ranks):
        chunk_plans(config, owner)


def describe_config(config: HarnessConfig) -> dict:
    aiv_count, ub_budget = hardware_limits(config)
    plans = [chunk_plans(config, owner) for owner in range(config.moe_ranks)]
    return {
        "status": "plan_only_no_device_validation",
        "protocol": PROTOCOL,
        "config": asdict(config),
        "world_size": config.world_size,
        "batch_size_factor": config.capacity / MAX_SEQUENCE_LENGTH,
        "windows": window_requirements(config),
        "unified_buffer_bytes": unified_buffer_requirements(config),
        "unified_buffer_budget_bytes": ub_budget,
        "assumed_aiv_count": aiv_count,
        "resource_validation": "declared planning inputs; not verified on a device",
        "moe_chunks": [
            [
                {
                    "tp_base": plan.tp_base,
                    "expert_interval": [plan.start, plan.end],
                    "rows": len(plan.rows),
                    "whole_group_rows": plan.total_rows,
                    "counts": plan.counts,
                }
                for plan in owner_plans
            ]
            for owner_plans in plans
        ],
        "note": (
            "Zero-row chunks still require combine_send; "
            "CPU planning is not NPU qualification."
        ),
    }


def validate_metadata(
    plan: ChunkPlan,
    layer_id: int,
    batch_info: list[int],
    counts: list[int],
) -> None:
    if tuple(batch_info) != plan.metadata(layer_id):
        raise AssertionError(
            f"metadata mismatch: expected {plan.metadata(layer_id)}, got {batch_info}"
        )
    if tuple(counts) != plan.counts:
        raise AssertionError(
            f"expert counts mismatch: expected {plan.counts}, got {counts}"
        )


class NativeHarness:
    """Own HCCL payload windows and a separate Gloo CPU control domain.

    Device operations are synchronous at measurement boundaries. Per-op NPU
    events enclose only the native call. The lifecycle timer also includes CPU
    metadata, FFN transform, synchronization and optional readiness hooks.
    """

    def __init__(self, config: HarnessConfig):
        validate_config(config)
        self.config = config
        self.rank = int(os.environ.get("RANK", "-1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self._ready = False
        self._failed = False
        self.payload_group: ProcessGroup | None = None
        self._round_keepalive: list[tuple] = []

    def __enter__(self) -> NativeHarness:
        config = self.config
        if not 0 <= self.rank < config.world_size or self.local_rank < 0:
            raise RuntimeError(
                "Native execution requires torchrun RANK, LOCAL_RANK and WORLD_SIZE"
            )
        if int(os.environ.get("WORLD_SIZE", "0")) != config.world_size:
            raise RuntimeError("WORLD_SIZE must equal attn-ranks + moe-ranks")
        # These are read by torch-npu/HCCL and the native host tiling. Explicit
        # CLI values are authoritative and must be set before importing them.
        os.environ["HCCL_BUFFSIZE"] = str(config.window_mb)
        os.environ["BATCH_SIZE_FACTOR"] = str(config.capacity / MAX_SEQUENCE_LENGTH)
        # Runtime-only dependencies: --help, --dry-run and CPU unit tests do not
        # import torch, torch_npu, vLLM or the native extension.
        import torch
        import torch.distributed as dist
        import torch_npu

        from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

        self.torch = torch
        self.dist = dist
        if dist.is_initialized():
            raise RuntimeError(
                "Use a fresh torchrun; the harness owns its communication groups"
            )
        torch.npu.set_device(self.local_rank)
        self.device = torch.device("npu", self.local_rank)
        self.dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
        ensure_afd_ascend_ops_loaded()
        self.operations = {
            "dispatch_send": torch.ops.afd_ascend.afd_async_dispatch_send,
            "dispatch_recv": torch.ops.afd_ascend.afd_async_dispatch_recv,
            "combine_send": torch.ops.afd_ascend.afd_async_combine_send,
            "combine_recv": torch.ops.afd_ascend.afd_async_combine_recv,
        }
        timeout = timedelta(seconds=config.timeout_seconds)
        dist.init_process_group("gloo", init_method="env://", timeout=timeout)
        self.control_group = dist.group.WORLD
        self._ready = True
        try:
            configs = [None] * config.world_size
            dist.all_gather_object(configs, asdict(config), group=self.control_group)
            if any(item != configs[0] for item in configs):
                raise RuntimeError(
                    "All ranks must supply the identical harness configuration"
                )
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = {"hccl_buffer_size": config.window_mb}
            self.payload_group = dist.new_group(
                ranks=list(range(config.world_size)),
                backend="hccl",
                timeout=timeout,
                pg_options=options,
            )
            backend = self.payload_group._get_backend(self.device)
            self.group_name = backend.get_hccl_comm_name(self.rank, init_comm=True)
            if not isinstance(self.group_name, str) or not self.group_name:
                raise RuntimeError("HCCL returned an empty communicator name")
            self.comm_args = torch.zeros(1, device=self.device, dtype=torch.float16)
            self.anchor = torch.zeros(1, device=self.device, dtype=self.dtype)
            self.plans = (
                chunk_plans(config, self.rank - config.attn_ranks)
                if self.rank >= config.attn_ranks
                else ()
            )
            self.dist.barrier(group=self.control_group)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # No device synchronization after failure: a stuck peer/kernel may make
        # cleanup hang. torchrun owns process termination in this case.
        if self._ready and exc_type is None and not self._failed:
            if self.payload_group is not None:
                self.dist.destroy_process_group(self.payload_group)
            self.dist.destroy_process_group()
            self._ready = False

    def gather_reports(self, local: dict) -> list[dict]:
        reports: list[dict] = [{} for _ in range(self.config.world_size)]
        self.dist.all_gather_object(reports, local, group=self.control_group)
        return reports

    def environment(self) -> dict:
        aiv_count, ub_budget = hardware_limits(self.config)
        versions = {}
        for package in ("torch", "torch-npu", "vllm-afd-plugin"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "not reported by package metadata"
        root = Path(__file__).resolve().parents[2]
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unavailable"
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("async_cam_*.py")
        }
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "host": platform.node(),
            "python": platform.python_version(),
            "packages": versions,
            "device": self.torch.npu.get_device_name(self.local_rank),
            "soc_version": self.config.soc_version,
            "declared_aiv_count": aiv_count,
            "declared_ub_budget_bytes": ub_budget,
            "hccl_group_name": self.group_name,
            "cann_home": os.environ.get("ASCEND_HOME_PATH"),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "git_revision": revision,
            "script_sha256": hashes,
            "command": [sys.executable, *sys.argv],
        }

    def _values(self, rank: int, layer_id: int, tokens: list[int]) -> torch.Tensor:
        torch, config = self.torch, self.config
        token_indices = torch.tensor(tokens, dtype=torch.int64).unsqueeze(1)
        positions = token_indices * config.hidden_size + torch.arange(
            config.hidden_size
        )
        values = (
            (
                positions * INPUT_INDEX_MULTIPLIER
                + rank * INPUT_RANK_MULTIPLIER
                + layer_id * INPUT_LAYER_MULTIPLIER
            )
            % VALUE_PERIOD
        ).float()
        values = (values / VALUE_DIVISOR - 1).to(self.dtype)
        # Nonzero and distinct max(abs(row)) values expose permuted quantization
        # scales. These anchors are exactly representable in FP16 and BF16.
        scale_exponents = (
            token_indices.flatten() + rank * SCALE_RANK_MULTIPLIER + layer_id
        ).remainder(SCALE_VARIATION_PERIOD)
        values[:, 0] = (
            QUANT_ANCHOR_NUMERATOR / QUANT_ANCHOR_DIVISOR * (2.0**scale_exponents)
        ).to(self.dtype)
        return values

    def _input(
        self, rank: int, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch, config = self.torch, self.config
        count = config.lengths[rank]
        values = self._values(rank, layer_id, list(range(count)))
        ids = torch.tensor(
            [route_ids(config, rank, token) for token in range(count)],
            dtype=torch.int32,
        )
        weights = torch.arange(1, config.top_k + 1, dtype=torch.float32).repeat(
            count, 1
        )
        weights += (
            torch.arange(count).remainder(WEIGHT_VARIATION_PERIOD).float() + rank
        ).unsqueeze(1) / WEIGHT_VARIATION_DIVISOR
        weights /= weights.sum(dim=1, keepdim=True)
        return values, ids, weights

    def _quantize_contract(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU golden for the versioned ``dispatch_send`` quantization contract.

        ``values`` is a CPU tensor of exactly the dtype that is handed to the
        operator, so the reference never sees input the device could not see.
        Returns the INT8 payload, the FP32 dequantization scales, and the FP32
        per-row ``amax`` the scales were derived from.

        This mirrors the kernel formula, not the other way round. For each row
        the kernel computes ``amax`` in FP32, divides the constant ``127.0`` by
        it to get ``dynamicScale``, multiplies the row by that scalar, rounds
        half-to-even, and stores ``1.0/dynamicScale``. Both FP32 divisions are
        reproduced here because the stored scale is compared against the device
        value; computing ``amax/127`` instead would differ in the last bit for
        most inputs.

        The multiplier bounds every element by ``amax * 127/amax == 127``, so
        the kernel needs no clamp and never emits a value outside [-127, 127].
        The clamp is kept here only as a guard against a non-finite row; it does
        not make an all-zero row well defined, since ``amax == 0`` gives an
        infinite multiplier on both sides.
        """
        torch = self.torch
        # float32 throughout: the kernel reduces and scales in FP32, and a
        # float64 intermediate here would round differently.
        widened = values.float()
        row_amax = widened.abs().amax(dim=1, keepdim=True)
        if not torch.isfinite(row_amax).all().item() or (row_amax == 0).any().item():
            # The kernel divides by this value and stores its reciprocal, so a
            # zero or non-finite row max makes the contract undefined rather
            # than merely lossy. Fail here instead of returning a NaN payload
            # that the clamp cannot repair and that would report as in-tolerance
            # downstream.
            raise ValueError(
                "dispatch quantization contract is undefined for a row whose "
                "amax is zero or non-finite"
            )
        dynamic_scale = QUANT_MULTIPLIER_NUMERATOR / row_amax
        quantized = (
            (widened * dynamic_scale)
            .round()
            .clamp(QUANT_CLAMP_LOW, QUANT_CLAMP_HIGH)
            .to(torch.int8)
        )
        return quantized, (1.0 / dynamic_scale).flatten(), row_amax.flatten()

    def _quantization_budget(self, amax: torch.Tensor, weights: torch.Tensor) -> float:
        """Worst-case quantization error of the weighted combine, in the contract.

        A row scaled by ``127/amax`` and rounded half-to-even has a per-element
        error of at most half a quantization step, ``amax/254``. Every element of
        that row is transformed by the same expert, which scales it by at most
        ``1 + (E-1)/32``, and is then weighted, so the error reaching one output
        element of one slot is bounded by ``amax/254`` times that factor times
        the weight. The top-k slots carry independent quantization errors, so
        their bounds add, and the contract reference adds one output-dtype
        rounding at its final cast on top. The factor of two below absorbs that
        final rounding and the FP32 accumulation order without measuring a
        quantity that is already known to be bounded.

        This is computed from the run's own ``amax`` and weights rather than
        restated from the end-to-end tolerance, so it answers the question it is
        labeled with instead of restating whatever band the other check allows.
        """
        config = self.config
        step = amax / (2 * QUANT_MULTIPLIER_NUMERATOR)
        transform = 1 + (config.total_experts - 1) / EXPERT_MULTIPLIER_DIVISOR
        per_slot = (step[:, None] * (transform * weights.abs())).sum(dim=1)
        return float(per_slot.max().item()) * 2

    def _dequantize(
        self, quantized: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        """Reference dequantization: FP32 multiply, then the operator dtype."""
        return (quantized.float() * scales[:, None]).to(self.dtype)

    def _expert(self, values: torch.Tensor, expert: int) -> torch.Tensor:
        """Total, injective expert transform so route swaps cannot cancel out."""
        return (
            values.float() * (1 + expert / EXPERT_MULTIPLIER_DIVISOR)
            + (expert % 7 - 3) / EXPERT_BIAS_DIVISOR
        ).to(self.dtype)

    def _expand_expert(self, values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Apply each token's routed expert transform in original order.

        Returns ``[N, K, H]`` where row ``t`` of slot ``k`` already holds the
        output of ``ids[t, k]``, rounded to the selected FP16/BF16 dtype.
        """
        torch = self.torch
        config = self.config
        expanded = torch.empty(
            (values.shape[0], config.top_k, values.shape[1]), dtype=self.dtype
        )
        for token in range(values.shape[0]):
            for slot in range(config.top_k):
                expanded[token, slot] = self._expert(
                    values[token : token + 1], int(ids[token, slot])
                )[0]
        return expanded

    def _combine_golden(
        self,
        expert_y: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """CPU golden in the selected FP16/BF16 output dtype.

        ``expert_y`` is ``[N, K, H]`` in original token/top-k order. Follow the
        kernel's arithmetic: multiply and accumulate in FP32 over the top-k
        slots, then round once to the operator's output dtype. The returned
        tensor is the acceptance baseline for the matching FP16/BF16 device
        output. Weights are used as supplied, without additional normalization.
        """
        torch = self.torch
        rows, _, hidden = expert_y.shape
        accumulate32 = torch.zeros((rows, hidden), dtype=torch.float32)
        for slot in range(self.config.top_k):
            weight = weights[:, slot : slot + 1]
            accumulate32 = accumulate32 + expert_y[:, slot, :].float() * weight
        return accumulate32.to(self.dtype)

    def _golden_dispatch(
        self, layer_id: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Golden dispatch pipeline from raw inputs only.

        Returns ``(x, ids, weights, quantized, scales, expert_output,
        raw_output)`` on the CPU. ``x`` is the value that is actually handed to
        the operator; every reference is computed from that same tensor, so no
        unrounded FP32 input can leak into the golden.

        ``expert_output`` is the expert transform of the dequantized dispatch
        payload, which is what the operator's own data path produces and what
        the combine contract reference sums. ``raw_output`` is the same
        transform applied to the unquantized input, so the gap between the two
        is the quantization error on its own, with no accumulation error mixed
        into it.

        The route table this golden implies comes from :func:`chunk_plans` and
        :func:`route_ids`, which expand the same raw ``expert_ids``; device
        metadata is compared against them rather than feeding them.
        """
        torch = self.torch
        x, ids, weights = self._input(self.rank, layer_id)
        if self.config.dynamic_quant:
            quantized, scales, _ = self._quantize_contract(x)
            routed = self._dequantize(quantized, scales)
        else:
            quantized = torch.zeros(0, dtype=torch.int8)
            scales = torch.zeros(x.shape[0], dtype=torch.float32)
            routed = x
        expert_output = self._expand_expert(routed, ids)
        raw_output = (
            self._expand_expert(x, ids) if self.config.dynamic_quant else expert_output
        )
        return (
            x,
            ids,
            weights,
            quantized,
            scales,
            expert_output,
            raw_output,
        )

    def _invoke(
        self,
        name: str,
        arguments: tuple,
        timed: bool,
        before_op: Callable[[str], None] | None,
        event_ms: dict[str, list[float]],
        host_ms: dict[str, list[float]],
    ):
        if before_op is not None:
            before_op(name)
        if timed:
            self.torch.npu.synchronize()
            self._round_keepalive.clear()
            start = self.torch.npu.Event(enable_timing=True)
            end = self.torch.npu.Event(enable_timing=True)
            start.record()
        host_start = time.perf_counter()
        result = self.operations[name](*arguments)
        if not timed and name.endswith("_send"):
            # Retain asynchronous inputs through the next same-stream fence.
            # In particular, never retain all full-capacity DR buffers for an
            # entire chunked round: only the currently pending send needs them.
            self._round_keepalive.append((arguments, result))
        if timed:
            end.record()
            self.torch.npu.synchronize()
            host_ms[name].append((time.perf_counter() - host_start) * 1000)
            event_ms[name].append(start.elapsed_time(end))
        return result

    def _watchdog_abort(self) -> None:
        print(
            f"rank {self.rank}: native round timed out; aborting worker",
            file=sys.stderr,
            flush=True,
        )
        os._exit(124)

    def run_round(
        self,
        layer_id: int,
        validate: bool = True,
        timed: bool = False,
        before_op: Callable[[str], None] | None = None,
        timed_ops: tuple[str, ...] | None = None,
    ) -> dict:
        if not 0 <= layer_id <= MAX_LAYER_INDEX:
            raise ValueError("layer_id must fit a nonnegative uint32")
        watchdog = threading.Timer(self.config.timeout_seconds, self._watchdog_abort)
        watchdog.daemon = True
        watchdog.start()
        try:
            with self.torch.inference_mode():
                return self._run_round(layer_id, validate, timed, before_op, timed_ops)
        except BaseException:
            self._failed = True
            raise
        finally:
            watchdog.cancel()

    def _run_round(
        self, layer_id: int, validate: bool, timed: bool, before_op, timed_ops
    ) -> dict:
        torch, config = self.torch, self.config
        failures: list[str] = []
        event_ms: dict[str, list[float]] = {name: [] for name in OP_NAMES}
        host_ms: dict[str, list[float]] = {name: [] for name in OP_NAMES}
        self._round_keepalive = []
        selected_ops = OP_NAMES if timed_ops is None else timed_ops
        if any(name not in OP_NAMES for name in selected_ops):
            raise ValueError("timed_ops contains an unknown operator")
        # Build the independent CPU golden and source tensors before timing.
        golden: dict | None = None
        if self.rank < config.attn_ranks:
            if validate:
                (
                    x,
                    ids,
                    weights,
                    _gold_quant,
                    _gold_scales,
                    gold_expert,
                    gold_raw,
                ) = self._golden_dispatch(layer_id)
                expected = self._combine_golden(gold_expert, weights)
                unquantized = None
                budget = 0.0
                if config.dynamic_quant:
                    unquantized = self._combine_golden(gold_raw, weights)
                    budget = self._quantization_budget(
                        x.float().abs().amax(dim=1), weights
                    )
                golden = {
                    "output": expected,
                    "unquantized": unquantized,
                    "budget": budget,
                }
            else:
                # Performance rounds need only source tensors. The untimed
                # precision preflight already checked the CPU golden.
                x, ids, weights = self._input(self.rank, layer_id)
            tensors = [tensor.to(self.device) for tensor in (x, ids, weights)]
            x, ids, weights = tensors
        torch.npu.synchronize()
        self.dist.barrier(group=self.control_group)
        lifecycle_start = time.perf_counter()
        chunks = rows = 0

        def invoke(name: str, arguments: tuple):
            return self._invoke(
                name,
                arguments,
                timed and name in selected_ops,
                before_op,
                event_ms,
                host_ms,
            )

        common = (
            config.hidden_size,
            config.top_k,
            config.moe_ranks,
            config.attn_ranks,
            config.experts_per_rank,
            self.rank,
            config.world_size,
        )
        if self.rank < config.attn_ranks:
            invoke(
                "dispatch_send",
                (
                    x,
                    ids,
                    self.comm_args,
                    0,
                    config.max_seq_len,
                    config.lengths[self.rank],
                    *common,
                    layer_id,
                    config.tp_size,
                    config.dynamic_quant,
                    self.group_name,
                ),
            )
            output = invoke(
                "combine_recv",
                (
                    self.anchor,
                    ids,
                    weights,
                    self.comm_args,
                    0,
                    config.lengths[self.rank],
                    *common,
                    self.group_name,
                ),
            )
            rows = config.lengths[self.rank]
        else:
            pending = {(plan.tp_base, plan.start): plan for plan in self.plans}
            next_expert = dict.fromkeys(range(0, config.attn_ranks, config.tp_size), 0)
            while pending:
                payload, scales, info, counts = invoke(
                    "dispatch_recv",
                    (
                        self.anchor,
                        self.comm_args,
                        0,
                        config.max_seq_len,
                        *common,
                        config.tp_size,
                        config.dynamic_quant,
                        self.group_name,
                    ),
                )
                if tuple(info.shape) != (
                    5 + config.tp_size + config.experts_per_rank * config.tp_size,
                ):
                    raise AssertionError(
                        f"Invalid compact metadata shape {tuple(info.shape)}"
                    )
                raw_info = info.cpu().tolist()
                raw_counts = counts.cpu().tolist()
                # Reading this DR's metadata completes the prior same-stream
                # combine_send and releases its retained input buffers.
                self._round_keepalive.clear()
                tp_base, start, end = raw_info[1], raw_info[3], raw_info[4]
                key = (tp_base, start)
                if key not in pending or start != next_expert.get(tp_base):
                    raise AssertionError(
                        f"Unexpected/replayed receive interval: {raw_info[:5]}"
                    )
                plan = pending.pop(key)
                # Exact routing metadata is also a safety prerequisite for
                # combine_send; reject it before feeding malformed DMA offsets.
                validate_metadata(plan, layer_id, raw_info, raw_counts)
                next_expert[tp_base] = end + 1
                count = len(plan.rows)
                expected_dtype = torch.int8 if config.dynamic_quant else self.dtype
                if payload.dtype != expected_dtype or tuple(payload.shape) != (
                    config.capacity,
                    config.hidden_size,
                ):
                    raise AssertionError(
                        "Dispatch payload has the wrong dtype or shape"
                    )
                if (
                    counts.dtype != torch.int64
                    or info.dtype != torch.int64
                    or tuple(counts.shape) != (config.experts_per_rank,)
                ):
                    raise AssertionError(
                        "Dispatch metadata/count dtype or shape mismatch"
                    )
                if scales.dtype != torch.float32 or tuple(scales.shape) != (
                    config.capacity if config.dynamic_quant else 1,
                ):
                    raise AssertionError(
                        "Dispatch scales have the wrong dtype or shape"
                    )
                if validate and count:
                    # Golden routing comes from the raw ``expert_ids`` of every
                    # source rank. The device metadata that just arrived is a
                    # compared object (see ``validate_metadata`` above), never an
                    # input to this reference, otherwise a wrong metadata field
                    # would be folded silently into the answer.
                    expected_payload = torch.empty(
                        (count, config.hidden_size), dtype=self.dtype
                    )
                    for source_rank in range(
                        plan.tp_base, plan.tp_base + config.tp_size
                    ):
                        positions = [
                            index
                            for index, (_, rank, _) in enumerate(plan.rows)
                            if rank == source_rank
                        ]
                        tokens = [plan.rows[index][2] for index in positions]
                        if tokens:
                            expected_payload[positions] = self._values(
                                source_rank, layer_id, tokens
                            )
                    if config.dynamic_quant:
                        # The source rank quantizes, so the golden quantizes each
                        # source's own values with the same contract. The
                        # quantized integers come from the same FP32 arithmetic
                        # on both sides and the transported scale is copied
                        # verbatim, so neither carries any tolerance.
                        expected_quant, expected_scales, _ = self._quantize_contract(
                            expected_payload
                        )
                        self._check(
                            payload[:count].cpu(),
                            expected_quant,
                            0,
                            0,
                            f"dispatch int8[{count} rows]",
                            failures,
                        )
                        self._check(
                            scales[:count].cpu(),
                            expected_scales,
                            0,
                            0,
                            "dispatch dequant scales",
                            failures,
                        )
                    else:
                        # No arithmetic happens on this path, so the payload must
                        # be bit-identical to the source values.
                        self._check(
                            payload[:count].cpu(),
                            expected_payload,
                            0,
                            0,
                            f"dispatch payload[{count} rows]",
                            failures,
                        )
                # Allocate at least one row: even a completely empty participant
                # must send its final completion notification using this info.
                result = torch.empty(
                    (max(count, 1), config.hidden_size),
                    device=self.device,
                    dtype=self.dtype,
                )
                if count:
                    values = payload[:count]
                    if config.dynamic_quant:
                        values = (values.float() * scales[:count, None]).to(self.dtype)
                    offset = 0
                    owner = self.rank - config.attn_ranks
                    for expert, expert_count in enumerate(plan.counts):
                        if expert_count:
                            result[offset : offset + expert_count] = self._expert(
                                values[offset : offset + expert_count],
                                owner * config.experts_per_rank + expert,
                            )
                            offset += expert_count
                invoke(
                    "combine_send",
                    (
                        result,
                        self.comm_args,
                        info,
                        0,
                        config.max_seq_len,
                        *common,
                        config.tp_size,
                        self.group_name,
                    ),
                )
                chunks += 1
                rows += count
        torch.npu.synchronize()
        self.dist.barrier(group=self.control_group)
        lifecycle_ms = (time.perf_counter() - lifecycle_start) * 1000
        self._round_keepalive.clear()
        if validate and self.rank < config.attn_ranks:
            assert golden is not None
            tolerance = 0.001 if config.dtype == "fp16" else 0.008
            actual = output.cpu()
            # Golden and device output must have the same selected dtype.
            self._check(
                actual,
                golden["output"],
                tolerance,
                tolerance,
                f"weighted combine vs CPU {config.dtype.upper()} golden",
                failures,
            )
            if config.dynamic_quant:
                # This is the only check whose operands are both golden tensors:
                # the contract reference consumes dequantized inputs and the
                # unquantized reference consumes the raw ones, so their distance
                # is the error the INT8 contract introduces and nothing else.
                self._check(
                    golden["output"],
                    golden["unquantized"],
                    golden["budget"],
                    0.0,
                    "quantization error budget (contract vs unquantized golden)",
                    failures,
                )
        return {
            "rank": self.rank,
            "layer_id": layer_id,
            "chunks": chunks,
            "rows": rows,
            "event_ms": event_ms,
            "host_ms": host_ms,
            "lifecycle_ms": lifecycle_ms,
            "failures": failures,
            "validated": validate,
        }

    def _check(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        atol: float,
        rtol: float,
        label: str,
        failures: list[str],
    ) -> None:
        """Compare one stage and record a diagnostic on any mismatch.

        The diagnostic separates the questions that must not be mixed:
        ``bitwise_mismatches`` answers data-movement equality, ``max_abs_error``
        and ``relative_l2`` answer numerical distance, and ``first_mismatch``
        names the offending index so a route or token can be identified.

        Zero tolerances require identical bit patterns, including signed zeros.
        Otherwise each element must satisfy ``atol + rtol * |expected|``.
        Diagnostics are emitted for failed comparisons; widening to FP64 below
        only computes error metrics and does not change the golden dtype.
        """
        torch = self.torch
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            failures.append(
                f"{label}: shape/dtype mismatch "
                f"{tuple(actual.shape)}/{actual.dtype} vs "
                f"{tuple(expected.shape)}/{expected.dtype}"
            )
            return
        if not torch.isfinite(actual).all().item() or not (
            torch.isfinite(expected).all().item()
        ):
            failures.append(
                f"{label}: non-finite element present "
                f"(actual finite={bool(torch.isfinite(actual).all().item())}, "
                f"expected finite={bool(torch.isfinite(expected).all().item())})"
            )
            return
        # Reduce differing bytes per element before counting, so diagnostics
        # have the same element units for INT8, FP16/BF16 and FP32 scales.
        bitwise = (
            actual.contiguous()
            .view(torch.uint8)
            .reshape(actual.numel(), actual.element_size())
        )
        expected_bits = (
            expected.contiguous()
            .view(torch.uint8)
            .reshape(expected.numel(), expected.element_size())
        )
        different_bits = (bitwise != expected_bits).any(dim=1)
        bitwise_mismatches = int(different_bits.sum().item())
        if not bitwise_mismatches:
            return
        flat_actual = actual.double().flatten()
        flat_expected = expected.double().flatten()
        difference = (flat_actual - flat_expected).abs()
        tolerance = atol + rtol * flat_expected.abs()
        outside = difference > tolerance if (atol or rtol) else different_bits
        over = int(outside.sum().item())
        if not over:
            return
        denominator = float(flat_expected.norm().item())
        detail = {
            "bitwise_mismatches": bitwise_mismatches,
            "max_abs_error": float(difference.max().item()),
            "relative_l2": float(difference.norm().item()) / denominator
            if denominator
            else float(difference.norm().item()),
            "over_tolerance": over,
            "first_mismatch_index": int(different_bits.nonzero()[0].item()),
        }
        failures.append(f"{label}: {detail}")
