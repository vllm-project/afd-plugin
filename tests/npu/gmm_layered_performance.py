# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""910C performance benchmarks for the two layered routed-expert operators.

Covers the DSV4-Flash W4A8 / EP4 FFN compute path. Geometry is derived from the
model constants rather than hard-coded as an arbitrary grid::

    T = per-rank source batch (the 64K dispatch chunk threshold by default)
    M = top-k * T / EP          rows this rank actually computes
    C = 262144 * BATCH_SIZE_FACTOR   rows this rank allocates

* ``gmm_swiglu_quant_v2_layered`` (W13) - grouped GEMM over
  ``[m_i, 4096] x [4096, 4096]`` per expert, SiLU gating, then per-token
  dynamic quantization. Emits ``int8 [C, 2048]`` and ``float32 [C]``.
* ``grouped_matmul_layered`` (W2) - grouped GEMM over
  ``[m_i, 2048] x [2048, 4096]`` per expert, emitting ``bfloat16 [C, 4096]``.

Run with::

    python tests/npu/gmm_layered_performance.py --validate-baseline

``SOC_VERSION=910c`` selects the SOC at *build* time, not here, and
``AFD_RUN_ASCEND_OP_RUNTIME`` gates the pytest suites; this harness is a script
and reads neither. It reads only ``LOCAL_RANK``, which selects the device and
defaults to 0. ``--help`` and ``--dry-run`` need only a standard Python
interpreter: they import no device package and launch no kernel. Add
``--report`` to archive the results as
``layered_computation_operator_benchmark.md`` beside this file.

Two timers, one pass
--------------------
Each sample records ``host_ms`` (a synchronized host interval around one unit,
including launch overhead) and ``event_ms`` (an NPU stream event interval around
the same unit, device time only).

Both are reported because either could dominate, but on 910C with this workload
they do not differ: across every measured case the host median stayed within 3%
of the device median. The kernels are long enough (~8 ms per layer at
``M = 98304``) that the next launch is enqueued while the current one runs, so
the host-side cost is hidden rather than serialized, and neither timer can be
read as the "real" one. A short-kernel workload would show the opposite, which
is the situation the pair exists to distinguish. Do not assume the two agree
without checking this table.

``C`` and ``M`` are not interchangeable
---------------------------------------
``C`` is the allocated row count and appears in every input and output shape, so
it enters the tiling cache key. ``M = sum(group_list)`` is the number of rows
actually computed and reaches the kernel only as data, so it does not. ``C``
therefore sets the address space and the tiling while ``M`` sets the work. They
are swept separately, and every reported row states both and their ratio.

Exact full capacity is unreachable, near-full is not
----------------------------------------------------
``M = top-k * T / EP`` carries a factor of three while ``C`` must be a power of
two, so ``M/C == 1`` exactly has no integer solution in ``T``. That does not cap
utilisation anywhere near the values a default sweep reaches: ``T`` is not
constrained to a power of two, only ``C`` is, so ``--source-tokens 43690
--capacity 65536`` gives ``M = 65535`` and ``M/C = 0.99998`` while staying under
the uint16 source-token bound. What this means for coverage is that "exactly
full" is unavailable, not that high utilisation is out of reach; the default
``T = 65536`` against ``C = 262144`` is 0.375, and the sweep is what states the
utilisation it actually reached.

The source batch straddles the uint16 boundary
----------------------------------------------
Some dispatch counters and token addresses are uint16, and the reference harness
rejects a per-rank source batch above ``UINT16_MAX = 65535``. The documented 64K
chunk threshold therefore sits one above that bound. ``T`` is a parameter; the
default keeps the documented 64K and the plan warns rather than silently
crossing the boundary.

Baselines
---------
W2 has a usable fused built-in, so it is compared against
``torch_npu.npu_grouped_matmul`` driven with byte-identical inputs. W13 has
none: the built-in fused operator derives its output width from the weight's own
last dimension, which for an ``[E, K, N // 8]`` int32-packed int4 weight is
``N // 8``, while the kernel unpacks int4 and expects ``N``. W13's baseline is
therefore the documented concatenation ``npu_grouped_matmul`` -> ``npu_swiglu``
-> ``npu_dynamic_quant``, which accepts the same W4A8 inputs. ``--validate-
baseline`` drives both the baseline and the fused operator with byte-identical
tensors at a small geometry and checks that they agree, because a baseline that
computes something else would make every speedup meaningless.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# DSV4-Flash routed-expert constants. EP4 splits a layer's full expert set over
# four FFN ranks with no TP split inside an expert.
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS_PER_RANK = 64
TOP_K = 6
EXPERT_PARALLEL = 4
# One tensor per remote MoE layer, selected on device. At 774 MiB of weights per
# layer, eight layers hold about 6 GiB resident while still exercising list
# traversal and layer selection. The full 43-layer model would need about
# 32.5 GiB of weights plus a 2 GiB activation and its pristine copy, which is
# more than half of one 910C die and leaves no headroom for the CAM windows.
LAYERS = 8

MAX_SEQUENCE_LENGTH = 1024 * 256
UINT16_MAX = 65535
INT4_PER_INT8 = 2
INT4_PER_INT32 = 8
NZ_FORMAT = 29
# The A8W4 MSD path consumes a compensation matrix scaled by the packed-int
# arithmetic width; the loader uses eight.
ASSIST_FACTOR = 8

# Two signed int4 values per int8, so the int8 last axis is half the logical int4
# extent; the int32 view that follows divides that by the four int8 per int32.
INT4_PER_INT8 = 2
INT8_PER_INT32 = 4
INT4_PER_INT32 = INT4_PER_INT8 * INT8_PER_INT32
W13_PACKED_N = 2 * INTERMEDIATE // INT4_PER_INT8
W2_PACKED_N = HIDDEN // INT4_PER_INT8
# The int32 view the operators consume: [E, K, packed_int8 // 4].
assert W13_PACKED_N // INT8_PER_INT32 == 2 * INTERMEDIATE // INT4_PER_INT32 == 512
assert W2_PACKED_N // INT8_PER_INT32 == HIDDEN // INT4_PER_INT32 == 512

DEFAULT_SOURCE_TOKENS = 65536
DEFAULT_CAPACITY = MAX_SEQUENCE_LENGTH

PERCENTILE_95 = 0.95
MILLISECONDS_PER_SECOND = 1000.0
REPORT_NAME = "layered_computation_operator_benchmark.md"

KERNELS = ("w13", "w2", "chain")
DISTRIBUTIONS = ("balanced", "skewed")
LAYER_MODES = ("single", "rotate")
# Eager launches each operator directly. ``aclgraph`` captures the measured unit
# into one reusable NPU graph and replays it, which is the launch path the FFN
# takes once its graph support lands. Both are measured by default, because the
# pair is the point: the two paths share no launch code, so a change that helps
# one can leave the other untouched.
EXECUTION_MODES = ("eager", "aclgraph")

# Geometry for the baseline differential check. This is deliberately *not* the
# model geometry: it is the small shape the precision suite already validated on
# device (``tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py`` uses
# K=256, N=256, E=4), which satisfies the NZ-layout constraints the operators
# enforce - ``k % 16 == 0``, ``n % 64 == 0``, ``m >= e``. A differential check
# asks whether two implementations agree, and that property does not depend on
# the shape being a real one, so the cheapest shape that exercises every
# quantization path is the right one.
GOLDEN_EXPERTS = 4
GOLDEN_K = 256
GOLDEN_N = 256
GOLDEN_ROWS = 8
# Counts are unequal so a wrong expert offset is visible, and non-zero so no
# expert is skipped.
GOLDEN_COUNTS = (3, 2, 2, 1)
# The A8W4 dequantized signal is of order 1e-4 and one int8 LSB is about 5e-06,
# so this bounds the comparison at a couple of LSB with a little headroom for a
# different seed than the precision suite measured.
GOLDEN_ATOL = 2e-5
GOLDEN_RTOL = 0.04
GOLDEN_SCALE_RTOL = 2**-7


def derive_valid_rows(
    source_tokens: int, top_k: int = TOP_K, ep: int = EXPERT_PARALLEL
) -> int:
    """Return M, the routed token-expert rows one FFN rank receives.

    ``top-k`` rows are emitted per source token and split evenly across the
    expert-parallel ranks. A non-integral quotient means these parameters cannot
    describe a balanced batch, so it is rejected rather than truncated: rounding
    would let the routing metadata disagree with the modelled load.
    """
    if source_tokens < 1:
        raise ValueError(f"source_tokens must be positive, got {source_tokens}")
    if ep < 1 or top_k < 1:
        raise ValueError(f"top_k and ep must be positive, got {top_k} and {ep}")
    numerator = top_k * source_tokens
    if numerator % ep:
        raise ValueError(
            f"top_k * source_tokens ({numerator}) is not divisible by ep ({ep}); "
            "a balanced batch cannot be described with these values"
        )
    return numerator // ep


def derive_capacity(factor: float) -> int:
    """Return C, the dispatch-receive capacity, from ``BATCH_SIZE_FACTOR``.

    Mirrors the host tiling: ``C = floor(262144 * factor)`` with the factor in
    ``(0, 1]`` and at least one row allocated.
    """
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"factor must be in (0, 1], got {factor}")
    capacity = int(MAX_SEQUENCE_LENGTH * factor)
    if capacity < 1:
        raise ValueError(f"factor {factor} allocates no rows")
    return capacity


def distribution_weights(
    experts: int,
    distribution: str,
    *,
    decay: float = 0.8,
    empty_fraction: float = 0.25,
) -> list[float]:
    """Return per-expert routing weights for ``distribution``.

    ``balanced`` is the steady-load ideal every expert shares equally.
    ``skewed`` keeps the same total row count but concentrates it on a prefix of
    experts and leaves a tail empty, which is what real routing produces. Both
    are compared at equal ``M``, so the only variable is the load shape - and
    the load shape is exactly what exercises group-size variance and idle
    experts, which a balanced sweep cannot reach.
    """
    if experts < 1:
        raise ValueError(f"experts must be positive, got {experts}")
    if distribution == "balanced":
        return [1.0] * experts
    if distribution != "skewed":
        raise ValueError(f"unknown distribution {distribution!r}")
    empty = min(int(experts * empty_fraction), experts - 1)
    live = experts - empty
    return [decay**index for index in range(live)] + [0.0] * empty


def allocate_counts(rows: int, weights: Sequence[float]) -> list[int]:
    """Split ``rows`` across experts proportional to ``weights``, summing exactly.

    Largest-remainder apportionment: floor first, then give the leftover rows to
    the entries with the largest fractional part. Plain rounding would not sum to
    ``rows``, and ``M`` is defined as that sum, so an off-by-one here would
    silently change the size of the measured matrix.
    """
    if rows < 0:
        raise ValueError(f"rows must be nonnegative, got {rows}")
    if not weights:
        raise ValueError("weights must be non-empty")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    exact = [rows * weight / total for weight in weights]
    counts = [int(value) for value in exact]
    for index in sorted(
        range(len(weights)),
        key=lambda position: exact[position] - counts[position],
        reverse=True,
    )[: rows - sum(counts)]:
        counts[index] += 1
    return counts


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize milliseconds using the nearest-rank p95 definition.

    Percentiles are taken over one configuration's own samples and never pooled
    across configurations, so a reported row always describes one measurement.
    """
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(PERCENTILE_95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def gemm_flops(rows: int, k: int, n: int) -> int:
    """Return the FLOP count of one dense GEMM (one multiply plus one add)."""
    return 2 * rows * k * n


@dataclass(frozen=True)
class Case:
    """One measured configuration: kernel, load shape, launch path, selection."""

    kernel: str
    source_tokens: int
    capacity: int
    distribution: str
    layer_mode: str
    execution_mode: str
    valid_rows: int
    counts: tuple[int, ...]

    def describe(self) -> str:
        # Five decimals, because three renders a near-full case (M/C = 0.99998)
        # as "1.000", which reads as the exact-full case that cannot occur.
        return (
            f"kernel={self.kernel} T={self.source_tokens} M={self.valid_rows} "
            f"C={self.capacity} M/C={self.valid_rows / self.capacity:.5f} "
            f"dist={self.distribution} layers={self.layer_mode} "
            f"exec={self.execution_mode}"
        )

    def key(self) -> tuple[str, str, str]:
        """Return the identity a comparison table groups on.

        The execution mode is deliberately excluded: the point of measuring both
        is to put the same configuration's two launch paths side by side.
        """
        return (self.kernel, self.distribution, self.layer_mode)


def build_cases(
    *,
    source_tokens: Sequence[int],
    capacities: Sequence[int],
    distributions: Sequence[str],
    layer_modes: Sequence[str],
    kernels: Sequence[str],
    execution_modes: Sequence[str],
    experts: int = EXPERTS_PER_RANK,
) -> list[Case]:
    """Return every requested configuration in a stable order.

    ``single`` repeats one layer, showing the steady-state cost of a fixed
    selection. ``rotate`` walks the whole layer list, which is what a model
    forward does and what the amortized per-layer figure is taken from.

    Every execution mode in ``execution_modes`` is planned for every combination,
    because the comparison between them is the point of that axis.

    The reference harness also rejects a configuration where one expert's rows
    exceed the capacity, because CAM chunks whole experts. That check is implied
    here rather than repeated: ``counts`` is an apportionment of ``M``, so the
    largest entry can never exceed ``M``, and ``M <= C`` is enforced above. A
    separate per-expert bound would be unreachable code.
    """
    if not source_tokens or not capacities or not distributions or not layer_modes:
        raise ValueError("every axis needs at least one value")
    if not kernels:
        raise ValueError("at least one kernel is needed")
    if not execution_modes:
        raise ValueError("at least one execution mode is needed")
    for mode in execution_modes:
        if mode not in EXECUTION_MODES:
            raise ValueError(f"unknown execution mode {mode!r}")
    cases: list[Case] = []
    for kernel in kernels:
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        for tokens in source_tokens:
            rows = derive_valid_rows(tokens)
            for capacity in capacities:
                if capacity < 1:
                    raise ValueError(f"capacity must be positive, got {capacity}")
                if rows > capacity:
                    raise ValueError(
                        f"M={rows} exceeds C={capacity}; a receive chunk must hold "
                        "every row it was sized for"
                    )
                for distribution in distributions:
                    counts = allocate_counts(
                        rows, distribution_weights(experts, distribution)
                    )
                    assert max(counts, default=0) <= rows, (
                        "an apportionment of M cannot exceed M"
                    )
                    for layer_mode in layer_modes:
                        for execution_mode in execution_modes:
                            cases.append(
                                Case(
                                    kernel=kernel,
                                    source_tokens=tokens,
                                    capacity=capacity,
                                    distribution=distribution,
                                    layer_mode=layer_mode,
                                    execution_mode=execution_mode,
                                    valid_rows=rows,
                                    counts=tuple(counts),
                                )
                            )
    return cases


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def plan_warnings(cases: Sequence[Case]) -> list[str]:
    """Return the modelling caveats the plan must surface before any run."""
    warnings: list[str] = []
    for tokens in sorted({case.source_tokens for case in cases}):
        if tokens > UINT16_MAX:
            warnings.append(
                f"source batch {tokens} exceeds the uint16 bound {UINT16_MAX}; "
                "dispatch counters and token addresses may not be representable"
            )
    non_power_of_two = sorted(
        capacity
        for capacity in {case.capacity for case in cases}
        if not is_power_of_two(capacity)
    )
    if non_power_of_two:
        warnings.append(
            f"capacities {non_power_of_two} are not powers of two; the reference "
            "harness constrains BATCH_SIZE_FACTOR-derived capacity to a power of two"
        )
    return warnings


def describe_plan(cases: Sequence[Case], args: argparse.Namespace) -> dict[str, Any]:
    """Return the JSON-serializable description printed by ``--dry-run``."""
    ratios = sorted({case.valid_rows / case.capacity for case in cases})
    return {
        "status": "dry-run; no device execution",
        "geometry": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts_per_rank": EXPERTS_PER_RANK,
            "top_k": TOP_K,
            "expert_parallel": EXPERT_PARALLEL,
            "layers": args.layers,
            # Shapes as the operators consume them: int32 storage with eight int4
            # nibbles per word, so the last axis is the logical width over eight.
            "w13_packed_weight": [EXPERTS_PER_RANK, HIDDEN, W13_PACKED_N // 4],
            "w2_packed_weight": [EXPERTS_PER_RANK, INTERMEDIATE, W2_PACKED_N // 4],
            "w13_scale_per_channel": [EXPERTS_PER_RANK, 2 * INTERMEDIATE],
            "w2_scale_per_channel": [EXPERTS_PER_RANK, 1, HIDDEN],
            "activation_w13": [None, HIDDEN],
            "activation_w2": [None, INTERMEDIATE],
        },
        "timing_contract": {
            "host_ms": "synchronized host interval of one measured unit",
            "event_ms": "NPU stream event interval of the same unit",
            "unit": (
                "one operator call under --layer-mode single; one full L-layer "
                "rotation under rotate, reported divided by L"
            ),
            "eager": "each operator call goes to the device as it is issued",
            "aclgraph": (
                "one measured unit is captured into a reusable NPUGraph after the "
                "warmup, then each sample is a replay of it; the input restore "
                "happens before the start event so it stays outside the interval"
            ),
            "warmup": (
                "absorbs the first tiling of each shape before any sample, because "
                "the tiling is cached per shape and tiling key; under aclgraph it "
                "also keeps that one-time work out of the captured graph"
            ),
        },
        "capacity_note": (
            "M/C == 1 exactly is unreachable: M = top-k * T / EP carries a factor "
            "of three while C must be a power of two. Near-full capacity is "
            "reachable because only C is constrained to a power of two; for "
            "example T=43690 with C=65536 gives M=65535, M/C=0.99998."
        ),
        "capacity_utilisation": ratios,
        "warnings": plan_warnings(cases),
        "case_count": len(cases),
        "cases": [case.describe() for case in cases],
    }


# --------------------------------------------------------------------------- #
#  Weight and activation construction
# --------------------------------------------------------------------------- #


def unpack_nibbles(torch: Any, packed: Any) -> tuple[Any, Any]:
    """Return the (low, high) signed int4 values of a packed int8 tensor.

    Arithmetic shifts sign-extend each nibble back to a signed four-bit value:
    ``(x << 4) >> 4`` isolates the low nibble and ``x >> 4`` the high one. The
    loader stores the first value of each pair in the low nibble, so the low
    nibble corresponds to the even index along the packed axis.
    """
    return (packed << 4) >> 4, packed >> 4


def pack_int4_weights(
    torch: Any,
    torch_npu: Any,
    experts: int,
    k: int,
    n: int,
) -> tuple[Any, Any, Any]:
    """Return one layer's packed weight, compensation, and per-channel scale.

    All three operands come back on the NPU. The operator itself takes all three
    and compares their devices, so a compensation left on the host is rejected by
    the dispatcher with "Expected all tensors to be on the same device" rather
    than by anything with a useful message.

    Reproduces the production W4A8 loader rather than randomizing an ND tensor,
    because a plain weight would tile and stream differently from a loaded one:
    two signed int4 values per int8 along the output axis, an FRACTAL_NZ format
    cast, then an int32 view.

    Uniform random int8 bytes are already valid packed int4 - each nibble read as
    a signed four-bit value is uniform on ``[-8, 7]`` - so the packed form is
    generated directly. Materializing an ``[E, K, N]`` int32 tensor to pack it
    down would cost four gigabytes per layer for no gain.

    The compensation is the column sum of the dequantized weight, times
    ``ASSIST_FACTOR``. Because the per-channel scale depends only on ``n``, that
    sum factors into a per-nibble-parity reduction, so only the running column
    sums are live instead of the full dequantized weight - which for a real layer
    would be 2 GiB.

    The scale is returned on the host because its first use is a bit-level
    reinterpretation to int64, which callers do on the CPU before transferring
    it; see ``build_weights``.
    """
    if n % INT4_PER_INT32:
        raise ValueError(f"n={n} must be a multiple of {INT4_PER_INT32}")
    packed = torch.randint(
        -128, 128, (experts, k, n // INT4_PER_INT8), dtype=torch.int8
    )
    scale = torch.rand((experts, n), dtype=torch.float32) * 1e-3 + 1e-4

    low_sum = torch.zeros((experts, n // INT4_PER_INT8), dtype=torch.float32)
    high_sum = torch.zeros_like(low_sum)
    block = max(k // 8, 1)
    for start in range(0, k, block):
        low, high = unpack_nibbles(torch, packed[:, start : start + block, :])
        low_sum += low.to(torch.float32).sum(dim=1)
        high_sum += high.to(torch.float32).sum(dim=1)

    # Even output columns take the low nibble of each packed byte, odd columns the
    # high one, which is the loader's little-endian packing order.
    column_sum = torch.empty((experts, n), dtype=torch.float32)
    column_sum[:, 0::2] = low_sum
    column_sum[:, 1::2] = high_sum
    compensation = (ASSIST_FACTOR * column_sum * scale).contiguous().npu()
    weight = (
        torch_npu.npu_format_cast(packed.contiguous().npu(), NZ_FORMAT)
        .view(torch.int32)
        .contiguous()
    )
    return weight, compensation, scale


def build_weights(torch: Any, torch_npu: Any, layers: int) -> dict[str, Any]:
    """Return per-layer W13 and W2 operands in the layout the operators expect.

    Both operators take a list with one entry per layer. Every layer carries
    different content so a wrong ``layer_index`` cannot pass unnoticed, and all
    layers share geometry so a single tiling covers the list.

    W13's per-channel scale is 2-D ``[E, 2I]``; W2's is 3-D ``[E, 1, H]``, which
    is what its per-channel path expects. Both carry fp32 scale bits inside an
    int64 element.
    """
    state: dict[str, list[Any]] = {
        name: [] for name in ("w13", "w2", "scale13", "scale2", "assist13", "assist2")
    }
    for _ in range(layers):
        weight, compensation, scale = pack_int4_weights(
            torch, torch_npu, EXPERTS_PER_RANK, HIDDEN, 2 * INTERMEDIATE
        )
        state["w13"].append(weight)
        state["assist13"].append(compensation)
        state["scale13"].append(scale.view(torch.int32).to(torch.int64).npu())

        weight, compensation, scale = pack_int4_weights(
            torch, torch_npu, EXPERTS_PER_RANK, INTERMEDIATE, HIDDEN
        )
        state["w2"].append(weight)
        state["assist2"].append(compensation)
        state["scale2"].append(
            scale.view(torch.int32)
            .to(torch.int64)
            .reshape(EXPERTS_PER_RANK, 1, HIDDEN)
            .npu()
        )
    return state


def build_activation(torch: Any, capacity: int, width: int) -> tuple[Any, Any]:
    """Return a capacity-sized int8 activation and its per-token fp32 scale."""
    activation = torch.randint(-64, 64, (capacity, width), dtype=torch.int8).npu()
    scale = torch.full((capacity,), 5e-3, dtype=torch.float32).npu()
    return activation, scale


def build_group_list(torch: Any, counts: Sequence[int]) -> Any:
    """Return the device-side per-expert counts (``group_list_type=1``)."""
    return torch.tensor(list(counts), dtype=torch.int64, device="npu")


def layer_index_tensor(torch: Any, index: int) -> Any:
    """Return the one-element int64 layer selector."""
    return torch.full((1,), index, dtype=torch.int64, device="npu")


# --------------------------------------------------------------------------- #
#  Layered operator calls
# --------------------------------------------------------------------------- #


def run_w13(torch, state, activation, activation_scale, group_list, layer_index):
    """Call the layered fused W13 operator (per-channel, per-token quant)."""
    return torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
        activation,
        state["w13"],
        state["scale13"],
        state["assist13"],
        activation_scale,
        group_list,
        layer_index,
        dequant_mode=0,
        quant_mode=0,
        group_list_type=1,
    )


def run_w2(torch, state, activation, activation_scale, group_list, layer_index):
    """Call the layered W2 operator."""
    return torch.ops.afd_ascend.grouped_matmul_layered(
        [activation],
        state["w2"],
        state["assist2"],
        state["scale2"],
        layer_index,
        group_list,
        per_token_scale=activation_scale,
        group_list_type=1,
        split_item=3,
        output_dtype=torch.bfloat16,
    )[0]


# --------------------------------------------------------------------------- #
#  Baseline calls
# --------------------------------------------------------------------------- #


def baseline_gmm(
    torch,
    torch_npu,
    weight,
    compensation,
    scale,
    activation,
    activation_scale,
    group_list,
):
    """One built-in W4A8 grouped matmul.

    ``torch`` is needed only for the ``bfloat16`` dtype constant, since the
    dtype enum lives on the framework rather than on the device module. Every
    baseline helper here takes both for that reason and nothing else.

    The built-in tiling reads ``dim(1)`` as the K-group count unconditionally, so
    the per-channel ``[E, N]`` scale is handed over as an ``[E, 1, N]`` view.
    Unlike the fused ``npu_grouped_matmul_swiglu_quant_v2``, this operator's
    infershape multiplies an int32 weight's last axis by eight, so N comes out
    right for packed int4 and the same weight tensor can be reused as-is.
    """
    return torch_npu.npu_grouped_matmul(
        x=[activation],
        weight=[weight],
        bias=[compensation],
        scale=[scale.unsqueeze(1)],
        per_token_scale=[activation_scale],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]


def baseline_w13(
    torch, torch_npu, state, activation, activation_scale, group_list, layer
):
    """Concatenated W4A8 baseline for W13: GMM, then SwiGLU, then dynamic quant."""
    gemm = baseline_gmm(
        torch,
        torch_npu,
        state["w13"][layer],
        state["assist13"][layer],
        state["scale13"][layer],
        activation,
        activation_scale,
        group_list,
    )
    gated = torch_npu.npu_swiglu(gemm, dim=-1)
    return torch_npu.npu_dynamic_quant(gated, quant_mode="pertoken")


def baseline_w2(
    torch, torch_npu, state, activation, activation_scale, group_list, layer
):
    """Built-in fused W2 baseline, driven with byte-identical inputs."""
    return baseline_gmm(
        torch,
        torch_npu,
        state["w2"][layer],
        state["assist2"][layer],
        state["scale2"][layer].reshape(EXPERTS_PER_RANK, HIDDEN),
        activation,
        activation_scale,
        group_list,
    )


def validate_baseline(torch, torch_npu) -> list[str]:
    """Check the concatenated W13 baseline against the fused layered operator.

    The question that matters is whether the baseline computes *the same thing*
    the fused operator does, because a baseline doing something else would make
    every speedup meaningless. Answering it differentially avoids any assumption
    about the assist matrix: the A8W4 compensation the kernel consumes is not a
    plain linear bias added to the mathematical result - the CPU golden in
    ``test_gmm_swiglu_quant_v2_layered_precision.py`` omits it entirely - so a
    from-scratch golden here would have to guess its semantics.

    Both sides are driven with byte-identical tensors instead, which is the same
    differential strategy the precision suite already validated on device. The
    geometry is its proven one: unequal per-expert counts so a wrong expert
    offset is visible, and a shape of its own so this check cannot inherit the
    tiling of any benchmark shape.

    **Each side gets its own copy of the activation.** The A8W4 MSD path may
    rewrite the activation buffer in place, so sharing one tensor would let
    whichever side ran first change what the second one reads, and the two would
    be compared on different inputs. Every call in the precision suite is cloned
    for the same reason.
    """
    rows = GOLDEN_ROWS
    counts = list(GOLDEN_COUNTS)
    assert sum(counts) == rows, "the golden counts must describe the golden rows"
    weight, compensation, scale = pack_int4_weights(
        torch, torch_npu, GOLDEN_EXPERTS, GOLDEN_K, GOLDEN_N
    )
    activation = torch.randint(-64, 64, (rows, GOLDEN_K), dtype=torch.int8).npu()
    activation_scale = torch.full((rows,), 5e-3, dtype=torch.float32).npu()
    group_list = build_group_list(torch, counts)
    encoded_scale = scale.view(torch.int32).to(torch.int64).npu()
    layer_index = layer_index_tensor(torch, 0)

    baseline_y, baseline_scale = baseline_w13(
        torch,
        torch_npu,
        {
            "w13": [weight],
            "assist13": [compensation],
            "scale13": [encoded_scale],
        },
        activation.clone(),
        activation_scale,
        group_list,
        0,
    )
    layered_y, layered_scale = run_w13(
        torch,
        {
            "w13": [weight],
            "scale13": [encoded_scale],
            "assist13": [compensation],
        },
        activation.clone(),
        activation_scale,
        group_list,
        layer_index,
    )
    torch_npu.npu.synchronize()

    context = (
        f"concatenated baseline vs fused operator: rows={rows} "
        f"experts={GOLDEN_EXPERTS} k={GOLDEN_K} n={GOLDEN_N} counts={counts}"
    )
    assert tuple(baseline_y.shape) == tuple(layered_y.shape), (
        f"{context}: baseline y {tuple(baseline_y.shape)} != "
        f"layered y {tuple(layered_y.shape)}"
    )
    baseline_scale_host = baseline_scale.cpu()
    layered_scale_host = layered_scale.cpu()
    baseline_values = baseline_y.cpu().float() * baseline_scale_host.unsqueeze(1)
    layered_values = layered_y.cpu().float() * layered_scale_host.unsqueeze(1)
    max_error = float((baseline_values - layered_values).abs().max())
    scale_max_error = float((baseline_scale_host - layered_scale_host).abs().max())
    # The per-token scale is an encoding choice, so the informative statistic is
    # the ratio between the two sides rather than their difference. A uniform
    # ratio near 128/127 means the two paths use a different int8 divisor
    # (2**7 against 2**7 - 1, a standard symmetric-quantization difference) with
    # the dequantized product still agreeing. A scattered ratio means they were
    # fed different data. The two cases need opposite responses, and a bare
    # pass/fail cannot tell them apart.
    ratio = (baseline_scale_host / layered_scale_host.abs().clamp_min(1e-30)).flatten()
    scale_ratio_min = float(ratio.min())
    scale_ratio_max = float(ratio.max())
    scale_rel_error = float(
        (
            (baseline_scale_host - layered_scale_host).abs()
            / layered_scale_host.abs().clamp_min(1e-30)
        ).max()
    )
    lsb_delta = (
        (baseline_y.cpu().to(torch.int32) - layered_y.cpu().to(torch.int32)).abs().max()
    )
    mismatch_count = int(
        (baseline_y.cpu().to(torch.int32) != layered_y.cpu().to(torch.int32))
        .sum()
        .item()
    )
    # Printed before asserting, not only carried in the assertion message, because
    # this is the one check whose result decides whether any later timing means
    # anything.
    report = (
        f"{context}: dequant_max_abs_err={max_error:.3e} "
        f"(rtol={GOLDEN_RTOL} atol={GOLDEN_ATOL}) "
        f"y_scale_max_abs_err={scale_max_error:.3e} "
        f"y_scale_max_rel_err={scale_rel_error:.3e} (rtol={GOLDEN_SCALE_RTOL}) "
        f"y_scale_ratio=[{scale_ratio_min:.6f}, {scale_ratio_max:.6f}] "
        f"int8_mismatch={mismatch_count}/{baseline_y.numel()} "
        f"int8_max_lsb_delta={int(lsb_delta)}"
    )
    print(report, file=sys.stderr)

    # The dequantized product is the physical quantity and the assertion that
    # matters: it is what the benchmark claims to compare. Keep it first so a
    # scale-convention difference cannot hide a real numerical divergence.
    torch.testing.assert_close(
        baseline_values,
        layered_values,
        rtol=GOLDEN_RTOL,
        atol=GOLDEN_ATOL,
        msg=report,
    )
    torch.testing.assert_close(
        baseline_scale_host,
        layered_scale_host,
        rtol=GOLDEN_SCALE_RTOL,
        atol=0.0,
        msg=report,
    )
    return [f"baseline_differential PASSED {report}"]


# --------------------------------------------------------------------------- #
#  Measurement
# --------------------------------------------------------------------------- #


def time_unit(
    torch_npu: Any,
    run: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    restore: Callable[[], None] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Run the warmup and sampling loops over ``run``, without capturing it.

    ``restore`` re-materializes an input the operator may have rewritten in
    place. It is called and drained by a synchronize *before* the start event, so
    the copy never enters either measured interval.
    """
    for _ in range(warmup):
        if restore is not None:
            restore()
        run()
    torch_npu.npu.synchronize()

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    host_samples: list[float] = []
    event_samples: list[float] = []
    for _ in range(iterations):
        if restore is not None:
            restore()
            torch_npu.npu.synchronize()

        host_start = time.perf_counter()
        start.record()
        run()
        end.record()
        torch_npu.npu.synchronize()
        host_samples.append(
            (time.perf_counter() - host_start) * MILLISECONDS_PER_SECOND
        )
        event_samples.append(start.elapsed_time(end))

    return {"host": summarize(host_samples), "event": summarize(event_samples)}


def measure(
    torch: Any,
    torch_npu: Any,
    run: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    restore: Callable[[], None] | None = None,
    execution_mode: str = "eager",
) -> dict[str, dict[str, float | int]]:
    """Time one measured unit under the requested launch path.

    ``eager`` calls the operators directly. ``aclgraph`` records one unit into a
    reusable ``NPUGraph`` and times replays of it, which is the launch path the
    FFN takes once its graph support lands.

    The warmup loop is not optional for either path: the tiling is cached per
    shape and tiling key, so the first call of every shape pays a tiling cost the
    later ones do not. Under ``aclgraph`` it matters twice over, because a warmup
    that ran inside the capture would record that one-time work into the graph
    and inflate every replay.
    """
    if execution_mode == "eager":
        return time_unit(
            torch_npu,
            run,
            warmup=warmup,
            iterations=iterations,
            restore=restore,
        )
    if execution_mode != "aclgraph":
        raise ValueError(f"unknown execution mode {execution_mode!r}")

    # Warm the operators outside the capture. Their first call resolves tiling
    # and allocates workspace; capturing that would make every replay pay it, and
    # would also bake the one-time allocations into the graph's memory pool.
    for _ in range(max(warmup, 1)):
        if restore is not None:
            restore()
        run()
    torch_npu.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    torch_npu.npu.synchronize()

    # The captured allocations live in the graph's own memory pool, not the
    # caching allocator's default stream pool, so replaying keeps writing to the
    # same buffers regardless of what Python still references.
    #
    # Capturability is not an assumption: capture of these two operators succeeded
    # on 910C, which is the evidence that neither the bindings nor the two op_api
    # entries read a tensor back to the host. What this harness does not check is
    # that a replay produces the same numbers as the eager call - it times the two
    # paths and nothing more - so a numerical graph-versus-eager comparison still
    # belongs to a precision suite.

    # Replays are timed the same way as eager units, with the input restored
    # outside the measured interval. The A8W4 MSD path still rewrites the
    # activation buffer it was handed, so a replay repeat needs the same
    # pristine content an eager repeat does.
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    host_samples: list[float] = []
    event_samples: list[float] = []
    for _ in range(iterations):
        if restore is not None:
            restore()
            torch_npu.npu.synchronize()
        host_start = time.perf_counter()
        start.record()
        graph.replay()
        end.record()
        torch_npu.npu.synchronize()
        host_samples.append(
            (time.perf_counter() - host_start) * MILLISECONDS_PER_SECOND
        )
        event_samples.append(start.elapsed_time(end))

    return {"host": summarize(host_samples), "event": summarize(event_samples)}


def layer_plan(torch, case: Case, layers: int) -> list[Any]:
    """Return the layer selectors one measured unit walks.

    ``single`` repeats the first layer, isolating steady-state cost. ``rotate``
    walks the whole list, which is what a model forward does and what the
    per-layer amortization is measured from.
    """
    if case.layer_mode == "single":
        return [layer_index_tensor(torch, 0)]
    return [layer_index_tensor(torch, index) for index in range(layers)]


def measure_case(torch, torch_npu, state, case: Case, args) -> dict[str, Any]:
    """Measure one case for a layered operator and its baseline.

    The measured unit is one operator call in ``single`` mode and one full
    L-layer rotation in ``rotate`` mode; rotation totals are reported divided by
    L so the two modes are directly comparable.

    The baseline walks the same layer sequence as the layered path, re-launching
    the built-in once per layer. That is exactly the cost the layered family
    replaces with one device-indexed call, so the comparison is the one the
    operator exists to win - not a same-shape kernel-versus-kernel race.
    """
    indices = layer_plan(torch, case, args.layers)
    slots = [int(index.item()) for index in indices]
    width = INTERMEDIATE if case.kernel == "w2" else HIDDEN
    activation, activation_scale = build_activation(torch, case.capacity, width)
    group_list = build_group_list(torch, case.counts)
    # The A8W4 MSD path may rewrite the activation buffer, so each timed unit gets
    # a pristine copy to restore from. Without this, repetition N would consume
    # the output of repetition N-1 and the timing would describe a different
    # problem than the one under test.
    pristine = activation.clone()

    def restore() -> None:
        activation.copy_(pristine)

    def layered() -> None:
        for index in indices:
            gated, gated_scale = run_w13(
                torch, state, activation, activation_scale, group_list, index
            )
            if case.kernel == "w2":
                continue
            if case.kernel == "chain":
                run_w2(torch, state, gated, gated_scale, group_list, index)

    def layered_w2_only() -> None:
        for index in indices:
            run_w2(torch, state, activation, activation_scale, group_list, index)

    def baseline() -> None:
        for slot in slots:
            gated, gated_scale = baseline_w13(
                torch, torch_npu, state, activation, activation_scale, group_list, slot
            )
            if case.kernel == "w2":
                continue
            if case.kernel == "chain":
                baseline_w2(
                    torch, torch_npu, state, gated, gated_scale, group_list, slot
                )

    def baseline_w2_only() -> None:
        for slot in slots:
            baseline_w2(
                torch, torch_npu, state, activation, activation_scale, group_list, slot
            )

    fn_layered = layered_w2_only if case.kernel == "w2" else layered
    fn_baseline = baseline_w2_only if case.kernel == "w2" else baseline

    result: dict[str, Any] = {
        "case": case.describe(),
        "layers": len(indices),
        # Carried separately from the prose so the comparison tables can group
        # one configuration's launch paths without re-parsing the label.
        "kernel": case.kernel,
        "distribution": case.distribution,
        "mode": case.layer_mode,
        "execution_mode": case.execution_mode,
        # The workload geometry, structured so the aggregation keys can include
        # it. Both --source-tokens and --capacity accept comma lists, so T, M and
        # C vary independently within one run; two results that differ only in
        # geometry would otherwise collide on the same key and the later one
        # would silently overwrite the earlier measurement.
        "source_tokens": case.source_tokens,
        "capacity": case.capacity,
        "valid_rows": case.valid_rows,
        "layered": measure(
            torch,
            torch_npu,
            fn_layered,
            warmup=args.warmup,
            iterations=args.iterations,
            # Every A8W4 MSD path - both operators and both their baselines - may
            # rewrite the activation buffer it was handed. Without the restore,
            # layer 1 of a rotation would consume layer 0's rewritten input and
            # the rotation would time a different problem than the model runs.
            restore=restore,
            execution_mode=case.execution_mode,
        ),
        "baseline": measure(
            torch,
            torch_npu,
            fn_baseline,
            warmup=args.warmup,
            iterations=args.iterations,
            restore=restore,
            execution_mode=case.execution_mode,
        ),
    }
    result["flops"] = {
        "w13": gemm_flops(case.valid_rows, HIDDEN, 2 * INTERMEDIATE),
        "w2": gemm_flops(case.valid_rows, INTERMEDIATE, HIDDEN),
        "chain": gemm_flops(case.valid_rows, HIDDEN, 2 * INTERMEDIATE)
        + gemm_flops(case.valid_rows, INTERMEDIATE, HIDDEN),
    }[case.kernel]
    result["weight_bytes"] = {
        "w13": EXPERTS_PER_RANK * HIDDEN * (W13_PACKED_N // INT8_PER_INT32) * 4,
        "w2": EXPERTS_PER_RANK * INTERMEDIATE * (W2_PACKED_N // INT8_PER_INT32) * 4,
        "chain": EXPERTS_PER_RANK
        * (
            HIDDEN * (W13_PACKED_N // INT8_PER_INT32)
            + INTERMEDIATE * (W2_PACKED_N // INT8_PER_INT32)
        )
        * 4,
    }[case.kernel]
    return result


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #


def per_layer_median_ms(result: dict[str, Any], which: str) -> float | None:
    """Return one timer's median for the measured unit, divided by layer count."""
    summary = result[which]["event"]
    if not summary.get("count"):
        return None
    return float(summary["median_ms"]) / result["layers"]


def render_results_table(results: Sequence[dict[str, Any]]) -> list[str]:
    """Render the per-case comparison table."""
    lines = [
        "| Case | Exec | Layers | Layered (ms/layer) | Baseline (ms/layer) | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        layered = per_layer_median_ms(result, "layered")
        baseline = per_layer_median_ms(result, "baseline")
        speedup = f"{baseline / layered:.3f}x" if layered and baseline else "n/a"
        exec_mode = result.get("execution_mode", "n/a")
        lines.append(
            f"| {result['case']} | {exec_mode} | {result['layers']} | "
            f"{layered:.4f} | {baseline:.4f} | {speedup} |"
            if layered is not None and baseline is not None
            else f"| {result['case']} | {exec_mode} | {result['layers']} | "
            f"{layered if layered is not None else 'n/a'} | n/a | n/a |"
        )
    return lines


def result_geometry(result: dict[str, Any]) -> tuple[int, int, int] | None:
    """Return ``(T, M, C)`` for one result, or None if it cannot be read.

    The aggregation tables key on this. ``--source-tokens`` and ``--capacity``
    both accept comma lists, so a single run can hold several workloads that
    share a kernel, distribution, layer mode and launch path and differ only
    here. Without it in the key the second workload would overwrite the first
    and the report would quietly show one geometry's numbers for all of them.
    """
    try:
        return (
            int(result["source_tokens"]),
            int(result["valid_rows"]),
            int(result["capacity"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def geometry_label(geometry: tuple[int, int, int]) -> str:
    """Render ``(T, M, C)`` compactly for a table cell."""
    tokens, rows, capacity = geometry
    return f"T={tokens} M={rows} C={capacity}"


def render_execution_mode_table(results: Sequence[dict[str, Any]]) -> list[str]:
    """Compare eager against aclgraph for each configuration and side.

    This is the table the two launch paths exist for. A speedup here is the
    graph's own contribution: both columns recompute the same weights on the
    same input with the same layer selection, so the only difference is how the
    work reaches the device. Reading it against the baseline speedup separates
    "the fused operator is faster" from "the graph launch is cheaper", which are
    independent wins.

    The key includes the workload geometry, because a run may sweep several
    ``--source-tokens`` or ``--capacity`` values that are otherwise identical.
    """
    by_case: dict[
        tuple[str, str, str, int, tuple[int, int, int]], dict[str, float]
    ] = {}
    for result in results:
        kernel = result.get("kernel")
        distribution = result.get("distribution")
        layer_mode = result.get("mode")
        exec_mode = result.get("execution_mode")
        geometry = result_geometry(result)
        if None in (kernel, distribution, layer_mode, exec_mode) or geometry is None:
            continue
        key = (
            str(kernel),
            str(distribution),
            str(layer_mode),
            int(result.get("layers", 1)),
            geometry,
        )
        for which in ("layered", "baseline"):
            value = per_layer_median_ms(result, which)
            if value is not None:
                by_case.setdefault(key, {})[f"{which}:{exec_mode}"] = value

    lines = [
        "| Kernel | Distribution | Geometry | Layers | Side | Eager (ms/layer) | "
        "ACL graph (ms/layer) | Graph speedup |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for kernel, distribution, layer_mode, layers, geometry in sorted(by_case):
        values = by_case[(kernel, distribution, layer_mode, layers, geometry)]
        for which in ("layered", "baseline"):
            eager = values.get(f"{which}:eager")
            graph = values.get(f"{which}:aclgraph")
            if eager is None or graph is None:
                continue
            speedup = f"{eager / graph:.3f}x" if graph else "n/a"
            lines.append(
                f"| {kernel} | {distribution} ({layer_mode}) | "
                f"{geometry_label(geometry)} | {layers} | "
                f"{which} | {eager:.4f} | {graph:.4f} | {speedup} |"
            )
    if len(lines) == 2:
        lines.append(
            "| (no configuration was measured under both launch paths) | | | | | | |"
        )
    return lines


def render_efficiency_table(results: Sequence[dict[str, Any]]) -> list[str]:
    """Render achieved compute and weight bandwidth from the layered timings."""
    lines = [
        "| Case | Effective TFLOP/s | Weight bandwidth (GB/s) |",
        "| --- | ---: | ---: |",
    ]
    for result in results:
        layered = per_layer_median_ms(result, "layered")
        if not layered:
            lines.append(f"| {result['case']} | n/a | n/a |")
            continue
        seconds = layered / MILLISECONDS_PER_SECOND
        lines.append(
            f"| {result['case']} | "
            f"{result['flops'] / seconds / 1e12:.2f} | "
            f"{result['weight_bytes'] / seconds / 1e9:.1f} |"
        )
    return lines


def render_timer_table(results: Sequence[dict[str, Any]]) -> list[str]:
    """Render host and device timings side by side.

    The gap between the two columns is the launch overhead each mode pays. It is
    reported explicitly because the layered family's stated benefit is removing
    per-layer host re-launches, which only ``host_ms`` reflects.
    """
    lines = [
        "| Case | Mode | Host median (ms/layer) | Device median (ms/layer) | "
        "Host/device |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        for which in ("layered", "baseline"):
            summary = result[which]["host"]
            device = result[which]["event"]
            if not summary.get("count") or not device.get("count"):
                continue
            host_ms = float(summary["median_ms"]) / result["layers"]
            device_ms = float(device["median_ms"]) / result["layers"]
            ratio = f"{host_ms / device_ms:.2f}" if device_ms else "n/a"
            lines.append(
                f"| {result['case']} | {which} | {host_ms:.4f} | "
                f"{device_ms:.4f} | {ratio} |"
            )
    return lines


def render_decomposition_table(results: Sequence[dict[str, Any]]) -> list[str]:
    """Separate the per-call cost from the per-unit fixed cost.

    ``single`` and ``rotate`` measure the same operator at the same workload, so
    the pair pins a two-parameter model: a timed unit with ``n`` calls costs
    ``n * c + f``, where ``c`` is the steady-state cost of one call and ``f`` is
    the fixed cost of a unit. Both reported values are already divided by the
    layer count, so ``single`` gives ``c + f`` and ``rotate`` gives ``c + f / L``.
    Subtracting and scaling recovers ``f``, and then ``c``.

    Only cases that measured both layer modes under the same launch path can be
    solved, so the table skips the rest rather than printing a number derived
    from one point. The execution mode joins the key: eager and aclgraph have
    different per-call and fixed costs, and mixing them would produce a fit for
    neither. The workload geometry joins the key too, so a run that sweeps
    several ``--source-tokens`` or ``--capacity`` values keeps one row per
    workload instead of overwriting all but the last. A result that lacks the
    identifying fields is skipped, which keeps the renderer usable on the
    synthetic results the CPU tests build.
    """
    by_case: dict[
        tuple[str, str, str, tuple[int, int, int]], dict[str, dict[str, Any]]
    ] = {}
    for result in results:
        kernel = result.get("kernel")
        mode = result.get("mode")
        distribution = result.get("distribution")
        geometry = result_geometry(result)
        if kernel is None or mode is None or distribution is None or geometry is None:
            continue
        exec_mode = str(result.get("execution_mode", "eager"))
        # Grouped by kernel, distribution, launch path and geometry. ``layers``
        # must not join the key: it is 1 under single and the resident count
        # under rotate, which is exactly the difference being solved for.
        by_case.setdefault((kernel, distribution, exec_mode, geometry), {})[mode] = (
            result
        )

    lines = [
        "| Kernel | Distribution | Exec | Geometry | Per-call `c` (ms) | "
        "Fixed `f` (ms) | Steady-state speedup |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (
        kernel,
        distribution,
        exec_mode,
        geometry,
    ), modes in sorted(by_case.items()):
        single = modes.get("single")
        rotate = modes.get("rotate")
        if single is None or rotate is None:
            continue
        layers = int(rotate.get("layers", 1))
        if layers < 2:
            continue
        row: list[tuple[float, float] | None] = []
        for which in ("layered", "baseline"):
            single_ms = per_layer_median_ms(single, which)
            rotate_ms = per_layer_median_ms(rotate, which)
            if single_ms is None or rotate_ms is None:
                row.append(None)
                continue
            # Both values are per-layer already: single = c + f, rotate = c + f/L.
            fixed = (single_ms - rotate_ms) * layers / (layers - 1)
            row.append((single_ms - fixed, fixed))
        layered, baseline = row
        if layered is None or baseline is None:
            continue
        speedup = f"{baseline[0] / layered[0]:.3f}x" if layered[0] else "n/a"
        lines.append(
            f"| {kernel} | {distribution} | {exec_mode} | "
            f"{geometry_label(geometry)} | {layered[0]:.4f} | "
            f"{layered[1]:.4f} | {speedup} |"
        )
    return lines


def render_report(
    args: argparse.Namespace,
    results: Sequence[dict[str, Any]],
    notes: Sequence[str],
) -> str:
    """Render the archived benchmark report as markdown."""
    lines = [
        "# Layered computation operator benchmark",
        "",
        "Generated by `tests/npu/gmm_layered_performance.py` for the two AFD-owned",
        "layered routed-expert operators on the DSV4-Flash W4A8 / EP4 FFN path.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
        f"| Hidden size `H` | {HIDDEN} |",
        f"| Expert intermediate `I` | {INTERMEDIATE} |",
        f"| Local experts `E` | {EXPERTS_PER_RANK} |",
        f"| top-k | {TOP_K} |",
        f"| Expert parallel | {EXPERT_PARALLEL} |",
        f"| Layers resident | {args.layers} |",
        f"| Warmup / iterations | {args.warmup} / {args.iterations} |",
        f"| Seed | {args.seed} |",
        "",
        "`M = top-k * T / EP` and `C = 262144 * BATCH_SIZE_FACTOR`. `M/C == 1`",
        "exactly is unreachable because `M` carries a factor of three while `C`",
        "must be a power of two. That does not cap utilisation near the values a",
        "default sweep reaches: only `C` is constrained to a power of two, so a",
        "source batch that is not one can sit just under capacity - `T = 43690`",
        "with `C = 65536` gives `M = 65535` and `M/C = 0.99998`. Each row above",
        "states the utilisation it actually ran at.",
        "",
        "The measured unit is one operator call under `single` and one full",
        "L-layer rotation under `rotate`, with rotation totals divided by `L` so",
        "both columns are per-layer. `w13` and `w2` are the two operators, `chain`",
        "is W13 followed by W2 - the complete routed-expert compute path.",
        "",
        "**`single` and `rotate` differ, and the gap is a fixture artefact rather",
        "than a property of the operator.** Every timed unit pays a fixed cost that",
        "is independent of the call count - the drain of the preceding `synchronize`",
        "and the pipeline fill before the first kernel starts. With one call per unit",
        "that cost is charged once; with eight back-to-back calls it is amortized",
        "across them. Fitting `n * c + f` to the two modes gives `f` per launch path",
        "in the decomposition table, and how large the gap looks is entirely `f`:",
        "under `eager` it is of order a millisecond and the two modes separate",
        "clearly, while under `aclgraph` it collapses and they nearly agree. Read",
        "`rotate` as the steady-state per-layer cost; `single` is the one-shot",
        "latency, which is a different and also useful number.",
        "",
        "The baseline column is not a like-for-like kernel race. For `w13` the",
        "baseline issues three launches per layer (grouped matmul, SwiGLU, dynamic",
        "quant) where the layered path issues one, so its margin is fusion. `chain`",
        "compares four launches against two.",
        "",
        "**A ratio below 1 does not mean the kernel is slower, and `w2` is the case",
        "that shows why.** Its steady-state figure under eager is slightly under",
        "1.00, but the decomposition separates the two causes: the per-call `c` of",
        "the two sides agree closely, while the layered side carries the larger",
        "fixed term. Nearly all of the difference is that term divided by the layer",
        "count, so it describes the launch path rather than the matrix multiply.",
        "Under a graph, which removes the fixed term, `w2` returns to parity while",
        'its `c` still matches. Read a single ratio as "is this configuration',
        'faster", not as "is this kernel faster"; only the decomposition separates',
        "those two questions.",
        "",
        "**Two launch paths are measured, and they answer different questions.**",
        "`eager` calls the operators directly. `aclgraph` records one measured unit",
        "into a reusable `NPUGraph` and times its replays, which is how the",
        "deployment would run the layered FFN under a captured graph. The graph",
        "speedup table compares them; it is the graph's own contribution, separate",
        "from the fusion contribution in the results table. Capturability is not",
        "assumed: the capture succeeding on 910C is the evidence that neither the",
        "bindings nor the two op_api entries read a tensor back to the host.",
        "",
        "**What the graph removes is the per-unit fixed cost, not compute.** Compare",
        "the fixed `f` of the decomposition table across the two launch paths: the",
        "eager value is of order a millisecond while the graphed one falls to the",
        "measurement floor, and the per-call `c` barely moves. Read the graphed `f`",
        "as having become unresolvable rather than as a precise ratio - it is a",
        "difference of two nearly equal numbers, so its own value is uncertain even",
        "though its collapse is not. The mechanism predicts exactly this: a captured",
        "unit has no per-call host dispatch left to hide. It is also why the two",
        "comparisons below give such different numbers for one change.",
        "",
        "**Both paths reach the deployment as a walk over the layer list, so",
        "`rotate` is the row that describes a real step.** Eager issues one call",
        "per layer back to back; the deployment's captured graph is replayed once",
        "per layer, also back to back. Either way a step pays the fixed cost of one",
        "timed unit per `L` layers, so a layer costs `c + f / L` and the `rotate`",
        "pair is the one to quote. The `single` pair instead charges a full `f` to",
        "every layer, which is the cost of an isolated call rather than of a step,",
        "so it overstates the gain and should not be quoted as a deployment figure.",
        "",
        "How the graph is captured does not change that. One graph holding the whole",
        "list replayed once and a one-layer graph replayed `L` times both cost",
        "`L * c + f`, because only the first replay pays the pipeline fill and the",
        "rest hide behind it. This harness captures the first form; the deployment",
        "uses the second.",
        "",
        "## Results",
        "",
    ]
    lines += render_results_table(results)
    lines += ["", "## Execution mode: eager versus ACL graph", ""]
    lines += render_execution_mode_table(results)
    lines += ["", "## Achieved efficiency", ""]
    lines += render_efficiency_table(results)
    lines += ["", "## Decomposed per-call cost", ""]
    lines += render_decomposition_table(results)
    lines += ["", "## Host versus device time", ""]
    lines += render_timer_table(results)
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    def int_list(value: str) -> list[int]:
        return [int(item) for item in value.split(",")]

    def str_list(value: str) -> list[str]:
        return value.split(",")

    parser.add_argument(
        "--source-tokens",
        type=int_list,
        default=[DEFAULT_SOURCE_TOKENS],
        help=(
            "Comma-separated per-rank source batch sizes. M derives as "
            "top-k * T / EP and must divide evenly."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int_list,
        default=[DEFAULT_CAPACITY],
        help="Comma-separated dispatch-receive capacities C (powers of two).",
    )
    parser.add_argument(
        "--distribution",
        type=str_list,
        default=list(DISTRIBUTIONS),
        help=f"Comma-separated routing shapes from {DISTRIBUTIONS}.",
    )
    parser.add_argument(
        "--layer-mode",
        type=str_list,
        default=list(LAYER_MODES),
        help=f"Comma-separated layer selection modes from {LAYER_MODES}.",
    )
    parser.add_argument(
        "--execution-mode",
        type=str_list,
        default=list(EXECUTION_MODES),
        help=(
            f"Comma-separated launch paths from {EXECUTION_MODES}. 'aclgraph' "
            "captures one measured unit into a reusable NPU graph and times its "
            "replays. Both are measured by default so the tables can compare "
            "them."
        ),
    )
    parser.add_argument(
        "--only",
        type=str_list,
        default=list(KERNELS),
        help=f"Comma-separated kernels from {KERNELS}.",
    )
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--validate-baseline",
        action="store_true",
        help="Check the concatenated W13 baseline against the fused operator first.",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const=str(Path(__file__).with_name(REPORT_NAME)),
        default=None,
        help=f"Archive results to this markdown file (default: {REPORT_NAME}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived plan; imports no device packages.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Reject anything the plan cannot honour before any device work starts."""
    allowed = {
        "distribution": DISTRIBUTIONS,
        "layer_mode": LAYER_MODES,
        "execution_mode": EXECUTION_MODES,
        "only": KERNELS,
    }
    for name, values in allowed.items():
        for value in getattr(args, name):
            if value not in values:
                raise ValueError(
                    f"unknown --{name.replace('_', '-')} value {value!r}; "
                    f"expected one of {values}"
                )
    if args.layers < 1:
        raise ValueError("--layers must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    return args


def require_runtime() -> tuple[Any, Any]:
    """Import torch and torch_npu, load the extension, and pin the device.

    The loader is the one the sibling layered tests use, not
    ``ensure_cam_async_ops_available``: that stronger check also requires the
    four ``afd_async_*`` communication operators to be registered, and this
    benchmark issues no collective at all. Requiring them would refuse to run on
    a build that has the two compute operators this file actually measures.
    """
    import os

    try:
        import torch
        import torch_npu
    except ImportError as error:
        raise SystemExit(
            f"this benchmark needs a working torch/torch_npu: {error}"
        ) from error

    from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

    ensure_afd_ascend_ops_loaded()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.config.allow_internal_format = True
    return torch, torch_npu


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args = validate_args(args)
        cases = build_cases(
            source_tokens=args.source_tokens,
            capacities=args.capacity,
            distributions=args.distribution,
            layer_modes=args.layer_mode,
            kernels=args.only,
            execution_modes=args.execution_mode,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.dry_run:
        print(json.dumps(describe_plan(cases, args), indent=2))
        return 0

    for warning in plan_warnings(cases):
        print(f"warning: {warning}", file=sys.stderr)

    torch, torch_npu = require_runtime()
    torch.manual_seed(args.seed)

    notes: list[str] = []
    if args.validate_baseline:
        # Runs before any full-size tensor is built: the check uses its own small
        # geometry and never touches the layer state, so a failing baseline should
        # not first cost the ~6 GiB of weights a measurement run needs. This is
        # also the reason it is cheap to run as a separate first step.
        notes += validate_baseline(torch, torch_npu)
        for note in notes:
            print(note, file=sys.stderr)

    state = build_weights(torch, torch_npu, args.layers)

    results = []
    for case in cases:
        print(f"measuring {case.describe()}", file=sys.stderr)
        result = measure_case(torch, torch_npu, state, case, args)
        results.append(result)
        layered = per_layer_median_ms(result, "layered")
        baseline = per_layer_median_ms(result, "baseline")
        print(
            f"  layered {layered:.4f} ms/layer, baseline {baseline:.4f} ms/layer",
            file=sys.stderr,
        )

    report = render_report(args, results, notes)
    print(report)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"archived to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
