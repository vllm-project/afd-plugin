# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare AFD layered A8W4 GMM with torch_npu's grouped matmul on an NPU.

Both MSD kernels may overwrite their int8 activation. A separate input slot is
used for each call in a timed block; all slots are restored before timing. The
reported latency is an NPU Event interval divided by calls per block. The CSV
P90 is the P90 of those block averages, not a single invocation's P90. Eager
intervals may contain Python submission gaps; graph replay largely removes
them. Neither result includes activation restoration or host data generation.
The two backends run in alternating order each round. Inputs, group counts,
weights, scales and biases are shared; only the activation pools are separate.
The pools use pool_slots * tokens * K bytes per backend, beyond graph/output
storage. Replaying a fixed shape and weights keeps caches warm; this measures
steady-state execution, not changing-layer or cold-cache behavior.
The CPU return interval starts immediately before the Python operator call (or
graph replay) and stops when it returns. The sync wall interval starts at the
same point and stops after recording and synchronizing the end Event. Both are
divided by calls per block, and exclude input restoration. These host intervals
must not be subtracted from Event time to infer pure kernel execution.

Example::

    python tools/benchmarks/gmm_layered_vs_builtin.py \
      --tokens 128,256 --k 2048 --n 4096 --experts 64 --layers 2 \
      --distributions uniform,skew --output /tmp/gmm_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

INT4_PER_INT32 = 8
NZ_FORMAT = 29
DEFAULT_SEED = 381
DEFAULT_ATOL = 0.02
DEFAULT_RTOL = 0.02
SCALE_MIN = 1e-4
SCALE_SPAN = 1e-3
SKEW_ACTIVE_DIVISOR = 2
GRAPH_WARMUP = 3


@dataclass(frozen=True)
class Case:
    tokens: int
    k: int
    n: int
    experts: int
    layers: int
    distribution: str


@dataclass
class Inputs:
    activation: torch.Tensor
    per_token_scale: torch.Tensor
    group_list: torch.Tensor
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    scales: list[torch.Tensor]


@dataclass(frozen=True)
class Timing:
    event_us: float
    cpu_return_us: float
    sync_wall_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", default="128,256", help="Comma-separated token counts"
    )
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--distributions", default="uniform,skew")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--pool-slots", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--output", type=Path, default=Path("gmm_layered_vs_builtin.csv")
    )
    return parser.parse_args()


def comma_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[str], list[str]]:
    tokens = [int(value) for value in comma_list(args.tokens)]
    distributions = comma_list(args.distributions)
    modes = comma_list(args.modes)
    if not tokens or any(value <= 0 for value in tokens):
        raise ValueError("--tokens must contain positive integers")
    if args.k <= 0 or args.n <= 0 or args.k % INT4_PER_INT32 or args.n % INT4_PER_INT32:
        raise ValueError("K and N must be positive multiples of 8")
    if args.layer_index is None:
        args.layer_index = min(1, args.layers - 1)
    if args.experts <= 0 or args.layers < 1 or not 0 <= args.layer_index < args.layers:
        raise ValueError("experts and layers must be positive; layer-index in range")
    if args.pool_slots <= 0 or args.warmup < 0 or args.rounds <= 0:
        raise ValueError("pool-slots and rounds must be positive; warmup nonnegative")
    if not distributions or set(distributions) - {"uniform", "skew"}:
        raise ValueError("distributions must be uniform and/or skew")
    if not modes or set(modes) - {"eager", "graph"}:
        raise ValueError("modes must be eager and/or graph")
    return tokens, distributions, modes


def expert_counts(tokens: int, experts: int, distribution: str, seed: int) -> list[int]:
    if distribution == "uniform":
        counts = [tokens // experts] * experts
        for expert in range(tokens % experts):
            counts[expert] += 1
    else:
        active = max(1, experts // SKEW_ACTIVE_DIVISOR)
        weights = [active - expert for expert in range(active)]
        counts = [0] * experts
        for token in range(tokens):
            bucket = token * sum(weights) // tokens
            running = 0
            for expert, weight in enumerate(weights):
                running += weight
                if bucket < running:
                    counts[expert] += 1
                    break
    random.Random(seed).shuffle(counts)
    return counts


def generate_inputs(case: Case, device: torch.device, seed: int) -> Inputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    activation = torch.randint(
        -64, 64, (case.tokens, case.k), generator=generator, dtype=torch.int8
    ).to(device)
    per_token_scale = (torch.rand(case.tokens, generator=generator) * 0.015 + 0.005).to(
        device
    )
    counts = expert_counts(case.tokens, case.experts, case.distribution, seed)
    group_list = torch.tensor(counts, dtype=torch.int64, device=device)

    weights = []
    biases = []
    scales = []
    for _ in range(case.layers):
        # Every int32 bit pattern is eight valid signed int4 values. This
        # avoids a much larger temporary tensor with unpacked nibbles.
        packed = torch.randint(
            -(1 << 31),
            1 << 31,
            (case.experts, case.k, case.n // INT4_PER_INT32),
            generator=generator,
            dtype=torch.int32,
        )
        weights.append(torch_npu.npu_format_cast(packed.to(device), NZ_FORMAT))
        float_scale = (
            torch.rand((case.experts, 1, case.n), generator=generator) * SCALE_SPAN
            + SCALE_MIN
        )
        bits = float_scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        scales.append(((bits << 32) | bits).to(device))
        biases.append(
            (torch.rand((case.experts, case.n), generator=generator) - 0.5).to(device)
        )
    return Inputs(activation, per_token_scale, group_list, weights, biases, scales)


def run_layered(
    inputs: Inputs, activation: torch.Tensor, layer_index: torch.Tensor
) -> torch.Tensor:
    return torch.ops.afd_ascend.grouped_matmul_layered(
        x=[activation],
        all_weight=inputs.weights,
        all_bias=inputs.biases,
        all_scale=inputs.scales,
        layer_index=layer_index,
        group_list=inputs.group_list,
        per_token_scale=inputs.per_token_scale,
        group_list_type=1,
        split_item=3,
        output_dtype=torch.bfloat16,
    )[0]


def run_builtin(inputs: Inputs, activation: torch.Tensor, layer: int) -> torch.Tensor:
    return torch_npu.npu_grouped_matmul(
        x=[activation],
        weight=[inputs.weights[layer]],
        bias=[inputs.biases[layer]],
        scale=[inputs.scales[layer]],
        per_token_scale=[inputs.per_token_scale],
        group_list=inputs.group_list,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]


def verify(
    inputs: Inputs,
    case: Case,
    selected_layer: int,
    device: torch.device,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    checked_layers = sorted({0, selected_layer, case.layers - 1})
    statistics_by_layer = {}
    for layer in checked_layers:
        layer_index = torch.tensor([layer], dtype=torch.int64, device=device)
        actual = run_layered(inputs, inputs.activation.clone(), layer_index)
        expected = run_builtin(inputs, inputs.activation.clone(), layer)
        torch.npu.synchronize()
        actual_float = actual.float()
        expected_float = expected.float()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(
                f"layer {layer}: shape/dtype mismatch {actual.shape}/{actual.dtype} "
                f"vs {expected.shape}/{expected.dtype}"
            )
        if not bool(torch.isfinite(actual_float).all()) or not bool(
            torch.isfinite(expected_float).all()
        ):
            raise AssertionError(f"layer {layer}: NaN/Inf in output")
        diff = (actual_float - expected_float).abs()
        max_abs = float(diff.max().item())
        max_rel = float((diff / expected_float.abs().clamp_min(atol)).max().item())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        statistics_by_layer[f"layer_{layer}_max_abs"] = max_abs
        statistics_by_layer[f"layer_{layer}_max_rel"] = max_rel
    print(
        f"  correctness: {json.dumps(statistics_by_layer, sort_keys=True)}",
        flush=True,
    )
    return statistics_by_layer


def restore_inputs(pool: list[torch.Tensor], original: torch.Tensor) -> None:
    for activation in pool:
        activation.copy_(original)
    torch.npu.synchronize()


def create_runner(
    name: str, inputs: Inputs, layer: int, layer_index: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "layered":
        return lambda activation: run_layered(inputs, activation, layer_index)
    return lambda activation: run_builtin(inputs, activation, layer)


def prepare_runner(
    name: str,
    mode: str,
    inputs: Inputs,
    layer: int,
    pool_slots: int,
    warmup: int,
    device: torch.device,
    atol: float,
    rtol: float,
) -> Callable[[], Timing]:
    layer_index = torch.tensor([layer], dtype=torch.int64, device=device)
    pool = [inputs.activation.clone() for _ in range(pool_slots)]
    run = create_runner(name, inputs, layer, layer_index)

    def call_all() -> list[torch.Tensor]:
        return [run(activation) for activation in pool]

    outputs: list[torch.Tensor] = []
    for _ in range(max(warmup, GRAPH_WARMUP if mode == "graph" else 0)):
        restore_inputs(pool, inputs.activation)
        outputs = call_all()
        torch.npu.synchronize()

    if mode == "graph":
        reference = run_builtin(inputs, inputs.activation.clone(), layer)
        torch.npu.synchronize()
        restore_inputs(pool, inputs.activation)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = call_all()
        torch.npu.synchronize()
        # The captured call wrote into pool. Keep graph outputs alive, then
        # restore every input before the first replay.
        replay = graph.replay
        restore_inputs(pool, inputs.activation)
        replay()
        torch.npu.synchronize()
        for slot, output in enumerate(outputs):
            if not bool(torch.isfinite(output.float()).all()):
                raise AssertionError(f"{name} graph slot {slot}: NaN/Inf in output")
            torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
    else:
        replay = call_all

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    def time_once() -> Timing:
        restore_inputs(pool, inputs.activation)
        start.record()
        cpu_call_start_ns = time.perf_counter_ns()
        result = replay()
        cpu_call_return_ns = time.perf_counter_ns()
        end.record()
        end.synchronize()
        sync_complete_ns = time.perf_counter_ns()
        if mode == "eager":
            # Retain eager outputs until their work has completed.
            outputs[:] = result
        return Timing(
            event_us=start.elapsed_time(end) * 1000 / pool_slots,
            cpu_return_us=(cpu_call_return_ns - cpu_call_start_ns) / 1000 / pool_slots,
            sync_wall_us=(sync_complete_ns - cpu_call_start_ns) / 1000 / pool_slots,
        )

    # This closure retains the graph, captured outputs, input pool and Events.
    return time_once


def percentile(values: list[float], percentile_value: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def cann_version() -> str:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    candidates = [Path(ascend_home) / "version.cfg"] if ascend_home else []
    candidates.append(Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"))
    candidates.extend(Path("/usr/local/Ascend").glob("cann-*/version.cfg"))
    for path in candidates:
        if path.is_file():
            return (
                path.read_text(encoding="utf-8", errors="replace")
                .strip()
                .replace("\n", "; ")
            )
    return "unknown"


def main() -> int:
    args = parse_args()
    tokens, distributions, modes = validate_args(args)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    ensure_afd_ascend_ops_loaded()
    device = torch.device(args.device)
    environment = {
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "cann_version": cann_version(),
        "python_version": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    for token_count in tokens:
        for distribution in distributions:
            case = Case(
                token_count, args.k, args.n, args.experts, args.layers, distribution
            )
            print(f"case={case}", flush=True)
            case_rows = []
            correctness_error = ""
            try:
                inputs = generate_inputs(case, device, args.seed)
                correctness = verify(
                    inputs, case, args.layer_index, device, args.atol, args.rtol
                )
            except Exception as exc:
                failures += len(modes)
                correctness_error = f"{type(exc).__name__}: {exc}"
                print(f"  correctness failed: {correctness_error}", flush=True)
                inputs = None
                correctness = {}
            for mode in modes:
                row = {
                    **environment,
                    "tokens": token_count,
                    "k": args.k,
                    "n": args.n,
                    "experts": args.experts,
                    "layers": args.layers,
                    "layer_index": args.layer_index,
                    "distribution": distribution,
                    "mode": mode,
                    "pool_slots": args.pool_slots,
                    "pool_input_bytes_per_backend": (
                        args.pool_slots * token_count * args.k
                    ),
                    "warmup": args.warmup,
                    "rounds": args.rounds,
                    "seed": args.seed,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "status": "pending",
                    "error": "",
                    "layered_median_us": "",
                    "layered_p90_us": "",
                    "builtin_median_us": "",
                    "builtin_p90_us": "",
                    "builtin_over_layered": "",
                    "layered_cpu_return_median_us": "",
                    "layered_cpu_return_p90_us": "",
                    "builtin_cpu_return_median_us": "",
                    "builtin_cpu_return_p90_us": "",
                    "layered_sync_wall_median_us": "",
                    "layered_sync_wall_p90_us": "",
                    "builtin_sync_wall_median_us": "",
                    "builtin_sync_wall_p90_us": "",
                    "correctness": json.dumps(correctness, sort_keys=True),
                    "layered_samples_us": "",
                    "builtin_samples_us": "",
                    "layered_cpu_return_samples_us": "",
                    "builtin_cpu_return_samples_us": "",
                    "layered_sync_wall_samples_us": "",
                    "builtin_sync_wall_samples_us": "",
                }
                if inputs is None:
                    row["status"] = "failed_correctness"
                    row["error"] = correctness_error
                else:
                    runners = {}
                    try:
                        runners = {
                            name: prepare_runner(
                                name,
                                mode,
                                inputs,
                                args.layer_index,
                                args.pool_slots,
                                args.warmup,
                                device,
                                args.atol,
                                args.rtol,
                            )
                            for name in ("layered", "builtin")
                        }
                        measurements: dict[str, list[Timing]] = {
                            "layered": [],
                            "builtin": [],
                        }
                        for round_index in range(args.rounds):
                            order = ("layered", "builtin")
                            if round_index % 2:
                                order = ("builtin", "layered")
                            for name in order:
                                measurements[name].append(runners[name]())
                        event_us = {
                            name: [sample.event_us for sample in measurements[name]]
                            for name in measurements
                        }
                        cpu_return_us = {
                            name: [
                                sample.cpu_return_us for sample in measurements[name]
                            ]
                            for name in measurements
                        }
                        sync_wall_us = {
                            name: [sample.sync_wall_us for sample in measurements[name]]
                            for name in measurements
                        }
                        layered_median = statistics.median(event_us["layered"])
                        builtin_median = statistics.median(event_us["builtin"])
                        row.update(
                            status="ok",
                            layered_median_us=layered_median,
                            layered_p90_us=percentile(event_us["layered"], 0.9),
                            builtin_median_us=builtin_median,
                            builtin_p90_us=percentile(event_us["builtin"], 0.9),
                            builtin_over_layered=builtin_median / layered_median,
                            layered_cpu_return_median_us=statistics.median(
                                cpu_return_us["layered"]
                            ),
                            layered_cpu_return_p90_us=percentile(
                                cpu_return_us["layered"], 0.9
                            ),
                            builtin_cpu_return_median_us=statistics.median(
                                cpu_return_us["builtin"]
                            ),
                            builtin_cpu_return_p90_us=percentile(
                                cpu_return_us["builtin"], 0.9
                            ),
                            layered_sync_wall_median_us=statistics.median(
                                sync_wall_us["layered"]
                            ),
                            layered_sync_wall_p90_us=percentile(
                                sync_wall_us["layered"], 0.9
                            ),
                            builtin_sync_wall_median_us=statistics.median(
                                sync_wall_us["builtin"]
                            ),
                            builtin_sync_wall_p90_us=percentile(
                                sync_wall_us["builtin"], 0.9
                            ),
                            layered_samples_us=json.dumps(event_us["layered"]),
                            builtin_samples_us=json.dumps(event_us["builtin"]),
                            layered_cpu_return_samples_us=json.dumps(
                                cpu_return_us["layered"]
                            ),
                            builtin_cpu_return_samples_us=json.dumps(
                                cpu_return_us["builtin"]
                            ),
                            layered_sync_wall_samples_us=json.dumps(
                                sync_wall_us["layered"]
                            ),
                            builtin_sync_wall_samples_us=json.dumps(
                                sync_wall_us["builtin"]
                            ),
                        )
                        print(
                            f"  {mode}: layered {layered_median:.2f} us, "
                            f"builtin {builtin_median:.2f} us, "
                            f"builtin/layered {row['builtin_over_layered']:.3f}x",
                            flush=True,
                        )
                        print(
                            "    CPU return median layered/builtin "
                            f"{row['layered_cpu_return_median_us']:.2f}/"
                            f"{row['builtin_cpu_return_median_us']:.2f} us; "
                            "sync wall median "
                            f"{row['layered_sync_wall_median_us']:.2f}/"
                            f"{row['builtin_sync_wall_median_us']:.2f} us",
                            flush=True,
                        )
                    except Exception as exc:
                        failures += 1
                        row["status"] = "failed_benchmark"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"  {mode} failed: {row['error']}", flush=True)
                    finally:
                        runners.clear()
                case_rows.append(row)
            rows.extend(case_rows)
            with args.output.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            del inputs
    print(f"wrote {args.output.resolve()}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
