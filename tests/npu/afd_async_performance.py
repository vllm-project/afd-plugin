# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: HCCL benchmarks for the four plugin-owned routed async CAM operators.
# Create: 2026-09-22
# Note: The timing harness checks precision before collecting device measurements.
# History: 2026-09-22 create
"""HCCL benchmarks for the four plugin-owned routed async CAM operators.

Run with torchrun on Ascend 910C or 950. --help and --dry-run need only Python.
See README.md beside this file for timing boundaries and launch examples.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from .afd_async_common import (
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
    )
else:
    from afd_async_common import (
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
    )

if TYPE_CHECKING:
    from torch.distributed import Store

MILLISECONDS_PER_SECOND = 1000.0
PERCENTILE_95 = 0.95
MAX_LAYER_INDEX = (1 << 32) - 1
OP_NAMES = ("dispatch_send", "dispatch_recv", "combine_send", "combine_recv")


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize milliseconds using the nearest-rank p95 definition."""
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(PERCENTILE_95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


class FocusSchedule:
    """CPU-store readiness protocol; never uses a collective on CAM windows.

    A readiness flag means the host is about to launch an op, not that its
    device kernel has started. Delays are experimental controls, not proof
    that peer waiting was removed; use a separate profiler run to verify it.
    """

    def __init__(
        self,
        store: Store,
        mode: str,
        rank: int,
        attn_ranks: int,
        moe_ranks: int,
        sender_stagger_ms: float,
        receiver_delay_ms: float,
        receiver_guard_ms: float,
    ) -> None:
        self.store = store
        self.mode = mode
        self.rank = rank
        self.attn_ranks = attn_ranks
        self.moe_ranks = moe_ranks
        self.sender_stagger_ms = sender_stagger_ms
        self.receiver_delay_ms = receiver_delay_ms
        self.receiver_guard_ms = receiver_guard_ms
        self.started: set[str] = set()

    def __call__(self, op_name: str) -> None:
        # Only the first call of each phase is deliberately delayed. Later
        # chunks must drain freely so a delayed sender cannot fill its window.
        if self.mode == "cycle" or op_name in self.started:
            return
        self.started.add(op_name)
        is_dispatch = op_name.startswith("dispatch_")
        is_send = op_name.endswith("_send")
        if is_dispatch:
            senders = range(self.attn_ranks)
            receivers = range(self.attn_ranks, self.attn_ranks + self.moe_ranks)
            phase = "dispatch"
            sender_index = self.rank
        else:
            senders = range(self.attn_ranks, self.attn_ranks + self.moe_ranks)
            receivers = range(self.attn_ranks)
            phase = "combine"
            sender_index = self.rank - self.attn_ranks
        if self.mode == "send":
            if is_send:
                self.store.wait([f"{phase}/recv/{rank}" for rank in receivers])
                delay_ms = (
                    self.receiver_guard_ms + sender_index * self.sender_stagger_ms
                )
                time.sleep(delay_ms / MILLISECONDS_PER_SECOND)
            else:
                self.store.set(f"{phase}/recv/{self.rank}", "ready")
        elif is_send:
            time.sleep(sender_index * self.sender_stagger_ms / MILLISECONDS_PER_SECOND)
            # Publish BEFORE launching. A blocking native send must not stop
            # the receiver from learning it should post its matching receive.
            self.store.set(f"{phase}/send/{self.rank}", "ready")
        else:
            self.store.wait([f"{phase}/send/{rank}" for rank in senders])
            time.sleep(self.receiver_delay_ms / MILLISECONDS_PER_SECOND)


def selected_ops(mode: str) -> tuple[str, ...]:
    if mode == "send":
        return ("dispatch_send", "combine_send")
    if mode == "recv":
        return ("dispatch_recv", "combine_recv")
    return OP_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--measure-mode", choices=("cycle", "send", "recv", "both"), default="both"
    )
    parser.add_argument("--sender-stagger-ms", type=float, default=50.0)
    parser.add_argument("--receiver-delay-ms", type=float, default=200.0)
    parser.add_argument("--receiver-guard-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, help="Rank-zero JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        if args.warmup < 0 or args.iterations < 1:
            raise ValueError("--warmup must be nonnegative and --iterations positive")
        for name in ("sender_stagger_ms", "receiver_delay_ms", "receiver_guard_ms"):
            value = vars(args)[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
        modes = (
            ("send", "recv") if args.measure_mode == "both" else (args.measure_mode,)
        )
        # The native tiling structure stores layer IDs as uint32.
        if len(modes) * (args.warmup + args.iterations) > MAX_LAYER_INDEX:
            raise ValueError("too many rounds for uint32 layer IDs")
    except ValueError as error:
        parser.error(str(error))

    plan = {
        "configuration": describe_config(config),
        "modes": modes,
        "warmup_per_mode": args.warmup,
        "iterations_per_mode": args.iterations,
        "sender_stagger_ms": args.sender_stagger_ms,
        "receiver_delay_ms": args.receiver_delay_ms,
        "receiver_guard_ms": args.receiver_guard_ms,
        "timing_contract": {
            "host_ms": "synchronized host latency of one native call, including waits",
            "event_ms": "NPU stream event interval of one native call, including waits",
            "lifecycle_ms": "round including control, metadata, transform and delays",
            "chunk_samples": "each receive/send chunk is a separate sample",
            "focus": "only selected send or receive ops enter focus summaries",
        },
        "validation": "an untimed precision round precedes any performance warmup",
    }
    if args.dry_run:
        print(json.dumps({"status": "dry-run; no device execution", **plan}, indent=2))
        return 0

    with NativeHarness(config) as harness:
        # Dependencies are loaded only inside the actual hardware execution path.
        # The rendezvous store is CPU based even though the data group is HCCL.
        store = harness.dist.distributed_c10d._get_default_store()
        precision = harness.run_round(0, validate=True, timed=False)
        failures = [
            failure
            for report in harness.gather_reports(precision)
            for failure in report["failures"]
        ]
        if failures:
            raise RuntimeError(f"precision preflight failed: {failures}")

        local_modes = []
        layer_id = 1
        for mode in modes:
            samples = []
            for iteration in range(args.warmup + args.iterations):
                round_store = harness.dist.PrefixStore(
                    f"afd-benchmark/{layer_id}", store
                )
                schedule = FocusSchedule(
                    round_store,
                    mode,
                    harness.rank,
                    config.attn_ranks,
                    config.moe_ranks,
                    args.sender_stagger_ms,
                    args.receiver_delay_ms,
                    args.receiver_guard_ms,
                )
                report = harness.run_round(
                    layer_id,
                    validate=False,
                    timed=True,
                    timed_ops=selected_ops(mode),
                    before_op=schedule,
                )
                layer_id += 1
                if report["failures"]:
                    raise RuntimeError(f"benchmark round failed: {report['failures']}")
                if iteration >= args.warmup:
                    samples.append(report)
            local_modes.append(
                {
                    "mode": mode,
                    "rank": harness.rank,
                    "selected_ops": selected_ops(mode),
                    "summary": {
                        timer: {
                            op: summarize(
                                [
                                    value
                                    for sample in samples
                                    for value in sample[timer][op]
                                ]
                            )
                            for op in selected_ops(mode)
                        }
                        for timer in ("host_ms", "event_ms")
                    },
                    "lifecycle": summarize(
                        [sample["lifecycle_ms"] for sample in samples]
                    ),
                    "samples": samples,
                }
            )
        gathered = harness.gather_reports(
            {
                "rank": harness.rank,
                "environment": harness.environment(),
                "modes": local_modes,
            }
        )
        if harness.rank == 0:
            result = {
                "status": "passed",
                **plan,
                "ranks": gathered,
                # Maxima are computed per iteration, not by pooling unrelated
                # per-rank percentiles or pretending chunk counts are uniform.
                "max_rank_lifecycle": {
                    mode: summarize(
                        [
                            max(
                                rank["modes"][mode_index]["samples"][iteration][
                                    "lifecycle_ms"
                                ]
                                for rank in gathered
                            )
                            for iteration in range(args.iterations)
                        ]
                    )
                    for mode_index, mode in enumerate(modes)
                },
            }
    # Publish success only after the owned process groups have been destroyed.
    if harness.rank == 0:
        result["cleanup_complete"] = True
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
