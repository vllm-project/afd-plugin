# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Check all four routed-only native operators against an independent CPU
#              oracle.
# Create: 2026-09-22
# Note: The CPU oracle checks the routed output after the four communication phases.
# History: 2026-09-22 create
"""Check all four routed-only native operators against an independent CPU oracle.

Example (four 910C devices, two Attention ranks and two MoE ranks)::

    torchrun --standalone --nproc-per-node=4 tests/npu/afd_async_precision.py \
        --dtype bf16 --dynamic-quant 1 --capacity 8 --iterations 3

Use a fresh torchrun for each dtype, quantization and routing configuration.
``--dry-run`` works with only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from .afd_async_common import (
        MAX_LAYER_INDEX,
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
    )
else:
    from afd_async_common import (
        MAX_LAYER_INDEX,
        NativeHarness,
        add_common_arguments,
        config_from_args,
        describe_config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Repeated reuse of the same HCCL windows",
    )
    parser.add_argument("--output", type=Path, help="Optional rank-zero JSON report")
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if not 1 <= args.iterations <= MAX_LAYER_INDEX + 1:
            raise ValueError("iterations must be positive and fit uint32 layer IDs")
    except ValueError as error:
        parser.error(str(error))
    report = describe_config(config)
    if args.dry_run:
        report["iterations"] = args.iterations
        print(json.dumps(report, indent=2))
        return 0
    try:
        with NativeHarness(config) as harness:
            report["environment"] = harness.gather_reports(harness.environment())
            rounds = []
            for iteration in range(args.iterations):
                local = harness.run_round(iteration, validate=True)
                gathered = harness.gather_reports(local)
                rounds.append(gathered)
                failures = sum(len(rank["failures"]) for rank in gathered)
                if harness.rank == 0:
                    print(
                        f"round {iteration}: {'FAIL' if failures else 'PASS'} "
                        f"({failures} mismatches)",
                        flush=True,
                    )
            passed = not any(rank["failures"] for ranks in rounds for rank in ranks)
            report.update(status="passed" if passed else "failed", rounds=rounds)
        report["cleanup_complete"] = True
        if harness.rank == 0:
            text = json.dumps(report, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0 if passed else 1
    except Exception as error:
        print(
            f"Native precision execution failed: {error}", file=sys.stderr, flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
