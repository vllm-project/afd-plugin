#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect small native tables and kernel type totals without reading traces."""

import argparse
import csv
import io
import json
import math
import tarfile
from pathlib import Path


SMALL_TABLES = frozenset(
    {
        "op_statistic.csv",
        "api_statistic.csv",
        "step_trace_time.csv",
        "communication.json",
        "communication_matrix.json",
    }
)
RUN_FILES = frozenset({"runtime.json", "profile.env", "profile-startup-check.txt"})
CSV_FIELD_LIMIT = 16 * 1024 * 1024
MICROSECONDS_PER_MILLISECOND = 1000


def summarize_types(path: Path) -> dict:
    groups: dict[str, tuple[int, float, float, float]] = {}
    rows = invalid = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        duration_column = next(
            (
                name
                for name in (
                    "Duration(us)", "Duration(\u03bcs)", "Duration(\u00b5s)"
                )
                if name in headers
            ),
            None,
        )
        if "Type" not in headers or duration_column is None:
            raise ValueError(
                f"Unsupported kernel table columns/units: {path}: {headers}"
            )
        for row in reader:
            rows += 1
            try:
                duration = float(row[duration_column].replace(",", ""))
            except (ValueError, AttributeError):
                invalid += 1
                continue
            if not math.isfinite(duration) or duration < 0:
                invalid += 1
                continue
            name = row["Type"] or "<empty>"
            count, total, minimum, maximum = groups.get(
                name, (0, 0.0, duration, duration)
            )
            groups[name] = (
                count + 1,
                total + duration,
                min(minimum, duration),
                max(maximum, duration),
            )
    return {
        "rows": rows,
        "invalid_duration_rows": invalid,
        "sum_kernel_ms": (
            sum(value[1] for value in groups.values()) / MICROSECONDS_PER_MILLISECOND
        ),
        "types": [
            {
                "type": name,
                "calls": value[0],
                "sum_ms": value[1] / MICROSECONDS_PER_MILLISECOND,
                "mean_us": value[1] / value[0],
                "min_us": value[2],
                "max_us": value[3],
            }
            for name, value in sorted(
                groups.items(), key=lambda item: item[1][1], reverse=True
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .tar.gz file")
    args = parser.parse_args()
    root = args.run_root.resolve()
    profile = root / "profile"
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(profile):
        parser.error("Output must be a new file outside profile/.")
    kernels = sorted(profile.rglob("kernel_details.csv"))
    if not kernels:
        parser.error("No exported kernel_details.csv under RUN_ROOT/profile/.")
    csv.field_size_limit(CSV_FIELD_LIMIT)
    processes = []
    sources = {}
    for index, kernel in enumerate(kernels, 1):
        relative = kernel.relative_to(root).as_posix()
        print(f"Reading CSV {index}/{len(kernels)}: {relative}", flush=True)
        processes.append({"source": relative, **summarize_types(kernel)})
        for path in kernel.parent.iterdir():
            if path.is_file() and path.name in SMALL_TABLES:
                sources[path.relative_to(root).as_posix()] = path
        for path in kernel.parent.parent.iterdir():
            if path.is_file() and (
                path.name == "profiler_metadata.json"
                or path.name.startswith("profiler_info_") and path.suffix == ".json"
            ):
                sources[path.relative_to(root).as_posix()] = path
    for path in root.iterdir():
        if path.is_file() and (
            path.name in RUN_FILES or ".wallclock-" in path.name
        ):
            sources[path.relative_to(root).as_posix()] = path
    result = {
        "run_root": str(root),
        "profile_process_exports": len(processes),
        "runtime_present": (root / "runtime.json").is_file(),
        "processes": processes,
        "native_files": [
            {"path": name, "bytes": path.stat().st_size}
            for name, path in sorted(sources.items())
        ],
        "limitations": [
            "All kernel types retained; grouping by Type merges distinct names/shapes.",
            "Whole capture totals include overlaps/waits "
            "and are not critical path time.",
            "No full trace, kernel CSV or operator CSV is copied into this package.",
            "Missing communication exports are preserved as missing, "
            "never treated as zero.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with tarfile.open(output, "x:gz") as archive:
        entry = tarfile.TarInfo("by_type.json")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
        for name, path in sorted(sources.items()):
            archive.add(path, arcname="source/" + name)
    print(f"Archive: {output} ({output.stat().st_size} bytes)")
    if not result["runtime_present"]:
        print("NOTE: runtime.json was not found for this run.")


if __name__ == "__main__":
    main()
