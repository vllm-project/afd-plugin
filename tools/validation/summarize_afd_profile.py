#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize exported AFD profiler tables without loading trace_view.json."""

import argparse
import csv
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path


DEFAULT_TOP_OPERATORS = 30
CSV_FIELD_LIMIT = 16 * 1024 * 1024
MICROSECONDS_PER_MILLISECOND = 1000
SOURCE_TABLE_NAMES = frozenset(
    {
        "kernel_details.csv",
        "operator_details.csv",
        "op_statistic.csv",
        "step_trace_time.csv",
        "api_statistic.csv",
        "communication.json",
        "communication_matrix.json",
    }
)
RUN_FILE_NAMES = frozenset(
    {"runtime.json", "profile.env", "profile-startup-check.txt", "profile-files.txt"}
)


def normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("μ", "u").replace("µ", "u"))


def number(value: str) -> float | None:
    try:
        result = float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None
    return result if math.isfinite(result) else None


def summarize_kernel(path: Path, top_count: int) -> dict:
    totals = Counter()
    counts = Counter()
    row_count = invalid_durations = invalid_timestamps = 0
    minimum_start = maximum_end = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = {normalized_header(key): key for key in reader.fieldnames or []}
        name_column = headers.get("name") or headers.get("opname")
        type_column = headers.get("type") or headers.get("optype")
        duration_column = headers.get("durationus")
        start_column = headers.get("starttimeus")
        if name_column is None or duration_column is None:
            return {
                "status": "unsupported_columns",
                "headers": reader.fieldnames,
                "note": "Need Name and Duration(us); units are never guessed.",
            }
        for row in reader:
            row_count += 1
            duration = number(row.get(duration_column, ""))
            if duration is None or duration < 0:
                invalid_durations += 1
                continue
            key = (
                row.get(name_column) or "<empty>",
                (row.get(type_column) or "") if type_column else "",
            )
            counts[key] += 1
            totals[key] += duration
            start = number(row.get(start_column, "")) if start_column else None
            if start is not None:
                minimum_start = (
                    min(minimum_start, start) if minimum_start is not None else start
                )
                end = start + duration
                maximum_end = max(maximum_end, end) if maximum_end is not None else end
            else:
                invalid_timestamps += 1
    return {
        "status": "ok",
        "rows": row_count,
        "valid_duration_rows": sum(counts.values()),
        "invalid_duration_rows": invalid_durations,
        "missing_start_rows": invalid_timestamps,
        "headers": reader.fieldnames,
        "sum_kernel_ms": sum(totals.values()) / MICROSECONDS_PER_MILLISECOND,
        "min_start_us": minimum_start,
        "max_end_us": maximum_end,
        "event_envelope_ms": (
            (maximum_end - minimum_start) / MICROSECONDS_PER_MILLISECOND
            if minimum_start is not None
            else None
        ),
        "top_operators": [
            {
                "name": key[0],
                "type": key[1],
                "calls": counts[key],
                "sum_ms": duration / MICROSECONDS_PER_MILLISECOND,
                "mean_us": duration / counts[key],
            }
            for key, duration in totals.most_common(top_count)
        ],
        "limitations": (
            "Whole captured window; overlapping task durations are summed. "
            "Not device busy time, step latency, bandwidth, or causal attribution. "
            "Timestamps retain CSV units/origin; cross-rank clocks are not aligned."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, required=True,
        help="One run's node directory containing profile/.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New directory for summaries and the transfer archive.",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_OPERATORS)
    parser.add_argument(
        "--include-tables", action="store_true",
        help="Also archive CSV/communication exports, never the full trace.",
    )
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    root = args.run_root.resolve()
    profile = root / "profile"
    if not profile.is_dir():
        parser.error(f"Missing profile directory: {profile}")
    if args.output.resolve() == root or args.output.resolve().is_relative_to(profile):
        parser.error("--output must be separate from the source profile directory")
    kernels = sorted(profile.rglob("kernel_details.csv"))
    if not kernels:
        parser.error(
            "No kernel_details.csv found; parse the raw profile "
            "with matching torch-npu first."
        )
    csv.field_size_limit(CSV_FIELD_LIMIT)
    # Distinct source paths retain role, worker/PID, and rank metadata without
    # assuming directory order or profiler rank is a physical NPU identifier.
    summaries = []
    files = []
    candidates = {}
    for path in sorted(profile.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        is_metadata = (
            path.name.startswith("profiler_info") and path.suffix == ".json"
            or path.name == "profiler_metadata.json"
        )
        selected = is_metadata or (
            args.include_tables and path.name in SOURCE_TABLE_NAMES
        )
        if selected:
            candidates[relative] = path
        if (
            is_metadata or path.name in SOURCE_TABLE_NAMES
            or path.name == "trace_view.json"
        ):
            files.append({
                "path": relative,
                "bytes": path.stat().st_size,
                "included_in_archive": selected,
            })
    for path in sorted(root.iterdir()):
        if path.is_file() and (
            path.name in RUN_FILE_NAMES or ".wallclock-" in path.name
        ):
            candidates[path.relative_to(root).as_posix()] = path
    for index, path in enumerate(kernels, 1):
        print(
            f"Summarizing {index}/{len(kernels)}: {path.relative_to(root)}", flush=True
        )
        summaries.append({
            "source": path.relative_to(root).as_posix(),
            **summarize_kernel(path, args.top),
        })
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.output / "summary.json"
    summary = {
        "run_root": str(root),
        "profile_process_exports": len(summaries),
        "include_tables": args.include_tables,
        "processes": summaries,
        "notes": [
            "Expected A4/F2: verify four Attention and two FFN processes "
            "using metadata.",
            "No trace_view.json file was read, modified, or archived.",
            "Top operators exclude other names; use native tables for a full review.",
            "Match active time windows/step counts before comparing rank totals.",
            "Summaries identify suspects; they cannot prove causal/critical paths.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    inventory_path = args.output / "files.csv"
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "bytes", "included_in_archive")
        )
        writer.writeheader()
        writer.writerows(files)
    archive = args.output / "profile-summary.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(summary_path, arcname="summary.json")
        bundle.add(inventory_path, arcname="files.csv")
        for relative, path in sorted(candidates.items()):
            bundle.add(path, arcname="source/" + relative)
    print(f"Archive: {archive} ({archive.stat().st_size} bytes)")
    if any(item["status"] != "ok" for item in summaries):
        print(
            "Some CSV schemas were not recognized; "
            "source column names are in summary.json."
        )


if __name__ == "__main__":
    main()
