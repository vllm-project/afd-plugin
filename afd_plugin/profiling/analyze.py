# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validate saved CAM captures and export traces and statistics offline.

The CLI combines manifests, worker JSONL and native_trace.py artifacts through
pipeline.py, retaining older Event/fragment reconstruction. Missing clocks,
records and unresolved links remain explicit quality issues.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from afd_plugin.profiling.native_trace import read_native_trace
from afd_plugin.profiling.pipeline import export_pipeline, reconstruct_pipeline

SCHEMA_VERSION = 1
NS_PER_US = 1000
ID_FIELDS = {
    "FRAGMENT": "fragment_id",
    "SOURCE_CALL": "source_call_id",
    "FFN_WORK": "ffn_work_id",
    "SPAN": "span_id",
}


def _text(record: dict, field: str) -> None:
    if not isinstance(record.get(field), str) or not record[field]:
        raise ValueError(f"{field} must be a nonempty string")


def _integer(record: dict, field: str, *, nullable: bool = False) -> None:
    if nullable and field in record and record[field] is None:
        return
    if type(record.get(field)) is not int or record[field] < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _references(record: dict, field: str) -> None:
    values = record.get(field)
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ValueError(f"{field} must be a list of nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate references")


def _validate_json(value: object, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("records must contain finite JSON values")
        return
    if not isinstance(value, (dict, list, tuple)) or id(value) in ancestors:
        raise ValueError("records must contain finite JSON values")
    ancestors.add(id(value))
    children: Iterable[object]
    if isinstance(value, dict):
        for key in value:
            if key is not None and not isinstance(key, (str, int, float)):
                raise ValueError("records must contain finite JSON values")
            _validate_json(key, ancestors)
        children = value.values()
    else:
        children = value
    for child in children:
        _validate_json(child, ancestors)
    ancestors.remove(id(value))


def _validate(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    _validate_json(record, set())
    _text(record, "record_type")
    kind = record["record_type"]
    if "session_id" in record:
        _text(record, "session_id")
    if kind == "SESSION":
        _text(record, "session_id")
        _integer(record, "schema_version")
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        _integer(record, "start_host_ns")
        if "worker_id" in record:
            _text(record, "worker_id")
        if record.get("native_trace_dir") is not None:
            _text(record, "native_trace_dir")
            _integer(record, "generation")
            _integer(record, "clock_uncertainty_ns")
            if type(record.get("clock_offset_ns")) is not int:
                raise ValueError("clock_offset_ns must be an integer")
        return
    if kind == "CLOCK_MAP":
        _text(record, "clock_domain")
        for field in (
            "device_origin_ns",
            "host_origin_ns",
            "uncertainty_ns",
            "valid_device_begin_ns",
            "valid_device_end_ns",
        ):
            _integer(record, field)
        if record["valid_device_end_ns"] < record["valid_device_begin_ns"]:
            raise ValueError("CLOCK_MAP has a reversed validity interval")
        return
    if kind == "SUMMARY":
        _text(record, "worker_id")
        for field in (
            "dropped_spans",
            "dropped_metadata",
            "unresolved_spans",
            "dropped_records",
        ):
            _integer(record, field)
        if "native_spans" in record:
            _integer(record, "native_spans")
        if "native_marks" in record:
            _integer(record, "native_marks")
        return
    if kind not in ID_FIELDS:
        raise ValueError(f"unsupported record_type: {kind}")
    _text(record, ID_FIELDS[kind])
    if kind in {"FRAGMENT", "SOURCE_CALL"}:
        _text(record, "group_id")
        _integer(record, "stage_idx")
    if kind == "FRAGMENT":
        _text(record, "request_id")
        _text(record, "mode")
        _integer(record, "token_start")
        _integer(record, "token_end")
        if record["token_end"] < record["token_start"]:
            raise ValueError("FRAGMENT has a reversed token interval")
    elif kind == "SOURCE_CALL":
        _integer(record, "layer_idx")
        _references(record, "fragment_ids")
        if "worker_id" in record:
            _text(record, "worker_id")
    elif kind == "FFN_WORK":
        _integer(record, "layer_idx")
        _references(record, "source_call_ids")
        if record.get("source_relation") == "temporal_source_candidate":
            _references(record, "source_workers")
            if not record["source_workers"]:
                raise ValueError("source_workers must not be empty")
    else:
        for field in ("worker_id", "exec_ref", "kind", "clock_domain", "producer_id"):
            _text(record, field)
        stream = record.get("stream_id")
        if not (isinstance(stream, str) and stream) and not (
            type(stream) is int and stream >= 0
        ):
            raise ValueError("stream_id must be a nonempty string or nonnegative int")
        if type(record.get("complete")) is not bool:
            raise ValueError("complete must be a boolean")
        _integer(record, "generation")
        if record.get("time_source") == "HOST":
            if type(record.get("host_epoch_offset_ns")) is not int:
                raise ValueError("host_epoch_offset_ns must be an integer")
            _integer(record, "clock_uncertainty_ns")
        _integer(record, "host_begin_ns")
        _integer(record, "host_end_ns", nullable=not record["complete"])
        for field in ("device_begin_ns", "device_end_ns"):
            _integer(record, field, nullable=True)
        for clock in ("host", "device"):
            begin, end = record[f"{clock}_begin_ns"], record[f"{clock}_end_ns"]
            if end is not None and (begin is None or end < begin):
                raise ValueError(f"SPAN has an invalid {clock} interval")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("records must contain finite JSON values")
    return number


def read_jsonl(
    paths: Iterable[str | Path], *, analyse_native: bool = False
) -> list[dict]:
    """Parse JSONL with source locations; reconstruct owns record schema validation."""
    records = []
    for path in paths:
        header = None
        with Path(path).open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line, parse_float=_finite_float, parse_constant=_finite_float
                    )
                    if header is None and (
                        not isinstance(record, dict)
                        or record.get("record_type") != "SESSION"
                    ):
                        raise ValueError("first record must be SESSION")
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                records.append(record)
                if header is None:
                    header = record
        if header is not None and header.get("native_trace_dir"):
            _validate(header)
            if header.get("identity", {}).get("ffn_execution_mode") != "full" and any(
                r.get("record_type") == "SUMMARY"
                and r.get("worker_id") == header["worker_id"]
                and r.get("native_marks", r.get("native_spans")) == 0
                for r in records
            ):
                continue
            directory = Path(path).parent / header["native_trace_dir"]
            if not directory.resolve().is_relative_to(Path(path).parent.resolve()):
                raise ValueError(f"{path}: native_trace_dir escapes session directory")
            records.extend(
                read_native_trace(directory, header, analyse_native=analyse_native)
            )
    return records


def _timing(span: dict, maps: Iterable[dict], start_host_ns: int) -> dict:
    begin, end = span["device_begin_ns"], span["device_end_ns"]
    timing = {
        "duration_ns": None,
        "relative_begin_us": None,
        "relative_end_us": None,
        "uncertainty_ns": None,
        "aligned": False,
    }
    if span.get("time_source") == "HOST" and span["host_end_ns"] is not None:
        begin, end = span["host_begin_ns"], span["host_end_ns"]
        offset = span["host_epoch_offset_ns"]
        timing.update(
            duration_ns=end - begin,
            relative_begin_us=(begin + offset - start_host_ns) / NS_PER_US,
            relative_end_us=(end + offset - start_host_ns) / NS_PER_US,
            uncertainty_ns=span["clock_uncertainty_ns"],
            aligned=True,
        )
        return timing
    if begin is None or end is None:
        return timing
    timing["duration_ns"] = end - begin
    for mapping in maps:
        if (
            mapping["valid_device_begin_ns"]
            <= begin
            <= end
            <= mapping["valid_device_end_ns"]
        ):
            offset = mapping["host_origin_ns"] - mapping["device_origin_ns"]
            timing.update(
                relative_begin_us=(begin + offset - start_host_ns) / NS_PER_US,
                relative_end_us=(end + offset - start_host_ns) / NS_PER_US,
                uncertainty_ns=mapping["uncertainty_ns"],
                aligned=True,
            )
            break
    return timing


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    total = 0
    previous_end = 0
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, previous_end))
        previous_end = max(previous_end, end)
    return total


def _correlate_work(
    works: dict[str, dict],
    calls: dict[str, dict],
    spans: dict[str, dict],
    execution_spans: dict[str, list[str]],
    errors: list[str],
) -> None:
    """Use rank, layer and causal time bounds; never pair calls by list order."""
    # shortcut: temporal candidates; add a side channel if exact IDs are required.
    windows = defaultdict(list)
    for call_id, call in calls.items():
        if "worker_id" not in call:
            continue
        bounds = {
            spans[span_id].get("stage", spans[span_id]["kind"]): spans[span_id][
                "timing"
            ]
            for span_id in execution_spans[call_id]
            if spans[span_id]["complete"] and spans[span_id]["timing"]["aligned"]
        }
        if "DISPATCH_SEND" not in bounds or "COMBINE_RECV" not in bounds:
            continue
        begin, end = bounds["DISPATCH_SEND"], bounds["COMBINE_RECV"]
        windows[(call["worker_id"], call["layer_idx"])].append(
            (
                begin["relative_begin_us"] - begin["uncertainty_ns"] / NS_PER_US,
                end["relative_end_us"] + end["uncertainty_ns"] / NS_PER_US,
                call_id,
            )
        )
    for work_id, work in works.items():
        if work.get("source_relation") != "temporal_source_candidate":
            continue
        work["source_call_ids"] = []
        receives = [
            spans[span_id]["timing"]
            for span_id in execution_spans[work_id]
            if spans[span_id].get("stage", spans[span_id]["kind"]) == "DISPATCH_RECV"
            and spans[span_id]["complete"]
            and spans[span_id]["timing"]["aligned"]
        ]
        if len(receives) != 1:
            errors.append(
                f"FFN_WORK {work_id}: missing aligned receive for correlation"
            )
            continue
        receive = receives[0]
        completed = receive["relative_end_us"]
        uncertainty = receive["uncertainty_ns"] / NS_PER_US
        for worker in work["source_workers"]:
            candidates = sorted(
                call_id
                for begin, end, call_id in windows[(worker, work["layer_idx"])]
                if begin <= completed + uncertainty and end >= completed - uncertainty
            )
            work["source_call_ids"].extend(candidates)
            if len(candidates) != 1:
                errors.append(
                    f"FFN_WORK {work_id}: {len(candidates)} source candidates "
                    f"for {worker}"
                )


def reconstruct(records: Iterable[dict]) -> dict:
    records = list(records)
    if any(
        r.get("record_type") == "SESSION" and r.get("view") == "pipeline"
        for r in records
    ):
        for record in records:
            if record.get("record_type") in {"SESSION", "SUMMARY"}:
                _validate(record)
        return reconstruct_pipeline(records)
    tables: dict[str, dict[str, dict]] = {kind: {} for kind in ID_FIELDS}
    headers = {}
    summaries = {}
    maps = defaultdict(list)
    session = None
    errors = []
    for record in records:
        _validate(record)
        kind = record["record_type"]
        if kind == "SESSION":
            identity = record.get("worker_id")
            if identity in headers:
                raise ValueError(f"duplicate SESSION for worker {identity!r}")
            headers[identity] = dict(record)
            common = {
                key: record[key]
                for key in (
                    "schema_version",
                    "session_id",
                    "start_host_ns",
                )
            }
            if session is not None and common != session:
                raise ValueError("SESSION headers disagree")
            session = common
        elif kind == "SUMMARY":
            worker = record["worker_id"]
            if worker in summaries:
                raise ValueError(f"duplicate SUMMARY: {worker}")
            summaries[worker] = dict(record)
        elif kind == "CLOCK_MAP":
            maps[record["clock_domain"]].append(dict(record))
        else:
            identifier = record[ID_FIELDS[kind]]
            if identifier in tables[kind]:
                raise ValueError(f"duplicate {kind} id: {identifier}")
            tables[kind][identifier] = dict(record)
    if session is None:
        raise ValueError("SESSION record is required")
    for worker, header in headers.items():
        summary = summaries.get(worker)
        if summary is None:
            if header.get("requires_summary"):
                errors.append(f"worker {worker}: missing SUMMARY")
            continue
        if any(
            summary[key]
            for key in (
                "dropped_spans",
                "dropped_metadata",
                "unresolved_spans",
                "dropped_records",
            )
        ):
            errors.append(f"worker {worker}: dropped or unresolved records")
        if summary.get("error"):
            errors.append(f"worker {worker}: {summary['error']}")
        if "native_spans" in summary:
            observed = sum(
                span["worker_id"] == worker for span in tables["SPAN"].values()
            )
            if observed != summary["native_spans"]:
                errors.append(
                    f"worker {worker}: expected {summary['native_spans']} native "
                    f"ranges, found {observed}"
                )
    for record in records:
        if record.get("session_id", session["session_id"]) != session["session_id"]:
            raise ValueError("records belong to different sessions")
    for domain, mappings in maps.items():
        mappings.sort(key=lambda item: item["valid_device_begin_ns"])
        for previous, current in zip(mappings, mappings[1:], strict=False):
            if previous["valid_device_end_ns"] > current["valid_device_begin_ns"]:
                raise ValueError(f"overlapping CLOCK_MAP intervals: {domain}")

    fragments, calls, works, spans = (tables[kind] for kind in ID_FIELDS)
    if calls.keys() & works.keys():
        raise ValueError("SOURCE_CALL and FFN_WORK ids must be disjoint")
    execution_spans = defaultdict(list)
    compute_intervals = defaultdict(list)
    layer_intervals = defaultdict(list)
    layers: dict[tuple[str, str], dict] = {}
    for span_id, span in sorted(spans.items()):
        execution_spans[span["exec_ref"]].append(span_id)
        if span["exec_ref"] not in calls and span["exec_ref"] not in works:
            errors.append(f"SPAN {span_id}: missing exec_ref {span['exec_ref']}")
        if not span["complete"]:
            errors.append(f"SPAN {span_id}: incomplete")
        span["timing"] = _timing(
            span,
            maps.get(span["clock_domain"], ()),
            session["start_host_ns"],
        )
        if not span["timing"]["aligned"]:
            errors.append(f"SPAN {span_id}: no CLOCK_MAP covers its device interval")
        if span["kind"] in {"COMPUTE", "COMMUNICATION", "WAIT"}:
            layer_key = (span["worker_id"], span["exec_ref"])
            metadata = works.get(span["exec_ref"], calls.get(span["exec_ref"], {}))
            layers.setdefault(
                layer_key,
                {
                    "worker_id": layer_key[0],
                    "execution_id": layer_key[1],
                    "role": "ffn" if span["exec_ref"] in works else "attention",
                    "layer_idx": metadata.get("layer_idx"),
                },
            )
            timing = span["timing"]
            if span["complete"] and timing["aligned"]:
                layer_intervals[(*layer_key, span["kind"])].append(
                    (
                        session["start_host_ns"]
                        + round(timing["relative_begin_us"] * NS_PER_US),
                        session["start_host_ns"]
                        + round(timing["relative_end_us"] * NS_PER_US),
                    )
                )
        if (
            span["complete"]
            and span["timing"]["duration_ns"] is not None
            and span["device_begin_ns"] is not None
            and span["device_end_ns"] is not None
            and (
                span["kind"] == "FFN_COMPUTE"
                or (span["kind"] == "COMPUTE" and span["exec_ref"] in works)
            )
        ):
            compute_intervals[(span["worker_id"], span["clock_domain"])].append(
                (span["device_begin_ns"], span["device_end_ns"]),
            )

    _correlate_work(works, calls, spans, execution_spans, errors)
    call_works = defaultdict(list)
    for work_id, work in sorted(works.items()):
        work["span_ids"] = execution_spans[work_id]
        if not work["span_ids"]:
            errors.append(f"FFN_WORK {work_id}: missing spans")
        for call_id in work["source_call_ids"]:
            if call_id not in calls:
                errors.append(f"FFN_WORK {work_id}: missing source_call {call_id}")
                continue
            if calls[call_id]["layer_idx"] != work["layer_idx"]:
                raise ValueError(f"layer_idx mismatch: {work_id} -> {call_id}")
            call_works[call_id].append(work_id)

    pipelines: dict[tuple[str, str, int, int], dict] = {}
    for call_id, call in sorted(calls.items()):
        call["span_ids"] = execution_spans[call_id]
        if not call["span_ids"]:
            errors.append(f"SOURCE_CALL {call_id}: missing spans")
        if call.get("coverage", "COMPLETE") != "COMPLETE":
            errors.append(f"SOURCE_CALL {call_id}: missing input snapshot")
        expected = set(call.get("expected_ffn_workers", []))
        observed = {
            spans[span_id]["worker_id"]
            for work_id in call_works[call_id]
            for span_id in works[work_id]["span_ids"]
        }
        if expected - observed:
            missing_workers = sorted(expected - observed)
            errors.append(
                f"SOURCE_CALL {call_id}: missing FFN workers {missing_workers}"
            )
        for fragment_id in call["fragment_ids"]:
            if fragment_id not in fragments:
                errors.append(f"SOURCE_CALL {call_id}: missing fragment {fragment_id}")
                continue
            fragment = fragments[fragment_id]
            if any(call[key] != fragment[key] for key in ("group_id", "stage_idx")):
                raise ValueError(f"group/stage mismatch: {call_id} -> {fragment_id}")
            key = (
                fragment["request_id"],
                call["group_id"],
                call["stage_idx"],
                call["layer_idx"],
            )
            pipeline = pipelines.setdefault(
                key,
                {
                    "request_id": key[0],
                    "group_id": key[1],
                    "stage_idx": key[2],
                    "layer_idx": key[3],
                    "fragment_ids": set(),
                    "source_call_ids": set(),
                    "attention_span_ids": set(),
                    "ffn_work_ids": set(),
                },
            )
            pipeline["fragment_ids"].add(fragment_id)
            pipeline["source_call_ids"].add(call_id)
            pipeline["attention_span_ids"].update(call["span_ids"])
            pipeline["ffn_work_ids"].update(call_works[call_id])
    output_pipelines = []
    for _, pipeline in sorted(pipelines.items()):
        for field in (
            "fragment_ids",
            "source_call_ids",
            "attention_span_ids",
            "ffn_work_ids",
        ):
            pipeline[field] = sorted(pipeline[field])
        by_worker = defaultdict(list)
        for work_id in pipeline["ffn_work_ids"]:
            worker_spans = defaultdict(list)
            for span_id in works[work_id]["span_ids"]:
                worker_spans[spans[span_id]["worker_id"]].append(span_id)
            for worker_id, span_ids in sorted(worker_spans.items()):
                by_worker[worker_id].append(
                    {"ffn_work_id": work_id, "span_ids": span_ids}
                )
        pipeline["ffn_workers"] = dict(by_worker)
        output_pipelines.append(pipeline)
    return {
        "session": session,
        "worker_sessions": list(headers.values()),
        "worker_summaries": summaries,
        "data_quality": "PARTIAL" if errors else "COMPLETE",
        "errors": errors,
        "clock_maps": dict(maps),
        "fragments": dict(sorted(fragments.items())),
        "source_calls": dict(sorted(calls.items())),
        "ffn_works": dict(sorted(works.items())),
        "spans": dict(sorted(spans.items())),
        "pipelines": output_pipelines,
        "layers": [
            {
                **layer,
                "duration_ns": {
                    phase.lower(): _union_duration(layer_intervals[(*key, phase)])
                    for phase in ("COMPUTE", "COMMUNICATION", "WAIT")
                },
            }
            for key, layer in sorted(layers.items())
        ],
        "ffn_compute_by_worker": [
            {
                "worker_id": worker,
                "clock_domain": domain,
                "duration_ns": _union_duration(intervals),
            }
            for (worker, domain), intervals in sorted(compute_intervals.items())
        ],
    }


def export_chrome_trace(records: Iterable[dict]) -> dict:
    result = reconstruct(records)
    if result.get("view") == "pipeline":
        return export_pipeline(result)
    spans = result.pop("spans")
    workers = sorted({span["worker_id"] for span in spans.values()})
    process_ids = {worker: index for index, worker in enumerate(workers, 1)}
    tracks = sorted(
        {
            (span["worker_id"], str(span["stream_id"]), span["producer_id"])
            for span in spans.values()
        }
    )
    thread_ids = {track: index for index, track in enumerate(tracks, 1)}
    events = []
    unaligned = []
    for worker in workers:
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": process_ids[worker],
                "args": {"name": worker},
            }
        )
    for (worker, stream, producer), thread_id in thread_ids.items():
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": process_ids[worker],
                "tid": thread_id,
                "args": {"name": f"{stream}:{producer}"},
            }
        )
    for span in spans.values():
        timing = span["timing"]
        if not timing["aligned"] or not span["complete"]:
            unaligned.append(span)
            continue
        track = (span["worker_id"], str(span["stream_id"]), span["producer_id"])
        events.append(
            {
                "name": span["kind"],
                "cat": "CAM host stage"
                if span.get("time_source") == "HOST"
                else "CAM device stage",
                "ph": "X",
                "pid": process_ids[span["worker_id"]],
                "tid": thread_ids[track],
                "ts": timing["relative_begin_us"],
                "dur": timing["duration_ns"] / NS_PER_US,
                "args": span,
            }
        )
    result.update(traceEvents=events, displayTimeUnit="ms", unaligned_spans=unaligned)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native",
        action="store_true",
        help="Parse raw native captures offline before exporting",
    )
    args = parser.parse_args(argv)
    paths = sorted(args.session_dir.rglob("*.jsonl"))
    if not paths:
        parser.error("session directory contains no JSONL files")
    try:
        result = export_chrome_trace(read_jsonl(paths, analyse_native=args.native))
    except ValueError as exc:
        parser.error(str(exc))
    with (
        gzip.open(args.output, "wt", encoding="utf-8")
        if args.output.suffix == ".gz"
        else args.output.open("w", encoding="utf-8")
    ) as output:
        json.dump(result, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
