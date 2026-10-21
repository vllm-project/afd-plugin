# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reconstruct and export aligned COMPUTE/COMMUNICATION/WAIT stream tracks.

analyze.py uses these helpers to align native marks, FFN graph DR/CS boundaries
or older spans with session clocks. Export preserves streams and dispatch
counts, reports incomplete coverage and never infers request/layer pairing.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

PHASES = frozenset({"COMPUTE", "COMMUNICATION", "WAIT"})
NS_PER_US = 1000
ATTENTION_FIELDS = ("microbatch_id", "layer_idx", "num_requests", "num_tokens")


def native_graph_pipeline(events: list[dict], header: dict) -> list[dict]:
    """Use native DR/CS device times; replay does not execute Python phase hooks."""
    boundaries = []
    streams = defaultdict(list)
    receives = sends = 0
    end = Decimal(0)
    for event in events:
        if event.get("ph") != "X" or "Physic Stream Id" not in event.get("args", {}):
            continue
        name = event.get("name", "").replace("_", "").lower()
        if "afdasyncdispatchrecv" in name:
            phases = ("WAIT", "COMPUTE")
            receives += 1
        elif "afdasynccombinesend" in name:
            phases = ("COMMUNICATION", "WAIT")
            sends += 1
        else:
            continue
        begin = Decimal(str(event["ts"]))
        duration = Decimal(str(event["dur"]))
        if (
            not begin.is_finite()
            or not duration.is_finite()
            or min(begin, duration) < 0
        ):
            raise ValueError("invalid graph kernel timestamp or duration")
        finish = begin + duration
        end = max(end, finish)
        streams[str(event["args"]["Physic Stream Id"])].append(
            (begin, finish, phases[0] == "WAIT")
        )
        for timestamp, phase in zip((begin, finish), phases, strict=True):
            boundaries.append(
                {
                    "name": f"afd.cam|{phase}",
                    "ts": timestamp,
                    "args": event["args"],
                }
            )
    boundaries.append({"name": "afd.cam|END", "ts": end})
    records = native_pipeline(boundaries, header)
    incomplete = not receives
    for kernels in streams.values():
        previous_end = Decimal(0)
        for index, (begin, finish, receive) in enumerate(sorted(kernels)):
            incomplete |= receive != (index % 2 == 0) or begin < previous_end
            previous_end = finish
        incomplete |= len(kernels) % 2 != 0
    records[-1].update(
        graph_receives=receives, graph_sends=sends, graph_incomplete=incomplete
    )
    return records


def native_pipeline(events: list[dict], header: dict) -> list[dict]:
    tracks = defaultdict(list)
    ends = []
    observed = 0
    for event in events:
        name = event.get("name", "")
        if not name.startswith("afd.cam|"):
            continue
        phase, *labels = name.removeprefix("afd.cam|").split("|")
        device = "Physic Stream Id" in event.get("args", {})
        if (phase == "END" and device) or (phase != "END" and not device):
            continue
        if phase != "END" and phase not in PHASES:
            raise ValueError("invalid pipeline phase")
        annotation = {}
        if labels:
            if (
                header["role"] != "attention"
                or phase != "COMMUNICATION"
                or len(labels) != len(ATTENTION_FIELDS)
                or any(not value.isascii() or not value.isdecimal() for value in labels)
            ):
                raise ValueError("invalid Attention pipeline annotation")
            annotation = dict(zip(ATTENTION_FIELDS, map(int, labels), strict=True))
        timestamp = int(Decimal(str(event["ts"])) * NS_PER_US)
        if timestamp < 0:
            raise ValueError("negative pipeline timestamp")
        observed += 1
        if phase == "END":
            ends.append(timestamp)
        else:
            tracks[str(event["args"]["Physic Stream Id"])].append(
                (timestamp, phase, annotation)
            )
    records = []
    end = ends[0] if len(ends) == 1 else None
    for stream, points in sorted(tracks.items()):
        points.sort(key=lambda point: point[0])
        for index, (begin, phase, annotation) in enumerate(points):
            finish = points[index + 1][0] if index + 1 < len(points) else end
            if finish is None:
                continue
            if end is not None:
                finish = min(finish, end)
            if finish <= begin:
                continue
            records.append(
                {
                    "record_type": "PIPELINE",
                    "worker_id": header["worker_id"],
                    "stream_id": stream,
                    "kind": phase,
                    "begin_ns": begin - header["clock_offset_ns"],
                    "end_ns": finish - header["clock_offset_ns"],
                    **({"annotation": annotation} if annotation else {}),
                }
            )
    records.append(
        {
            "record_type": "NATIVE_SUMMARY",
            "worker_id": header["worker_id"],
            "observed_marks": observed,
            "end_markers": len(ends),
        }
    )
    return records


def reconstruct_pipeline(records: list[dict]) -> dict:
    headers = {}
    summaries: dict[str, dict] = {}
    native: dict[str, dict] = {}
    intervals = defaultdict(list)
    session = None
    for record in records:
        kind = record["record_type"]
        worker = record["worker_id"]
        if kind == "SESSION":
            common = {
                key: record[key]
                for key in ("schema_version", "session_id", "start_host_ns")
            }
            if session is not None and session != common:
                raise ValueError("SESSION headers disagree")
            if worker in headers:
                raise ValueError("duplicate pipeline SESSION")
            if record.get("view") != "pipeline" or record.get("role") not in {
                "attention",
                "ffn",
            }:
                raise ValueError("invalid pipeline worker header")
            session = common
            headers[worker] = record
        elif kind in {"SUMMARY", "NATIVE_SUMMARY"}:
            target = summaries if kind == "SUMMARY" else native
            if worker in target:
                raise ValueError(f"duplicate {kind}")
            target[worker] = record
        elif kind == "PIPELINE":
            if (
                record["kind"] not in PHASES
                or not all(
                    type(record[key]) is int and record[key] >= 0
                    for key in ("begin_ns", "end_ns")
                )
                or record["end_ns"] < record["begin_ns"]
            ):
                raise ValueError("invalid pipeline interval")
            if "annotation" in record and (
                record["kind"] != "COMMUNICATION"
                or not isinstance(record["annotation"], dict)
                or set(record["annotation"]) != set(ATTENTION_FIELDS)
                or any(
                    type(value) is not int or value < 0
                    for value in record["annotation"].values()
                )
            ):
                raise ValueError("invalid Attention pipeline annotation")
            intervals[(worker, str(record["stream_id"]))].append(record)
        else:
            raise ValueError("pipeline capture contains detailed records")
    if session is None or any(worker not in headers for worker, _ in intervals):
        raise ValueError("missing pipeline SESSION")
    errors = []
    for worker in headers:
        summary = summaries.get(worker)
        if summary is None:
            errors.append(f"worker {worker}: missing SUMMARY")
            continue
        if summary.get("error") or any(
            summary.get(key, 0)
            for key in (
                "dropped_spans",
                "dropped_metadata",
                "unresolved_spans",
                "dropped_records",
            )
        ):
            errors.append(f"worker {worker}: dropped, unresolved or failed capture")
        expected = summary.get("native_marks", 0)
        observed = native.get(worker, {})
        if headers[worker].get("identity", {}).get("ffn_execution_mode") == "full" and (
            not observed.get("graph_receives")
            or observed.get("graph_receives") != observed.get("graph_sends")
            or observed.get("graph_incomplete", True)
        ):
            errors.append(
                f"worker {worker}: missing or incomplete native graph kernels"
            )
        if expected and (
            observed.get("observed_marks") != expected
            or observed.get("end_markers") != 1
        ):
            errors.append(f"worker {worker}: missing native pipeline markers")
    tracks = []
    for (worker, stream), values in sorted(intervals.items()):
        merged: list[tuple[int, int, str]] = []
        annotations = []
        for value in sorted(values, key=lambda value: value["begin_ns"]):
            if merged and value["begin_ns"] < merged[-1][1]:
                raise ValueError("overlapping states in one pipeline stream")
            begin, end = value["begin_ns"], value["end_ns"]
            if "annotation" in value:
                if headers[worker]["role"] != "attention":
                    raise ValueError("FFN pipeline cannot carry Attention annotations")
                annotations.append({"begin_ns": begin, **value["annotation"]})
            if (
                merged
                and merged[-1][1] == begin
                and merged[-1][2] == value["kind"]
                and "annotation" not in value
            ):
                merged[-1] = (merged[-1][0], end, value["kind"])
            else:
                merged.append((begin, end, value["kind"]))
        tracks.append(
            {
                "worker_id": worker,
                "role": headers[worker]["role"],
                "stream_id": stream,
                "intervals": merged,
                **({"annotations": annotations} if annotations else {}),
            }
        )
    return {
        "session": session,
        "view": "pipeline",
        "worker_sessions": list(headers.values()),
        "worker_summaries": summaries,
        "pipelines": tracks,
        "data_quality": "PARTIAL" if errors else "COMPLETE",
        "errors": errors,
    }


def export_pipeline(result: dict) -> dict:
    events = []
    workers = {
        header["worker_id"]: (index, header["role"])
        for index, header in enumerate(result["worker_sessions"], 1)
    }
    for worker, (pid, role) in workers.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": f"{role}:{worker}"},
            }
        )
    start = result["session"]["start_host_ns"]
    for index, track in enumerate(result.pop("pipelines"), 1):
        pid = workers[track["worker_id"]][0]
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": index,
                "args": {"name": f"stream {track['stream_id']}"},
            }
        )
        annotations = {
            entry["begin_ns"]: {key: entry[key] for key in ATTENTION_FIELDS}
            for entry in track.get("annotations", [])
        }
        for begin, end, phase in track["intervals"]:
            events.append(
                {
                    "name": phase,
                    "cat": phase,
                    "ph": "X",
                    "pid": pid,
                    "tid": index,
                    "ts": (begin - start) / NS_PER_US,
                    "dur": (end - begin) / NS_PER_US,
                    **({"args": annotations[begin]} if begin in annotations else {}),
                }
            )
    return {**result, "traceEvents": events, "displayTimeUnit": "ms"}
