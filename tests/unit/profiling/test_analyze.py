# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json

import pytest

from afd_plugin.profiling.analyze import export_chrome_trace, read_jsonl, reconstruct


def header(**updates):
    return {
        "record_type": "SESSION",
        "schema_version": 1,
        "session_id": "legacy",
        "worker_id": "a0",
        "generation": 3,
        "start_host_ns": 10_000_000_000,
        "clock_offset_ns": 100,
        "clock_uncertainty_ns": 50,
        **updates,
    }


@pytest.mark.parametrize("array", [False, True])
def test_legacy_native_trace_keeps_device_precision_and_skips_host_duplicates(
    tmp_path, array
):
    native = tmp_path / "native"
    native.mkdir()
    event = {
        "name": "afd.cam|3|a0:call:1|COMPUTE|ATTENTION|D|1",
        "ph": "X",
        "tid": 47,
        "ts": "10000000.125",
        "dur": 5.5,
        "args": {"Physic Stream Id": 47},
    }
    events = [event, {**event, "args": {}}]
    payload = events if array else {"traceEvents": events}
    (native / "trace_view.json").write_text(json.dumps(payload))
    rows = [
        header(native_trace_dir="native"),
        {
            "record_type": "SOURCE_CALL",
            "source_call_id": "a0:call:1",
            "group_id": "g",
            "stage_idx": 0,
            "layer_idx": 7,
            "fragment_ids": [],
        },
    ]
    path = tmp_path / "a0.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    records = read_jsonl([path])
    span = next(row for row in records if row["record_type"] == "SPAN")
    assert span["device_begin_ns"] == 10_000_000_125
    trace = export_chrome_trace(records)
    assert trace["data_quality"] == "COMPLETE", trace["errors"]
    phases = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert len(phases) == 1
    assert phases[0]["ts"] == 0.025
    assert phases[0]["dur"] == 5.5


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_jsonl_nonfinite_numbers_keep_source_line(tmp_path, number):
    path = tmp_path / "worker.jsonl"
    path.write_text(json.dumps(header()) + '\n{"extra":[' + number + "]}\n")
    with pytest.raises(ValueError, match="worker.jsonl:2:.*finite JSON"):
        read_jsonl([path])


def test_native_trace_path_cannot_escape_session(tmp_path):
    path = tmp_path / "worker.jsonl"
    path.write_text(json.dumps(header(native_trace_dir="../escape")) + "\n")
    with pytest.raises(ValueError, match="escapes session"):
        read_jsonl([path])


@pytest.mark.parametrize(
    "change", [{"schema_version": 99}, {"identity": {"x": float("nan")}}]
)
def test_reconstruct_rejects_invalid_session(change):
    with pytest.raises(ValueError):
        reconstruct([header(**change)])
