# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest
from pydantic import ValidationError

from afd_plugin.profiling.analyze import (
    export_chrome_trace,
    main,
    read_jsonl,
    reconstruct,
)
from afd_plugin.profiling.control import StartRequest
from afd_plugin.profiling.native_trace import read_native_trace
from afd_plugin.profiling.storage import archive_native_trace

EPOCH_NS = 1_790_000_000_000_000_000


def capture(worker, role, offset=0):
    return [
        {
            "record_type": "SESSION",
            "schema_version": 1,
            "session_id": "test",
            "worker_id": worker,
            "role": role,
            "view": "pipeline",
            "generation": 1,
            "clock_offset_ns": offset,
            "clock_uncertainty_ns": 10,
            "start_host_ns": EPOCH_NS,
            "native_trace_dir": f"{worker}/lightweight",
        },
        {
            "record_type": "SUMMARY",
            "worker_id": worker,
            "native_marks": 4,
            "dropped_spans": 0,
            "dropped_metadata": 0,
            "unresolved_spans": 0,
            "dropped_records": 0,
            "error": None,
        },
    ]


def events(offset=0):
    result = []
    for index, phase in enumerate(("COMPUTE", "COMMUNICATION", "WAIT", "END")):
        event: dict = {
            "ph": "X",
            "name": f"afd.cam|{phase}",
            "dur": 0,
            "ts": str((EPOCH_NS + offset + index * 10000) // 1000),
            "args": {"Physic Stream Id": 47}
            if phase != "END"
            else {"domain": "afd.cam"},
        }
        result.append(event)
        if phase != "END":
            result.append({**event, "args": {"event_type": "marker"}})
    return result


def native(root, worker, data):
    directory = root / worker / "lightweight"
    directory.mkdir(parents=True)
    (directory / "trace_view.json").write_text(json.dumps({"traceEvents": data}))
    return directory


def graph_events(offset=0):
    events = []
    for name, begin, duration in (
        ("AfdAsyncDispatchRecv", 0, 5),
        ("GroupedMatmul", 5, 3),
        ("AfdAsyncCombineSend", 8, 2),
        ("AfdAsyncDispatchRecv", 15, 5),
        ("GroupedMatmul", 20, 3),
        ("AfdAsyncCombineSend", 23, 2),
    ):
        events.append(
            {
                "ph": "X",
                "name": f"aclnn{name}_{name}",
                "ts": str((EPOCH_NS + offset) // 1000 + begin),
                "dur": duration,
                "args": {"Physic Stream Id": 13},
            }
        )
    return events


@pytest.mark.parametrize("archive", [False, True])
def test_graph_pipeline_reads_native_kernels_without_python_marks(tmp_path, archive):
    rows = capture("f0", "ffn", offset=50000)
    rows[0]["identity"] = {"ffn_execution_mode": "full"}
    rows[1]["native_marks"] = 0
    directory = native(tmp_path, "f0", list(reversed(graph_events(offset=50000))))
    if archive:
        archive_native_trace(directory)
    path = tmp_path / "f0.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    result = reconstruct(read_jsonl([path]))
    assert result["data_quality"] == "COMPLETE", result["errors"]
    assert result["pipelines"][0]["intervals"] == [
        (EPOCH_NS, EPOCH_NS + 5000, "WAIT"),
        (EPOCH_NS + 5000, EPOCH_NS + 8000, "COMPUTE"),
        (EPOCH_NS + 8000, EPOCH_NS + 10000, "COMMUNICATION"),
        (EPOCH_NS + 10000, EPOCH_NS + 20000, "WAIT"),
        (EPOCH_NS + 20000, EPOCH_NS + 23000, "COMPUTE"),
        (EPOCH_NS + 23000, EPOCH_NS + 25000, "COMMUNICATION"),
    ]


@pytest.mark.parametrize("missing", ["receive", "send", "both", "prefix"])
def test_graph_pipeline_reports_missing_or_unpaired_native_kernels(tmp_path, missing):
    rows = capture("f0", "ffn")
    rows[0]["identity"] = {"ffn_execution_mode": "full"}
    rows[1]["native_marks"] = 0
    events = graph_events()
    if missing == "prefix":
        events = events[2:4]
    else:
        removed = {"receive": "DispatchRecv", "send": "CombineSend", "both": "AfdAsync"}
        events = [event for event in events if removed[missing] not in event["name"]]
    directory = native(tmp_path, "f0", events)
    result = reconstruct(rows + read_native_trace(directory, rows[0]))
    assert result["data_quality"] == "PARTIAL"
    assert "incomplete native graph kernels" in result["errors"][0]


@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity"])
def test_graph_pipeline_rejects_invalid_kernel_duration(tmp_path, duration):
    rows = capture("f0", "ffn")
    rows[0]["identity"] = {"ffn_execution_mode": "full"}
    events = graph_events()
    events[0]["dur"] = duration
    directory = native(tmp_path, "f0", events)
    with pytest.raises(ValueError, match="invalid graph kernel"):
        read_native_trace(directory, rows[0])


def test_attention_and_ffn_pipeline_align_without_layers_or_request_ids(tmp_path):
    records = []
    for worker, role, offset in (("a0", "attention", 0), ("f0", "ffn", 50000)):
        rows = capture(worker, role, offset)
        directory = native(tmp_path, worker, events(offset))
        records.extend(rows + read_native_trace(directory, rows[0]))
    result = reconstruct(records)
    assert result["data_quality"] == "COMPLETE", result["errors"]
    tracks = result["pipelines"]
    assert [t["role"] for t in tracks] == ["attention", "ffn"]
    assert tracks[0]["intervals"] == tracks[1]["intervals"]
    assert [x[2] for x in tracks[0]["intervals"]] == [
        "COMPUTE",
        "COMMUNICATION",
        "WAIT",
    ]
    trace = export_chrome_trace(records)
    phases = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert len(phases) == 6 and not any("args" in event for event in phases)
    assert all(event["dur"] == 10 for event in phases)
    assert not any(
        key in trace for key in ("source_calls", "fragments", "ffn_works", "layers")
    )


def test_native_end_clips_queued_work_at_capture_boundary(tmp_path):
    data = events()
    # Device WAIT was queued after the native host END; exclude it from this window.
    for event in data:
        if event["name"] == "afd.cam|WAIT":
            event["ts"] = str((EPOCH_NS + 50000) // 1000)
    rows = capture("a0", "attention")
    records = read_native_trace(native(tmp_path, "a0", data), rows[0])
    phases = [r for r in records if r["record_type"] == "PIPELINE"]
    assert [r["kind"] for r in phases] == ["COMPUTE", "COMMUNICATION"]
    assert phases[-1]["end_ns"] == EPOCH_NS + 30000


def test_attention_dispatch_annotation_survives_alignment_and_export(tmp_path):
    offset = 50000
    rows = capture("a0", "attention", offset)
    data = events(offset)
    for event in data:
        if event["name"] == "afd.cam|COMMUNICATION":
            event["name"] += "|1|7|3|9"
    records = rows + read_native_trace(native(tmp_path, "a0", data), rows[0])
    expected = {"microbatch_id": 1, "layer_idx": 7, "num_requests": 3, "num_tokens": 9}
    track = reconstruct(records)["pipelines"][0]
    assert track["annotations"] == [{"begin_ns": EPOCH_NS + 10000, **expected}]
    trace = export_chrome_trace(records)
    phases = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert len(phases) == 3
    assert phases[1]["args"] == expected
    assert phases[1]["ts"] == 10
    assert all("args" not in e for e in (phases[0], phases[2]))


@pytest.mark.parametrize(
    "role,phase,suffix",
    [
        ("ffn", "COMMUNICATION", "|0|1|2|3"),
        ("attention", "COMPUTE", "|0|1|2|3"),
        ("attention", "COMMUNICATION", "|0|1|2"),
        ("attention", "COMMUNICATION", "|0|1|-2|3"),
        ("attention", "COMMUNICATION", "|0|1|True|3"),
    ],
)
def test_invalid_attention_annotation_rejected(tmp_path, role, phase, suffix):
    rows = capture("a0", role)
    data = events()
    for event in data:
        if event["name"] == f"afd.cam|{phase}":
            event["name"] += suffix
    with pytest.raises(ValueError, match="annotation"):
        read_native_trace(native(tmp_path, "a0", data), rows[0])


def test_missing_native_marker_is_partial(tmp_path):
    rows = capture("a0", "attention")
    records = rows + read_native_trace(native(tmp_path, "a0", events()[:-1]), rows[0])
    assert reconstruct(records)["data_quality"] == "PARTIAL"


def test_native_archive_roundtrip_and_compact_gzip_export(tmp_path):
    rows = capture("a0", "attention")
    directory = native(tmp_path, "a0", events())
    original = (directory / "trace_view.json").read_bytes()
    archive = archive_native_trace(directory)
    assert archive.exists() and not directory.exists()
    with tarfile.open(archive) as data:
        native_file = data.extractfile("lightweight/trace_view.json")
        assert native_file is not None
        assert native_file.read() == original
    (tmp_path / "a0.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert (
        export_chrome_trace(read_jsonl([tmp_path / "a0.jsonl"]))["data_quality"]
        == "COMPLETE"
    )
    output = tmp_path / "pipeline.json.gz"
    main([str(tmp_path), "--output", str(output)])
    with gzip.open(output, "rt") as source:
        assert json.load(source)["data_quality"] == "COMPLETE"
    assert len(list((tmp_path / "a0").iterdir())) == 1


def test_archive_io_failure_keeps_raw_capture(tmp_path, monkeypatch):
    directory = tmp_path / "lightweight"
    directory.mkdir()
    (directory / "raw").write_bytes(b"raw-native-data")
    monkeypatch.setattr(
        "afd_plugin.profiling.storage.os.fsync",
        lambda _: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError, match="disk failure"):
        archive_native_trace(directory)
    assert (directory / "raw").read_bytes() == b"raw-native-data"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "name", ["../escape", "/tmp/escape", "other/trace_view.json", "."]
)
def test_native_archive_rejects_paths_outside_capture(tmp_path, name):
    directory = tmp_path / "lightweight"
    with tarfile.open(tmp_path / "lightweight.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="unsafe"):
        read_native_trace(directory, capture("a0", "attention")[0])


def test_long_lightweight_duration_keeps_full_limit():
    assert StartRequest(mode="LIGHTWEIGHT", duration=86400.0).duration == 86400
    with pytest.raises(ValidationError, match="one hour"):
        StartRequest(mode="FULL", duration=86400.0)
    with pytest.raises(ValidationError):
        StartRequest(mode="LIGHTWEIGHT", duration=604801.0)


def test_pipeline_mark_count_rejects_boolean_summary():
    rows = capture("a0", "attention")
    rows[1]["native_marks"] = False
    with pytest.raises(ValueError, match="native_marks"):
        reconstruct(rows)
