# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import threading

import pytest

from afd_plugin.profiling.storage import JsonRecordWriter


def test_writer_flushes_records_and_counts_utf8_bytes(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"index": 1}, {"name": "中文", "values": [1, 2]}]
    writer = JsonRecordWriter(path)
    for record in records:
        assert writer.submit(record)

    assert writer.close(2.0)
    assert writer.close(0.0)
    assert [json.loads(line) for line in path.read_text().splitlines()] == records
    assert writer.snapshot() == {
        "written_records": 2,
        "dropped_records": 0,
        "written_bytes": path.stat().st_size,
        "error": None,
        "closed": True,
        "pending_records": 0,
    }
    assert writer.submit({"late": True}) is False
    assert writer.snapshot()["dropped_records"] == 1


def test_writer_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("existing trace\n")

    with pytest.raises(FileExistsError):
        JsonRecordWriter(path)

    assert path.read_text() == "existing trace\n"


def test_writer_enforces_byte_budget_and_drops_remaining_records(tmp_path):
    path = tmp_path / "records.jsonl"
    record = {"value": 1}
    line_size = len((json.dumps(record) + "\n").encode("utf-8"))
    writer = JsonRecordWriter(path, max_records=10, max_bytes=2 * line_size)
    for _ in range(5):
        writer.submit(record)

    assert writer.close(2.0)
    snapshot = writer.snapshot()
    assert snapshot["written_records"] == 2
    assert snapshot["dropped_records"] == 3
    assert snapshot["written_bytes"] == path.stat().st_size == 2 * line_size
    assert "max_bytes exceeded" in snapshot["error"]
    assert snapshot["pending_records"] == 0
    assert snapshot["closed"] is True


@pytest.mark.parametrize("record", [{"value": float("nan")}, {"value": object()}])
def test_writer_contains_serialization_errors(tmp_path, record):
    writer = JsonRecordWriter(tmp_path / "records.jsonl")
    assert writer.submit(record)

    assert writer.close(2.0)
    snapshot = writer.snapshot()
    assert snapshot["written_records"] == 0
    assert snapshot["dropped_records"] == 1
    assert snapshot["error"] is not None
    assert snapshot["pending_records"] == 0
    assert writer.submit({"late": True}) is False


def test_writer_queue_full_and_close_timeout_are_visible(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    dumps = json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return dumps(*args, **kwargs)

    monkeypatch.setattr(json, "dumps", blocked_dumps)
    writer = JsonRecordWriter(tmp_path / "records.jsonl", max_records=1)
    try:
        assert writer.submit({"index": 0})
        assert entered.wait(2.0)
        assert writer.submit({"index": 1})
        assert writer.submit({"index": 2}) is False
        assert writer.snapshot()["pending_records"] == 2
        assert writer.close(0.0) is False
        assert writer.snapshot()["closed"] is False
        assert writer.submit({"index": 3}) is False
    finally:
        release.set()
        assert writer.close(2.0)

    assert writer.snapshot()["written_records"] == 2
    assert writer.snapshot()["dropped_records"] == 2
    assert writer.snapshot()["pending_records"] == 0
    assert writer.snapshot()["error"] is None
