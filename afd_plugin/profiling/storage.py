# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Persist worker records and finalized LIGHTWEIGHT native output off-thread.

WorkerSession's bounded JSONL writer reports drops, errors and flush status.
RuntimeProfiler's archive thread verifies durable gzip output before deleting
raw files. analyze.py/native_trace.py consume these artifacts offline.
"""

from __future__ import annotations

import gzip
import json
import os
import queue
import shutil
import tarfile
import threading
from collections.abc import Callable
from pathlib import Path

ARCHIVE_BUFFER_BYTES = 1024 * 1024


class JsonRecordWriter:
    """Transfer records through a bounded, lock-based queue to one writer.

    Written counters cover flushed batches; an IO failure can leave a partial
    final batch in the file, identified by the error in the snapshot.
    """

    def __init__(
        self,
        path: Path,
        max_records: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
        *,
        on_change: Callable[[], None] = lambda: None,
    ) -> None:
        if max_records <= 0 or max_bytes <= 0:
            raise ValueError("max_records and max_bytes must be positive")
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max_records)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._on_change = on_change
        self._closed = False
        self._written_records = 0
        self._dropped_records = 0
        self._written_bytes = 0
        self._pending_records = 0
        self._error: str | None = None
        self._file = path.open("x", encoding="utf-8", newline="\n")
        self._thread = threading.Thread(
            target=self._run, name="afd-cam-profile-writer", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._file.close()
            raise

    def submit(self, record: dict) -> bool:
        """Transfer ownership; the caller must not mutate the submitted record."""
        with self._lock:
            if self._stopping.is_set():
                self._dropped_records += 1
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped_records += 1
                return False
            self._pending_records += 1
            self._ready.set()
            return True

    def close(self, timeout_s: float) -> bool:
        """Stop admission and wait at most timeout_s; inspect snapshot for errors."""
        with self._lock:
            self._stopping.set()
            self._ready.set()
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "written_records": self._written_records,
                "dropped_records": self._dropped_records,
                "written_bytes": self._written_bytes,
                "error": self._error,
                "closed": self._closed,
                "pending_records": self._pending_records,
            }

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = reason
            self._stopping.set()
        self._on_change()

    def _drop(self, count: int) -> None:
        with self._lock:
            self._dropped_records += count
            self._pending_records -= count

    def _write_batch(self, records: list[dict]) -> None:
        lines = []
        batch_bytes = 0
        for record in records:
            try:
                line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                line_bytes = len(line.encode("utf-8"))
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
                break
            if self._written_bytes + batch_bytes + line_bytes > self._max_bytes:
                self._fail(f"max_bytes exceeded: limit={self._max_bytes}")
                break
            lines.append(line)
            batch_bytes += line_bytes

        if lines:
            self._file.writelines(lines)
            self._file.flush()
            with self._lock:
                self._written_records += len(lines)
                self._written_bytes += batch_bytes
                self._pending_records -= len(lines)
        self._drop(len(records) - len(lines))

    def _run(self) -> None:
        batch = []
        try:
            while not (self._stopping.is_set() and self._queue.empty()):
                self._ready.wait()
                self._ready.clear()
                try:
                    batch = [self._queue.get_nowait()]
                except queue.Empty:
                    continue
                for _ in range(63):
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                self._write_batch(batch)
                batch = []
                if not self._queue.empty():
                    self._ready.set()
                if self._error is not None:
                    break
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            self._drop(len(batch))
        finally:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._drop(1)
            try:
                self._file.close()
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._closed = True
            self._on_change()


def archive_native_trace(directory: Path) -> Path:
    """Replace a finalized native directory only after a durable verified archive."""
    target = directory.with_name(directory.name + ".tar.gz")
    temporary = target.with_name(target.name + ".tmp")
    if directory.is_symlink() or any(
        path.is_symlink() for path in directory.rglob("*")
    ):
        raise ValueError("native capture contains symlinks")
    try:
        with temporary.open("xb") as output:
            with tarfile.open(fileobj=output, mode="w:gz", compresslevel=1) as archive:
                archive.add(directory, arcname=directory.name)
            output.flush()
            os.fsync(output.fileno())
        # Consume the gzip footer before deleting the original data.
        with gzip.open(temporary, "rb") as source:
            while source.read(ARCHIVE_BUFFER_BYTES):
                pass
        temporary.replace(target)
        parent = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        shutil.rmtree(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
