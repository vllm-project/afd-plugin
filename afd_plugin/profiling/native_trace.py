# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Import native TorchNPU artifacts for offline LIGHTWEIGHT reconstruction.

analyze.py safely extracts archives and optionally runs TorchNPU parsing with
--native. This adapter passes eager MSTX marks or FFN graph DR/CS boundaries
to pipeline.py; parsing never runs on inference workers.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from afd_plugin.profiling.pipeline import native_graph_pipeline, native_pipeline

RANGE_PREFIX = "afd.cam|"
NS_PER_US = 1000


def read_native_trace(
    directory: Path, header: dict, *, analyse_native: bool = False
) -> list[dict]:
    with native_directory(directory) as root:
        if analyse_native and not list(root.rglob("trace_view.json")):
            # Offline only: native parsing must not compete with live inference.
            from torch_npu.profiler.profiler import analyse

            for raw in root.glob("*_ascend_pt"):
                analyse(str(raw), max_process_number=1)
        return _read_native_directory(root, header)


@contextmanager
def native_directory(directory: Path):
    archive_path = directory.with_name(directory.name + ".tar.gz")
    if not archive_path.exists():
        yield directory
        return
    with TemporaryDirectory(prefix="afd-cam-native-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if (
                    not (member.isfile() or member.isdir())
                    or not (root / member.name).resolve().is_relative_to(root)
                    or not Path(member.name).parts
                    or Path(member.name).parts[0] != directory.name
                ):
                    raise ValueError("unsafe native capture archive")
            archive.extractall(root, filter="data")
        yield root / directory.name


def _native_events(paths: list[Path]) -> Iterator[tuple[Path, dict]]:
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
        events = trace if isinstance(trace, list) else trace["traceEvents"]
        for event in events:
            yield path, event


def _read_native_directory(directory: Path, header: dict) -> list[dict]:
    paths = sorted(directory.rglob("trace_view.json"))
    if not paths:
        raise ValueError(
            f"{directory}: native trace not exported; run "
            "torch_npu.profiler.profiler.analyse on the capture directory first"
        )
    if header.get("view") == "pipeline":
        events = [event for _, event in _native_events(paths)]
        if header.get("identity", {}).get("ffn_execution_mode") == "full":
            return native_graph_pipeline(events, header)
        return native_pipeline(events, header)
    records = []
    worker = header["worker_id"]
    domain = f"{worker}:native"
    for path, event in _native_events(paths):
        if event.get("ph") != "X" or not event.get("name", "").startswith(RANGE_PREFIX):
            continue
        _, generation, reference, phase, stage, clock, sequence = event["name"].split(
            "|"
        )
        if int(generation) != header["generation"]:
            raise ValueError(f"{path}: native range belongs to another generation")
        if phase not in {"COMPUTE", "COMMUNICATION", "WAIT"} or clock not in {
            "H",
            "D",
        }:
            raise ValueError(f"{path}: invalid native phase or clock")
        device = "Physic Stream Id" in event.get("args", {})
        # Stream annotations also export a host launch range; keep device time.
        if device != (clock == "D"):
            continue
        begin = int(Decimal(str(event["ts"])) * NS_PER_US)
        duration = int(Decimal(str(event["dur"])) * NS_PER_US)
        if begin < 0 or duration < 0:
            raise ValueError(f"{path}: negative native timestamp or duration")
        end = begin + duration
        records.append(
            {
                "record_type": "SPAN",
                "span_id": f"{worker}:{generation}:{sequence}",
                "worker_id": worker,
                "exec_ref": reference,
                "kind": phase,
                "stage": stage,
                "clock_domain": domain,
                "producer_id": str(event["tid"]),
                "stream_id": event["args"]["Physic Stream Id"] if device else "host",
                "host_begin_ns": begin,
                "host_end_ns": end,
                "device_begin_ns": begin if device else None,
                "device_end_ns": end if device else None,
                "generation": int(generation),
                "complete": True,
                "time_source": "NATIVE" if device else "HOST",
                "host_epoch_offset_ns": -header["clock_offset_ns"],
                "clock_uncertainty_ns": header["clock_uncertainty_ns"],
            }
        )
    device_spans = [r for r in records if r["device_begin_ns"] is not None]
    if device_spans:
        begin = min(r["device_begin_ns"] for r in device_spans)
        end = max(r["device_end_ns"] for r in device_spans)
        records.append(
            {
                "record_type": "CLOCK_MAP",
                "clock_domain": domain,
                "device_origin_ns": begin,
                "host_origin_ns": begin - header["clock_offset_ns"],
                "uncertainty_ns": header["clock_uncertainty_ns"],
                "valid_device_begin_ns": begin,
                "valid_device_end_ns": end,
                "quality": "NATIVE",
            }
        )
    return records
