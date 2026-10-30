#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One isolated native rank; run.py sets visibility/vendor paths before import."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

WORLD = 10
SENDERS = 8
TARGETS = 2
BATCH = 3072
HIDDEN = 4096
EXPERTS = 32
INTERMEDIATE = 2048
TOP_K = 8
ROUTES_PER_TARGET = 4
ROUTING_GROUPS = 8
ROWS_PER_EXPERT = BATCH // ROUTING_GROUPS
EXPANDED_ROWS = BATCH * ROUTES_PER_TARGET
CS_TP = 2
CS_CELL_ROWS = 48
CS_TARGET_ROWS = BATCH // CS_TP
CS_MAX_SEQUENCE = 16384
DISPATCH_MAX_SEQUENCE = 4096
CS_TOP_K = 6
CS_TRACE_ELEMENTS = 48 * 16 * 4
DISPATCH_TRACE_ELEMENTS = 48 * 32 * 4
STATE_HALF_ELEMENTS = 16
BATCH_INFO_FIELDS = 5
CS_GRAPH_CALLS = 64
DS_GRAPH_CALLS = 16
CS_GMM_PAIRS = 128
DS_GMM_PAIRS = 256
DR_GMM_PAIRS = 64
SEED = 20261022
WARMUP_REPLAYS = 3
GMM_GROUP_ROWS = BATCH // EXPERTS
GMM_SCALE = 0.005
INT4_COMPENSATION = 8
QUANT_MAX = 127
QUANT_PERIOD = 255
CS_VALUE_PERIOD = 251
CS_ROW_PERIOD = 97
CS_SENDER_MULTIPLIER = 101
CS_COLUMN_PERIOD = 3
DS_ROW_MULTIPLIER = 17
DS_RANK_MULTIPLIER = 23
WEIGHT_SCALE_RANGE = 1e-3
WEIGHT_SCALE_MIN = 1e-4
GMM_INPUT_LOW = -64
GMM_INPUT_HIGH = 64
ROOT = Path(__file__).resolve().parents[3]


def cs_payload(sender: int) -> torch.Tensor:
    rows = torch.arange(BATCH).reshape(EXPERTS, CS_TP, CS_CELL_ROWS)
    values = (rows % CS_ROW_PERIOD + sender * CS_SENDER_MULTIPLIER) % CS_VALUE_PERIOD
    return (values.reshape(-1, 1) + torch.arange(HIDDEN) % CS_COLUMN_PERIOD).to(
        torch.bfloat16
    )


def ds_payload(sender: int) -> torch.Tensor:
    values = (
        torch.arange(BATCH).view(-1, 1) * DS_ROW_MULTIPLIER
        + torch.arange(HIDDEN)
        + sender * DS_RANK_MULTIPLIER
    ) % QUANT_PERIOD - QUANT_MAX
    values[:, 0] = QUANT_MAX
    return values.to(torch.int8)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_report(extension: Path):
    vendor = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    artifacts = [extension, Path(os.environ["AFD_CUST_OPAPI_LIB_PATH"])]
    artifacts.extend(vendor.rglob("*.o"))
    artifacts.extend(vendor.rglob("libcust_optiling.so"))
    sources = {}
    for mode in ("combine_send", "dispatch_send", "dispatch_recv"):
        path = (
            ROOT
            / f"csrc/npu/ascend_kernels/afd_async_{mode}/op_kernel/afd_async_{mode}.h"
        )
        if path.is_file():
            sources[mode] = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
    versions = {}
    for name in (
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/version.info",
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info",
        "/usr/local/Ascend/driver/version.info",
    ):
        path = Path(name)
        if path.is_file():
            versions[name] = path.read_text(errors="replace")
    try:
        smi = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi = str(exc)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "hostname": platform.node(),
        "device": str(torch.npu.get_device_properties(0)),
        "npu_smi": smi,
        "versions": versions,
        "source_sha256_lf": sources,
        "artifacts_sha256": {str(path): fingerprint(path) for path in artifacts},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "HCCL_IF_IP",
                "HCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "HCCL_BUFFSIZE",
                "BATCH_SIZE_FACTOR",
                "ASCEND_CUSTOM_OPP_PATH",
                "AFD_CUST_OPAPI_LIB_PATH",
            )
        },
    }


class Fixture:
    def __init__(self, case, rank: int, comm_name: str, stream):
        self.case, self.rank, self.comm_name, self.stream = (
            case,
            rank,
            comm_name,
            stream,
        )
        self.mode = case["mode"]
        self.sender = rank >= TARGETS if self.mode == "cs" else rank < SENDERS
        self.sender_index = rank - TARGETS if self.mode == "cs" else rank
        self.active = self.sender and self.sender_index < case["fan_in"]
        self.measured = not self.sender if self.mode == "dr" else self.active
        self.anchor = torch.zeros(1, dtype=torch.bfloat16, device="npu")
        trace_elements = (
            CS_TRACE_ELEMENTS if self.mode == "cs" else DISPATCH_TRACE_ELEMENTS
        )
        self.trace = torch.zeros(trace_elements, dtype=torch.float16, device="npu")
        self.graph = None
        self.graph_outputs = []
        self.graph_traces = []
        self.graph_calls = (
            CS_GRAPH_CALLS
            if self.mode == "cs"
            else DS_GRAPH_CALLS
            if self.mode == "ds"
            else 1
        )
        if self.mode == "cs":
            if self.sender:
                self.x = cs_payload(self.sender_index).npu()
                offset = self.sender_index * CS_TARGET_ROWS
                self.batch_info = torch.tensor(
                    [BATCH, 0, 0, 0, EXPERTS - 1, offset, offset]
                    + [CS_CELL_ROWS] * (EXPERTS * CS_TP),
                    dtype=torch.int64,
                    device="npu",
                )
            else:
                self.x = torch.empty(
                    (BATCH, HIDDEN), dtype=torch.bfloat16, device="npu"
                )
                self.batch_info = torch.zeros(
                    BATCH_INFO_FIELDS + CS_TP + EXPERTS * CS_TP,
                    dtype=torch.int64,
                    device="npu",
                )
                self.window = torch.empty(
                    STATE_HALF_ELEMENTS + SENDERS * CS_TARGET_ROWS * HIDDEN,
                    dtype=torch.float16,
                    device="npu",
                )
        elif self.sender:
            self.x = ds_payload(rank).to(torch.bfloat16).npu()
            local = torch.arange(BATCH).view(
                -1, 1
            ) % ROUTING_GROUPS * ROUTES_PER_TARGET + torch.arange(ROUTES_PER_TARGET)
            self.expert_ids = torch.cat((local, local + EXPERTS), dim=1).int().npu()

    def cs(self, trace):
        return torch.ops.afd_ascend.afd_async_combine_send(
            self.x,
            trace,
            self.batch_info,
            0,
            CS_MAX_SEQUENCE,
            HIDDEN,
            CS_TOP_K,
            SENDERS,
            TARGETS,
            EXPERTS,
            self.rank if self.sender else TARGETS,
            WORLD,
            CS_TP,
            self.comm_name,
        )

    def ds(self, trace):
        return torch.ops.afd_ascend.afd_async_dispatch_send(
            self.x,
            self.expert_ids,
            trace,
            0,
            DISPATCH_MAX_SEQUENCE,
            BATCH,
            HIDDEN,
            TOP_K,
            TARGETS,
            SENDERS,
            EXPERTS,
            self.rank,
            WORLD,
            0,
            1,
            1,
            self.comm_name,
        )

    def dr(self, trace):
        return torch.ops.afd_ascend.afd_async_dispatch_recv(
            self.anchor,
            trace,
            0,
            DISPATCH_MAX_SEQUENCE,
            HIDDEN,
            TOP_K,
            TARGETS,
            SENDERS,
            EXPERTS,
            self.rank,
            WORLD,
            1,
            1,
            self.comm_name,
        )

    def helper(self, action: int):
        # Helper is outside every graph/event interval. The diagnostic kernel
        # writes comm_args, never XOut (native CS returns only a completion token).
        self.batch_info[:2].copy_(
            torch.tensor(
                [SENDERS * CS_TARGET_ROWS, action], dtype=torch.int64, device="npu"
            )
        )
        self.cs(self.window)

    def validate_cs(self, fan_in: int):
        with torch.npu.stream(self.stream):
            self.helper(1)
        self.stream.synchronize()
        states = self.window[:STATE_HALF_ELEMENTS].view(torch.int32).cpu()
        expected_states = torch.tensor(
            [s * CS_TARGET_ROWS + 1 if s < fan_in else 0 for s in range(SENDERS)],
            dtype=torch.int32,
        )
        torch.testing.assert_close(states, expected_states, rtol=0, atol=0)
        actual = (
            self.window[STATE_HALF_ELEMENTS:].view(torch.bfloat16).reshape(-1, HIDDEN)
        )
        for sender in range(fan_in):
            expected = cs_payload(sender).reshape(EXPERTS, CS_TP, CS_CELL_ROWS, HIDDEN)
            expected = expected[:, self.rank].reshape(CS_TARGET_ROWS, HIDDEN)
            start = sender * CS_TARGET_ROWS
            torch.testing.assert_close(
                actual[start : start + CS_TARGET_ROWS].cpu(), expected, rtol=0, atol=0
            )

    def validate_metadata(self, metadata: torch.Tensor) -> int:
        source = int(metadata[1])
        expected = torch.tensor(
            [
                EXPANDED_ROWS,
                source,
                0,
                0,
                EXPERTS - 1,
                (self.rank - SENDERS) * EXPANDED_ROWS,
            ]
            + [ROWS_PER_EXPERT] * EXPERTS,
            dtype=torch.int64,
        )
        torch.testing.assert_close(metadata, expected, rtol=0, atol=0)
        return source

    def validate_payload(self, outputs, source: int):
        expanded, scales, _, counts = outputs
        torch.testing.assert_close(
            counts.cpu(),
            torch.full((EXPERTS,), ROWS_PER_EXPERT, dtype=torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scales[:EXPANDED_ROWS].cpu(), torch.ones(EXPANDED_ROWS), rtol=0, atol=0
        )
        values = ds_payload(source)
        # Check each expert independently to avoid a 48 MiB expanded CPU golden.
        for expert in range(EXPERTS):
            rows = torch.arange(expert // ROUTES_PER_TARGET, BATCH, ROUTING_GROUPS)
            start = expert * ROWS_PER_EXPERT
            torch.testing.assert_close(
                expanded[start : start + ROWS_PER_EXPERT].cpu(),
                values[rows],
                rtol=0,
                atol=0,
            )

    def drain(self, fan_in: int, replay: bool = False):
        seen = []
        for _ in range(fan_in):
            with torch.npu.stream(self.stream):
                if replay:
                    self.graph.replay()
                    outputs = self.graph_outputs[0]
                else:
                    outputs = self.dr(self.trace)
            self.stream.synchronize()
            source = self.validate_metadata(outputs[2].cpu())
            self.validate_payload(outputs, source)
            seen.append(source)
        if sorted(seen) != list(range(fan_in)):
            raise AssertionError(f"Incomplete/duplicate dispatch sources: {seen}")

    def publish(self, fan_in: int):
        if self.sender and self.sender_index < fan_in:
            with torch.npu.stream(self.stream):
                self.ds(self.trace)
        torch.npu.synchronize()
        dist.barrier()

    def initialize(self):
        with torch.npu.stream(self.stream):
            if self.mode == "cs":
                self.cs(self.trace) if self.sender else self.helper(2)
            elif self.sender:
                self.ds(self.trace)
        # DR starts concurrently with initial DS: all ten ranks must enter MC2.
        if self.mode != "cs" and not self.sender:
            self.drain(SENDERS)
        torch.npu.synchronize()
        dist.barrier()
        if self.mode == "cs" and not self.sender:
            self.validate_cs(SENDERS)
        dist.barrier()

    def capture(self):
        if self.mode == "dr":
            self.publish(SENDERS)
        if self.measured:
            operation = {"cs": self.cs, "ds": self.ds, "dr": self.dr}[self.mode]
            self.graph_traces = [
                torch.zeros_like(self.trace) for _ in range(self.graph_calls)
            ]
            torch.npu.synchronize()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, stream=self.stream):
                self.graph_outputs = [operation(trace) for trace in self.graph_traces]
        torch.npu.synchronize()
        dist.barrier()
        # Capture may execute a receive, depending on runtime. Republish ALL slots
        # and drain them once so inactive sources cannot leak into later samples.
        if self.mode == "dr":
            self.publish(SENDERS)
            if not self.sender:
                self.drain(SENDERS)
        torch.npu.synchronize()
        dist.barrier()
        for _ in range(WARMUP_REPLAYS):
            if self.mode == "dr":
                self.publish(self.case["fan_in"])
                if not self.sender:
                    self.drain(self.case["fan_in"], replay=True)
            elif self.measured:
                with torch.npu.stream(self.stream):
                    self.graph.replay()
            torch.npu.synchronize()
            dist.barrier()
        if self.mode == "ds":
            # Also works on runtimes where capture itself only records kernels.
            self.publish(self.case["fan_in"])
            if not self.sender:
                self.drain(self.case["fan_in"])
        torch.npu.synchronize()
        dist.barrier()


class Background:
    def __init__(self, mode: str, stream):
        self.stream = stream
        self.retained = []
        self.w13, self.s13, self.a13 = self.weight(HIDDEN, 2 * INTERMEDIATE, False)
        self.w2, self.s2, self.a2 = self.weight(INTERMEDIATE, HIDDEN, True)
        self.pristine = torch.randint(
            GMM_INPUT_LOW, GMM_INPUT_HIGH, (BATCH, HIDDEN), dtype=torch.int8
        ).npu()
        pairs = {"cs": CS_GMM_PAIRS, "ds": DS_GMM_PAIRS, "dr": DR_GMM_PAIRS}[mode]
        self.inputs = [self.pristine.clone() for _ in range(pairs)]
        self.groups = torch.full(
            (EXPERTS,), GMM_GROUP_ROWS, dtype=torch.int64, device="npu"
        )
        self.index = torch.zeros(1, dtype=torch.int64, device="npu")
        self.scale = torch.full((BATCH,), GMM_SCALE, device="npu")
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for activation in self.inputs:
                self.pair(activation)
        torch.npu.synchronize()
        self.restore()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            for activation in self.inputs:
                self.retained.extend(self.pair(activation))
        torch.npu.synchronize()

    def weight(self, k: int, n: int, scale_3d: bool):
        packed = torch.randint(-128, 128, (EXPERTS, k, n // 2), dtype=torch.int8)
        scales = (
            (torch.rand((EXPERTS, n)) * WEIGHT_SCALE_RANGE + WEIGHT_SCALE_MIN)
            .half()
            .float()
        )
        sums = []
        for expert in packed:
            signed = expert.to(torch.int16)
            low = ((signed & 15) + INT4_COMPENSATION) % 16 - INT4_COMPENSATION
            high = signed >> 4
            sums.append(torch.stack((low.sum(0), high.sum(0)), dim=-1).reshape(n))
        offset = (INT4_COMPENSATION * torch.stack(sums).float() * scales).npu()
        encoded = scales.view(torch.int32).to(torch.int64)
        if scale_3d:
            encoded = encoded.unsqueeze(1)
        nz = torch_npu.npu_format_cast(packed.npu(), 29)
        self.retained.append(nz)
        return nz.view(torch.int32).contiguous(), encoded.npu(), offset

    def pair(self, activation):
        gated, scales = torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
            activation,
            [self.w13],
            [self.s13],
            [self.a13],
            self.scale,
            self.groups,
            self.index,
            dequant_mode=0,
            quant_mode=0,
            group_list_type=1,
        )
        output = torch.ops.afd_ascend.grouped_matmul_layered(
            [gated],
            [self.w2],
            [self.a2],
            [self.s2],
            self.index,
            self.groups,
            per_token_scale=scales,
            group_list_type=1,
            split_item=3,
            output_dtype=torch.bfloat16,
        )[0]
        return gated, scales, output

    def restore(self):
        with torch.npu.stream(self.stream):
            for activation in self.inputs:
                activation.copy_(self.pristine)
        torch.npu.synchronize()

    def replay(self):
        with torch.npu.stream(self.stream):
            self.graph.replay()


def timed_replay(fixture: Fixture, background: Background | None):
    count = fixture.case["fan_in"] if fixture.mode == "dr" else 1
    events = [
        (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
        for _ in range(count)
    ]
    deferred = fixture.mode == "dr" and fixture.case["readback"] == "deferred"
    snapshots = (
        [torch.empty_like(fixture.graph_outputs[0][2]) for _ in range(count)]
        if deferred
        else []
    )
    # Snapshot allocation/event construction precede background submission.
    seen, metadata = [], []
    if background:
        background.replay()
    for index, (start, end) in enumerate(events):
        with torch.npu.stream(fixture.stream):
            start.record(fixture.stream)
            fixture.graph.replay()
            end.record(fixture.stream)
            if deferred:
                snapshots[index].copy_(fixture.graph_outputs[0][2])
        if not deferred:
            fixture.stream.synchronize()
            if fixture.mode == "dr":
                metadata.append(fixture.graph_outputs[0][2].cpu())
    torch.npu.synchronize()
    elapsed = [
        start.elapsed_time(end) * 1000 / fixture.graph_calls for start, end in events
    ]
    if any(not math.isfinite(value) or value <= 0 for value in elapsed):
        raise ValueError(f"Invalid NPU event times: {elapsed}")
    if fixture.mode == "dr":
        if deferred:
            metadata = [snapshot.cpu() for snapshot in snapshots]
        seen = [fixture.validate_metadata(info) for info in metadata]
        if sorted(seen) != list(range(count)):
            raise AssertionError(f"Timed DR consumed incorrect sources: {seen}")
        # Full payload checks are outside the ready-group timing. Setup/warmup
        # validate every source; the last replay is additionally checked here.
        fixture.validate_payload(fixture.graph_outputs[0], seen[-1])
    return elapsed, seen


def profile_context(case, measured: bool, directory: Path):
    if case["profile"] == "none" or not measured:
        return nullcontext()
    metrics = (
        torch_npu.profiler.AiCMetrics.PipeUtilization
        if case["profile"] == "pipe"
        else torch_npu.profiler.AiCMetrics.L2Cache
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=metrics,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    rank = args.rank
    vendor = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    helper_manifest = vendor / "issue487_helper_manifest.json"
    is_target_helper = case["mode"] == "cs" and rank < TARGETS
    if is_target_helper:
        if (
            json.loads(helper_manifest.read_text())["purpose"]
            != "issue487_cs_target_only"
        ):
            raise ValueError("Invalid CS target helper")
    elif helper_manifest.exists():
        raise ValueError("A timed rank must never load the CS target helper vendor")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.extension.resolve()))
    # Resolve all required registrations before any participant enters MC2.
    _ = (
        torch.ops.afd_ascend.afd_async_combine_send,
        torch.ops.afd_ascend.afd_async_dispatch_send,
        torch.ops.afd_ascend.afd_async_dispatch_recv,
    )
    if case["gmm"]:
        _ = (
            torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered,
            torch.ops.afd_ascend.grouped_matmul_layered,
        )
    report = {
        "case": case,
        "rank": rank,
        "correctness": False,
        "elapsed_us": [],
        "consumed_sources": [],
        "environment": environment_report(args.extension),
    }
    (args.case.parent / f"environment-rank{rank}.json").write_text(
        json.dumps(report["environment"], indent=2) + "\n"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{case['master_addr']}:{case['master_port']}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=case["group_timeout"]),
    )
    plans = [None] * WORLD
    dist.all_gather_object(plans, case["plan_sha256"])
    if len(set(plans)) != 1:
        raise ValueError("S/R plans differ; use identical matrix/rendezvous parameters")
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": 512}
    group = dist.new_group(
        list(range(WORLD)),
        backend="hccl",
        pg_options=options,
        timeout=timedelta(seconds=case["group_timeout"]),
    )
    comm_name = str(group._get_backend(torch.device("npu")).get_hccl_comm_name(rank))
    comm_stream, compute_stream = torch.npu.Stream(), torch.npu.Stream()
    torch.npu.set_stream_limit(comm_stream, vector_num=case["aiv"])
    torch.npu.set_stream_limit(
        compute_stream, cube_num=case["gmm_aic"], vector_num=2 * case["gmm_aic"]
    )
    with torch.inference_mode():
        fixture = Fixture(case, rank, comm_name, comm_stream)
        torch.npu.synchronize()
        print(f"rank {rank}: initialize all ten MC2 participants", flush=True)
        fixture.initialize()
        fixture.capture()
        background = (
            Background(case["mode"], compute_stream)
            if case["gmm"] and fixture.measured
            else None
        )
        dist.barrier()
        print(f"rank {rank}: measurement start", flush=True)
        with profile_context(
            case, fixture.measured, args.case.parent / f"profile-rank{rank}"
        ):
            for round_index in range(case["rounds"]):
                samples = case["samples"] if case["mode"] == "dr" else 1
                for sample in range(samples):
                    if background:
                        background.restore()
                    if case["mode"] == "dr":
                        fixture.publish(case["fan_in"])
                    else:
                        if case["mode"] == "cs" and not fixture.sender:
                            with torch.npu.stream(comm_stream):
                                fixture.helper(0)
                        torch.npu.synchronize()
                        dist.barrier()
                    # Remote targets are idle during CS/DS; senders are idle
                    # during DR. No host publication delay is inside the events.
                    if fixture.measured:
                        with torch.profiler.record_function(
                            f"issue487-r{round_index}-s{sample}"
                        ):
                            elapsed, sources = timed_replay(fixture, background)
                        report["elapsed_us"].extend(elapsed)
                        report["consumed_sources"].append(sources)
                    torch.npu.synchronize()
                    dist.barrier()
                    if not fixture.sender:
                        if case["mode"] == "cs":
                            fixture.validate_cs(case["fan_in"])
                        elif case["mode"] == "ds":
                            fixture.drain(case["fan_in"])
                    torch.npu.synchronize()
                    dist.barrier()
    report["correctness"] = True
    destination = args.case.parent / f"rank{rank}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"rank {rank}: PASS; {destination}", flush=True)
    dist.barrier()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
