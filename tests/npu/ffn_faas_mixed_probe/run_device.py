# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One bounded, fail-fast mixed-core probe; no model or performance run."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tests.e2e.process_utils import terminate_process_groups

HIDDEN = 2048
TOP_K = 6
MAX_ROUTES = 96
SCRATCH_BYTES = 9_587_456
GUARD_BYTES = 64
STATUS_WORDS = 16
KERNEL_TIMEOUT_SECONDS = 20
WORKER_TIMEOUT_SECONDS = 120
POLL_SECONDS = 0.01
ATOL = 0.06
RTOL = 0.02
NRMSE_LIMIT = 0.02
# name, returned code, tokens, whether routes belong to this F
CASES = (
    ("idle", 1, 0, False),
    ("error", -1, 0, False),
    ("zero-work", 0, 0, False),
    ("zero-owned", 0, 1, False),
    ("one-token", 0, 1, True),
    ("two-token", 0, 2, True),
    ("repeat96", 0, 16, True),
    ("reuse-one", 0, 1, True),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def worker(args: argparse.Namespace) -> None:
    # Isolate all device imports and ownership in the bounded worker process.
    import torch
    import torch_npu  # noqa: F401

    report: dict = {
        "passed": False,
        "phase": "prepare",
        "library": str(args.library),
        "library_sha256": digest(args.library),
        "input_sha256": digest(args.input),
        "tiling_sha256": digest(args.tiling),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "device": args.device,
        "pairs": args.pairs,
        "cases": [],
    }
    output = args.output_dir / "result.json"

    def wait_kernel() -> None:
        event = torch.npu.Event()
        event.record()
        deadline = time.monotonic() + KERNEL_TIMEOUT_SECONDS
        while not event.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("mixed-core kernel event exceeded its deadline")
            time.sleep(POLL_SECONDS)

    try:
        payload = torch.load(args.input, map_location="cpu", weights_only=True)
        assert payload["expected_experts"].shape == (MAX_ROUTES, HIDDEN)
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        library = ctypes.CDLL(str(args.library))
        geometry = library.faas_probe_geometry_do
        geometry.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        geometry.restype = None
        compute = library.faas_probe_compute_do
        compute.argtypes = [ctypes.c_uint32, *([ctypes.c_void_p] * 10)]
        compute.restype = None
        stream = torch.npu.current_stream().npu_stream
        empty_status = torch.full((4 * args.pairs, STATUS_WORDS), -1, dtype=torch.int32)
        status = empty_status.to(device)
        report["phase"] = "geometry"
        write_report(output, report)
        geometry(args.pairs, stream, status.data_ptr())
        wait_kernel()
        observed = status.cpu()
        report["geometry"] = observed.tolist()
        for kind in range(2):
            subblocks = 1 if kind == 0 else 2
            for block in range(args.pairs * subblocks):
                row = observed[kind * args.pairs + block]
                assert row[:4].tolist() == [
                    kind,
                    block,
                    block % subblocks,
                    subblocks,
                ], row.tolist()
                assert int(row[4]) == args.pairs and int(row[5]) == 0, row.tolist()
        assert torch.all(observed[3 * args.pairs :] == -1)
        write_report(output, report)

        x = payload["x"].to(device)
        w13, w2 = (payload[name].to(device) for name in ("w13", "w2"))
        ids_cpu = torch.zeros((16, TOP_K), dtype=torch.int32)
        ids = ids_cpu.to(device)
        controls = torch.tensor([0, 0, 1], dtype=torch.int32).to(device)
        scratch = torch.full(
            (SCRATCH_BYTES + 2 * GUARD_BYTES,), 0xA5, dtype=torch.uint8
        ).to(device)
        sentinel = torch.full((MAX_ROUTES, HIDDEN), float("nan"), dtype=torch.bfloat16)
        y = sentinel.to(device)
        tiling = torch.frombuffer(
            bytearray(args.tiling.read_bytes()), dtype=torch.uint8
        )
        assert tiling.numel() == 400
        tiling = tiling.to(device)
        for name, code, tokens, owned in CASES:
            report["phase"] = name
            write_report(output, report)
            status.copy_(empty_status)
            y.copy_(sentinel)
            ids.copy_(ids_cpu if owned else torch.full_like(ids_cpu, -1))
            controls.copy_(torch.tensor([code, tokens, 1], dtype=torch.int32))
            compute(
                args.pairs,
                stream,
                x.data_ptr(),
                ids.data_ptr(),
                w13.data_ptr(),
                w2.data_ptr(),
                controls.data_ptr(),
                scratch.data_ptr() + GUARD_BYTES,
                y.data_ptr(),
                status.data_ptr(),
                tiling.data_ptr(),
            )
            wait_kernel()
            observed = status.cpu()
            report["last_status"] = observed.tolist()
            write_report(output, report)
            assert torch.all(observed[: 3 * args.pairs, 5] == (1 if code else 4))
            guarded = scratch.cpu()
            assert torch.all(guarded[:GUARD_BYTES] == 0xA5)
            assert torch.all(guarded[-GUARD_BYTES:] == 0xA5)
            case: dict = {"case": name, "code": code, "tokens": tokens, "passed": True}
            if code == 0:
                actual = y.cpu()
                expected = payload["expected_experts"][: tokens * TOP_K]
                if not owned:
                    expected = torch.zeros_like(expected)
                torch.save(actual, args.output_dir / f"{name}.pt")
                try:
                    torch.testing.assert_close(
                        actual[: tokens * TOP_K], expected, atol=ATOL, rtol=RTOL
                    )
                    difference = actual[: tokens * TOP_K].float() - expected.float()
                    nrmse = float(
                        (
                            difference.square().sum()
                            / expected.float().square().sum().clamp_min(1e-24)
                        ).sqrt()
                    )
                    assert nrmse <= NRMSE_LIMIT, nrmse
                    assert torch.isnan(actual[tokens * TOP_K :]).all()
                    case.update(compared_values=expected.numel(), nrmse=nrmse)
                    if name == "repeat96":
                        second_error = difference[16:32].square().sum()
                        second_reference = expected[16:32].float().square().sum()
                        case["second_m_tile_nrmse"] = float(
                            (second_error / second_reference.clamp_min(1e-24)).sqrt()
                        )
                except BaseException:
                    torch.save(guarded, args.output_dir / f"{name}-scratch.pt")
                    raise
            report["cases"].append(case)
            write_report(output, report)
        report.update(passed=True, phase="complete", same_scratch_all_cases=True)
        write_report(output, report)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(output, report)
        # A timed-out native launch is not safe for normal device destructors.
        if isinstance(error, TimeoutError):
            os._exit(2)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tiling", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pairs", type=int, choices=(2,), default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env={**os.environ, "PYTHONOPTIMIZE": "0"},
        )
        try:
            code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        finally:
            failures = terminate_process_groups(
                [process],
                termination_timeout_s=5,
                poll_interval_s=0.1,
                reap_timeout_s=5,
                process_name="FFN mixed probe",
            )
            write_report(
                args.output_dir / "driver.json",
                {"exit_code": process.poll(), "cleanup_failures": failures},
            )
        if code != 0 or failures:
            raise RuntimeError("mixed probe failed; inspect the preserved result/log")


if __name__ == "__main__":
    main()
