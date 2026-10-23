# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real Attention clients of separately launched, full-checkpoint FFN services.

Normal runs compare changing routes/layers and repeatedly reuse transport slots.
Fault runs wait for a test driver to stop/kill F after paired readiness, and
must observe a failure rather than return a successful inference result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from afd_plugin.ffn_faas.client import AttentionSession
from afd_plugin.ffn_faas.model import CheckpointDescription
from afd_plugin.ffn_faas.npu import (
    NPUExecutor,
    SessionPeerFailureError,
    UnsafeSessionCleanupError,
)
from tests.npu.ffn_faas_native import (
    ATOL,
    MAX_NORMALIZED_RMSE,
    RTOL,
    CheckpointOracle,
    make_work,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attention-ranks", type=int, default=4)
    parser.add_argument("--attention-tp", type=int, default=2)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=156)
    parser.add_argument("--tp-rows-aligned", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "normal",
            "idle-stop",
            "inflight-stop",
            "killed-peer",
            "lost-store",
            "signal-stop",
        ],
        default="normal",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank, device = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    checkpoint = CheckpointDescription.read(args.model)
    spec = checkpoint.session_spec(
        attention_ranks=args.attention_ranks,
        attention_tp=args.attention_tp,
        ffn_ranks=args.ffn_ranks,
        capacity=8,
        slots=2,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "mode": args.mode,
        "spec": json.loads(spec.to_json()),
        "comparisons": [],
        "memory_samples": [],
    }
    executor = NPUExecutor(
        spec,
        rank=rank,
        device=device,
        host="127.0.0.1",
        port=args.port,
        timeout=10,
        startup_timeout=120,
        auto_initialize=False,
    )
    session = AttentionSession(executor, tp_rows_aligned=args.tp_rows_aligned)
    oracle = CheckpointOracle(checkpoint)
    unsafe = False
    started = time.monotonic()
    failed = False
    armed = False
    try:
        executor.initialize()
        session.warmup()
        if args.mode != "normal":
            if args.mode == "inflight-stop":
                work = make_work(spec, rank, 1)
                session.send(
                    work.device_x,
                    work.device_ids,
                    work.device_probabilities,
                    layer=work.layer,
                )
            armed_file = args.output_dir / f"armed-{rank}"
            temporary = armed_file.with_suffix(".pending")
            temporary.write_text(json.dumps({"pid": os.getpid(), "rank": rank}) + "\n")
            temporary.replace(armed_file)
            armed = True
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                executor.check_peers()
                time.sleep(0.01)
            raise TimeoutError("test driver did not trigger a detectable FFN failure")
        for index in range(1, args.rounds + 1):
            work = make_work(spec, rank, index)
            shape_case = index % 26
            if args.tp_rows_aligned:
                count = (0, 1, 8, 9, 16, 17)[(index + rank // spec.attention_tp) % 6]
                generator = torch.Generator().manual_seed(9100 + index * 17 + rank)
                work.x = torch.randn(
                    (count, spec.hidden_size), generator=generator
                ).bfloat16()
                work.ids = torch.randint(
                    spec.num_experts,
                    (count, spec.top_k),
                    generator=generator,
                    dtype=torch.int32,
                )
                work.probabilities = torch.softmax(
                    torch.randn((count, spec.top_k), generator=generator), dim=-1
                )
                if index % 4 == 0:
                    work.ids.fill_(0)
                elif index % 4 == 1:
                    work.ids.fill_(spec.num_experts - 1)
                work.device_x, work.device_ids, work.device_probabilities = (
                    work.x.npu(),
                    work.ids.npu(),
                    work.probabilities.npu(),
                )
            elif shape_case in (1, 2, 3, 4, 5):
                # Repeated unequal/zero TP shapes must agree the same chunk
                # generations while the other DP progresses independently.
                lengths = {1: (11, 0), 2: (11, 3), 3: (0, 11), 4: (3, 11), 5: (0, 0)}
                count = lengths[shape_case][rank % spec.attention_tp]
                work.x = torch.full(
                    (count, spec.hidden_size), 0.125, dtype=torch.bfloat16
                )
                work.ids = torch.arange(spec.top_k, dtype=torch.int32)[None].repeat(
                    count, 1
                )
                work.probabilities = torch.full((count, spec.top_k), 1.0 / spec.top_k)
                work.device_x, work.device_ids, work.device_probabilities = (
                    work.x.npu(),
                    work.ids.npu(),
                    work.probabilities.npu(),
                )
            sequence_before = session.sequence
            session.send(
                work.device_x,
                work.device_ids,
                work.device_probabilities,
                layer=work.layer,
            )
            if rank // spec.attention_tp:
                time.sleep(0.003)  # test-only slow independent source
            actual = session.receive().cpu()
            expected = oracle.reference(work).combined
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
            denominator = (
                float(expected.float().square().mean().sqrt())
                if expected.numel()
                else 0.0
            )
            error = (
                float((actual.float() - expected.float()).square().mean().sqrt())
                if expected.numel()
                else 0.0
            )
            nrmse = error / max(denominator, 1e-12)
            assert nrmse <= MAX_NORMALIZED_RMSE, nrmse
            report["comparisons"].append(
                {
                    "index": index,
                    "layer": work.layer,
                    "tokens": len(work.x),
                    "nrmse": nrmse,
                    "transport_chunks": session.sequence - sequence_before,
                }
            )
            if index in (26, 78, args.rounds):
                report["memory_samples"].append(
                    {
                        "index": index,
                        "allocated": torch.npu.memory_allocated(device),
                        "reserved": torch.npu.memory_reserved(device),
                        "store_keys": int(executor.store.num_keys()),
                    }
                )
        if args.tp_rows_aligned:
            report["shape_cells_created"] = sum(
                executor.store.check([f"chunk-shape/{peer}"])
                for peer in range(spec.attention_ranks)
            )
            assert report["shape_cells_created"] == 0
        report.update(result="passed", tp_rows_aligned=args.tp_rows_aligned)
    except BaseException as error:
        failed = True
        detected = time.monotonic()
        trigger_file = args.output_dir.parent / "fault-triggered.json"
        if armed and isinstance(error, SessionPeerFailureError):
            confirmation_deadline = time.monotonic() + 2
            while (
                not trigger_file.exists() and time.monotonic() < confirmation_deadline
            ):
                time.sleep(0.01)
        trigger = (
            json.loads(trigger_file.read_text()) if trigger_file.exists() else None
        )
        executor.publish_failure(error)
        report["error"] = traceback.format_exc()
        expected_fault = (
            armed
            and trigger is not None
            and trigger["case"] == args.mode
            and isinstance(error, SessionPeerFailureError)
            and trigger["requested_monotonic"] <= detected
            and (
                args.mode == "killed-peer"
                or (args.mode == "lost-store" and "Store" in str(error))
                or "standalone FFN stop aborts" in str(error)
            )
        )
        report.update(
            result="expected_failure" if expected_fault else "failed",
            paired_ready_and_armed=armed,
            fault_trigger=trigger,
            failure_detected_monotonic=detected,
            failure_type=type(error).__name__,
        )
    finally:
        try:
            session.close(abort=failed)
            report["resources_released"] = True
        except UnsafeSessionCleanupError:
            unsafe = True
            report["resources_released"] = False
            report["cleanup_error"] = traceback.format_exc()
        oracle.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["sequences"] = session.sequence
        (args.output_dir / f"rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    code = int(report["result"] == "failed")
    if unsafe:
        os._exit(code)
    if code:
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
