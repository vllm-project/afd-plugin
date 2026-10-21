# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in unmodified CAM communication and numerical regression probes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.npu.async_cam_common import HarnessConfig, NativeHarness

pytestmark = pytest.mark.npu


def test_profile_wire_roundtrip(tmp_path):
    world_size = 4
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip(
            "requires AFD_RUN_ASCEND_OP_RUNTIME=1 and allocated Ascend 910C devices"
        )
    if os.environ.get("SOC_VERSION", "").lower() != "910c":
        pytest.skip("native CAM is built only for SOC_VERSION=910c (ascend910_93)")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if len([device for device in visible.split(",") if device.strip()]) < world_size:
        pytest.skip(f"requires at least {world_size} devices allocated with npu run")
    pytest.importorskip("torch_npu")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tests.npu.test_cam_profile_wire",
        "--probe",
        str(tmp_path),
    ]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible.split(",")[:world_size])
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate(timeout=5)
        (tmp_path / "wire.log").write_text(stdout)
        pytest.fail(
            f"CAM wire subprocess exceeded 240 seconds; see {tmp_path / 'wire.log'}"
        )
    (tmp_path / "wire.log").write_text(stdout)
    assert process.returncode == 0, stdout
    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    if any(report["status"] == "unsupported_soc" for report in reports):
        pytest.skip(
            "; ".join(
                report["reason"]
                for report in reports
                if report["status"] == "unsupported_soc"
            )
        )
    assert all(report["status"] == "passed" for report in reports)


def run_probe() -> dict:
    config = HarnessConfig(
        attn_ranks=2,
        moe_ranks=2,
        tp_size=2,
        lengths=(8, 8),
        dtype="bf16",
        dynamic_quant=1,
        routing="sparse",
        capacity=16,
        max_seq_len=16,
        timeout_seconds=60,
    )
    os.environ["HCCL_BUFFSIZE"] = str(config.window_mb)
    os.environ["BATCH_SIZE_FACTOR"] = str(config.capacity / 262144)
    import torch
    import torch_npu  # noqa: F401

    device_name = torch.npu.get_device_name(int(os.environ["LOCAL_RANK"]))
    if not device_name.startswith(("Ascend910_93", "Ascend910C")):
        return {
            "status": "unsupported_soc",
            "reason": f"native CAM requires ascend910_93/910C, found {device_name}",
        }
    from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

    ensure_cam_async_ops_available()
    saved_batches = []
    rounds = []
    with NativeHarness(config) as harness:
        dispatch_recv = harness.operations["dispatch_recv"]

        def profile_recv(*args):
            result = dispatch_recv(*args)
            assert len(result) == 4
            info = result[2]
            assert info.dtype == torch.int64
            assert tuple(info.shape) == (
                5 + config.tp_size * (1 + config.experts_per_rank),
            )
            saved_batches.append((info, info.cpu().tolist()))
            return result

        harness.operations["dispatch_recv"] = profile_recv
        for layer_id in (1, 7, 1):
            report = harness.run_round(layer_id)
            assert not report["failures"], report["failures"]
            rounds.append(report)
        for batch, expected in saved_batches:
            assert batch.cpu().tolist() == expected, (
                "batch metadata changed after window reuse"
            )
        if harness.rank >= config.attn_ranks:
            assert len(saved_batches) == sum(report["chunks"] for report in rounds)
            assert len({batch.data_ptr() for batch, _ in saved_batches}) == len(
                saved_batches
            )
        gathered = harness.gather_reports({"rank": harness.rank, "rounds": rounds})
        if config.routing == "sparse" and config.moe_ranks == 2:
            empty_owner = gathered[-1]["rounds"]
            assert all(
                report["rows"] == 0 and report["chunks"] == 1 for report in empty_owner
            )
        if config.routing == "sparse" or config.moe_ranks == 1:
            assert all(
                report["chunks"] >= 2
                for report in gathered[config.attn_ranks]["rounds"]
            )
        return {
            "status": "passed",
            "environment": harness.environment(),
            "metadata_count": len(saved_batches),
            "ranks": gathered,
        }


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe":
        raise SystemExit("usage: test_cam_profile_wire.py --probe OUTPUT")
    watchdog = threading.Timer(180, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        result = run_probe()
        path = Path(sys.argv[2]) / f"rank-{os.environ['RANK']}.json"
        path.write_text(json.dumps(result, indent=2))
    finally:
        watchdog.cancel()
