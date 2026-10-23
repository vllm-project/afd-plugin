# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""torchrun --nproc-per-node F -m afd_plugin.ffn_faas serve MODEL ..."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from afd_plugin.ffn_faas.runner import FFNModelRunner, WorkerConfig
from afd_plugin.ffn_faas.service import FFNService
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--attention-tp", type=int, required=True)
    parser.add_argument("--ffn-ranks", type=int, required=True)
    parser.add_argument("--devices", help="comma-separated allocated logical NPU IDs")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--window-mb", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=29700)
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()
    rank, local_rank, world = (
        int(os.environ.get(key, default))
        for key, default in (
            ("RANK", "0"),
            ("LOCAL_RANK", "0"),
            ("WORLD_SIZE", "1"),
        )
    )
    if world != args.ffn_ranks:
        parser.error("external launcher WORLD_SIZE must equal --ffn-ranks")
    devices = (
        tuple(map(int, args.devices.split(",")))
        if args.devices
        else tuple(range(world))
    )
    if len(devices) <= local_rank or len(set(devices)) != len(devices):
        parser.error("--devices must supply distinct allocated devices for local ranks")
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("timeouts must be positive")
    checkpoint = V4CheckpointDescription.read(args.model)
    spec = checkpoint.session_spec(
        attention_ranks=args.attention_ranks,
        attention_tp=args.attention_tp,
        ffn_ranks=args.ffn_ranks,
        capacity=args.capacity,
        slots=args.slots,
    )
    service = FFNService(rank=rank, model_identity=checkpoint.identity)
    address = service.start_http(args.status_host, args.status_port + local_rank)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda signum, frame: service.request_signal_stop())
    print(
        json.dumps({"rank": rank, "management": address, "session": spec.fingerprint}),
        flush=True,
    )
    unsafe = False
    code = 0
    # Keep owned weights/windows alive through reporting and the final process
    # exit, including when run() cannot prove device/peer quiescence.
    runner = FFNModelRunner(
        service,
        checkpoint,
        spec,
        WorkerConfig(
            devices[local_rank],
            rank,
            args.host,
            args.port,
            args.timeout,
            args.startup_timeout,
            args.window_mb,
        ),
    )
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
        code = 1
        unsafe = not service.snapshot()["resources_released"]
    finally:
        status = json.dumps(service.snapshot(), indent=2) + "\n"
        print(status, flush=True)
        if args.status_file is not None:
            args.status_file.parent.mkdir(parents=True, exist_ok=True)
            args.status_file.with_name(
                f"{args.status_file.stem}-rank-{rank}.json"
            ).write_text(status)
        service.close_http()
    if unsafe:
        # Never run HCCL/Python destructors against possibly live remote
        # accesses. Process isolation is the final cleanup boundary on failure.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
