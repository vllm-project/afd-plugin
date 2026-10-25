# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""torchrun -m afd_plugin.ffn_faas serve MODEL: independent F, existing CAM."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from contextlib import suppress
from pathlib import Path

from afd_plugin.ffn_faas.runner import FFNModelRunner
from afd_plugin.ffn_faas.service import FFNService
from afd_plugin.ffn_faas.settings import FFNConfig
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription


def _report_exception() -> None:
    """Diagnostics must not choose how an unsafe native session exits."""
    with suppress(BaseException):
        traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--attention-tp", type=int, required=True)
    parser.add_argument("--ffn-ranks", type=int, required=True)
    parser.add_argument("--devices", help="allocated logical NPU IDs on this node")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--full-graph", action="store_true")
    parser.add_argument("--ffn-two-stream", action="store_true")
    parser.add_argument("--ffn-compute-aic-num", type=int)
    parser.add_argument("--ffn-compute-aiv-num", type=int)
    parser.add_argument("--ffn-combine-aiv-num", type=int)
    parser.add_argument("--hccl-buffer-size", type=int)
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=29700)
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()
    try:
        rank, local_rank, world = (
            int(os.environ.get(key, default))
            for key, default in (
                ("RANK", "0"),
                ("LOCAL_RANK", "0"),
                ("WORLD_SIZE", "1"),
            )
        )
        if world != args.ffn_ranks:
            raise ValueError("F-only launcher WORLD_SIZE must equal --ffn-ranks")
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        devices = (
            tuple(map(int, args.devices.split(",")))
            if args.devices
            else tuple(range(local_world))
        )
        if (
            not 0 <= local_rank < len(devices)
            or len(set(devices)) != len(devices)
            or min(devices) < 0
        ):
            raise ValueError("--devices must supply distinct local NPU IDs")
        status_port = args.status_port + local_rank if args.status_port else 0
        if not 0 <= status_port < 65536:
            raise ValueError("invalid per-rank management port")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
            raise ValueError("set VLLM_USE_V2_MODEL_RUNNER=1 on both A and F")
        if os.environ.get("AFD_ASYNC_CAM_LAYERED_GMM") != "1":
            raise ValueError("standalone V4 requires AFD_ASYNC_CAM_LAYERED_GMM=1")
        settings = FFNConfig(
            model=args.model,
            attention_ranks=args.attention_ranks,
            attention_tp=args.attention_tp,
            ffn_ranks=args.ffn_ranks,
            rank=rank,
            local_rank=local_rank,
            device=devices[local_rank],
            host=args.host,
            port=args.port,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            full_graph=args.full_graph,
            ffn_two_stream=args.ffn_two_stream,
            ffn_compute_aic_num=args.ffn_compute_aic_num,
            ffn_compute_aiv_num=args.ffn_compute_aiv_num,
            ffn_combine_aiv_num=args.ffn_combine_aiv_num,
            hccl_buffer_size=args.hccl_buffer_size,
            shutdown_timeout=args.shutdown_timeout,
        )
        checkpoint = V4CheckpointDescription.read(args.model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    service = FFNService(rank=rank, model_path=str(checkpoint.root))
    address = service.start_http(args.status_host, status_port)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda signum, frame: service.request_signal_stop())
    print(
        json.dumps({"rank": rank, "management": address, "backend": "CAM"}), flush=True
    )
    runner = FFNModelRunner(service, checkpoint, settings)
    code = 0
    resources_released = False
    try:
        try:
            runner.run()
        except BaseException:
            code = 1
            _report_exception()
        try:
            snapshot = service.snapshot()
            # Capture the release proof before fallible reporting. If querying
            # it fails, retain the conservative default through finalization.
            resources_released = snapshot["resources_released"] is True
            status = json.dumps(snapshot, indent=2) + "\n"
            print(status, flush=True)
            if args.status_file is not None:
                args.status_file.parent.mkdir(parents=True, exist_ok=True)
                args.status_file.with_name(
                    f"{args.status_file.stem}-rank-{rank}.json"
                ).write_text(status)
        except BaseException:
            code = 1
            _report_exception()
        finally:
            try:
                service.close_http()
            except BaseException:
                code = 1
                _report_exception()
    finally:
        if not resources_released:
            # Keep runner/backend owners reachable until process isolation,
            # even if status writes, tracebacks, HTTP close or flush fail.
            try:
                for stream in (sys.stdout, sys.stderr):
                    with suppress(BaseException):
                        stream.flush()
            finally:
                os._exit(code or 1)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
