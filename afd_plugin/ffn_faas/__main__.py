# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""External-launcher entry point for a standalone FFN platform adapter."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from contextlib import suppress
from pathlib import Path

from afd_plugin.ffn_faas.platforms import SUPPORTED_BACKENDS, get_platform_adapter
from afd_plugin.ffn_faas.runner import FFNModelRunner
from afd_plugin.ffn_faas.service import FFNService


def _report_exception() -> None:
    """Diagnostics must not choose how an unsafe native session exits."""
    with suppress(BaseException):
        traceback.print_exc()


def main() -> int:
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="ascend-cam"
    )
    selected, _ = selection.parse_known_args()
    adapter = get_platform_adapter(selected.backend)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="ascend-cam")
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=29700)
    parser.add_argument("--status-file", type=Path)
    adapter.add_arguments(parser)
    args = parser.parse_args()
    try:
        launch = adapter.configure(args)
        status_port = args.status_port + launch.local_rank if args.status_port else 0
        if not 0 <= status_port < 65536:
            raise ValueError("invalid per-rank management port")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    service = FFNService(
        rank=launch.rank,
        model_path=launch.model_path,
        backend_name=launch.backend_name,
        initial_status=launch.initial_status,
    )
    address = service.start_http(args.status_host, status_port)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda signum, frame: service.request_signal_stop())
    print(
        json.dumps(
            {"rank": launch.rank, "management": address, "backend": launch.backend_name}
        ),
        flush=True,
    )
    runner = FFNModelRunner(
        service, launch.backend_factory, shutdown_timeout=launch.shutdown_timeout
    )
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
                    f"{args.status_file.stem}-rank-{launch.rank}.json"
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
