# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""How a pod learns who it is, and whether its node is clean.

Kept free of the rendezvous (and therefore of torch) so the whole of it stays
unit-testable without a cluster or an accelerator.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable
from pathlib import Path

POD_INDEX_ENV = "AFD_E2E_POD_INDEX"
JOB_COMPLETION_INDEX_ENV = "JOB_COMPLETION_INDEX"
POD_ADDRESSES_ENV = "AFD_E2E_POD_ADDRESSES"
POD_IP_ENV = "POD_IP"
HOSTNAME_ENV = "HOSTNAME"
E2E_RUN_ID_ENV = "AFD_E2E_RUN_ID"
PROC_ROOT = Path("/proc")


def resolve_pod_index(
    num_pods: int,
    *,
    environment: os._Environ[str] | dict[str, str] | None = None,
) -> int:
    """Resolve this pod's index from the environment, in precedence order.

    An explicit override wins, then the Indexed Job's completion index, then the
    trailing integer of the hostname for hand-applied manifests.
    """
    environment = os.environ if environment is None else environment
    for name in (POD_INDEX_ENV, JOB_COMPLETION_INDEX_ENV):
        raw_value = environment.get(name)
        if raw_value:
            return _checked_index(int(raw_value), num_pods, name)
    hostname = environment.get(HOSTNAME_ENV, "")
    trailing = hostname.rsplit("-", 1)[-1]
    if trailing.isdecimal():
        return _checked_index(int(trailing), num_pods, f"HOSTNAME={hostname}")
    raise RuntimeError(
        f"cannot resolve this pod's index: set {POD_INDEX_ENV}, run under an "
        f"Indexed Job ({JOB_COMPLETION_INDEX_ENV}), or use a hostname ending "
        f"in its index (HOSTNAME={hostname!r})",
    )


def _checked_index(index: int, num_pods: int, source: str) -> int:
    if not 0 <= index < num_pods:
        raise RuntimeError(
            f"pod index {index} from {source} is outside 0..{num_pods - 1}",
        )
    return index


def local_address(
    *,
    environment: os._Environ[str] | dict[str, str] | None = None,
) -> str:
    """This pod's own reachable address."""
    environment = os.environ if environment is None else environment
    pod_ip = environment.get(POD_IP_ENV)
    if pod_ip:
        return pod_ip
    return socket.gethostbyname(socket.gethostname())


def _resolve_host(host: str) -> object:
    """Resolve a hostname, ignoring the port the resolver also wants."""
    return socket.getaddrinfo(host, None)


def wait_for_address(
    host: str,
    timeout_s: float,
    *,
    poll_interval_s: float = 2.0,
    resolve: Callable[[str], object] | None = None,
) -> None:
    """Block until ``host`` resolves.

    A pod's DNS record can lag its own start by a few seconds. The rendezvous
    store client resolves once and fails outright rather than retrying, so
    without this a slow DNS publish looks like a store outage.
    """
    resolve = _resolve_host if resolve is None else resolve
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while True:
        try:
            resolve(host)
            return
        except OSError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{host} did not resolve within {timeout_s:.0f}s: {last_error!r}",
            )
        time.sleep(poll_interval_s)


def find_stale_run_markers(
    run_id: str,
    *,
    proc_root: Path = PROC_ROOT,
) -> list[str]:
    """Report local processes carrying a *different* run's E2E marker.

    A leftover process holds devices and ports; without this pre-flight the
    symptom is an unexplained rendezvous hang rather than a named failure.
    """
    marker = f"{E2E_RUN_ID_ENV}=".encode()
    survivors: list[str] = []
    if not proc_root.is_dir():
        return survivors
    for entry in sorted(proc_root.iterdir(), key=lambda path: path.name):
        if not entry.name.isdecimal():
            continue
        try:
            environment = (entry / "environ").read_bytes()
        except OSError:
            continue
        for item in environment.split(b"\0"):
            if not item.startswith(marker):
                continue
            value = item[len(marker) :].decode(errors="replace")
            if not value.startswith(run_id):
                survivors.append(f"pid {entry.name}: {E2E_RUN_ID_ENV}={value}")
    return survivors


def parse_key_values(values: list[str], *, option: str) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options into a mapping."""
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not name:
            raise ValueError(f"{option} expects KEY=VALUE, got {value!r}")
        parsed[name] = content
    return parsed
