# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Case deployment: protocol order, role commands, readiness and owned cleanup."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.case import (
    CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    CLEANUP_PROFILE_STRICT,
    WORKER_SHUTDOWN_TIMEOUT_ENV,
    Case,
    CaseEnvironment,
    DboConfig,
    Deployment,
    JsonValue,
)
from tests.e2e.multi_pod.layout import RoleSlot
from tests.e2e.process_utils import (
    kill_processes_matching_environment,
    terminate_process_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESS_POLL_INTERVAL_S = 0.2
PROCESS_REAP_TIMEOUT_S = 5
LOG_THREAD_JOIN_TIMEOUT_S = 2
READY_POLL_INTERVAL_S = 2
MRV2_EVIDENCE_PREFIX = "AFD E2E execution: "
NPU_ASYNC_PROCESS_KILL_TIMEOUT_S = 120
# Starting the first role is not waiting for its connector rendezvous or API.
CONNECTOR_PROTOCOLS = (
    ("gpu", "P2pNcclAFDConnector", ("ffn", "attention")),
    ("gpu", "DeepEPM2NAFDConnector", ("ffn", "attention")),
    ("npu", "CAMP2pAFDConnector", ("ffn", "attention")),
    ("npu", "CAMAsyncAFDConnector", ("attention", "ffn")),
)
DEPLOYMENT_ARGUMENTS = frozenset(
    (
        "--config",
        "--additional-config",
        "--served-model-name",
        "--data-parallel-size",
        "--tensor-parallel-size",
        "--data-parallel-size-local",
        "--data-parallel-start-rank",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--headless",
        "--worker-extension-cls",
        "--compilation-config",
        "--enforce-eager",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--max-cudagraph-capture-size",
        "--cudagraph-capture-sizes",
        "--enable-dbo",
        "--dbo-decode-token-threshold",
        "--dbo-prefill-token-threshold",
        "--shutdown-timeout",
        "--host",
        "--port",
    )
)


@dataclass(frozen=True)
class ResolvedDeployment:
    case: Case
    environment: CaseEnvironment
    role_devices: tuple[tuple[str, tuple[str, ...]], ...]
    launch_order: tuple[str, ...]
    slots: tuple[RoleSlot, ...] = ()

    def command(self, role: str, probe_arguments: tuple[str, ...] = ()) -> list[str]:
        profile = self.case.deployment
        layout = dict(profile.roles)[role]
        dbo = profile.dbo_for_role(role)
        env = self.environment
        slot = next((slot for slot in self.slots if slot.role == role), None)
        cmd = [
            env.vllm_bin,
            "serve",
            env.model,
            "--shutdown-timeout",
            str(profile.shutdown_timeout_s),
            "--served-model-name",
            f"{self.case.served_model_name_prefix}-{role}",
            "--data-parallel-size",
            str(layout.dp),
            "--tensor-parallel-size",
            str(layout.tp),
            "--enable-expert-parallel",
        ]
        if profile.ffn is not None:
            afd: dict[str, JsonValue] = {
                "role": role,
                "connector": profile.connector,
                "host": env.afd_host if slot is None else slot.afd_host,
                "port": env.afd_port,
                "num_attention_ranks": profile.attention.rank_count,
                "num_ffn_ranks": profile.ffn.rank_count,
            }
            if profile.connector == "CAMAsyncAFDConnector":
                afd["async"] = True
            if profile.compute_gate_on_attention:
                afd["compute_gate_on_attention"] = True
            if profile.connector_extra_config:
                afd["connector_extra_config"] = dict(profile.connector_extra_config)
            cmd.extend(
                [
                    "--enable-ep-weight-filter",
                    "--additional-config",
                    json.dumps(
                        {"afd": afd, **dict(layout.additional_config)},
                        separators=(",", ":"),
                    ),
                ]
            )
        if slot is not None and slot.spans_pods:
            cmd.extend(
                [
                    "--data-parallel-size-local",
                    str(slot.dp_size_local),
                    "--data-parallel-address",
                    slot.dp_address,
                    "--data-parallel-rpc-port",
                    str(slot.dp_rpc_port),
                ]
            )
            # A leader must omit start-rank: vLLM otherwise selects external LB
            # and rejects its headless peers.
            if slot.headless:
                cmd.extend(
                    [
                        "--data-parallel-start-rank",
                        str(slot.dp_start_rank),
                        "--headless",
                    ]
                )
        elif role == "attention" and profile.pin_attention_address:
            cmd.extend(
                [
                    "--data-parallel-address",
                    env.afd_host if slot is None else slot.dp_address,
                ]
            )
        cmd.extend(probe_arguments)
        # Native baseline emits DBO before graph flags; preserve existing CLI.
        if profile.is_baseline:
            cmd.extend(dbo_arguments(dbo))
        if profile.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(profile.max_num_seqs)])
        if profile.max_num_batched_tokens is not None:
            cmd.extend(
                ["--max-num-batched-tokens", str(profile.max_num_batched_tokens)]
            )
        if profile.compilation_config:
            cmd.extend(
                [
                    "--compilation-config",
                    json.dumps(dict(profile.compilation_config), separators=(",", ":")),
                ]
            )
        elif layout.uses_graph:
            capture_size = str(profile.cudagraph_capture_size)
            cmd.extend(
                [
                    "--max-cudagraph-capture-size",
                    capture_size,
                    "--cudagraph-capture-sizes",
                    capture_size,
                    "--compilation-config",
                    json.dumps(
                        {"cudagraph_mode": "FULL_DECODE_ONLY"}, separators=(",", ":")
                    ),
                ]
            )
        else:
            cmd.append("--enforce-eager")
        if not profile.is_baseline:
            cmd.extend(dbo_arguments(dbo))
        port = env.api_port_base + (role == "ffn")
        if slot is None or not slot.headless:
            cmd.extend(["--host", env.api_host, "--port", str(port)])
        cmd.extend(layout.vllm_args)
        cmd.extend(self.case.common_vllm_args)
        return cmd


def dbo_arguments(dbo: DboConfig | None) -> list[str]:
    if dbo is None:
        return []
    flags = ["--enable-dbo"]
    if dbo.decode_token_threshold is not None:
        flags.extend(["--dbo-decode-token-threshold", str(dbo.decode_token_threshold)])
    if dbo.prefill_token_threshold is not None:
        flags.extend(
            ["--dbo-prefill-token-threshold", str(dbo.prefill_token_threshold)]
        )
    return flags


def launch_order(profile: Deployment) -> tuple[str, ...]:
    if profile.is_baseline:
        return ("baseline",)
    for backend, connector, order in CONNECTOR_PROTOCOLS:
        if (profile.backend, profile.connector) == (backend, connector):
            return order
    raise ValueError(
        f"unsupported connector protocol: {profile.backend}/{profile.connector}"
    )


def resolve(
    case: Case, environment: CaseEnvironment, *, slots: tuple[RoleSlot, ...] = ()
) -> ResolvedDeployment:
    profile = case.deployment
    if environment.device_backend != profile.backend:
        raise ValueError(
            f"Case requires {profile.backend}, got {environment.device_backend}"
        )
    if profile.cleanup_profile not in (
        CLEANUP_PROFILE_STRICT,
        CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    ):
        raise ValueError(f"unknown cleanup profile {profile.cleanup_profile!r}")
    if profile.cleanup_profile == CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY and (
        profile.backend != "npu" or profile.connector != "CAMAsyncAFDConnector"
    ):
        raise ValueError("legacy NPU async cleanup requires the NPU async CAM protocol")
    if any(role.uses_graph for _, role in profile.roles) and (
        profile.cudagraph_capture_size is None or profile.cudagraph_capture_size < 1
    ):
        raise ValueError("graph deployment requires a positive capture size")
    arguments = (
        *case.common_vllm_args,
        *(arg for _, layout in profile.roles for arg in layout.vllm_args),
    )
    for argument in arguments:
        name = argument.split("=", 1)[0].replace("_", "-")
        if name in DEPLOYMENT_ARGUMENTS:
            raise ValueError(f"{name} must be declared by Deployment or Check")
    required = (
        sum(len(slot.devices) for slot in slots)
        if slots
        else sum(role.rank_count for _, role in profile.roles)
    )
    if len(environment.devices) < required:
        raise ValueError(
            f"Case needs {required} devices, got {len(environment.devices)}"
        )
    if slots:
        role_bindings = tuple(
            (
                slot.role,
                tuple(environment.devices[int(device)] for device in slot.devices),
            )
            for slot in slots
        )
        return ResolvedDeployment(
            case, environment, role_bindings, launch_order(profile), slots
        )
    bound = []
    offset = 0
    for role, layout in profile.roles:
        bound.append((role, environment.devices[offset : offset + layout.rank_count]))
        offset += layout.rank_count
    return ResolvedDeployment(case, environment, tuple(bound), launch_order(profile))


class ServiceDeployment:
    """The single owner of all local service processes for one deployment."""

    def __init__(
        self,
        resolved: ResolvedDeployment,
        *,
        run_id: str,
        probe_arguments: tuple[str, ...] = (),
        probe_environment: dict[str, str] | None = None,
        extra_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.resolved = resolved
        self.run_id = run_id
        self.probe_arguments = probe_arguments
        self.probe_environment = probe_environment or {}
        self.extra_environment = extra_environment or {}
        self.processes: list[subprocess.Popen[str]] = []
        self.processes_by_role: dict[str, subprocess.Popen[str]] = {}
        self.log_threads: list[threading.Thread] = []
        self.execution_evidence: list[tuple[str, str]] = []

    def start(self) -> None:
        for role in self.resolved.launch_order:
            self.start_role(role)
        self.wait_ready()

    def start_role(self, role: str) -> None:
        """Launch one local role; multi-pod owns barriers between these calls."""
        resolved = self.resolved
        env = resolved.environment
        role_devices = dict(resolved.role_devices)
        env.output_path.mkdir(parents=True, exist_ok=True)
        command = resolved.command(role, self.probe_arguments)
        visible = ",".join(role_devices[role])
        process_env = os.environ.copy()
        process_env.update(self.extra_environment)
        backend = resolved.case.deployment.backend
        process_env[
            "ASCEND_RT_VISIBLE_DEVICES" if backend == "npu" else "CUDA_VISIBLE_DEVICES"
        ] = visible
        process_env["VLLM_PLUGINS"] = (
            ("ascend" if backend == "npu" else "")
            if resolved.case.deployment.is_baseline
            else ("ascend,afd" if backend == "npu" else "afd")
        )
        process_env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        process_env["AFD_E2E_RUN_ID"] = self.run_id
        process_env["AFD_E2E_PROCESS_ROLE"] = role
        process_env["PYTHONUNBUFFERED"] = "1"
        process_env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "18000")
        worker_timeout = resolved.case.deployment.worker_shutdown_timeout_s
        if worker_timeout is not None:
            process_env.setdefault(WORKER_SHUTDOWN_TIMEOUT_ENV, str(worker_timeout))
        process_env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
        process_env.pop("AFD_E2E_HASH_INPUT_IDS", None)
        process_env.pop("AFD_E2E_HASH_REQUIRE_GRAPH", None)
        process_env.pop("AFD_E2E_NATIVE_BASELINE", None)
        process_env.pop("AFD_E2E_MRV2_EVIDENCE_START", None)
        if backend == "npu":
            process_env.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
        if resolved.case.deployment.dbo_for_role(role) is not None:
            process_env["VLLM_LOGGING_LEVEL"] = "DEBUG"
        process_env.update(self.probe_environment)
        inherited = process_env.get("PYTHONPATH", "")
        process_env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(REPO_ROOT), inherited) if path
        )
        print(f"\n=== Starting {role}: {shlex.join(command)} ===", flush=True)
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.processes.append(process)
        self.processes_by_role[role] = process
        thread = threading.Thread(target=self.stream, args=(role, process), daemon=True)
        self.log_threads.append(thread)
        thread.start()
        self.ensure_alive()

    def stream(self, role: str, process: subprocess.Popen[str]) -> None:
        path = self.resolved.environment.output_path / f"{role}.log"
        assert process.stdout is not None
        with path.open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{role}] {line}", end="", flush=True)
                if MRV2_EVIDENCE_PREFIX in line:
                    self.execution_evidence.append((role, line.strip()))

    def ensure_alive(self) -> None:
        for process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"service {process.pid} exited with {process.returncode}"
                )

    def wait_ready(
        self, *, path: str = "/v1/models", on_poll: Callable[[], None] | None = None
    ) -> None:
        env = self.resolved.environment
        url = f"http://{env.api_host}:{env.api_port_base}{path}"
        deadline = time.monotonic() + env.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            self.ensure_alive()
            if on_poll is not None:
                on_poll()
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(READY_POLL_INTERVAL_S)
        raise TimeoutError(f"service did not become ready at {url}: {last_error}")

    def close(self) -> None:
        legacy_async = (
            self.resolved.case.deployment.cleanup_profile
            == CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY
        )
        ffn = self.processes_by_role.get("ffn")
        failures = terminate_process_groups(
            self.processes,
            termination_timeout_s=self.resolved.case.deployment.termination_timeout_s,
            poll_interval_s=PROCESS_POLL_INTERVAL_S,
            reap_timeout_s=PROCESS_REAP_TIMEOUT_S,
            deferred_sigkill_pgids=(ffn.pid,)
            if legacy_async and ffn is not None
            else (),
        )
        if legacy_async:
            failures.extend(
                kill_processes_matching_environment(
                    {"AFD_E2E_RUN_ID": self.run_id, "AFD_E2E_PROCESS_ROLE": "ffn"},
                    timeout_s=NPU_ASYNC_PROCESS_KILL_TIMEOUT_S,
                    poll_interval_s=PROCESS_POLL_INTERVAL_S,
                    process_name="FFN",
                )
            )
        for thread in self.log_threads:
            thread.join(timeout=LOG_THREAD_JOIN_TIMEOUT_S)
            if thread.is_alive():
                failures.append("service log reader did not stop")
        if failures:
            raise RuntimeError("; ".join(failures))
