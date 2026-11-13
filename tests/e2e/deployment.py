# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Case deployment: protocol order, role commands, readiness and owned cleanup."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.case import (
    ATTENTION_ROLE,
    CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    CLEANUP_PROFILE_STRICT,
    FFN_ROLE,
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
DECODE_GRAPH_COMPILATION_CONFIG = {"cudagraph_mode": "FULL_DECODE_ONLY"}
# The standalone FFN service's torchrun (F-only) world listens next to the AFD
# rendezvous port; its rank r management endpoint is the FFN API port + r.
FFN_SERVICE_MASTER_PORT_OFFSET = 1
FFN_SERVICE_STATUS_FILE = "ffn-service.json"
FFN_SERVICE_HTTP_TIMEOUT_S = 5
FFN_SERVICE_PROCESS_KILL_TIMEOUT_S = 120
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
        if role == FFN_ROLE and profile.ffn_service is not None:
            # The service has no vLLM Worker to host the evidence extension.
            return self.ffn_service_command()
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
            cmd.extend(
                [
                    "--enable-ep-weight-filter",
                    "--additional-config",
                    json.dumps(
                        {
                            "afd": self.afd_config(role, slot),
                            **dict(layout.additional_config),
                        },
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
                    json.dumps(DECODE_GRAPH_COMPILATION_CONFIG, separators=(",", ":")),
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

    def afd_config(self, role: str, slot: RoleSlot | None) -> dict[str, JsonValue]:
        """The ``additional_config["afd"]`` one role's service starts with."""
        profile = self.case.deployment
        assert profile.ffn is not None
        env = self.environment
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
        if profile.ffn_service is not None:
            afd["afd_process_group_timeout_s"] = profile.ffn_service.afd_timeout_s
        return afd

    def ffn_service_command(self) -> list[str]:
        """``torchrun`` command of the standalone FFN service on this host.

        One process per FFN rank. The AFD configuration, execution mode and
        DBO match the FFN half of the ``vllm serve`` deployment, so the
        unchanged Attention command pairs with it. Request-shaping arguments
        (``common_vllm_args``, batch and capture budgets) belong to Attention,
        whose control payloads drive every FFN step and graph capture.
        """
        profile = self.case.deployment
        service, ffn = profile.ffn_service, profile.ffn
        assert service is not None and ffn is not None
        env = self.environment
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={ffn.rank_count}",
            "--nnodes=1",
            f"--master-addr={env.afd_host}",
            f"--master-port={env.afd_port + FFN_SERVICE_MASTER_PORT_OFFSET}",
            "-m",
            "afd_plugin.ffn_faas",
            "serve",
            env.model,
            "--backend",
            service.backend,
            "--data-parallel-size",
            str(ffn.dp),
            "--tensor-parallel-size",
            str(ffn.tp),
            "--enable-expert-parallel",
            "--additional-config",
            json.dumps(
                {
                    "afd": self.afd_config(FFN_ROLE, None),
                    **dict(ffn.additional_config),
                },
                separators=(",", ":"),
            ),
            "--receiver-shutdown-timeout",
            str(service.receiver_shutdown_timeout_s),
            "--status-host",
            env.api_host,
            "--status-port",
            str(self.ffn_service_status_port(0)),
            "--status-file",
            str(self.ffn_service_status_file),
        ]
        if ffn.uses_graph:
            cmd.extend(
                [
                    "--compilation-config",
                    json.dumps(DECODE_GRAPH_COMPILATION_CONFIG, separators=(",", ":")),
                ]
            )
        else:
            cmd.append("--enforce-eager")
        cmd.extend(dbo_arguments(profile.dbo_for_role(FFN_ROLE)))
        cmd.extend(ffn.vllm_args)
        return cmd

    def ffn_service_status_port(self, rank: int) -> int:
        """Management port of one FFN service rank (the FFN API port + rank)."""
        return self.environment.api_port_base + 1 + rank

    @property
    def ffn_service_status_file(self) -> Path:
        """Base status file; rank r writes ``ffn-service-rank-<r>.json``."""
        return self.environment.output_path / FFN_SERVICE_STATUS_FILE


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
    if profile.ffn_service is not None and (profile.ffn is None or slots):
        raise ValueError("the FFN service runs a local FFN role on one host")
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
        if self.resolved.case.deployment.ffn_service is not None:
            self.wait_ffn_service_ready()
        self.wait_ready()

    def wait_ffn_service_ready(self) -> None:
        """Wait until every FFN service rank reports ``/ready`` (paired with A).

        ``/ready`` means paired and receiving; CUDA graphs are captured later,
        driven by Attention, whose API comes up only after that.
        """
        profile = self.resolved.case.deployment
        assert profile.ffn is not None
        env = self.resolved.environment
        pending = set(range(profile.ffn.rank_count))
        deadline = time.monotonic() + env.startup_timeout
        last_error: BaseException | None = None
        while pending and time.monotonic() < deadline:
            self.ensure_alive()
            for rank in sorted(pending):
                try:
                    with urllib.request.urlopen(
                        self.ffn_service_url(rank, "/ready"),
                        timeout=FFN_SERVICE_HTTP_TIMEOUT_S,
                    ) as response:
                        if response.status == 200:
                            pending.discard(rank)
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
            if pending:
                time.sleep(READY_POLL_INTERVAL_S)
        if pending:
            raise TimeoutError(
                f"FFN service ranks {sorted(pending)} were not ready: {last_error}"
            )

    def ffn_service_url(self, rank: int, path: str) -> str:
        env = self.resolved.environment
        port = self.resolved.ffn_service_status_port(rank)
        return f"http://{env.api_host}:{port}{path}"

    def finish(self) -> None:
        """Stop a deployment whose protocol defines a clean stop; else no-op.

        The standalone FFN service stops idle: ``/stop`` on every rank cancels
        its idle control receive, then Attention shuts down right away, since
        releasing DeepEP buffers is a barrier over every M2N rank. The service
        launcher must exit 0 and every rank must report ``stopped`` with
        released resources. Other deployments stop in ``close``.
        """
        profile = self.resolved.case.deployment
        service = profile.ffn_service
        if service is None:
            return
        assert profile.ffn is not None
        for rank in range(profile.ffn.rank_count):
            request = urllib.request.Request(
                self.ffn_service_url(rank, "/stop"), method="POST"
            )
            with urllib.request.urlopen(request, timeout=FFN_SERVICE_HTTP_TIMEOUT_S):
                pass
        failures = terminate_process_groups(
            [self.processes_by_role[ATTENTION_ROLE]],
            termination_timeout_s=profile.termination_timeout_s,
            poll_interval_s=PROCESS_POLL_INTERVAL_S,
            reap_timeout_s=PROCESS_REAP_TIMEOUT_S,
        )
        if failures:
            raise RuntimeError("; ".join(failures))
        returncode = self.processes_by_role[FFN_ROLE].wait(
            timeout=service.stop_timeout_s
        )
        if returncode != 0:
            raise RuntimeError(f"FFN service exited with {returncode}")
        status_file = self.resolved.ffn_service_status_file
        for rank in range(profile.ffn.rank_count):
            rank_file = status_file.with_name(f"{status_file.stem}-rank-{rank}.json")
            status = json.loads(rank_file.read_text())
            if status["phase"] != "stopped" or status["resources_released"] is not True:
                raise RuntimeError(
                    f"FFN service rank {rank} did not stop cleanly: {status}"
                )

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
        # torchrun starts the FFN service workers outside the launcher's process
        # group, so the service also needs the environment-matched sweep.
        ffn_service = self.resolved.case.deployment.ffn_service is not None
        if legacy_async or ffn_service:
            failures.extend(
                kill_processes_matching_environment(
                    {"AFD_E2E_RUN_ID": self.run_id, "AFD_E2E_PROCESS_ROLE": "ffn"},
                    timeout_s=FFN_SERVICE_PROCESS_KILL_TIMEOUT_S
                    if ffn_service
                    else NPU_ASYNC_PROCESS_KILL_TIMEOUT_S,
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
