# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Resource and environment parsing shared by E2E entrypoints."""

import argparse
import os
import sys
from pathlib import Path

from tests.e2e.case import (
    DEFAULT_API_PORT,
    DEFAULT_CONNECTOR_PORT,
    DEFAULT_STARTUP_TIMEOUT_S,
    CaseEnvironment,
)


def add_case_environment_arguments(
    parser: argparse.ArgumentParser,
    case_ids: tuple[str, ...],
    *,
    devices_required: bool = True,
    startup_option: str = "--startup-timeout",
) -> None:
    parser.add_argument("--scenario", choices=case_ids, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--device-backend", choices=("gpu", "npu"), required=True)
    parser.add_argument("--devices", required=devices_required, default="")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument(
        "--api-port-base",
        type=int,
        default=int(os.environ.get("AFD_E2E_API_PORT_BASE", str(DEFAULT_API_PORT))),
    )
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument(
        "--afd-port",
        type=int,
        default=int(os.environ.get("AFD_E2E_AFD_PORT", str(DEFAULT_CONNECTOR_PORT))),
    )
    parser.add_argument(
        startup_option,
        dest="startup_timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_S,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print configuration and commands without starting services",
    )


def parse_case_environment(
    case_ids: tuple[str, ...], argv: list[str] | None = None
) -> tuple[str, CaseEnvironment, bool]:
    """Read resources in the subprocess; collection never calls this function."""
    parser = argparse.ArgumentParser(description="Run a model-declared E2E Case")
    add_case_environment_arguments(parser, case_ids)
    args = parser.parse_args(argv)
    return (
        args.scenario,
        case_environment(args),
        args.describe,
    )


def case_environment(args: argparse.Namespace) -> CaseEnvironment:
    """Bind parsed resources; model and placement decisions live elsewhere."""
    return CaseEnvironment(
        model=str(Path(args.model).expanduser()),
        vllm_bin=args.vllm_bin,
        device_backend=args.device_backend,
        devices=tuple(
            value.strip() for value in args.devices.split(",") if value.strip()
        ),
        output_path=args.output_path,
        api_host=args.api_host,
        api_port_base=args.api_port_base,
        afd_host=args.afd_host,
        afd_port=args.afd_port,
        startup_timeout=args.startup_timeout,
    )


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def model_case_command(
    module: str,
    scenario: str,
    output_path: Path,
    *,
    backend: str,
    default_devices: tuple[str, ...] = ("0", "1", "2", "3"),
) -> list[str]:
    """Bind a model entry to resources without selecting its topology or checks."""
    prefix = f"AFD_{backend.upper()}_E2E"
    return [
        sys.executable,
        "-m",
        module,
        "--scenario",
        scenario,
        "--model",
        required_env(f"{prefix}_MODEL"),
        "--vllm-bin",
        os.environ.get(f"{prefix}_VLLM_BIN", "vllm"),
        "--device-backend",
        backend,
        "--devices",
        os.environ.get("AFD_E2E_DEVICES", ",".join(default_devices)),
        "--output-path",
        str(output_path),
    ]


def devices_from_env(name: str, expected_count: int) -> list[str]:
    devices = [item.strip() for item in required_env(name).split(",") if item.strip()]
    if len(devices) != expected_count:
        raise RuntimeError(f"{name} must contain exactly {expected_count} devices")
    if len(devices) != len(set(devices)):
        raise RuntimeError(f"{name} devices must be unique")
    return devices


def prepend_env_paths(env: dict[str, str], name: str, *paths: Path) -> None:
    existing = [path for path in env.get(name, "").split(os.pathsep) if path]
    env[name] = os.pathsep.join(
        dict.fromkeys([*(str(path) for path in paths), *existing]),
    )


def model_multi_pod_command(module: str, scenario: str, layout: str) -> list[str]:
    """Bind an existing GPU pod to a model-owned case; provisioning stays outside."""
    command = [
        sys.executable,
        "-m",
        module,
        "multi-pod",
        "--scenario",
        scenario,
        "--pod-layout",
        layout,
        "--run-id",
        required_env("AFD_E2E_RUN_ID"),
        "--model",
        required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "gpu",
        "--output-path",
        required_env("AFD_E2E_GSM8K_OUTPUT"),
        "--store-host",
        required_env("AFD_E2E_STORE_HOST"),
    ]
    for variable, option in (
        ("AFD_E2E_STORE_PORT", "--store-port"),
        ("AFD_E2E_DEVICES", "--devices"),
    ):
        if value := os.environ.get(variable):
            command.extend([option, value])
    for value in os.environ.get("AFD_E2E_POD_ENV", "").split(";"):
        if value.strip():
            command.extend(["--pod-env", value.strip()])
    return command
