# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V2-Lite multi-pod AFD E2E cases.

This test runs as *one pod's slice* of an already-provisioned multi-pod
deployment: the k8s pods or Docker containers must already exist -- brought
up by hand, following `.agents/skills/run-e2e/resources/k8-multi-pod.md` --
before this test is invoked once inside each of them. Every test decision --
launch order, readiness, evaluation, teardown -- is made by the in-pod
runner itself; this only supplies what the runner cannot infer from its own
pod's environment.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import run_runner

# The layout is a second axis, orthogonal to the scenario id, so accuracy
# evidence stays comparable with the single-host rows.
POD_LAYOUTS = {
    "2pod-role-split": "2A0F,0A2F",
    "2pod-interleaved": "1A1F,1A1F",
}
MULTI_POD_CASES = [
    ("afd-graph-2a2f", "2pod-role-split"),
    ("afd-graph-2a2f", "2pod-interleaved"),
]
DEEPSEEK_V2_LITE_MAX_MODEL_LEN = 4096


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def build_runner_command(scenario: str, layout_name: str) -> list[str]:
    """Build this pod's in-pod runner argv for one case.

    Assumes this process is itself running inside one pod of an
    already-provisioned deployment. Pod identity (AFD_E2E_POD_INDEX /
    JOB_COMPLETION_INDEX / HOSTNAME) and peer addresses
    (AFD_E2E_POD_ADDRESSES, or the rendezvous store's own address-exchange
    barrier) are resolved by the runner itself from its environment, so this
    only supplies what it cannot infer: the scenario, the shared rendezvous
    store host, and where to write results.

    AFD_E2E_RUN_ID need not match across pods -- it only tags this pod's own
    stale-process pre-flight and its launched processes -- but each
    invocation should still get its own value, or a leftover process from a
    prior run on this same pod could be mistaken for part of this run.
    """
    layout = POD_LAYOUTS[layout_name]
    command = [
        sys.executable,
        "-m",
        "tests.e2e.multi_pod.runner",
        "--scenario",
        scenario,
        "--pod-layout",
        layout,
        "--run-id",
        _required_env("AFD_E2E_RUN_ID"),
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--gsm8k-output-path",
        _required_env("AFD_E2E_GSM8K_OUTPUT"),
        "--store-host",
        _required_env("AFD_E2E_STORE_HOST"),
        f"--common-vllm-arg=--max-model-len={DEEPSEEK_V2_LITE_MAX_MODEL_LEN}",
    ]
    store_port = os.environ.get("AFD_E2E_STORE_PORT")
    if store_port:
        command.extend(["--store-port", store_port])
    for value in os.environ.get("AFD_E2E_POD_ENV", "").split(";"):
        if value.strip():
            command.extend(["--pod-env", value.strip()])
    return command


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    MULTI_POD_CASES,
    ids=[f"{scenario}-{layout}" for scenario, layout in MULTI_POD_CASES],
)
def test_multi_pod(scenario: str, layout_name: str) -> None:
    run_runner(build_runner_command(scenario, layout_name))
