# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3 MoE multi-pod AFD E2E cases on the DeepEP M2N connector.

This test runs as *one pod's slice* of an already-provisioned multi-pod
deployment: the k8s pods or Docker containers must already exist -- brought
up by hand, following `.agents/skills/run-e2e/resources/k8-multi-pod.md` --
before this test is invoked once inside each of them. Every test decision --
launch order, readiness, evaluation, teardown -- is made by the in-pod
runner itself; this only supplies what the runner cannot infer from its own
pod's environment.

DeepEP M2N needs FFN ranks first (afdEP-comm) and every NVLink domain to hold
an equal, contiguous block of world ranks (DeepEP), so the two pods split by
role with equal rank counts: Attention on one pod, FFN on the other. The
ranks then span two NVLink domains, so dispatch and combine cross pods over
NCCL GIN (RDMA).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_case_entry,
)
from tests.e2e.environment import model_multi_pod_command
from tests.e2e.models.qwen3_moe.cases import MULTI_POD_IDS

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
QWEN3_MOE_REPO_ID = "Qwen/Qwen3-30B-A3B"
# The layout is a second axis, orthogonal to the scenario id, so accuracy
# evidence stays comparable with the single-host rows.
POD_LAYOUTS = {
    "2pod-role-split": "2A0F,0A2F",
}
MULTI_POD_CASES = tuple(
    (scenario, layout) for scenario in MULTI_POD_IDS for layout in POD_LAYOUTS
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Make GSM8K and Qwen3-30B-A3B available in this pod for the runner."""
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)
        if not os.environ.get("AFD_GPU_E2E_MODEL"):
            os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(QWEN3_MOE_REPO_ID))
        yield


def build_runner_command(scenario: str, layout_name: str) -> list[str]:
    """Build this pod's in-pod runner argv for one case.

    Assumes this process is itself running inside one pod of an
    already-provisioned deployment. Pod identity (AFD_E2E_POD_INDEX /
    JOB_COMPLETION_INDEX / HOSTNAME) and peer addresses
    (AFD_E2E_POD_ADDRESSES, or the rendezvous store's own address-exchange
    barrier) are resolved by the runner itself from its environment, so this
    only supplies what it cannot infer: the scenario, the shared rendezvous
    store host, and where to write results.
    """
    return model_multi_pod_command(
        "tests.e2e.models.qwen3_moe.cases", scenario, POD_LAYOUTS[layout_name]
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("scenario", "layout_name"),
    MULTI_POD_CASES,
    ids=[f"{scenario}-{layout}" for scenario, layout in MULTI_POD_CASES],
)
def test_multi_pod_deepep_m2n(scenario: str, layout_name: str) -> None:
    run_case_entry(
        build_runner_command(scenario, layout_name),
        Path(_required_env("AFD_E2E_GSM8K_OUTPUT")),
    )
