# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import subprocess
import sys

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    resolve_class_from_qualname,
)

GPU_RUNTIME_CLASS_PATHS = [
    ATTENTION_WORKER_FQCN,
    ATTENTION_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    "afd_plugin.v1.worker:AFDAttentionWorker",
    "afd_plugin.v1.worker:AFDAttentionModelRunnerV2",
]

NPU_WORKER_CLASS_PATHS = [
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
]

NPU_MODEL_RUNNER_CLASS_PATHS = [
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
]

# vLLM resolves the platform and worker before the worker imports its model
# runner. Keep that production order explicit: a standalone vLLM-Ascend model
# runner import can enter device_op before the ops package finishes loading.
NPU_RUNTIME_CLASS_PATHS = NPU_WORKER_CLASS_PATHS + NPU_MODEL_RUNNER_CLASS_PATHS

_NPU_RUNTIME_RESOLUTION_SCRIPT = """
import sys

from vllm.platforms import current_platform

assert current_platform.device_type == "npu"

import afd_plugin

afd_plugin.register_afd()

from afd_plugin.validation import resolve_class_from_qualname

for qualname in sys.argv[1:]:
    cls = resolve_class_from_qualname(qualname)
    assert cls.__module__.startswith("afd_plugin.v1.worker.npu")

"""


def test_gpu_attention_class_path_selects_mrv2():
    assert (
        ATTENTION_MODEL_RUNNER_FQCN == "afd_plugin.v1.worker.AFDAttentionModelRunnerV2"
    )


@pytest.mark.parametrize(
    "qualname",
    GPU_RUNTIME_CLASS_PATHS + NPU_RUNTIME_CLASS_PATHS,
)
def test_runtime_class_paths_are_plugin_paths(qualname):
    assert qualname.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", GPU_RUNTIME_CLASS_PATHS)
def test_gpu_runtime_class_paths_resolve_when_vllm_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker")


_GPU_WORKER_IMPORT_SCRIPT = """
import sys

import afd_plugin.v1.worker.attention_model_runner_v2
import afd_plugin.v1.worker.attention_worker
import afd_plugin.v1.worker.ffn_worker

loaded = sorted(
    name
    for name in sys.modules
    if name.startswith("afd_plugin.connectors.gpu.deepep_m2n")
    or name == "afd_plugin.v1.worker.ffn_routed_experts_model_runner"
)
assert not loaded, loaded
"""


@pytest.mark.vllm_runtime
def test_gpu_workers_and_runners_import_no_connector_backend():
    """Importing the GPU workers and the Attention runner loads no DeepEP M2N
    module and not the routed-experts runner.

    Why: workers and runners stay connector-agnostic; P2P deployments must
    not pay for, or depend on, another connector's backend.
    """
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    result = subprocess.run(
        [sys.executable, "-c", _GPU_WORKER_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "GPU worker import loaded a connector backend:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.vllm_runtime
def test_npu_runtime_class_paths_resolve_when_vllm_ascend_is_available():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _NPU_RUNTIME_RESOLUTION_SCRIPT,
            *NPU_RUNTIME_CLASS_PATHS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "NPU runtime class resolution failed in an isolated process:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
