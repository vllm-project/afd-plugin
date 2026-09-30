# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolated CPU contracts for the NPU worker's runner import boundary."""

from __future__ import annotations

import subprocess
import sys

import pytest

_NPU_WORKER_IMPORT_SCRIPT = """
import importlib
import os
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_PLUGINS"] = "ascend"
import torch
import torch_npu


def forbid_init(*args, **kwargs):
    raise AssertionError("CPU import contract attempted NPU initialization")


with ExitStack() as guards:
    for target, name in (
        (torch_npu.npu, "_lazy_init"),
        (torch_npu._C, "_npu_init"),
        (torch_npu._C, "_npu_setDevice"),
    ):
        guards.enter_context(patch.object(target, name, side_effect=forbid_init))

    from vllm.platforms import current_platform

    assert current_platform.device_type == "npu"
    blocked = {
        "afd_plugin.v1.worker.attention_model_runner",
        "afd_plugin.v1.worker.attention_model_runner_v2",
        "afd_plugin.v1.worker.ffn_model_runner",
        "afd_plugin.v1.worker.npu.attention_model_runner",
        "afd_plugin.v1.worker.npu.attention_model_runner_v2",
        "afd_plugin.v1.worker.npu.ffn_model_runner",
    }
    assert blocked.isdisjoint(sys.modules)
    # None entries reject even transient imports of a forbidden runner.
    for name in blocked:
        sys.modules[name] = None

    import afd_plugin

    afd_plugin.register_afd()
    importlib.import_module("afd_plugin.v1.worker.attention_metadata")
    importlib.import_module("afd_plugin.model_executor.models.remote_ffn")
    worker_module = importlib.import_module(
        "afd_plugin.v1.worker.npu.attention_worker"
    )
    assert all(sys.modules[name] is None for name in blocked)

    use_v2 = sys.argv[1] == "v2"
    suffix = "_v2" if use_v2 else ""
    selected = "afd_plugin.v1.worker.npu.attention_model_runner" + suffix
    del sys.modules[selected]
    selected_module = importlib.import_module(selected)
    runner_cls = getattr(
        selected_module,
        "AFDNPUAttentionModelRunnerV2" if use_v2 else "AFDNPUAttentionModelRunner",
    )
    worker = object.__new__(worker_module.AFDNPUAttentionWorker)
    worker.use_v2_model_runner = use_v2
    model_config = object()
    worker.vllm_config = SimpleNamespace(model_config=model_config)
    device = torch.device("npu", 0)
    worker._init_device = lambda: device
    for name in (
        "assert_compatible_afd_stack",
        "fail_if_unsupported_npu_afd_features",
        "fix_all2all_backend_for_afd",
        "validate_npu_model_runner_v2_config",
        "init_workspace_manager",
    ):
        guards.enter_context(patch.object(worker_module, name))
    guards.enter_context(
        patch.object(worker_module, "npu_afd_num_ubatches", return_value=1)
    )
    guards.enter_context(
        patch.object(worker_module, "get_afd_model_config", return_value=model_config)
    )
    constructor = guards.enter_context(
        patch.object(runner_cls, "__init__", return_value=None)
    )
    worker.init_device()
    assert type(worker.model_runner) is runner_cls
    constructor.assert_called_once_with(worker.vllm_config, device)
    assert all(sys.modules[name] is None for name in blocked - {selected})
"""


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("runner_version", ["v1", "v2"])
def test_npu_attention_worker_imports_only_selected_plugin_runner(runner_version):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")
    pytest.importorskip("torch_npu")

    result = subprocess.run(
        [sys.executable, "-c", _NPU_WORKER_IMPORT_SCRIPT, runner_version],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"NPU {runner_version} worker import boundary failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
