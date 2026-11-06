# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Load Async FFN host orchestration without device/runtime dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock


def load_host_module(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[3] / (name.replace(".", "/") + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def load_async_modules(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    npu = MagicMock()
    npu.get_stream_limit.return_value = {"cube_core_num": 24, "vector_core_num": 48}
    stub("torch", inference_mode=lambda: lambda function: function, npu=npu)
    for name in (
        "torch.distributed",
        "torch.distributed.distributed_c10d",
        "torch.distributed.rendezvous",
        "vllm.logger",
        "vllm.distributed",
        "vllm.utils.torch_utils",
        "vllm_ascend.utils",
        "afd_plugin.compat.npu.ops",
        "afd_plugin.compat.npu.forward_context",
        "afd_plugin.compat.npu.profiler",
        "afd_plugin.model_executor.npu.async_cam_layered",
        "afd_plugin.v1.worker.ffn_metadata",
    ):
        stub(name)
    stub("afd_plugin.connectors.metadata", AFDTransferState=object)
    load_host_module(monkeypatch, "afd_plugin.connectors.base")
    load_host_module(monkeypatch, "tests.unit.fakes.connector")
    group = load_host_module(monkeypatch, "afd_plugin.distributed.afd_process_group")
    stub("afd_plugin.distributed", init_afd_process_group=group.init_afd_process_group)
    connector = load_host_module(monkeypatch, "afd_plugin.connectors.npu.async_cam")
    runtime = load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.async_ffn_model_runner"
    )
    return group, connector, runtime
