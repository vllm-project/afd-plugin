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
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        # A cached package attribute otherwise keeps the previous module alive.
        monkeypatch.setitem(vars(parent), child_name, module)
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
        "vllm.v1.worker.gpu.shutdown",
        "afd_plugin.compat.npu.ops",
        "afd_plugin.compat.npu.forward_context",
        "afd_plugin.compat.npu.profiler",
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_layered",
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_mega",
        "afd_plugin.model_executor.ffn.npu.expert_weights",
        "afd_plugin.v1.worker.ffn_metadata",
    ):
        stub(name)
    stub("vllm.forward_context")
    load_host_module(monkeypatch, "afd_plugin.connectors.metadata")
    load_host_module(monkeypatch, "afd_plugin.connectors.base")
    load_host_module(monkeypatch, "tests.unit.fakes.connector")
    group = load_host_module(monkeypatch, "afd_plugin.distributed.afd_process_group")
    stub("afd_plugin.distributed", init_afd_process_group=group.init_afd_process_group)
    connector = load_host_module(monkeypatch, "afd_plugin.connectors.npu.async_cam")
    runtime = load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.async_ffn_model_runner"
    )
    return group, connector, runtime


def load_ffn_owners(monkeypatch, modules):
    _, _, async_runtime = modules
    for name in (
        "vllm.compilation.monitor",
        "vllm.config",
        "vllm.forward_context",
        "vllm.platforms",
        "afd_plugin.v1.worker.attention_metadata",
        "afd_plugin.v1.worker.cuda_graph",
        "vllm.distributed.parallel_state",
        "vllm.v1.worker.worker_base",
        "vllm.v1.worker.workspace",
        "vllm_ascend.ascend_config",
        "vllm_ascend.cpu_binding",
        "vllm_ascend.worker.worker",
    ):
        module = ModuleType(name)
        module.__dict__["__getattr__"] = lambda attr: MagicMock(name=attr)
        if name == "vllm_ascend.worker.worker":
            module.__dict__["NPUWorker"] = object
        monkeypatch.setitem(sys.modules, name, module)
    load_host_module(monkeypatch, "afd_plugin.model_executor.models.model_utils")
    load_host_module(monkeypatch, "afd_plugin.connectors.factory")
    sync_runtime = load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.ffn_model_runner"
    )
    # Worker selection needs the runner class, not the device connector runtime.
    mega_connector = ModuleType("afd_plugin.connectors.npu.async_mega")
    monkeypatch.setattr(
        mega_connector, "CAMAsyncMegaAFDConnector", object, raising=False
    )
    monkeypatch.setitem(sys.modules, mega_connector.__name__, mega_connector)
    load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.async_mega_ffn_model_runner"
    )
    worker = load_host_module(monkeypatch, "afd_plugin.v1.worker.npu.ffn_worker")
    return sync_runtime, async_runtime, worker
