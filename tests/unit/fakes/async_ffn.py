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


def load_mxfp_executor(monkeypatch, torch):
    """Fake only the native MLP boundary; keep host selection/slicing real."""
    from types import SimpleNamespace

    modules = {
        "torch_npu": dict(float4_e2m1fn_x2="fp4", float8_e8m0fnu="e8m0"),
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts": dict(
            MoEWeights=SimpleNamespace
        ),
        "vllm_ascend.ops.fused_moe.dataclass.moe_mlp": dict(
            MoEMlpComputeInput=SimpleNamespace
        ),
        "vllm_ascend.ops.fused_moe.dataclass.moe_quant": dict(
            MoEMxfpParams=SimpleNamespace, MoEQuantParams=SimpleNamespace
        ),
        "vllm_ascend.quantization.quant_type": dict(
            QuantType=SimpleNamespace(W4A8MXFP="mxfp")
        ),
    }

    def apply_mlp(*, mlp_compute_input, quant_method):
        return mlp_compute_input.hidden_states * mlp_compute_input.weights.w1, None

    modules["vllm_ascend.ops.fused_moe.moe_mlp"] = dict(apply_moe_mlp=apply_mlp)
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    executor = load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.async_cam_mxfp"
    ).AsyncCAMMXFPExecutor
    config = SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    layers = []
    for index, multiplier in ((3, 2), (7, 4)):
        owner = SimpleNamespace(
            w13_weight=torch.tensor(multiplier),
            w2_weight=torch.tensor(1),
            w13_weight_scale=None,
            w2_weight_scale=None,
            dynamic_eplb=False,
            activation="silu",
            quant_method=SimpleNamespace(quant_method=None),
        )
        layers.append(
            (
                index,
                SimpleNamespace(
                    experts=SimpleNamespace(
                        quant_type="mxfp",
                        routed_experts=owner,
                        shared_experts=None,
                        moe_config=config,
                    )
                ),
            )
        )
    return executor(layers)
