# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Load Async FFN host orchestration without device/runtime dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


def mxfp_runtime(monkeypatch, torch):
    """Isolate the native MX kernels while running layer/row selection on CPU."""

    def local_mlp(*, mlp_compute_input, quant_method):
        args = mlp_compute_input
        if args.dynamic_scale is not None or args.group_list_type != 1:
            raise ValueError(
                "the native BF16 path requires expert counts and no CAM scales"
            )
        gains = torch.repeat_interleave(args.weights.w1, args.group_list)
        return args.hidden_states * gains[:, None], None

    modules: dict[str, dict[str, object]] = {
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
        "vllm_ascend.ops.fused_moe.moe_mlp": dict(apply_moe_mlp=local_mlp),
        "vllm_ascend.quantization.quant_type": dict(
            QuantType=SimpleNamespace(W4A8MXFP="fp4")
        ),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    executor = load_host_module(
        monkeypatch, "afd_plugin.connectors.npu.async_cam_runtime.afd_async_mxfp"
    ).AfdAsyncMXFPExecutor
    config = SimpleNamespace(
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    layers = {}
    for index in (3, 7):
        owner = torch.nn.Module()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            weight = torch.tensor([index, index + 1], dtype=torch.bfloat16)
            owner.register_parameter(
                name, torch.nn.Parameter(weight, requires_grad=False)
            )
        owner.dynamic_eplb = False
        owner.activation = "silu"
        owner.quant_method = SimpleNamespace(quant_method=None)
        layers[index] = SimpleNamespace(
            experts=SimpleNamespace(
                quant_type="fp4",
                routed_experts=owner,
                _shared_experts=None,
                moe_config=config,
            )
        )
    return executor, layers


def eager_mxfp_runner(monkeypatch, torch, runtime, layers, receive, send):
    """Start the actual eager runner with a host model/connector boundary."""
    from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
        AsyncCAMGraphStartup,
        resolve_ffn_mode,
    )
    from tests.unit.fakes.connector import FakeFFNSession
    from tests.unit.fakes.vllm_config import npu_vllm_config

    class Model:
        def get_mxfp_expert_layers(self):
            return layers

        def get_experts_layer_indices(self):
            return tuple(layers)

    module = ModuleType("afd_plugin.model_executor.models.npu.deepseek_v4")
    module.__dict__["AFDDeepseekV4ForCausalLM"] = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    graph_startup = AsyncCAMGraphStartup(1, 1, 2)
    connector = SimpleNamespace(
        dynamic_quant=0,
        expert_per_rank=2,
        is_initialized=False,
        graph_startup=graph_startup,
        ffn_session=FakeFFNSession,
        init_afd_connector=lambda: resolve_ffn_mode([graph_startup.plan]),
        recv_attn_output=receive,
        send_ffn_output=send,
    )
    cfg = npu_vllm_config(
        role="ffn",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
    )
    session = runtime.AsyncNPUFFNModelRunner(cfg, torch.device("cpu"), connector)
    session.attach_model(Model(), parameter_bytes=0)
    session.initialize_afd_connector()
    return session
