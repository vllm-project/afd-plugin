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


class GPUAsyncHostCase:
    """Host receive/compute/reply fixture; no device or graph simulation."""

    def __init__(self, runtime, torch):
        from collections import deque
        from types import SimpleNamespace

        from afd_plugin.connectors import (
            AFDTransferContext,
            AFDTransferMetadata,
        )
        from tests.unit.fakes.profiler import StepProfiler

        self.arrivals: deque = deque()
        self.replies: list[tuple] = []
        self.events: list[str] = []
        self.context = SimpleNamespace(
            dp_metadata="not async",
            additional_kwargs={},
            all_moe_layers=[],
        )
        runner = object.__new__(runtime.GPUAsyncFFNModelRunner)
        runner.vllm_config = SimpleNamespace()
        runner.prof = StepProfiler()
        runner.session = SimpleNamespace(
            keep_serving=lambda stop: not stop,
            before_first_step=lambda: self.events.append("receiver_ready"),
            after_step=lambda: self.events.append("step_finished"),
        )
        runner._first_step = True
        runner._padded_hidden = None
        runner._padded_shared = None
        runner._padded_counts = None
        runner._padded_buckets = ()
        runner._padded_graphs = {}
        runner._padded_max_routed = 0
        runner._padded_max_shared = 0

        def receive(**kwargs):
            if not self.arrivals:
                raise TimeoutError("idle")
            payload = self.arrivals.popleft()
            if isinstance(payload, BaseException):
                raise payload
            return payload

        def send(routed, context, *, shared_output):
            self.replies.append((routed, context, shared_output))

        def compute(hidden_states, layer_idx, *, group_list, expand_x_shared):
            return runtime.AFDF2ATransferPayload(
                routed_output=hidden_states * 2,
                shared_output=expand_x_shared,
            )

        runner.connector = SimpleNamespace(
            ffn_size=1,
            recv_attn_output=receive,
            send_ffn_output=send,
            extra_info=SimpleNamespace(recv_poll_timeout_ms=1),
        )
        runner.model = SimpleNamespace(compute_ffn_output=compute)
        self.runner = runner
        self.torch = torch
        self.runtime = runtime
        self.make_context = AFDTransferContext
        self.make_metadata = AFDTransferMetadata.create_attention_metadata

    def add(self, *, layer_idx=3, stage_idx=1, routed_rows=2, shared_rows=1):
        hidden = self.torch.ones(routed_rows, 4)
        shared = self.torch.full((shared_rows, 4), 3.0)
        state = self.runtime.GpuAsyncTransferState(
            routed_tokens=routed_rows,
            shared_tokens=shared_rows,
            group_list=self.torch.tensor([routed_rows], dtype=self.torch.int32),
            expand_x_shared=shared,
        )
        payload = self.runtime.AFDA2FTransferPayload(
            hidden_states=hidden,
            context=self.make_context(
                metadata=self.make_metadata(
                    layer_idx=layer_idx,
                    stage_idx=stage_idx,
                    seq_len=routed_rows,
                ),
                states=state,
            ),
        )
        self.arrivals.append(payload)
        return payload
