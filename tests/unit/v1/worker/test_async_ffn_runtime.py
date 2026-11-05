# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MRV2 loading, eager isolation and graph-only resource lifetime contracts."""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
import torch  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.v1.worker.npu import async_ffn_model_runner as runtime  # noqa: E402
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker  # noqa: E402
from tests.unit.fakes.vllm_config import npu_vllm_config  # noqa: E402


def config():
    cfg = npu_vllm_config(
        role="ffn",
        connector="CAMAsyncAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        compute_gate_on_attention=True,
        async_dp=True,
    )
    cfg.model_config.hf_config = SimpleNamespace(
        hidden_size=32, num_experts_per_tok=2, n_routed_experts=4, num_hidden_layers=6
    )
    cfg.model_config.dtype = torch.bfloat16
    cfg.scheduler_config.max_num_batched_tokens = 8
    return cfg


@pytest.fixture
def harness(monkeypatch):
    # These CPU tensor tests exercise execution/loading, while the configuration
    # and real-device contracts validate admission separately.
    monkeypatch.setattr(
        runtime, "validate_npu_model_runner_v2_config", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runtime, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    cfg = config()
    connector = CAMAsyncAFDConnector(
        2,
        0,
        cfg,
        AFDConfig(
            role="ffn",
            connector="CAMAsyncAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=2,
            compute_gate_on_attention=True,
            async_dp=True,
        ),
        0,
    )
    connector._initialized = True
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    runner = runtime.AsyncNPUFFNModelRunner(cfg, torch.device("cpu"), connector)
    runner.layer_ids = (2, 5)
    calls: list[str | tuple] = []
    payloads: deque[AFDA2FTransferPayload] = deque()

    def receive(**kwargs):
        calls.append("receive")
        return payloads.popleft()

    def send(output, context, **kwargs):
        calls.append(("send", context.states.token_nums_rankid_layeridx))

    monkeypatch.setattr(connector, "recv_attn_output", receive)
    monkeypatch.setattr(connector, "send_ffn_output", send)
    runner.model = SimpleNamespace()
    return SimpleNamespace(
        runner=runner, connector=connector, calls=calls, payloads=payloads
    )


def payload(layer, rows):
    state = AFDAsyncTransferState(
        token_nums_rankid_layeridx=torch.tensor([rows, 0, layer, 0, 1]),
        group_list=torch.tensor([rows, 0]),
        dynamic_scales=torch.ones(8),
    )
    return AFDA2FTransferPayload(
        torch.ones(8, 32, dtype=torch.int8),
        AFDTransferContext(AFDTransferMetadata(0, 0, [8]), state),
    )


def test_eager_preserves_layer_loop_device_metadata_and_return_headers(harness):
    h = harness
    first, zero = payload(5, 3), payload(2, 0)
    h.payloads.extend((first, zero))
    observed = []

    def execute(hidden, scales, groups, header):
        observed.append((hidden.shape[0], groups, header))
        return hidden.to(torch.bfloat16)

    h.runner._layered_executor = execute
    assert h.runner.execute_connector_driven_step()
    assert [(rows, int(header[2])) for rows, _, header in observed] == [
        (8, 5),
        (8, 2),
    ]
    returned = [
        call[1] for call in h.calls if isinstance(call, tuple) and call[0] == "send"
    ]
    assert returned[0] is first.context.states.token_nums_rankid_layeridx
    assert returned[1] is zero.context.states.token_nums_rankid_layeridx


@pytest.mark.parametrize("fail", [False, True])
def test_full_receiver_exit_requires_success_even_during_shutdown(harness, fail):
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = harness.runner
    harness.runner.effective_mode = "full"
    worker._ffn_thread = None
    worker._ffn_loop_error = None
    worker._bind_cpus_once = lambda: None

    def serve():
        worker._ffn_shutdown_event.set()
        if fail:
            raise RuntimeError("partial device completion")

    worker._run_ffn_server_loop = serve
    worker.start_ffn_server_loop()
    assert isinstance(worker._ffn_thread, threading.Thread)
    worker._ffn_thread.join(3)
    assert not worker._ffn_thread.is_alive()
    assert harness.runner._graph_receiver_exited is not fail
    assert (worker._ffn_loop_error is not None) is fail
