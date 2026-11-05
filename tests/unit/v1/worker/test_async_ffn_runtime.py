# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async CAM eager wire outputs for layered execution."""

from __future__ import annotations

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
        calls.append(("send", output, context.states.token_nums_rankid_layeridx))

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
        torch.arange(8 * 32, dtype=torch.int8).reshape(8, 32),
        AFDTransferContext(AFDTransferMetadata(0, 0, [8]), state),
    )


def test_independent_model_attachment_exposes_original_owner(harness):
    model = harness.runner.get_model()
    harness.runner.model = None
    harness.connector._initialized = False
    harness.runner.attach_model(model, parameter_bytes=123)
    assert harness.runner.get_model() is model


@pytest.mark.parametrize("boundary", ["already_loaded", "paired", "negative_size"])
def test_independent_attachment_rejects_invalid_lifetime(harness, boundary):
    model = harness.runner.get_model()
    if boundary != "already_loaded":
        harness.runner.model = None
    harness.connector._initialized = boundary == "paired"
    with pytest.raises((RuntimeError, ValueError)):
        harness.runner.attach_model(
            model, parameter_bytes=-1 if boundary == "negative_size" else 123
        )


def test_receiver_exit_rejects_pending_graph_work(harness):
    harness.runner._graph_pending = True
    with pytest.raises(RuntimeError, match="pending work; retain resources"):
        harness.runner.mark_receiver_exited()


def test_eager_preserves_layer_loop_device_metadata_and_return_headers(harness):
    h = harness
    first, zero = payload(5, 3), payload(2, 0)
    h.payloads.extend((first, zero))
    first.context.states.group_list = torch.tensor([1, 2])
    first.context.states.dynamic_scales = torch.arange(1, 9, dtype=torch.float32)

    def execute(hidden, scales, groups, header):
        # Toy experts have gains 1/2 and layer gain equal to the layer ID.
        # Only the sent result is asserted; this does not simulate CANN.
        gains = torch.repeat_interleave(torch.tensor([1.0, 2.0]), groups)
        rows = gains.numel()
        output = torch.zeros_like(hidden, dtype=torch.bfloat16)
        output[:rows] = (
            hidden[:rows].float() * scales[:rows, None] * gains[:, None] * header[2]
        ).to(torch.bfloat16)
        return output

    h.runner._layered_executor = execute
    expected = [torch.zeros(8, 32), torch.zeros(8, 32)]
    expected[0][:3] = (
        first.hidden_states[:3].float() * torch.tensor([5.0, 20.0, 30.0])[:, None]
    )
    assert h.runner.execute_connector_driven_step()
    returned = [
        call for call in h.calls if isinstance(call, tuple) and call[0] == "send"
    ]
    for sent, received_payload, hidden in zip(
        returned, (first, zero), expected, strict=True
    ):
        torch.testing.assert_close(sent[1], hidden.to(torch.bfloat16), rtol=0, atol=0)
        assert sent[2] is received_payload.context.states.token_nums_rankid_layeridx
