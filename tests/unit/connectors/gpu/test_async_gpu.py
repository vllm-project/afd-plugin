# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU routing and on-wire connector behavior, without simulating GPU execution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402

from afd_plugin.config import parse_afd_config  # noqa: E402
from afd_plugin.connectors.gpu.async_gpu import (  # noqa: E402
    ConnectorShutdown,
    GpuAsyncAFDConnector,
    plan_dispatch,
)
from afd_plugin.connectors.gpu.async_gpu_protocol import (  # noqa: E402
    FLAG_REPLY_READY,
    FLAG_SHUTDOWN_BIT,
    SlotHeader,
    decode_header_words,
)
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDTransferContext,
    AFDTransferMetadata,
)
from tests.unit.fakes.async_gpu import RecordingAsyncWindow  # noqa: E402
from tests.unit.fakes.vllm_config import npu_vllm_config  # noqa: E402


@pytest.fixture
def connector():
    config = npu_vllm_config(
        connector="GpuAsyncAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        use_ubatching=True,
        num_ubatches=2,
    )
    config.model_config.dtype = torch.float32
    vars(config.model_config.hf_text_config).update(
        hidden_size=3,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=0,
        num_hidden_layers=4,
    )
    instance = GpuAsyncAFDConnector(
        0,
        0,
        config,
        parse_afd_config(config, validate=False),
        0,
    )
    instance.window = RecordingAsyncWindow()
    instance._initialized = True
    instance._seq_device = torch.zeros(2, dtype=torch.int32)
    return instance


def _context(stage):
    return AFDTransferContext(
        AFDTransferMetadata.create_attention_metadata(
            layer_idx=1,
            stage_idx=stage,
            seq_len=2,
        )
    )


def test_routing_matches_independent_expert_weighted_sum():
    ids = torch.tensor([[0, 3], [2, 1], [0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    hidden = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    plan = plan_dispatch(ids, weights, ffn_size=2, expert_per_rank=2)
    result = torch.zeros_like(hidden)
    offset = 0
    for expert, count in enumerate(plan.counts.tolist()):
        segment = slice(offset, offset + count)
        tokens = plan.expand_idx[segment].long()
        partial = hidden[tokens] * (expert + 1)
        result.index_add_(0, tokens, partial * plan.weights[segment, None])
        offset += count
    expected = sum(
        hidden * (ids[:, slot, None] + 1) * weights[:, slot, None]
        for slot in range(ids.shape[1])
    )
    torch.testing.assert_close(result, expected)


def test_equal_shape_stages_send_independent_routing_headers(connector):
    # A queued transfer can read its header after the peer stage runs.
    connector.window = RecordingAsyncWindow(defer_headers=True)
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weights = torch.ones((2, 2))
    for stage, expert in enumerate((0, 3)):
        connector.send_attn_output(
            hidden,
            _context(stage),
            topk_ids=torch.full((2, 2), expert),
            topk_weights=weights,
        )
    messages = connector.window.messages
    headers = [
        decode_header_words(msg["header"].tolist(), connector.layout)
        for msg in messages
    ]
    assert [header.expert_counts for header in headers] == [
        (4, 0),
        (0, 0),
        (0, 0),
        (0, 4),
    ]
    assert [msg["ring"] for msg in messages] == [0, 0, 1, 1]
    assert [msg["flag_value"].item() for msg in messages] == [1, 1, 1, 1]


def test_combine_reads_each_reply_before_clearing_its_marker(connector):
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    connector.window.replies = {0: hidden, 1: 2 * hidden}
    connector.send_attn_output(
        hidden,
        _context(0),
        topk_ids=torch.zeros((2, 2), dtype=torch.int32),
        topk_weights=torch.ones((2, 2)),
    )
    output = connector.recv_ffn_output(hidden)
    torch.testing.assert_close(output, 3 * hidden)
    assert connector.window.events == [
        ("wait", 0, 0, FLAG_REPLY_READY),
        ("read", 0, 0),
        ("clear", 0, 0),
        ("wait", 1, 0, FLAG_REPLY_READY),
        ("read", 1, 0),
        ("clear", 1, 0),
    ]


def test_shutdown_waits_for_every_attention_producer(connector):
    terminal = SlotHeader(0, 0, 0, FLAG_SHUTDOWN_BIT, 0, (0, 0))
    connector.window.arrivals.extend(
        [
            SimpleNamespace(region=0, ring=0, header=terminal),
            SimpleNamespace(region=1, ring=0, header=terminal),
        ]
    )
    session = connector.ffn_session()
    assert session.keep_serving(shutdown_requested=True)
    with pytest.raises(ConnectorShutdown, match="All Attention"):
        connector.recv_attn_output(timeout_ms=1)
    assert not session.keep_serving(shutdown_requested=True)
    session.validate_shutdown()
