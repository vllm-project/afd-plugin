# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU routing and on-wire connector behavior, without simulating GPU execution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch  # noqa: E402

from afd_plugin.connectors.gpu.async_nvshmem.connector import (  # noqa: E402
    FAILURE_STORE_KEY,
    ConnectorShutdown,
    plan_dispatch,
)
from afd_plugin.connectors.gpu.async_nvshmem.protocol import (  # noqa: E402
    FLAG_REPLY_READY,
    FLAG_SHUTDOWN_BIT,
    SlotHeader,
    decode_header_words,
)
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDTransferContext,
    AFDTransferMetadata,
)
from tests.unit.fakes.async_gpu import (  # noqa: E402
    RecordingAsyncWindow,
    recording_async_connector,
)


@pytest.fixture
def connector():
    return recording_async_connector()


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
    assert session.keep_serving(shutdown_requested=False)
    assert not session.keep_serving(shutdown_requested=True)
    with pytest.raises(ConnectorShutdown, match="All Attention"):
        connector.recv_attn_output(timeout_ms=1)
    assert not session.keep_serving(shutdown_requested=True)
    session.validate_shutdown()


def test_dispatch_sends_current_batch_size_when_shapes_change(connector):
    connector.window = RecordingAsyncWindow(defer_headers=True)
    for count in range(1, 17):
        hidden = torch.ones(count, 3)
        connector.window.replies = {rank: torch.zeros_like(hidden) for rank in range(2)}
        connector.send_attn_output(
            hidden,
            AFDTransferContext(
                AFDTransferMetadata.create_attention_metadata(
                    layer_idx=1, stage_idx=0, seq_len=count
                )
            ),
            topk_ids=torch.zeros((count, 2), dtype=torch.int32),
            topk_weights=torch.ones((count, 2)),
        )
        for message in connector.window.messages[-2:]:
            assert (
                decode_header_words(
                    message["header"].tolist(), connector.layout
                ).num_tokens
                == count
            )
        connector.recv_ffn_output(hidden)


@pytest.mark.parametrize("raises_poll_timeout", [False, True])
def test_drain_expires_when_a_producer_disappears(
    connector, monkeypatch, raises_poll_timeout
):
    clock = iter((0.0, 0.0, float(connector.afd_config.afd_process_group_timeout_s)))
    monkeypatch.setattr(
        "afd_plugin.connectors.gpu.async_nvshmem.connector.time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )

    def run_one():
        if raises_poll_timeout:
            raise TimeoutError("empty poll")

    with pytest.raises(TimeoutError, match="drain timed out"):
        connector.ffn_session().drain(run_one)
    assert b"drain timed out" in connector.window.runtime.store.get(FAILURE_STORE_KEY)
    assert not connector.window.closed


def test_close_failure_notifies_other_ranks_without_freeing_live_window(
    connector, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    hidden = torch.ones(2, 3)
    connector.send_attn_output(
        hidden,
        _context(0),
        topk_ids=torch.zeros((2, 2), dtype=torch.int32),
        topk_weights=torch.ones((2, 2)),
    )
    with pytest.raises(RuntimeError, match="unconsumed dispatch"):
        connector.close()
    assert b"unconsumed dispatch" in connector.window.runtime.store.get(
        FAILURE_STORE_KEY
    )
    with pytest.raises(RuntimeError, match="peer failed.*unconsumed dispatch"):
        connector.close()
    assert not connector.window.closed


def test_close_fails_when_a_peer_reports_failed_drain(connector):
    connector.window.runtime.store.set(
        FAILURE_STORE_KEY, "rank 3: receiver did not drain"
    )
    with pytest.raises(RuntimeError, match="peer failed.*rank 3"):
        connector.close()
    assert not connector.window.closed


def test_close_deadline_publishes_failure_when_peer_never_arrives(
    connector, monkeypatch
):
    connector.is_attention = False
    connector.receiver_finished = True
    clock = iter((0.0, float(connector.afd_config.afd_process_group_timeout_s)))
    monkeypatch.setattr(
        "afd_plugin.connectors.gpu.async_nvshmem.connector.time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )
    with pytest.raises(TimeoutError, match="close timed out"):
        connector.close()
    assert b"close timed out" in connector.window.runtime.store.get(FAILURE_STORE_KEY)
    assert not connector.window.closed
