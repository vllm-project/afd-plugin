# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import fastafd_moe  # noqa: E402
from tests.unit.fakes.fastafd import recording_connector  # noqa: E402

IDS = torch.tensor([[1, 5], [2, 6]], dtype=torch.int32)
WEIGHTS = torch.tensor([[0.7, 0.3], [0.6, 0.4]])


def _forward_context(monkeypatch, connector, *, stage_idx=0, ubatch_idx=None):
    monkeypatch.setattr(
        fastafd_moe,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(connector=connector, stage_idx=stage_idx),
    )
    context = SimpleNamespace()
    if ubatch_idx is not None:
        context.ubatch_idx = ubatch_idx
    monkeypatch.setattr(fastafd_moe, "get_forward_context", lambda: context)


@pytest.mark.parametrize(
    ("stage_idx", "ubatch_idx", "lane"),
    [(0, None, 0), (1, None, 1), (0, 1, 1)],
)
def test_ops_send_the_routing_payload_on_the_microbatch_lane(
    monkeypatch,
    stage_idx,
    ubatch_idx,
    lane,
):
    events: list[object] = []
    payloads: list[tuple] = []
    _forward_context(
        monkeypatch,
        recording_connector(events, payloads),
        stage_idx=stage_idx,
        ubatch_idx=ubatch_idx,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    local = fastafd_moe._fastafd_dispatch(hidden_states, IDS, WEIGHTS)
    output = fastafd_moe._fastafd_combine(local, hidden_states, IDS, WEIGHTS)

    assert events == [("dispatch", lane), ("combine", lane)]
    for _op, ids, weights in payloads:
        assert torch.equal(ids, IDS)
        assert torch.equal(weights, WEIGHTS)
    assert torch.equal(local, hidden_states)
    assert torch.equal(output, hidden_states * 2)


def test_ops_need_afd_forward_metadata(monkeypatch):
    monkeypatch.setattr(
        fastafd_moe,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="needs AFD forward metadata"):
        fastafd_moe._fastafd_dispatch(torch.zeros(2, 4), IDS, WEIGHTS)


def test_ops_require_the_fastafd_connector(monkeypatch):
    _forward_context(monkeypatch, object())

    with pytest.raises(RuntimeError, match="requires FastAFDAFDConnector"):
        fastafd_moe._fastafd_dispatch(torch.zeros(2, 4), IDS, WEIGHTS)
