# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async Mega dispatch payload building contracts."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.npu.async_mega import dispatch  # noqa: E402


def test_plain_tp_layout_slices_replicated_tokens():
    layout = dispatch.plain_tp_layout(parent_tokens=5, tp_rank=1, tp_size=2)
    assert layout.padded_tokens == 6
    assert layout.local_token_slice == slice(3, 6)
    assert layout.local_tokens == 3


def test_plain_payload_pads_padding_rows_with_distinct_route_ids():
    # 5 real tokens over TP2 pad to 6; rank1's slice (rows 3..5) holds the pad row.
    layout = dispatch.plain_tp_layout(parent_tokens=5, tp_rank=1, tp_size=2)
    hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    ids = torch.zeros((5, 2), dtype=torch.int32)

    payload = dispatch.build_mega_dispatch_payload(
        hidden[3:], weights[3:], ids[3:], layout
    )

    assert payload.topk_ids.shape == (3, 2)
    # The pad row must carry arange(top_k) so each route maps to a distinct slot.
    assert torch.equal(payload.topk_ids[2], torch.arange(2, dtype=torch.int32))
    # Real rows keep their (zeroed) route ids.
    assert torch.equal(payload.topk_ids[:2], torch.zeros((2, 2), dtype=torch.int32))


def test_plain_payload_sequence_parallel_passes_through():
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    ids = torch.ones((3, 2), dtype=torch.int32)

    layout = dispatch.CAMDispatchLayout(
        parent_tokens=3,
        padded_tokens=3,
        local_token_slice=slice(0, 3),
        requires_tp_all_gather=False,
        use_sequence_parallel=True,
        tp_rank=0,
        tp_size=1,
    )
    payload = dispatch.build_mega_dispatch_payload(hidden, weights, ids, layout)

    assert payload.layout.padded_tokens == 3
    assert payload.topk_ids is ids
