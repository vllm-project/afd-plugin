# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async Mega send/receive helper contracts."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata  # noqa: E402
from afd_plugin.connectors.npu.async_mega import remote  # noqa: E402


def _context(stage):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=3, stage_idx=stage, seq_len=2
        )
    )


def test_send_then_receive_caches_output():
    calls = []
    value = torch.ones(2, 1)

    def recv(**kwargs):
        calls.append("recv")
        return value

    connector = SimpleNamespace(
        send_attn_output=lambda *args, **kwargs: calls.append("send"),
        recv_ffn_output=recv,
    )
    receive = remote.send_and_build_receive(connector, value, _context(0))
    assert calls == ["send"]
    assert receive() is value and receive() is value
    assert calls == ["send", "recv"]
