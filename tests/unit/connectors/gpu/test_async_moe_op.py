# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opaque async ops preserve connector messages and native stage metadata."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu import async_moe_op  # noqa: E402, F401
from afd_plugin.model_executor import models  # noqa: E402
from tests.unit.fakes.connector import ProxyConnector  # noqa: E402


def test_async_ops_use_each_native_child_stage(monkeypatch):
    messages: list[tuple] = []
    sidecar = SimpleNamespace(connector=ProxyConnector(messages), stage_idx=0)
    monkeypatch.setattr(
        models, "get_afd_metadata_from_forward_context", lambda: sidecar
    )
    hidden = torch.ones(2, 4)
    weights = torch.ones(2, 1)
    ids = torch.zeros(2, 1, dtype=torch.int32)
    for stage in (0, 1):
        sidecar.stage_idx = stage
        torch.ops.vllm.afd_async_gpu_dispatch(hidden, weights, ids, 3)
        output = torch.ops.vllm.afd_async_gpu_receive(hidden)
        torch.testing.assert_close(output, hidden * 0.25)
    sent = [message for message in messages if message[0] == "send"]
    received = [message for message in messages if message[0] == "recv"]
    assert [message[2].metadata.stage_idx for message in sent] == [0, 1]
    assert [message[2].metadata.layer_idx for message in sent] == [3, 3]
    assert [message[2] for message in received] == [0, 1]
    assert all(set(message[3]) == {"topk_weights", "topk_ids"} for message in sent)
