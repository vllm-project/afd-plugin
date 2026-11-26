# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 Async CAM dispatch, shared overlap and layout restoration on CPU."""

from collections.abc import Callable
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402

from afd_plugin.config import AFD_ASYNC_CONNECTOR  # noqa: E402
from afd_plugin.connectors.npu.async_cam_runtime import (  # noqa: E402
    async_cam_dbo,
    async_cam_layout,
)
from afd_plugin.model_executor.ffn.npu import cam_runner  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v4_moe as model  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402


@pytest.mark.parametrize(
    "sp", [False, True], ids=["replicated-tp", "sequence-parallel"]
)
def test_draft_dispatch_restores_its_own_layout_and_adds_shared_once(
    cam_moe, monkeypatch, sp
):
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    ids = torch.arange(10, dtype=torch.int32).reshape(5, 2) % 3
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    monkeypatch.setattr(
        model, "compute_attention_gate_topk", lambda *_a, **_kw: (weights, ids)
    )
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    events: list[str] = []
    metadata = install_proxy_context(monkeypatch, [])
    metadata.stage_idx = 1

    def local(rows):
        return rows if sp else torch.cat((rows[3:], torch.zeros_like(rows[:1])))

    def send(states, context, **payload):
        events.append("send")
        assert context.metadata.layer_idx == 43
        assert context.metadata.stage_idx == 1
        assert context.metadata.seq_lens == [5 if sp else 3]
        assert payload.keys() == {"topk_ids", "topk_weights"}
        torch.testing.assert_close(states, local(hidden))
        torch.testing.assert_close(payload["topk_ids"], local(ids))
        torch.testing.assert_close(payload["topk_weights"], local(weights))

    def recv(*, ref_tensor, ubatch_idx, context):
        events.append("recv")
        assert ubatch_idx == context.metadata.stage_idx == 1
        torch.testing.assert_close(ref_tensor, local(hidden))
        return ref_tensor + 100

    class Shared(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["prefix"] == "mtp.0.mlp.shared_experts"
            assert kwargs["is_sequence_parallel"] == sp

        def forward(self, states):
            events.append("shared")
            assert states is hidden
            return states * 3

    block = cam_moe(
        gate=True,
        connector=AFD_ASYNC_CONNECTOR,
        layer_idx=43,
        prefix="mtp.0.mlp",
        is_draft_layer=True,
        sp=sp,
        shared_cls=Shared,
    )
    metadata.connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=recv)
    hooks: list[Callable] = []
    monkeypatch.setattr(async_cam_dbo, "dbo_enabled", lambda: True)
    monkeypatch.setattr(async_cam_dbo, "dbo_maybe_run_recv_hook", lambda: None)
    monkeypatch.setattr(async_cam_dbo, "dbo_register_recv_hook", hooks.append)

    def yield_to_peer(states, *, role):
        events.append("yield")
        metadata.stage_idx = 0
        hooks.pop()()
        return states

    def gather(local_output, dim):
        events.append("gather")
        torch.testing.assert_close(local_output, local(hidden) + 100)
        return torch.cat((hidden, torch.zeros_like(hidden[:1]))) + 100

    monkeypatch.setattr(cam_runner, "maybe_apply_dbo_yield", yield_to_peer)
    monkeypatch.setattr(async_cam_layout, "tensor_model_parallel_all_gather", gather)

    torch.testing.assert_close(block(hidden), hidden * 4 + 100)
    assert events == ["send", "shared", "yield", "recv"] + ([] if sp else ["gather"])
