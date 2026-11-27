# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 Async CAM dispatch, shared overlap and layout restoration on CPU."""

import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402

from afd_plugin.connectors.npu.async_cam_runtime import (  # noqa: E402
    async_cam_dbo,
    async_cam_layout,
)
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.vllm_config import (  # noqa: E402
    async_moe_config,
    attention_moe_vllm_config,
)


@pytest.fixture
def target_dbo_config(sp):
    config = attention_moe_vllm_config(enforce_eager=True)
    parallel = config.parallel_config
    parallel.enable_dbo = True
    parallel.enable_expert_parallel = True
    parallel.tensor_parallel_size = 2
    parallel.data_parallel_size = 2 if sp else 1
    parallel.all2all_backend = (
        "allgather_reducescatter" if sp else "flashinfer_all2allv"
    )
    config.additional_config = async_moe_config(
        tensor_parallel_size=parallel.tensor_parallel_size,
        num_attention_ranks=parallel.tensor_parallel_size * parallel.data_parallel_size,
    ).additional_config
    return config


@pytest.mark.parametrize(
    "sp", [False, True], ids=["replicated-tp", "sequence-parallel"]
)
def test_target_dbo_restores_its_own_layout_and_adds_shared_once(
    cam_moe, cam_factory_on_cpu, target_dbo_config, monkeypatch, sp
):  # long-test: check the wire handoff and restored output together
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    ids = torch.arange(10, dtype=torch.int32).reshape(5, 2) % 3
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
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
        assert context.metadata.layer_idx == 3
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
            assert kwargs["prefix"] == "model.layers.3.mlp.shared_experts"
            assert kwargs["is_sequence_parallel"] == sp

        def forward(self, states):
            events.append("shared")
            assert states is hidden
            return states * 3

    block = cam_moe(
        gate=True,
        vllm_config=target_dbo_config,
        shared_cls=Shared,
    )
    block.gate.weight_fp32 = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def select(logits, **kwargs):
        events.append("gate")
        torch.testing.assert_close(logits, hidden @ block.gate.weight_fp32.T)
        assert kwargs["norm_type"] == 0
        return weights, ids, None

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_moe_gating_top_k=select)
    )
    monkeypatch.setattr(
        torch.ops.vllm, "afd_balance_routing_ids_", lambda *_: None, raising=False
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

    monkeypatch.setattr(cam_factory_on_cpu, "maybe_apply_dbo_yield", yield_to_peer)
    monkeypatch.setattr(async_cam_layout, "tensor_model_parallel_all_gather", gather)

    torch.testing.assert_close(block(hidden), hidden * 4 + 100)
    assert events == ["gate", "send", "shared", "yield", "recv"] + (
        [] if sp else ["gather"]
    )
