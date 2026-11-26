# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 model-to-CAMP2P contract on CPU."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402

from afd_plugin.model_executor.ffn import remote  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v4_moe as model  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402


@pytest.mark.parametrize("explicit", [False, True], ids=["context-ids", "model-ids"])
def test_ffn_gate_sends_local_ids(cam_moe, monkeypatch, explicit):
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    local_ids = torch.tensor([4, 7], dtype=torch.int32)
    context_ids = torch.arange(5) if explicit else local_ids
    monkeypatch.setattr(
        model, "get_forward_context", lambda: SimpleNamespace(input_ids=context_ids)
    )
    block = cam_moe(gate=False)
    hidden = torch.ones(2, 4)
    output = block(hidden, input_ids=local_ids if explicit else None)

    send, _, recv = events
    assert send[1] is hidden and recv[1] is hidden
    assert send[3].keys() == {"input_ids"}
    torch.testing.assert_close(send[3]["input_ids"], local_ids.long())
    torch.testing.assert_close(output, hidden * 0.25)
    assert list(block.parameters()) == []


def test_ffn_gate_requires_ids_before_send(cam_moe, monkeypatch):
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    monkeypatch.setattr(
        model, "get_forward_context", lambda: SimpleNamespace(input_ids=None)
    )
    with pytest.raises(RuntimeError, match="requires input_ids"):
        cam_moe(gate=False)(torch.zeros(2, 4))
    assert events == []


@pytest.mark.parametrize(
    "has_shared", [False, True], ids=["routed-only", "with-shared"]
)
def test_sync_gate_preserves_routes_and_stage_across_yield(
    cam_moe, monkeypatch, has_shared
):
    events: list[tuple] = []
    stage = SimpleNamespace(ubatch_idx=0)
    metadata = install_proxy_context(monkeypatch, events)
    monkeypatch.setattr(remote, "get_forward_context", lambda: stage)

    def switch_stage(hidden, *, role):
        events.append(("yield", role))
        stage.ubatch_idx = 1 - stage.ubatch_idx
        metadata.stage_idx = stage.ubatch_idx
        return hidden

    monkeypatch.setattr(remote, "maybe_apply_dbo_yield", switch_stage)

    class Shared(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(3.0))

        def forward(self, states):
            events.append(("shared", states))
            return states * self.scale

    block = cam_moe(gate=True, shared_cls=Shared if has_shared else None)
    expected_weights = {"gate.weight", "gate.e_score_correction_bias"}
    if has_shared:
        expected_weights.add("shared_experts.scale")
    assert set(block.state_dict()) == expected_weights
    for owner in (0, 1):
        hidden = torch.full((2, 4), 4.0 + owner)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32) + owner
        weights = torch.full((2, 2), 0.75 + owner)
        monkeypatch.setattr(
            model,
            "compute_attention_gate_topk",
            lambda *_a, routes=(weights, ids), **_kw: routes,
        )
        events.clear()
        output = block(hidden)

        assert [e[0] for e in events] == (
            ["send", "shared", "yield", "recv"]
            if has_shared
            else ["send", "yield", "recv"]
        )
        _, states, ctx, payload = events[0]
        assert states is hidden
        assert ctx.metadata.layer_idx == 3
        assert ctx.metadata.seq_lens == [2]
        assert ctx.metadata.stage_idx == events[-1][2] == owner
        assert payload.keys() == {"routing"}
        assert payload["routing"][0] is ids and payload["routing"][1] is weights
        if has_shared:
            assert events[1][1] is hidden
        torch.testing.assert_close(output, hidden * (3.25 if has_shared else 0.25))


def test_sync_gate_requires_forward_metadata(cam_moe, monkeypatch):
    monkeypatch.setattr(remote, "get_afd_metadata_from_forward_context", lambda: None)
    monkeypatch.setattr(
        model,
        "compute_attention_gate_topk",
        lambda *_a, **_kw: (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32)),
    )
    with pytest.raises(RuntimeError, match="AFD forward metadata"):
        cam_moe(gate=True)(torch.ones(1, 4))


def test_hash_gate_sends_routes_from_the_checkpoint_table(cam_moe, monkeypatch):
    block = cam_moe(
        gate=True,
        routing_config={
            "mlp_layer_types": ["moe"] * 3 + ["hash_moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 2.5,
        },
    )
    table = torch.tensor([[2, 1], [0, 2], [1, 0], [0, 1]], dtype=torch.int32)
    gate_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    block.load_state_dict({"gate.weight": gate_weight, "gate.tid2eid": table})
    # Supply the FP32 weight produced by Ascend's post-load hook.
    block.gate.weight_fp32 = block.gate.weight.detach().float()
    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    normalized = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    def select(*, x, input_ids, tid2eid, routed_scaling_factor, **kwargs):
        torch.testing.assert_close(x, hidden.float() @ gate_weight.T)
        return normalized * routed_scaling_factor, tid2eid[input_ids], None

    monkeypatch.setattr(
        torch.ops._C_ascend, "moe_gating_top_k_hash", select, raising=False
    )
    monkeypatch.setattr(
        torch.ops.vllm, "afd_balance_routing_ids_", lambda *_: None, raising=False
    )
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)

    output = block(hidden, input_ids=torch.tensor([2, -1]))

    ids, weights = events[0][3]["routing"]
    torch.testing.assert_close(ids, table[[2, 0]])
    torch.testing.assert_close(weights, normalized * 2.5)
    torch.testing.assert_close(output, hidden * 0.25)
