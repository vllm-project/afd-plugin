# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402
from vllm.model_executor.layers.fused_moe.router import (  # noqa: E402
    fused_topk_router,
)

from afd_plugin.config import AFD_ASYNC_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402

ROUTING_HIDDEN_SIZE = 16
ROUTING_NUM_EXPERTS = 8
ROUTING_TOP_K = 2


class _FakeConnector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor * 0.25


class _PassthroughNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _FakeAttention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states


def _install_fake_forward_context(monkeypatch, events, *, stage_idx=2):
    connector = _FakeConnector(events)
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=9)
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=stage_idx),
    )

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", record_yield)
    return afd_metadata


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["dense", "moe"])
def test_native_decoder_forward_calls_remote_proxy_once(
    monkeypatch,
    layer_idx,
):
    events: list[tuple] = []
    afd_metadata = _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(adapter.native, "DeepseekAttention", _FakeAttention)

    layer = object.__new__(adapter.AFDDeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.use_mha = True
    layer.use_sequence_parallel_moe = False
    layer.routed_scaling_factor = 4.0
    layer.input_layernorm = _PassthroughNorm()
    layer.self_attn = _FakeAttention()
    layer.post_attention_layernorm = _PassthroughNorm()
    layer.mlp = adapter.RemoteFFNProxy(layer_idx=layer_idx)

    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)
    output, residual = layer(
        torch.arange(2),
        hidden_states,
        None,
    )

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    sent_metadata = events[0][2].metadata
    assert sent_metadata.layer_idx == layer_idx
    assert sent_metadata.stage_idx == 2
    assert sent_metadata.seq_lens == [2]
    assert events[1][2] == "attention"
    assert events[2][2] == 2
    assert afd_metadata.stage_idx == 2
    assert torch.equal(output, hidden_states * 0.25)
    assert torch.equal(residual, hidden_states)


@pytest.mark.parametrize(
    ("mlp_type", "expected_scale"),
    [("dense", 0.25), ("moe", 1.0)],
)
def test_ffn_compute_applies_dense_fp16_scaling_once(
    monkeypatch,
    mlp_type,
    expected_scale,
):
    class FakeDenseMLP(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.clone()

    class FakeMoE(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.clone()

    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", FakeDenseMLP)
    layer = object.__new__(adapter.AFDDeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.compute_gate_on_attention = False
    layer.routed_scaling_factor = 4.0
    layer.mlp = FakeDenseMLP() if mlp_type == "dense" else FakeMoE()
    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)

    output = layer.compute_ffn_output(hidden_states)

    assert torch.equal(output, hidden_states * expected_scale)


def test_gate_proxy_sends_routing_payload(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    topk_weights = torch.tensor([[0.75, 0.25]])
    topk_ids = torch.tensor([[1, 3]])
    router_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    gate_calls = []

    def compute_gate_topk(**kwargs):
        gate_calls.append(kwargs)
        return topk_weights, topk_ids, router_logits

    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "compute_gate_topk",
        compute_gate_topk,
    )
    proxy = object.__new__(adapter.GateOnlyRemoteMoE)
    adapter.RemoteFFNProxy.__init__(proxy, layer_idx=3)
    proxy.gate = nn.Linear(4, 4, bias=False)
    proxy.vllm_config = object()
    proxy.config = object()
    proxy.top_k = 2
    proxy.gate_router = object()
    hidden_states = torch.ones(1, 4)

    output = proxy(hidden_states)

    assert len(gate_calls) == 1
    assert gate_calls[0]["gate"] is proxy.gate
    assert gate_calls[0]["gate_router"] is proxy.gate_router
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    send_kwargs = events[0][3]
    assert send_kwargs["router_logits"] is router_logits
    assert send_kwargs["topk_ids"] is topk_ids
    assert send_kwargs["topk_weights"] is topk_weights
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_experts_proxy_sends_router_logits(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    proxy = adapter.AFDAttentionFusedMoE(
        layer_idx=3,
        external_routing=True,
    )
    hidden_states = torch.ones(1, 4)
    router_logits = torch.ones(1, 8)

    output = proxy(hidden_states, router_logits)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    context = events[0][2]
    assert context.metadata.layer_idx == 3
    assert context.metadata.stage_idx == 1
    assert context.states is None
    assert events[0][3]["router_logits"] is router_logits
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_experts_proxy_without_external_routing_ships_hidden_states(
    monkeypatch,
):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    proxy = adapter.AFDAttentionFusedMoE(
        layer_idx=2,
        external_routing=False,
    )
    hidden_states = torch.ones(1, 4)

    output = proxy(hidden_states, hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert events[0][3] == {}
    assert torch.equal(output, hidden_states * 0.25)


COMBINED_VALUE = 1.0
ROUTE_WEIGHT = 0.5


class _InPlaceCombineConnector(_FakeConnector):
    """Combine writes the routed output into the Attention buffer, as DeepEP
    M2N does."""

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        # Record what was sent; combine overwrites the buffer afterwards.
        super().send_attn_output(hidden_states.clone(), context, **kwargs)

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor.fill_(COMBINED_VALUE)


class _LogitsGate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros(hidden_states.shape[0], ROUTING_NUM_EXPERTS), None


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states):
        return hidden_states * self.factor


def _routed_experts_moe(*, shared_experts, routed_scaling_factor, tp_size=1):
    moe = object.__new__(adapter.AFDDeepseekV2RoutedExpertsMoE)
    nn.Module.__init__(moe)
    moe.gate = _LogitsGate()
    moe.router = SimpleNamespace(
        select_experts=lambda hidden_states, router_logits: (
            torch.full((hidden_states.shape[0], ROUTING_TOP_K), ROUTE_WEIGHT),
            torch.zeros((hidden_states.shape[0], ROUTING_TOP_K), dtype=torch.int32),
        ),
    )
    moe.routed_scaling_factor = routed_scaling_factor
    moe.tp_size = tp_size
    moe.experts = adapter.AFDAttentionFusedMoE(layer_idx=3, external_routing=True)
    moe.shared_experts = shared_experts
    return moe


@pytest.mark.parametrize(
    ("shared_factor", "routed_scaling_factor"),
    [(None, 1.0), (3.0, 1.0), (3.0, 2.0)],
    ids=["routed-only", "shared", "shared-scaled"],
)
def test_m2n_moe_sends_topk_and_adds_shared_output_of_dispatched_input(
    monkeypatch, shared_factor, routed_scaling_factor
):
    """The DeepSeek M2N MoE sends top-k routes scaled by routed_scaling_factor
    and returns combine plus the shared experts of the input it dispatched.

    Why: combine overwrites the Attention buffer and sums weight * expert, so
    the shared MLP must read the input before combine, and the routed scale
    must ride in the weights.
    """
    events: list[tuple] = []
    afd_metadata = _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    afd_metadata.connector = _InPlaceCombineConnector(events)
    moe = _routed_experts_moe(
        shared_experts=None if shared_factor is None else _Scale(shared_factor),
        routed_scaling_factor=routed_scaling_factor,
    )
    hidden_value = 2.0
    hidden_states = torch.full((2, ROUTING_HIDDEN_SIZE), hidden_value)

    output = moe(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    send_kwargs = events[0][3]
    assert set(send_kwargs) == {"topk_ids", "topk_weights"}
    assert torch.equal(
        send_kwargs["topk_weights"],
        torch.full((2, ROUTING_TOP_K), ROUTE_WEIGHT * routed_scaling_factor),
    )
    shared_value = 0.0 if shared_factor is None else hidden_value * shared_factor
    assert torch.equal(
        output,
        torch.full((2, ROUTING_HIDDEN_SIZE), COMBINED_VALUE + shared_value),
    )


def test_remote_proxy_requires_forward_metadata(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
        adapter.RemoteFFNProxy(layer_idx=0)(torch.ones(1, 4))


def test_remote_proxy_exchanges_cam_during_profile(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(in_profile_run=True, ubatch_idx=0),
    )
    hidden_states = torch.ones(2, 4)

    output = adapter.RemoteFFNProxy(layer_idx=0)(hidden_states)

    assert torch.equal(output, hidden_states * 0.25)
    assert [event[0] for event in events] == ["send", "yield", "recv"]


def test_synchronous_model_forward_delegates_to_native(monkeypatch):
    calls = []
    expected = torch.ones(1, 4)

    def native_forward(instance, *args):
        calls.append((instance, args))
        return expected

    monkeypatch.setattr(adapter.native.DeepseekV2Model, "forward", native_forward)
    model = object.__new__(adapter.AFDDeepseekV2Model)
    nn.Module.__init__(model)
    model.afd_config = AFDConfig(role="attention")
    positions = torch.arange(1)

    output = adapter.AFDDeepseekV2Model.forward(
        model,
        None,
        positions,
        None,
    )

    assert output is expected
    assert calls == [(model, (None, positions, None, None))]


def test_async_connector_dispatches_to_schedule_adapter(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward

    expected = torch.ones(1, 4)
    calls = []

    def async_forward(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "run_model_forward",
        async_forward,
    )
    monkeypatch.setattr(
        adapter.native.DeepseekV2Model,
        "forward",
        lambda *_args: pytest.fail("native synchronous forward was called"),
    )
    model = object.__new__(adapter.AFDDeepseekV2Model)
    nn.Module.__init__(model)
    model.afd_config = AFDConfig(
        role="attention",
        connector=AFD_ASYNC_CONNECTOR,
    )
    positions = torch.arange(1)

    output = adapter.AFDDeepseekV2Model.forward(
        model,
        None,
        positions,
        None,
    )

    assert output is expected
    assert calls == [(model, None, positions, None, None)]


def test_decoder_inherits_native_forward_without_override():
    assert "forward" not in adapter.AFDDeepseekV2DecoderLayer.__dict__
    assert (
        adapter.AFDDeepseekV2DecoderLayer.forward
        is adapter.native.DeepseekV2DecoderLayer.forward
    )


def test_attention_gate_balances_only_profile_routing(monkeypatch):
    pytest.importorskip("vllm_ascend")
    from vllm.forward_context import override_forward_context

    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_attention_gate as gate_module,
    )

    monkeypatch.delenv("AFD_FORCE_BALANCED_TOPK_IDS", raising=False)
    logits = torch.zeros(3, 4)
    weights = torch.full((3, 2), 0.5)
    hot_ids = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)

    class Gate(nn.Module):
        e_score_correction_bias = None

        def forward(self, hidden_states):
            return logits, None

    connector = SimpleNamespace(
        select_experts=lambda **kwargs: (weights.clone(), hot_ids.clone())
    )
    monkeypatch.setattr(
        gate_module,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(connector=connector),
    )
    config = SimpleNamespace(n_routed_experts=4, n_shared_experts=0)
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    for is_profile in (False, True, False):
        context = SimpleNamespace(
            in_profile_run=is_profile,
            additional_kwargs={"in_profile_run": is_profile},
        )
        with override_forward_context(context):
            actual_weights, actual_ids, actual_logits = gate_module.compute_gate_topk(
                gate=Gate(),
                vllm_config=vllm_config,
                config=config,
                top_k=2,
                hidden_states=torch.zeros(3, 4),
            )
        expected_ids = torch.tensor([[0, 1], [2, 3], [0, 1]]) if is_profile else hot_ids
        torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(actual_weights, weights)
        assert actual_logits is logits


class _RecordingGate(nn.Module):
    """Linear gate returning ``(logits, bias)`` like vLLM's ReplicatedLinear."""

    def __init__(self, calls: list) -> None:
        super().__init__()
        self.calls = calls
        self.weight = nn.Parameter(
            torch.randn(ROUTING_NUM_EXPERTS, ROUTING_HIDDEN_SIZE)
        )

    def forward(self, hidden_states):
        self.calls.append(hidden_states)
        return hidden_states @ self.weight.t(), None


def _routing_proxy(gate: nn.Module, *, renormalize: bool = True):
    """A proxy that routes on Attention, configured as a model sets it up."""
    proxy = adapter.RemoteFFNProxy(layer_idx=3)
    proxy.gate = gate
    proxy.top_k = ROUTING_TOP_K
    proxy.renormalize = renormalize
    return proxy


def test_default_proxy_is_parameter_free_and_sends_no_routes(monkeypatch):
    """Without a gate the proxy sends only hidden states.

    Why: connectors other than M2N run the whole MoE block on FFN; adding the
    routing settings to the base proxy must not change their payload.
    """
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    proxy = adapter.RemoteFFNProxy(layer_idx=0)

    proxy(torch.ones(2, 4))

    assert list(proxy.parameters()) == []
    assert proxy.tp_size == 1
    assert events[0][3] == {}


def test_routing_proxy_routes_on_attention_and_sends_topk(monkeypatch):
    """Gate and top-k run once on Attention; only the routes go to the connector.

    Why: DeepEP M2N dispatches by topk_ids/topk_weights and its combine returns
    the weighted sum, so the proxy must not send logits or apply weights.
    """
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    gate_calls: list = []
    topk_calls: list[dict] = []
    topk_weights = torch.tensor([[0.75, 0.25]])
    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

    def fake_fused_topk(**kwargs):
        topk_calls.append(kwargs)
        return topk_weights, topk_ids, torch.zeros_like(topk_ids)

    monkeypatch.setattr(adapter, "fused_topk", fake_fused_topk)
    proxy = _routing_proxy(_RecordingGate(gate_calls), renormalize=False)
    hidden_states = torch.ones(1, ROUTING_HIDDEN_SIZE)

    output = proxy(hidden_states)

    assert gate_calls == [hidden_states]
    assert len(topk_calls) == 1
    call = topk_calls[0]
    assert call["hidden_states"] is hidden_states
    assert torch.equal(call["gating_output"], hidden_states @ proxy.gate.weight.t())
    assert call["topk"] == ROUTING_TOP_K
    assert call["renormalize"] is False
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    context = events[0][2]
    assert context.metadata.layer_idx == 3
    assert context.metadata.stage_idx == 1
    assert events[0][3] == {"topk_ids": topk_ids, "topk_weights": topk_weights}
    assert events[2][2] == 1
    assert torch.equal(output, hidden_states * 0.25)


def test_routing_proxy_owns_gate_at_native_checkpoint_path():
    """The gate is the proxy's only parameter, under ``gate``.

    Why: the proxy replaces ``mlp``, so ``model.layers.N.mlp.gate.weight``
    from the checkpoint loads into it with no name mapping.
    """
    proxy = _routing_proxy(_RecordingGate([]))

    assert set(dict(proxy.named_parameters())) == {"gate.weight"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused top-k needs CUDA")
@pytest.mark.parametrize("renormalize", [True, False])
def test_routing_proxy_matches_native_router(monkeypatch, renormalize):
    """On GPU the proxy's routes equal vLLM's native fused top-k router.

    Why: the FFN applies the weights the Attention side computes; any drift
    from native routing changes the model's output.
    """
    torch.manual_seed(0)
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=0)
    gate = _RecordingGate([]).cuda().to(torch.bfloat16)
    proxy = _routing_proxy(gate, renormalize=renormalize)
    hidden_states = torch.randn(
        33, ROUTING_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )

    proxy(hidden_states)

    sent = events[0][3]
    router_logits, _ = gate(hidden_states)
    native_weights, native_ids = fused_topk_router.FusedTopKRouter(
        top_k=ROUTING_TOP_K,
        global_num_experts=ROUTING_NUM_EXPERTS,
        renormalize=renormalize,
    )._compute_routing(hidden_states, router_logits, indices_type=None)
    assert torch.equal(sent["topk_ids"], native_ids)
    assert torch.equal(sent["topk_weights"], native_weights)

    reference = torch.softmax(router_logits.float(), dim=-1)
    reference_weights, reference_ids = reference.topk(ROUTING_TOP_K, dim=-1)
    if renormalize:
        reference_weights = reference_weights / reference_weights.sum(-1, keepdim=True)
    assert torch.equal(
        sent["topk_ids"].long().sort(-1).values, reference_ids.sort(-1).values
    )
    torch.testing.assert_close(
        sent["topk_weights"].sort(-1).values,
        reference_weights.sort(-1).values,
        rtol=1e-3,
        atol=1e-3,
    )


# Fake combine output scale of ``_FakeConnector.recv_ffn_output``.
COMBINE_SCALE = 0.25
# Added to the peer rank's rows by the fake all-gather, to tell them apart.
PEER_OFFSET = 100.0


class _TokenGate(nn.Module):
    """Logits that make expert ``token % ROUTING_NUM_EXPERTS`` win per row."""

    def forward(self, hidden_states):
        logits = torch.zeros(hidden_states.shape[0], ROUTING_NUM_EXPERTS)
        experts = hidden_states[:, 0].long() % ROUTING_NUM_EXPERTS
        logits[torch.arange(hidden_states.shape[0]), experts] = 1.0
        return logits, None


def _torch_topk(*, hidden_states, gating_output, topk, renormalize):
    weights, ids = torch.topk(gating_output.softmax(dim=-1), topk, dim=-1)
    return weights, ids.to(torch.int32), None


def _chunk_for_rank(tp_rank: int, tp_size: int):
    """vLLM's ``sequence_parallel_chunk`` for one rank: pad, then slice."""

    def chunk(x):
        padded = torch.nn.functional.pad(x, (0, 0, 0, (-x.shape[0]) % tp_size))
        rows = padded.shape[0] // tp_size
        return padded[tp_rank * rows : (tp_rank + 1) * rows].clone()

    return chunk


def _tokens(num_tokens: int):
    """Token ``i`` has value ``i`` in every channel."""
    return torch.arange(num_tokens, dtype=torch.float32)[:, None].repeat(
        1, ROUTING_HIDDEN_SIZE
    )


def _tp_routing_proxy(monkeypatch, tp_size: int):
    monkeypatch.setattr(adapter, "fused_topk", _torch_topk)
    proxy = _routing_proxy(_TokenGate())
    proxy.tp_size = tp_size
    return proxy


def test_routing_proxy_tp1_dispatches_every_token(monkeypatch):
    """Without Attention TP the proxy dispatches the whole batch unsplit.

    Why: TP = 1 must keep the existing data path.
    """

    def no_split(x):
        raise AssertionError("TP = 1 must not split")

    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(adapter.native, "sequence_parallel_chunk", no_split)
    hidden_states = _tokens(3)

    output = _tp_routing_proxy(monkeypatch, tp_size=1)(hidden_states)

    sends = [event for event in events if event[0] == "send"]
    assert len(sends) == 1
    assert torch.equal(sends[0][1], hidden_states)
    assert torch.equal(output, hidden_states * COMBINE_SCALE)


@pytest.mark.parametrize(
    ("tp_rank", "sent_tokens", "padding_rows"),
    [(0, [0, 1], []), (1, [2, 0], [1])],
)
def test_routing_proxy_tp2_rank_dispatches_its_slice_and_gathers(
    monkeypatch, tp_rank, sent_tokens, padding_rows
):
    """With TP 2 and 3 tokens, rank 0 dispatches tokens 0-1 and rank 1 token 2
    plus one padding row whose experts are all ``UNROUTED_EXPERT_ID``; the
    combined slices are all-gathered and trimmed back to 3 tokens.

    Why: TP peers hold the same tokens; each must dispatch a disjoint slice
    so every token reaches its experts exactly once.
    """
    tp_size = 2
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(
        adapter.native, "sequence_parallel_chunk", _chunk_for_rank(tp_rank, tp_size)
    )
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    gathered_inputs = []

    def all_gather(local_output, dim):
        gathered_inputs.append(local_output.clone())
        return torch.cat([local_output, local_output + PEER_OFFSET], dim=dim)

    monkeypatch.setattr(adapter.native, "tensor_model_parallel_all_gather", all_gather)

    output = _tp_routing_proxy(monkeypatch, tp_size)(_tokens(3))

    (send,) = [event for event in events if event[0] == "send"]
    sent_states, sent_kwargs = send[1], send[3]
    assert sent_states[:, 0].tolist() == sent_tokens
    topk_ids = sent_kwargs["topk_ids"]
    unrouted = (topk_ids == adapter.UNROUTED_EXPERT_ID).all(dim=1)
    assert unrouted.nonzero().flatten().tolist() == padding_rows
    assert not (topk_ids[~unrouted] == adapter.UNROUTED_EXPERT_ID).any()
    (local_output,) = gathered_inputs
    assert torch.equal(local_output, sent_states * COMBINE_SCALE)
    expected = torch.cat([local_output, local_output + PEER_OFFSET])[:3]
    assert torch.equal(output, expected)


def test_routing_proxy_tp4_with_one_token_routes_only_rank0(monkeypatch):
    """One decode token over TP 4: rank 0 routes it, ranks 1-3 send one
    unrouted padding row each.

    Why: decode batches can be smaller than the TP size; padding must never
    reach an expert.
    """
    tp_size = 4
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(
        adapter.native,
        "tensor_model_parallel_all_gather",
        lambda local, dim: torch.cat([local] * tp_size, dim=dim),
    )
    for tp_rank in range(tp_size):
        monkeypatch.setattr(
            adapter.native, "sequence_parallel_chunk", _chunk_for_rank(tp_rank, tp_size)
        )
        monkeypatch.setattr(
            adapter.native,
            "get_tensor_model_parallel_rank",
            lambda rank=tp_rank: rank,
        )

        output = _tp_routing_proxy(monkeypatch, tp_size)(_tokens(1))

        assert output.shape == (1, ROUTING_HIDDEN_SIZE)
    unrouted = [
        bool((event[3]["topk_ids"] == adapter.UNROUTED_EXPERT_ID).all())
        for event in events
        if event[0] == "send"
    ]
    assert unrouted == [False, True, True, True]


@pytest.mark.parametrize(
    ("tp_rank", "sent_tokens", "padding_rows"),
    [(0, [0, 1], []), (1, [2, 0], [1])],
)
def test_m2n_moe_tp2_rank_dispatches_its_slice_and_gathers_shared_sum(
    monkeypatch, tp_rank, sent_tokens, padding_rows
):
    """With TP 2 and 3 tokens the DeepSeek M2N MoE routes and dispatches only
    its slice, unroutes the padding row, adds the shared output of the slice
    before combine overwrites it, and all-gathers routed plus shared.

    Why: TP peers hold the same tokens; each must dispatch a disjoint slice,
    and the replicated shared MLP must see the same rows as combine.
    """
    tp_size = 2
    shared_factor = 3.0
    events: list[tuple] = []
    afd_metadata = _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    afd_metadata.connector = _InPlaceCombineConnector(events)
    monkeypatch.setattr(
        adapter.native, "sequence_parallel_chunk", _chunk_for_rank(tp_rank, tp_size)
    )
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    gathered_inputs = []

    def all_gather(local_output, dim):
        gathered_inputs.append(local_output.clone())
        return torch.cat([local_output, local_output + PEER_OFFSET], dim=dim)

    monkeypatch.setattr(adapter.native, "tensor_model_parallel_all_gather", all_gather)
    moe = _routed_experts_moe(
        shared_experts=_Scale(shared_factor),
        routed_scaling_factor=1.0,
        tp_size=tp_size,
    )

    output = moe(_tokens(3))

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    sent_states, sent_kwargs = events[0][1], events[0][3]
    assert sent_states[:, 0].tolist() == sent_tokens
    unrouted = (sent_kwargs["topk_ids"] == adapter.UNROUTED_EXPERT_ID).all(dim=1)
    assert unrouted.nonzero().flatten().tolist() == padding_rows
    (local_output,) = gathered_inputs
    expected_local = COMBINED_VALUE + shared_factor * torch.tensor(
        sent_tokens, dtype=torch.float32
    )[:, None].expand(-1, ROUTING_HIDDEN_SIZE)
    assert torch.equal(local_output, expected_local)
    assert torch.equal(
        output, torch.cat([local_output, local_output + PEER_OFFSET])[:3]
    )
