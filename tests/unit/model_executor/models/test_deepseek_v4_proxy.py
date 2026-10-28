# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    override_forward_context,
)
from vllm.v1.worker.workspace import use_workspace_lane  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402
from tests.unit.fakes.deepseek_v4 import (  # noqa: E402
    Connector as _FakeConnector,
)
from tests.unit.fakes.deepseek_v4 import (  # noqa: E402
    DecoderCase,
    WorkspaceCase,
    mhc_step,
)


def test_remote_v4_ffn_sends_token_ids(monkeypatch):
    events: list[tuple] = []
    connector = _FakeConnector(events)
    afd_metadata = SimpleNamespace(
        connector=connector,
        stage_idx=9,
    )
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=2, slot_mapping={}),
    )

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", record_yield)
    proxy = adapter.RemoteDeepseekV4FFN(layer_idx=3)
    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)
    input_ids = torch.tensor([11, 13], dtype=torch.int32)

    output = proxy(hidden_states, input_ids)

    assert adapter.AFDDeepseekV4ForCausalLM.afd_requires_input_ids
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    sent_context = events[0][2]
    assert sent_context.metadata.layer_idx == 3
    assert sent_context.metadata.stage_idx == 2
    assert sent_context.metadata.seq_lens == [2]
    assert sent_context.states is None
    assert set(events[0][3]) == {"input_ids"}
    assert events[0][3]["input_ids"] is input_ids
    assert events[0][3]["input_ids"].dtype is torch.int32
    assert events[1][2] == "attention"
    assert events[2][2] == 2
    assert afd_metadata.stage_idx == 2
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_v4_ffn_preserves_ids_in_padding_slots(monkeypatch):
    events: list[tuple] = []
    connector = _FakeConnector(events)
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(
            ubatch_idx=0,
            slot_mapping={"model.layers.0.attn": torch.tensor([5, 6, -1, -1])},
        ),
    )
    monkeypatch.setattr(
        adapter,
        "maybe_apply_dbo_yield",
        lambda hidden_states, *, role: hidden_states,
    )
    proxy = adapter.RemoteDeepseekV4FFN(layer_idx=0)
    hidden_states = torch.ones((4, 4), dtype=torch.float16)
    input_ids = torch.tensor([11, 13, 17, 19], dtype=torch.int32)

    proxy(hidden_states, input_ids)

    sent_input_ids = events[0][3]["input_ids"]
    assert sent_input_ids is input_ids
    assert sent_input_ids.dtype is torch.int32
    assert sent_input_ids.tolist() == [11, 13, 17, 19]
    assert input_ids.tolist() == [11, 13, 17, 19]


@pytest.mark.parametrize(
    "input_ids",
    [None, torch.ones((2, 1), dtype=torch.int64), torch.ones(3, dtype=torch.int64)],
    ids=["missing", "not-one-dimensional", "token-count-mismatch"],
)
def test_remote_v4_ffn_validates_token_ids_before_metadata_lookup(
    monkeypatch,
    input_ids,
):
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: pytest.fail("metadata lookup must follow token-id validation"),
    )
    proxy = adapter.RemoteDeepseekV4FFN(layer_idx=0)

    error = RuntimeError if input_ids is None else ValueError
    message = (
        "requires input_ids"
        if input_ids is None
        else "one-dimensional and token-aligned"
    )
    with pytest.raises(error, match=message):
        proxy(torch.ones((2, 4)), input_ids)


def test_v4_decoder_forward_rejects_ffn_role(monkeypatch):
    class FakeMissingLayer(nn.Module):
        pass

    monkeypatch.setattr(adapter.native, "PPMissingLayer", FakeMissingLayer)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.attn = FakeMissingLayer()

    with pytest.raises(RuntimeError, match="decoder forward is Attention-owned"):
        layer.forward(
            torch.ones((2, 4)),
            positions=torch.arange(2),
            input_ids=None,
            mega_gate_metadata=None,
        )


def test_v4_ffn_compute_rejects_attention_role_before_input_ids():
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.ffn = adapter.RemoteDeepseekV4FFN(layer_idx=0)

    with pytest.raises(RuntimeError, match="FFN compute is FFN-role only"):
        layer.compute_ffn_output(torch.ones((2, 4)), input_ids=None)


def test_v4_ffn_compute_requires_input_ids(monkeypatch):
    class FakeMoE(nn.Module):
        pass

    monkeypatch.setattr(adapter.native, "DeepseekV4MoE", FakeMoE)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.ffn = FakeMoE()

    with pytest.raises(RuntimeError, match="FFN compute requires input_ids"):
        layer.compute_ffn_output(torch.ones((2, 4)), input_ids=None)


@pytest.fixture
def decoder_case(monkeypatch):
    def build(**kwargs):
        case = DecoderCase(adapter, **kwargs)
        monkeypatch.setattr(adapter.native, "mhc_fused_post_pre_tilelang", mhc_step)
        monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", case.record_yield)
        return case

    return build


@pytest.mark.parametrize(
    "real_tokens,num_stages,all_padding,ffn_value",
    [
        (0, 2, False, 1),
        (2, 2, False, 28),
        (4, 2, True, 28),
        (None, 2, False, 28),
        (0, 1, False, 28),
    ],
    ids=[
        "empty-stage-1",
        "nonempty-stage",
        "positive-capture-dummy",
        "unknown-extent",
        "not-ubatched",
    ],
)
def test_v4_decoder_preserves_padded_ffn_protocol_for_stage_extent(
    decoder_case,
    real_tokens,
    num_stages,
    all_padding,
    ffn_value,
):
    case = decoder_case(
        real_tokens=real_tokens,
        num_stages=num_stages,
        all_padding=all_padding,
    )
    output, residual, post_mix, res_mix = case.forward()

    assert torch.equal(output, torch.full_like(case.x, ffn_value * 0.25))
    assert torch.equal(residual, torch.full_like(case.x, 14))
    assert torch.equal(post_mix, torch.full_like(case.x, 26))
    assert torch.equal(res_mix, torch.full_like(case.x, 38))
    assert [event[0] for event in case.events] == ["send", "yield", "recv"]
    sent = case.events[0]
    assert torch.equal(sent[1], torch.full_like(case.x, ffn_value))
    assert sent[2].metadata.layer_idx == 3
    assert sent[2].metadata.stage_idx == (1 if num_stages > 1 else 0)
    assert sent[2].metadata.seq_lens == [4]
    assert sent[3]["input_ids"] is case.input_ids
    assert case.input_ids.tolist() == [11, 13, 17, 19]
    assert case.events[1][2] == "attention"
    assert case.events[2][2] == (1 if num_stages > 1 else 0)


def test_v4_decoder_without_afd_metadata_preserves_attention_error(decoder_case):
    case = decoder_case()
    case.context.additional_kwargs.clear()
    error = ValueError("native attention failed")
    case.layer.attn.failure = error

    with pytest.raises(ValueError, match="native attention failed") as caught:
        case.forward()

    assert caught.value is error
    assert case.events == []


def test_v4_decoder_resumes_attention_when_stage_has_real_tokens(decoder_case):
    case = decoder_case(real_tokens=0)
    first_output = case.forward()[0]
    assert torch.equal(first_output, torch.full_like(case.x, 0.25))
    case.metadata.tokens_unpadded_lens = [2]
    case.x.fill_(12)

    second_output = case.forward()[0]

    assert torch.equal(second_output, torch.full_like(case.x, 10))
    assert [event[0] for event in case.events] == ["send", "yield", "recv"] * 2
    assert torch.equal(case.events[3][1], torch.full_like(case.x, 40))
    assert case.events[3][2].metadata.seq_lens == [4]
    assert case.events[3][3]["input_ids"] is case.input_ids


@pytest.mark.parametrize("prior_override", [False, True])
def test_v4_workspace_reservation_error_restores_owner_and_forward_context(
    monkeypatch, prior_override
):
    error = ValueError("native reservation failed")
    case = WorkspaceCase(monkeypatch, num_lanes=2, error=error)
    lane = 0 if prior_override else 1
    case.ubatch = 0 if prior_override else 1
    if prior_override:
        case.install_prior_slot_override()
    # Public context/returned-storage observations establish restoration; no
    # private selector/array identity assertions are used.
    with use_workspace_lane(lane), override_forward_context(case.outer_context):
        outside = get_forward_context()
        before = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        before.fill_(71)
        with pytest.raises(ValueError, match="native reservation failed") as caught:
            adapter._reserve_flashmla_ubatch_workspace(case.model, case.config)
        after = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        assert caught.value is error
        assert get_forward_context() is outside
        assert after.data_ptr() == before.data_ptr()
        assert after.item() == 71


def test_v4_workspace_reservation_success_restores_instance_owner_and_context(
    monkeypatch,
):
    case = WorkspaceCase(monkeypatch, num_lanes=2)
    case.install_prior_slot_override()
    # Fully reserve the original selected owner first so the helper must keep
    # this returned storage while routing the remaining independent slots.
    before = case.request(0, 0)
    before.fill_(37)
    with override_forward_context(case.outer_context):
        outside = get_forward_context()
        adapter._reserve_flashmla_ubatch_workspace(case.model, case.config)
        # Existing override selects its own owner, not the helper's last slot.
        case.ubatch = 0
        after = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        assert get_forward_context() is outside
        assert after.data_ptr() == before.data_ptr()
        assert after.item() == 37
