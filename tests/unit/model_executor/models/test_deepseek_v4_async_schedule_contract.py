# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

native = pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402


def test_dsv4_afd_schedule_executes_two_layers_and_two_stages():
    events: list[tuple] = []
    pending = {}
    stage_context = {
        0: SimpleNamespace(
            ubatch_idx=0,
            input_ids=(11, 13),
            attn_metadata="metadata-0",
        ),
        1: SimpleNamespace(
            ubatch_idx=1,
            input_ids=(17, 19),
            attn_metadata="metadata-1",
        ),
    }
    layers = [SimpleNamespace(layer_idx=0), SimpleNamespace(layer_idx=1)]

    class FakeConnector:
        def send(self, layer_idx, stage_idx, context, hc_post, hc_comb):
            assert context.ubatch_idx == stage_idx
            pending[layer_idx, stage_idx] = (hc_post, hc_comb, context)
            events.append(("send", layer_idx, stage_idx))

        def recv(self, layer_idx, stage_idx):
            hc_post, hc_comb, context = pending.pop((layer_idx, stage_idx))
            assert hc_post == f"hc-post-{layer_idx}-{stage_idx}"
            assert hc_comb == f"hc-comb-{layer_idx}-{stage_idx}"
            assert context is stage_context[stage_idx]
            events.append(("recv", layer_idx, stage_idx))

    connector = FakeConnector()

    def compute(layer, stage_idx):
        context = stage_context[stage_idx]
        assert context.ubatch_idx == stage_idx
        assert context.input_ids == ((11, 13), (17, 19))[stage_idx]
        assert context.attn_metadata == f"metadata-{stage_idx}"
        pending[layer.layer_idx, stage_idx] = (
            f"hc-post-{layer.layer_idx}-{stage_idx}",
            f"hc-comb-{layer.layer_idx}-{stage_idx}",
            context,
        )
        events.append(("compute", layer.layer_idx, stage_idx))

    def send(layer, stage_idx):
        hc_post, hc_comb, context = pending[layer.layer_idx, stage_idx]
        connector.send(layer.layer_idx, stage_idx, context, hc_post, hc_comb)

    def recv(layer, stage_idx):
        connector.recv(layer.layer_idx, stage_idx)

    adapter._run_two_stage_async_moe_schedule(layers, compute, send, recv)

    assert events == [
        ("compute", 0, 0),
        ("send", 0, 0),
        ("compute", 0, 1),
        ("recv", 0, 0),
        ("send", 0, 1),
        ("compute", 1, 0),
        ("recv", 0, 1),
        ("send", 1, 0),
        ("compute", 1, 1),
        ("recv", 1, 0),
        ("send", 1, 1),
        ("recv", 1, 1),
    ]
    assert pending == {}


def test_dsv4_afd_schedule_propagates_post_send_failure():
    events: list[tuple] = []
    layer = SimpleNamespace(layer_idx=0)

    def compute(_layer, stage_idx):
        events.append(("compute", stage_idx))
        if stage_idx == 1:
            raise RuntimeError("stage-one failed")

    def send(_layer, stage_idx):
        events.append(("send", stage_idx))

    with pytest.raises(RuntimeError, match="stage-one failed"):
        adapter._run_two_stage_async_moe_schedule(
            [layer],
            compute,
            send,
            lambda *_args: events.append(("recv",)),
        )

    assert events == [("compute", 0), ("send", 0), ("compute", 1)]


@pytest.mark.parametrize(
    ("input_ids", "inputs_embeds"),
    ((None, None), (object(), object())),
)
def test_dsv4_afd_ubatch_rejects_inputs_without_token_ids(
    monkeypatch,
    input_ids,
    inputs_embeds,
):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: object(),
    )

    with pytest.raises(NotImplementedError, match="requires token input_ids"):
        model.forward(input_ids, object(), None, inputs_embeds)


def test_dsv4_model_uses_native_forward_without_afd_stage_plan(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    sentinel = object()
    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: None,
    )
    monkeypatch.setattr(native.DeepseekV4Model, "forward", lambda *_args: sentinel)

    assert model.forward(None, object(), None, object()) is sentinel


def test_dsv4_constructor_signatures_match_target():
    for adapter_class, native_class in (
        (adapter.AFDDeepseekV4DecoderLayer, native.DeepseekV4DecoderLayer),
        (adapter.AFDDeepseekV4Model, native.DeepseekV4Model),
    ):
        assert inspect.signature(adapter_class.__init__) == inspect.signature(
            native_class.__init__,
        )
