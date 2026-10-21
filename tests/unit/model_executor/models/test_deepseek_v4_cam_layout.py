# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    async_cam_layout,  # noqa: E402
    deepseek_v4_attention_gate,
)
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402


def _make_proxy(monkeypatch, *, use_sequence_parallel: bool, in_profile_run: bool):
    proxy = object.__new__(adapter.AFDDeepseekV4AttentionGateRemoteMoE)
    torch.nn.Module.__init__(proxy)
    proxy.use_sequence_parallel_moe = use_sequence_parallel
    proxy.shared_experts = None
    proxy.layer_idx = 3
    connector = SimpleNamespace()
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(
            connector=connector,
            stage_idx=0,
            profile_pipeline=None,
            profile_batch=[(0, 3, 15)],
        ),
    )
    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", lambda states, **kw: states)
    topk_weights = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    topk_ids = torch.arange(32, dtype=torch.int32).reshape(16, 2)
    monkeypatch.setattr(
        deepseek_v4_attention_gate,
        "compute_attention_gate_topk",
        lambda _proxy, _hidden_states, **_kwargs: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(
            in_profile_run=in_profile_run,
        ),
    )
    return proxy, topk_weights, topk_ids


@pytest.mark.parametrize("profiling", [False, True])
def test_dsv4_profile_gate_shards_plain_tp8_and_restores(monkeypatch, profiling):
    proxy, topk_weights, topk_ids = _make_proxy(
        monkeypatch,
        use_sequence_parallel=False,
        in_profile_run=True,
    )
    tp_group = SimpleNamespace(world_size=8, rank_in_group=3)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    gathered_output = hidden_states + 100
    monkeypatch.setattr(
        async_cam_layout,
        "tensor_model_parallel_all_gather",
        lambda _tensor, token_dim: gathered_output,
    )
    sends = []
    execution = SimpleNamespace(
        tracer=SimpleNamespace(transition=Mock(), release_execution=Mock())
    )
    monkeypatch.setattr(
        adapter, "begin_profile_call", lambda _: execution if profiling else None
    )

    def send(_connector, local_hidden_states, context, **kwargs):
        assert context.metadata.layer_idx == proxy.layer_idx
        assert context.metadata.stage_idx == 0
        # Counts describe the unpadded batch, not this rank's two-token shard.
        assert context.profile_annotation == ("0|3|3|15" if profiling else "")
        sends.append((local_hidden_states, kwargs))
        return lambda: local_hidden_states + 100

    monkeypatch.setattr(adapter, "send_async_cam_remote_call", send)

    output = proxy(hidden_states)

    sent_hidden_states, send_kwargs = sends[0]
    assert torch.equal(sent_hidden_states, hidden_states[6:8])
    assert torch.equal(send_kwargs["topk_weights"], topk_weights[6:8])
    assert torch.equal(send_kwargs["topk_ids"], topk_ids[6:8])
    assert torch.equal(output, gathered_output)


def test_dsv4_gate_keeps_sequence_parallel_tokens_rank_local(monkeypatch):
    proxy, topk_weights, topk_ids = _make_proxy(
        monkeypatch,
        use_sequence_parallel=True,
        in_profile_run=False,
    )
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=3),
    )
    hidden_states = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    sends = []

    def send(_connector, local_hidden_states, context, **kwargs):
        sends.append((local_hidden_states, kwargs))
        return lambda: local_hidden_states + 1

    monkeypatch.setattr(adapter, "send_async_cam_remote_call", send)

    output = proxy(hidden_states)

    sent_hidden_states, send_kwargs = sends[0]
    assert sent_hidden_states is hidden_states
    assert send_kwargs["topk_weights"] is topk_weights
    assert send_kwargs["topk_ids"] is topk_ids
    assert torch.equal(output, hidden_states + 1)
