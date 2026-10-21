# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV2 enters SP at the MoE boundary, including one-token stages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

pytest.importorskip("vllm")

from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v2_async_cam_forward as schedule,
)


@pytest.mark.parametrize("num_tokens", [1, 3, 4])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_decoder_first_and_later_moe_own_collectives(monkeypatch, num_tokens, tp_rank):
    tp_size = 2
    full = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    padded = torch.nn.functional.pad(full, (0, 0, 0, (-num_tokens) % tp_size))
    local = padded.chunk(tp_size)[tp_rank]
    events = []

    def gather(value, dim):
        events.append("gather")
        torch.testing.assert_close(value, local)
        assert dim == 0
        return padded

    def reduce_scatter(value, dim):
        events.append("reduce_scatter")
        torch.testing.assert_close(value, padded)
        assert dim == 0
        return local

    def chunk(value):
        events.append("chunk_residual")
        torch.testing.assert_close(value, full)
        return local

    def attention(*, positions, hidden_states):
        events.append("attention")
        assert positions.shape[0] == num_tokens
        torch.testing.assert_close(hidden_states, full)
        return hidden_states

    def norm(hidden, residual=None):
        return hidden if residual is None else (hidden, residual)

    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(adapter.native, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(
        adapter.native, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    monkeypatch.setattr(adapter.native, "sequence_parallel_chunk", chunk)
    layer = SimpleNamespace(
        cam_profile_enabled=False,
        use_sequence_parallel_moe=True,
        input_layernorm=norm,
        post_attention_layernorm=norm,
        self_attn=attention,
        use_mha=True,
        compute_gate_on_attention=False,
    )
    for already_sp in (False, True):
        events.clear()
        result = adapter.AFDDeepseekV2DecoderLayer.compute_attn_output(
            layer,
            torch.arange(num_tokens),
            local if already_sp else full,
            local if already_sp else None,
            already_sequence_parallel=already_sp,
        )
        torch.testing.assert_close(result[0], local)
        torch.testing.assert_close(result[1], local)
        assert events == (
            ["gather", "attention", "reduce_scatter"]
            if already_sp
            else ["attention", "reduce_scatter", "chunk_residual"]
        )


@pytest.mark.parametrize("dense_tail", [False, True])
def test_single_stage_gathers_once_at_dense_or_output_boundary(monkeypatch, dense_tail):
    # Even a single real token needs an explicit gather: its SP shard also has
    # one row and shape comparison cannot tell the two layouts apart.
    full = torch.tensor([[3.0, 4.0]])
    residual = torch.tensor([[7.0, 8.0]])
    layouts = []
    gathers: list[Tensor] = []
    layer_calls = []

    class MoE:
        is_moe_layer = True
        use_sequence_parallel_moe = True
        mlp = SimpleNamespace(shared_experts=None)

        def __init__(self, index):
            self.layer_idx = index

        def compute_attn_output(
            self,
            positions,
            states,
            previous_residual,
            scaling,
            *,
            already_sequence_parallel,
        ):
            layer_calls.append(already_sequence_parallel)
            topk = torch.ones(1, 1)
            return full.clone(), residual.clone(), topk, topk.to(torch.int32), None

    class Dense:
        is_moe_layer = False

        def __call__(self, positions, states, previous_residual, scaling):
            assert len(gathers) == 1
            torch.testing.assert_close(states, full)
            torch.testing.assert_close(previous_residual, residual)
            return states, previous_residual

    def prepare(states, weights, ids, logits, *, use_sequence_parallel):
        assert use_sequence_parallel
        dispatch_layout = SimpleNamespace()
        layouts.append(dispatch_layout)
        return SimpleNamespace(
            hidden_states=states,
            topk_weights=weights,
            topk_ids=ids,
            router_logits=logits,
            layout=dispatch_layout,
        )

    def gather(states, dim):
        assert dim == 0
        gathers.append(states)
        torch.testing.assert_close(states, torch.cat((full, residual), dim=-1))
        return torch.cat((states, torch.zeros_like(states)))

    monkeypatch.setattr(schedule, "prepare_cam_dispatch_payload", prepare)
    monkeypatch.setattr(
        schedule, "restore_cam_dispatch_output", lambda states, dispatch_layout: states
    )
    monkeypatch.setattr(schedule, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(
        schedule, "maybe_apply_dbo_yield", lambda states, **kwargs: states
    )
    layers: list[MoE | Dense] = [MoE(0), MoE(1)]
    if dense_tail:
        layers.append(Dense())
    result = schedule.run_attention_gate_afd_forward(
        SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers)),
        full,
        None,
        torch.arange(1),
        SimpleNamespace(
            stage_idx=0,
            profile_pipeline=None,
            connector=SimpleNamespace(
                send_attn_output=lambda *args, **kwargs: None,
                recv_ffn_output=lambda ref_tensor, ubatch_idx, context=None: ref_tensor,
            ),
        ),
    )
    assert layer_calls == [False, True]
    assert len(layouts) == 2
    assert len(gathers) == 1
    torch.testing.assert_close(result[0], full)
    torch.testing.assert_close(result[1], residual)
