# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV2 enters SP at the MoE boundary, including one-token stages."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

pytest.importorskip("vllm")

from vllm.config import ParallelConfig  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    async_cam_layout as layout,
)
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v2_async_cam_forward as schedule,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


@pytest.mark.parametrize("dp_size,expected", [(1, False), (2, True)])
def test_target_sp_requires_dp_and_tp(dp_size, expected):
    # Exercise the pinned property's derivation without starting distributed
    # groups or running ParallelConfig's environment-dependent initialization.
    config = SimpleNamespace(
        data_parallel_size=dp_size,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )
    assert ParallelConfig.use_sequence_parallel_moe.fget(config) is expected


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


@pytest.mark.parametrize("use_sp", [False, True])
def test_two_stages_preserve_dense_prefix_and_restore_once(monkeypatch, use_sp):
    tp_size = 2
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    metadata = layout.AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=(
            AsyncMoeStage(slice(0, 1), slice(0, 1), input_tokens=2),
            AsyncMoeStage(slice(1, 2), slice(1, 4), input_tokens=4),
        ),
        parent_input_tokens=6,
        use_sequence_parallel=use_sp,
    )
    group = SimpleNamespace(world_size=tp_size, rank_in_group=0)
    monkeypatch.setattr(layout, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        schedule, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(schedule, "get_forward_context", lambda: context)
    active: list[int] = []

    def override(stage_context):
        active[:] = [stage_context.ubatch_idx]
        return nullcontext()

    monkeypatch.setattr(schedule, "override_forward_context", override)
    monkeypatch.setattr(schedule, "log_async_moe_stage_attention", lambda *args: None)
    events: list[str | tuple[str, int] | tuple[str, int, int]] = []
    complete_stages = {}

    class Dense:
        is_moe_layer = False

        def __call__(self, positions, states, residual, scaling):
            assert states.shape == hidden.shape
            assert residual is None
            events.append("dense")
            return states + 10, states + 100

    class MoE:
        is_moe_layer = True
        mlp = SimpleNamespace(shared_experts=None)

        def __init__(self, index):
            self.layer_idx = index

        def compute_attn_output(
            self, positions, states, residual, scaling, *, already_sequence_parallel
        ):
            stage_idx = active[0]
            stage = metadata.stages[stage_idx]
            assert already_sequence_parallel is (use_sp and self.layer_idx == 2)
            expected_rows = stage.actual_tokens if use_sp else stage.input_tokens
            if already_sequence_parallel:
                expected_rows = stage.input_tokens // tp_size
            assert states.shape[0] == expected_rows
            assert positions.shape[0] == (
                stage.actual_tokens if use_sp else stage.input_tokens
            )
            assert scaling.shape[-1] == positions.shape[0]
            if self.layer_idx == 1:
                if use_sp:
                    pad = stage.input_tokens - stage.actual_tokens
                    states = torch.nn.functional.pad(states, (0, 0, 0, pad))
                    residual = torch.nn.functional.pad(residual, (0, 0, 0, pad))
                complete_stages[stage_idx] = torch.cat((states, residual), dim=-1)
                if use_sp:
                    states = states.chunk(tp_size)[group.rank_in_group]
                    residual = residual.chunk(tp_size)[group.rank_in_group]
            events.append(("attention", self.layer_idx, stage_idx))
            topk = torch.ones((states.shape[0], 1))
            return states, residual, topk, topk.to(torch.int32), None

    # Model the communication boundary separately; layout boundary behavior is
    # covered by the shared layout tests. Here we check the stage schedule.
    monkeypatch.setattr(
        schedule, "restore_cam_dispatch_output", lambda states, dispatch_layout: states
    )
    # A non-None layout is part of the outstanding-send invariant.
    dispatch_layout = SimpleNamespace()

    def prepare(states, weights, ids, logits, *, use_sequence_parallel):
        assert use_sequence_parallel is use_sp
        return SimpleNamespace(
            hidden_states=states,
            topk_weights=weights,
            topk_ids=ids,
            router_logits=logits,
            layout=dispatch_layout,
        )

    monkeypatch.setattr(schedule, "prepare_cam_dispatch_payload", prepare)
    connector = SimpleNamespace(
        send_attn_output=lambda states, transfer, **kwargs: events.append(
            ("send", transfer.metadata.stage_idx)
        ),
        recv_ffn_output=lambda ref_tensor, ubatch_idx: ref_tensor,
    )
    gathers: list[int] = []

    def gather(states, dim):
        stage_idx = len(gathers)
        gathers.append(stage_idx)
        assert dim == 0
        torch.testing.assert_close(
            states, complete_stages[stage_idx].chunk(tp_size)[group.rank_in_group]
        )
        return complete_stages[stage_idx]

    monkeypatch.setattr(layout, "tensor_model_parallel_all_gather", gather)
    output, residual = schedule.run_async_moe_ubatch_afd_forward(
        SimpleNamespace(
            layers=[Dense(), MoE(1), MoE(2)],
            start_layer=0,
            end_layer=3,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    use_sequence_parallel_moe=use_sp, pipeline_parallel_size=1
                )
            ),
        ),
        hidden,
        None,
        torch.arange(6),
        SimpleNamespace(connector=connector),
        metadata,
        torch.ones(1, 6),
    )
    assert events[0] == "dense"
    assert gathers == ([0, 1] if use_sp else [])
    torch.testing.assert_close(output[:4], hidden[:4] + 10)
    torch.testing.assert_close(residual[:4], hidden[:4] + 100)
    assert output[4:].count_nonzero() == 0
    assert residual[4:].count_nonzero() == 0


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
            connector=SimpleNamespace(
                send_attn_output=lambda *args, **kwargs: None,
                recv_ffn_output=lambda ref_tensor, ubatch_idx: ref_tensor,
            ),
        ),
    )
    assert layer_calls == [False, True]
    assert len(layouts) == 2
    assert len(gathers) == 1
    torch.testing.assert_close(result[0], full)
    torch.testing.assert_close(result[1], residual)
