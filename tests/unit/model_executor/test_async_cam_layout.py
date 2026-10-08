# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

pytest.importorskip("vllm")

from afd_plugin.model_executor.models.npu import async_cam_layout  # noqa: E402
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    AsyncMoeUbatchMetadata,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


@pytest.mark.parametrize("tp_rank", range(4))
def test_sp_layout_shards_stages_and_restores_global_tokens_once(monkeypatch, tp_rank):
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=(
            AsyncMoeStage(slice(0, 1), slice(0, 5), input_tokens=8),
            AsyncMoeStage(slice(0, 1), slice(5, 12), input_tokens=8),
        ),
        parent_input_tokens=13,
        use_sequence_parallel=True,
    )
    global_hidden = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    global_residual = global_hidden + 100
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=tp_rank),
    )
    gathered: list[Tensor] = []
    physical_stages = [
        torch.cat((global_hidden[:5], torch.zeros(3, 2))),
        torch.cat((global_hidden[5:12], torch.zeros(1, 2))),
    ]

    def all_gather(local, token_dim):
        assert token_dim == 0
        stage = physical_stages[len(gathered)]
        assert torch.equal(local, stage[tp_rank * 2 : (tp_rank + 1) * 2])
        gathered.append(local)
        return stage

    monkeypatch.setattr(
        async_cam_layout, "tensor_model_parallel_all_gather", all_gather
    )
    stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
        global_hidden,
        global_residual,
        torch.arange(13),
        torch.ones(2, 13),
        metadata,
    )
    assert gathered == []  # Embeddings are global; no gather is needed to stage them.
    assert [x.tolist() for x in stage_inputs.positions] == [
        list(range(5)),
        list(range(5, 12)),
    ]
    assert [tuple(x.shape) for x in stage_inputs.llama_4_scaling] == [(2, 5), (2, 7)]
    assert [x.shape[0] for x in stage_inputs.hidden_states] == [2, 2]
    for hidden in stage_inputs.hidden_states:
        dispatch = async_cam_layout.prepare_cam_dispatch_payload(
            hidden,
            torch.ones(2, 1),
            torch.zeros(2, 1),
            None,
            use_sequence_parallel=True,
        )
        assert dispatch.hidden_states is hidden
        assert (
            async_cam_layout.restore_cam_dispatch_output(hidden, dispatch.layout)
            is hidden
        )
    assert gathered == []  # Remote FFN retains the model's rank-local token layout.
    restored = async_cam_layout.restore_async_moe_stage_outputs(
        stage_inputs.hidden_states, metadata
    )
    expected = global_hidden.clone()
    expected[12].zero_()
    assert torch.equal(restored, expected)
    assert len(gathered) == 2


def test_replicated_layout_removes_and_restores_parent_padding():
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 3), input_tokens=3),
            AsyncMoeStage(slice(0, 1), slice(3, 5), input_tokens=2),
        ],
        parent_input_tokens=8,
        use_sequence_parallel=False,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    positions = torch.arange(8)

    stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
    )

    assert [stage[:, 0].tolist() for stage in stage_inputs.hidden_states] == [
        [0.0, 2.0, 4.0],
        [6.0, 8.0],
    ]
    assert [stage.tolist() for stage in stage_inputs.positions] == [
        [0, 1, 2],
        [3, 4],
    ]
    restored = async_cam_layout.restore_async_moe_stage_outputs(
        stage_inputs.hidden_states,
        metadata,
    )
    assert torch.equal(restored[:5], hidden_states[:5])
    assert torch.count_nonzero(restored[5:]) == 0


def test_plain_tp_cam_boundary_shards_and_restores_replicated_tokens(monkeypatch):
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    router_logits = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    padded_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100

    monkeypatch.setattr(
        async_cam_layout,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_output,
    )

    expected_hidden_rows = (
        hidden_states[:3],
        torch.cat((hidden_states[3:], hidden_states.new_zeros((1, 2)))),
    )
    for tp_rank in range(2):
        tp_group.rank_in_group = tp_rank
        payload = async_cam_layout.prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=False,
        )

        assert torch.equal(payload.hidden_states, expected_hidden_rows[tp_rank])
        assert payload.hidden_states.shape[0] == 3
        assert payload.topk_weights.shape[0] == 3
        assert payload.topk_ids.shape[0] == 3
        assert payload.router_logits is not None
        assert payload.router_logits.shape[0] == 3
        assert payload.layout.parent_tokens == 5
        assert payload.layout.padded_tokens == 6
        assert payload.layout.requires_tp_all_gather is True

        local_output = padded_output[tp_rank * 3 : (tp_rank + 1) * 3]
        restored = async_cam_layout.restore_cam_dispatch_output(
            local_output,
            payload.layout,
        )
        assert torch.equal(restored, padded_output[:5])
