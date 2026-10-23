# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("vllm")

from afd_plugin.model_executor.models.npu import async_cam_layout  # noqa: E402


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
        expected_ids = (
            topk_ids[:3]
            if tp_rank == 0
            else torch.cat((topk_ids[3:], topk_ids.new_zeros((1, 2))))
        )
        assert torch.equal(payload.topk_ids, expected_ids)
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


@pytest.mark.parametrize("num_tokens,tp_size", [(13, 2), (1, 4), (0, 2)])
@pytest.mark.parametrize("skip_padding_routes", [False, True])
def test_padding_ids_preserve_real_zero_expert_routes(
    monkeypatch, num_tokens, tp_size, skip_padding_routes
):
    tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)
    # Expert 0, zero input and zero probability are legal for every real row.
    hidden_states = torch.zeros(num_tokens, 2, dtype=torch.bfloat16)
    probabilities = torch.zeros(num_tokens, 6)
    ids = torch.zeros(num_tokens, 6, dtype=torch.int32)
    local_tokens = (num_tokens + tp_size - 1) // tp_size
    for rank in range(tp_size):
        tp_group.rank_in_group = rank
        payload = async_cam_layout.prepare_cam_dispatch_payload(
            hidden_states,
            probabilities,
            ids,
            None,
            use_sequence_parallel=False,
            skip_padding_routes=skip_padding_routes,
        )
        first_row = rank * local_tokens
        real_rows = max(0, min(local_tokens, num_tokens - first_row))
        expected = torch.zeros(local_tokens, 6, dtype=torch.int32)
        if skip_padding_routes:
            expected[real_rows:] = -1
        assert torch.equal(payload.topk_ids, expected)
        assert payload.hidden_states.shape[0] == local_tokens
        assert torch.equal(payload.topk_weights, torch.zeros(local_tokens, 6))
        assert payload.layout.parent_tokens == num_tokens
        assert payload.layout.padded_tokens == local_tokens * tp_size


@pytest.mark.parametrize(
    "use_sequence_parallel,tp_size", [(False, 2), (True, 2), (False, 1)]
)
def test_unpadded_routes_share_input_storage(
    monkeypatch, use_sequence_parallel, tp_size
):
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp_size, rank_in_group=0),
    )
    hidden_states = torch.zeros(4, 2)
    probabilities = torch.zeros(4, 6)
    ids = torch.zeros(4, 6, dtype=torch.int32)
    payload = async_cam_layout.prepare_cam_dispatch_payload(
        hidden_states,
        probabilities,
        ids,
        None,
        use_sequence_parallel=use_sequence_parallel,
        skip_padding_routes=True,
    )
    assert payload.hidden_states.data_ptr() == hidden_states.data_ptr()
    assert payload.topk_weights.data_ptr() == probabilities.data_ptr()
    assert payload.topk_ids.data_ptr() == ids.data_ptr()
    assert torch.count_nonzero(payload.topk_ids) == 0
