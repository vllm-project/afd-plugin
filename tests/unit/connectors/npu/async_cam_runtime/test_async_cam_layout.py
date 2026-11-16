# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("vllm")

from afd_plugin.connectors.npu.async_cam_runtime import async_cam_layout  # noqa: E402


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
