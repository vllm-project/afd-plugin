# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check the token layout around DSV4's TP-sharded shared experts."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_shared_experts as shared,
)


def test_sp_shared_experts_gather_before_mlp_and_reduce_scatter_after(monkeypatch):
    expert = shared.AFDDeepseekV4SharedExperts.__new__(
        shared.AFDDeepseekV4SharedExperts
    )
    torch.nn.Module.__init__(expert)
    expert.is_sequence_parallel = True
    expert.weights_replicated = False
    monkeypatch.setattr(shared, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        shared, "get_forward_context", lambda: SimpleNamespace(num_tokens=5)
    )

    full_tokens = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    local_tokens = full_tokens[3:]
    calls = []

    def gather(tokens, dim):
        assert dim == 0
        assert torch.equal(tokens, local_tokens)
        calls.append("gather")
        return full_tokens

    def local_mlp(tokens):
        assert torch.equal(tokens, full_tokens[:5])
        calls.append("mlp")
        return tokens + 10

    def reduce_scatter(tokens, dim):
        assert dim == 0
        assert torch.equal(tokens, torch.tensor([[10.0], [11], [12], [13], [14], [0]]))
        calls.append("reduce_scatter")
        return tokens[3:]

    monkeypatch.setattr(shared, "tensor_model_parallel_all_gather", gather)
    monkeypatch.setattr(shared, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(expert, "_run_local_mlp", local_mlp)

    assert torch.equal(expert(local_tokens), torch.tensor([[13.0], [14], [0]]))
    assert calls == ["gather", "mlp", "reduce_scatter"]


def test_tp_shared_experts_all_reduce_partial_output(monkeypatch):
    expert = shared.AFDDeepseekV4SharedExperts.__new__(
        shared.AFDDeepseekV4SharedExperts
    )
    torch.nn.Module.__init__(expert)
    expert.is_sequence_parallel = False
    expert.weights_replicated = False
    monkeypatch.setattr(shared, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(expert, "_run_local_mlp", lambda x: x + 1)
    monkeypatch.setattr(shared, "tensor_model_parallel_all_reduce", lambda x: x * 4)

    tokens = torch.tensor([[2.0]])
    assert torch.equal(expert(tokens), torch.tensor([[12.0]]))


def test_sp_replicated_shared_experts_keep_local_tokens(monkeypatch):
    expert = shared.AFDDeepseekV4SharedExperts.__new__(
        shared.AFDDeepseekV4SharedExperts
    )
    torch.nn.Module.__init__(expert)
    expert.is_sequence_parallel = True
    expert.weights_replicated = True
    monkeypatch.setattr(shared, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(expert, "_run_local_mlp", lambda x: x + 1)

    tokens = torch.tensor([[2.0]])
    assert torch.equal(expert(tokens), torch.tensor([[3.0]]))


def test_tp_replicated_shared_experts_gather_disjoint_outputs(monkeypatch):
    expert = shared.AFDDeepseekV4SharedExperts.__new__(
        shared.AFDDeepseekV4SharedExperts
    )
    torch.nn.Module.__init__(expert)
    expert.is_sequence_parallel = False
    expert.weights_replicated = True
    monkeypatch.setattr(shared, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        shared, "get_tp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    full_tokens = torch.arange(5, dtype=torch.float32).unsqueeze(-1)

    def local_mlp(tokens):
        assert torch.equal(tokens, torch.tensor([[3.0], [4], [0]]))
        return tokens + 10

    def gather(tokens, dim):
        assert dim == 0
        assert torch.equal(tokens, torch.tensor([[13.0], [14], [10]]))
        return torch.tensor([[10.0], [11], [12], [13], [14], [10]])

    monkeypatch.setattr(expert, "_run_local_mlp", local_mlp)
    monkeypatch.setattr(shared, "tensor_model_parallel_all_gather", gather)

    assert torch.equal(expert(full_tokens), full_tokens + 10)
