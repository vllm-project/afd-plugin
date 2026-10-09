# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check DSV2 shared-expert arithmetic with distinct TP-rank token inputs."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def shared_output():
    # Load the production function without importing the optional NPU runtime.
    path = Path("afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py")
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_shared_output"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    "use_sp,replicated", [(True, False), (True, True), (False, False)]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_shared_experts_match_full_mlp(shared_output, use_sp, replicated, rank):
    # Three real tokens plus SP padding; ranks receive different token rows.
    tokens = torch.tensor([[0.2, -0.4], [0.8, 0.1], [1.3, -0.6], [0.0, 0.0]])
    gate = torch.tensor([[0.2, 0.7], [-0.3, 0.5], [0.8, -0.4], [0.1, 0.6]])
    up = torch.tensor([[0.4, -0.1], [0.6, 0.9], [-0.2, 0.8], [0.7, 0.3]])
    down = torch.tensor([[0.5, -0.8, 0.3, 0.6], [0.2, 0.4, -0.7, 0.9]])

    def mlp(x, gate_weight, up_weight, down_weight):
        return (
            torch.nn.functional.silu(x @ gate_weight.T) * (x @ up_weight.T)
        ) @ down_weight.T

    expected = mlp(tokens, gate, up, down)
    gate_shards, up_shards, down_shards = (
        gate.chunk(2, dim=0),
        up.chunk(2, dim=0),
        down.chunk(2, dim=1),
    )
    token_shards = tokens.chunk(2)
    local = token_shards[rank] if use_sp else tokens
    peer_input = token_shards[1 - rank] if use_sp else tokens

    def gather(value, dim):
        nonlocal peer_input
        assert use_sp and not replicated
        assert dim == 0
        torch.testing.assert_close(value, local)
        peer_input = tokens
        return tokens

    def forward_shared(value):
        if replicated:
            return mlp(value, gate, up, down)
        # Model the native TP all-reduce, including the peer's own tokens.
        partial = mlp(value, gate_shards[rank], up_shards[rank], down_shards[rank])
        peer = 1 - rank
        return partial + mlp(
            peer_input, gate_shards[peer], up_shards[peer], down_shards[peer]
        )

    shared = Mock(side_effect=forward_shared)
    shared.gate_up_proj.tp_size = 1 if replicated else 2
    layer = SimpleNamespace(
        use_sequence_parallel_moe=use_sp,
        mlp=SimpleNamespace(shared_experts=shared),
    )
    shared_output["get_tensor_model_parallel_rank"] = lambda: rank
    shared_output["tensor_model_parallel_all_gather"] = gather
    output = shared_output["compute_shared_output"](layer, local)
    reference = expected.chunk(2)[rank] if use_sp else expected
    torch.testing.assert_close(output, reference)
