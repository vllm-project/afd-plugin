# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Regression for shared TP weights receiving different SP token rows."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


def test_shared_experts_match_full_mlp():
    # Load the real helper without requiring the optional NPU runtime.
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
    tokens = torch.tensor([[0.2, -0.4], [1.3, -0.6]])
    gate = torch.tensor([[0.2, 0.7], [-0.3, 0.5]])
    up = torch.tensor([[0.4, -0.1], [0.6, 0.9]])
    down = torch.tensor([[0.5, -0.8], [0.2, 0.4]])

    def mlp(x, g, u, d):
        return (torch.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T

    expected = mlp(tokens, gate, up, down)
    shards = list(zip(gate.chunk(2), up.chunk(2), down.chunk(2, dim=1), strict=True))

    def check_rank(rank):
        peer = 1 - rank
        peer_input = tokens[peer : peer + 1]

        def gather(value, dim):
            nonlocal peer_input
            peer_input = tokens
            return tokens

        def forward_shared(value):
            # Emulate the native TP reduction using each rank's actual input.
            return mlp(value, *shards[rank]) + mlp(peer_input, *shards[peer])

        shared = Mock(side_effect=forward_shared)
        shared.gate_up_proj.tp_size = 2
        layer = SimpleNamespace(
            use_sequence_parallel_moe=True, mlp=SimpleNamespace(shared_experts=shared)
        )
        namespace["get_tensor_model_parallel_rank"] = lambda rank=rank: rank
        namespace["tensor_model_parallel_all_gather"] = gather
        output = namespace["compute_shared_output"](layer, tokens[rank : rank + 1])
        torch.testing.assert_close(output, expected[rank : rank + 1])

    check_rank(0)
    check_rank(1)
