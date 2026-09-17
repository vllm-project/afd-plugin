# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Tests for the async GPU connector's Dynamo-opaque dispatch/receive ops."""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from afd_plugin.connectors.gpu.async_moe_op import (
    register_async_moe_ops,
)


def test_async_moe_ops_are_registered_once() -> None:
    first = register_async_moe_ops()
    second = register_async_moe_ops()
    assert first == second


def test_async_moe_ops_fake_impls_shape() -> None:
    dispatch, receive = register_async_moe_ops()
    with FakeTensorMode():
        hidden = torch.randn(6, 16)
        ids = torch.zeros(6, 2, dtype=torch.int64)
        weights = torch.ones(6, 2)
        sent = dispatch(hidden, weights, ids, 0)
        out = receive(sent)
    assert sent.shape == hidden.shape
    assert out.shape == hidden.shape


def test_async_moe_ops_are_dynamo_opaque() -> None:
    """Tracing must split at the ops, not reach into the connector.

    Before the ops existed, tracing the async MoE forward ran into the
    connector's Python -- NVSHMEM pointer views, host caches, ctypes -- and
    Dynamo raised Unsupported, first on a logger call. Exporting with fake
    tensors asserts the tracer captures the dispatch/receive as opaque calls
    with the deferred-receive control flow in between. The ops only register
    CUDA kernels, so nothing here executes them.
    """
    dispatch, receive = register_async_moe_ops()

    def proxy(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor):
        sent = dispatch(hidden, weights, ids, 3)
        returned = receive(sent)
        return dispatch(returned, weights, ids, 4)

    with FakeTensorMode():
        hidden = torch.randn(4, 8)
        ids = torch.zeros(4, 2, dtype=torch.int64)
        weights = torch.ones(4, 2)
        graph_module, _ = torch._dynamo.export(proxy)(hidden, ids, weights)

    op_targets = [
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert sum("afd_async_dispatch" in t for t in op_targets) == 2
    assert sum("afd_async_recv" in t for t in op_targets) == 1
