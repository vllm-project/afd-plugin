# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

native = pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_async_cam_forward as async_forward,
)


def test_dsv4_async_metadata_lazily_delegates_to_async_cam_forward(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    metadata = object()
    input_ids = object()
    positions = object()
    intermediate_tensors = object()
    sentinel = object()
    calls: list[tuple[object, ...]] = []

    def run_async(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: metadata,
    )
    monkeypatch.setattr(
        async_forward,
        "run_async_moe_ubatch_forward",
        run_async,
    )

    result = model.forward(input_ids, positions, intermediate_tensors)

    assert result is sentinel
    assert calls == [
        (model, input_ids, positions, intermediate_tensors, metadata, None)
    ]


def test_dsv4_without_async_metadata_uses_native_forward(monkeypatch):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    sentinel = object()
    calls: list[tuple[object, ...]] = []

    def native_forward(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        adapter,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda: None,
    )
    monkeypatch.setattr(
        native.DeepseekV4Model,
        "forward",
        native_forward,
    )

    result = model.forward(None, object(), None, object())

    assert result is sentinel
    assert len(calls) == 1


def test_async_cam_forward_resolves_forward_context_entrypoint():
    assert callable(async_forward.get_forward_context)
    assert (
        "get_forward_context"
        in async_forward.run_async_moe_ubatch_forward.__code__.co_names
    )
