# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The AFD ubatch wrapper's handling of steps the splitter did not divide."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode  # noqa: E402

from afd_plugin.v1.worker import ubatch_wrapper  # noqa: E402
from afd_plugin.v1.worker.ubatch_wrapper import AFDUBatchWrapper  # noqa: E402


def _wrapper(monkeypatch, *, mode, num_tokens, captured):
    wrapper = object.__new__(AFDUBatchWrapper)
    wrapper.cudagraph_wrapper = None  # the AFD wrapper never builds one
    wrapper.cudagraphs = dict.fromkeys(captured)
    wrapper.runnable = lambda *a, **k: "eager"
    context = SimpleNamespace(
        ubatch_slices=None,
        cudagraph_runtime_mode=mode,
        batch_descriptor=SimpleNamespace(num_tokens=num_tokens),
    )
    monkeypatch.setattr(ubatch_wrapper, "get_forward_context", lambda: context)
    return wrapper


def test_uncaptured_full_step_runs_eagerly_instead_of_asserting(monkeypatch):
    # A decode bucket the splitter declines to divide reaches the non-ubatch
    # path in FULL mode with no graph for its key. Upstream would assert on
    # the cudagraph_wrapper the AFD wrapper never constructs.
    wrapper = _wrapper(monkeypatch, mode=CUDAGraphMode.FULL, num_tokens=8, captured=())
    assert wrapper() == "eager"


def test_captured_or_non_full_steps_keep_upstream_behaviour(monkeypatch):
    calls: list[str] = []

    def upstream_call(self, *args, **kwargs):
        calls.append("upstream")
        return "upstream"

    monkeypatch.setattr(ubatch_wrapper.UBatchWrapper, "__call__", upstream_call)
    captured = _wrapper(
        monkeypatch, mode=CUDAGraphMode.FULL, num_tokens=8, captured=(8,)
    )
    assert captured() == "upstream"
    eager_mode = _wrapper(
        monkeypatch, mode=CUDAGraphMode.NONE, num_tokens=8, captured=()
    )
    assert eager_mode() == "upstream"
    assert calls == ["upstream", "upstream"]
