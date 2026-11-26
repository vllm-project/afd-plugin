# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DSV4 Attention-side Hash id selection helper.

``local_hash_input_ids`` decides which token ids Attention sends towards the FFN
role for a Hash layer. The helper is pure tensor logic, which matters here: an
ids/token misalignment would let a token-keyed router select experts for the
wrong tokens without raising anywhere, so the alignment behaviour is worth
testing without an Ascend device.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from collections.abc import Iterator
from importlib.machinery import ModuleSpec

import pytest

# The gate calls torch.ops.vllm.afd_balance_routing_ids_; this module runs
# without the AFD worker installer, so the op is registered here.
pytestmark = pytest.mark.usefixtures("registered_routing_ids_op")

torch = pytest.importorskip("torch")

_STUB_MODULES = (
    "vllm",
    "vllm.forward_context",
    "vllm.logger",
)


@contextlib.contextmanager
def _vllm_stub() -> Iterator[None]:
    """Expose a minimal ``vllm`` surface for the duration of one import.

    Importing the model package pulls in ``vllm.forward_context``. The stub is
    removed again immediately afterwards so a partial ``vllm`` cannot mask the
    real "vllm is not installed" failure for other test modules in the same
    session.
    """

    missing = [name for name in _STUB_MODULES if name not in sys.modules]
    if not missing:
        yield
        return

    builders: dict[str, dict[str, object]] = {
        "vllm": {},
        "vllm.forward_context": {
            "DPMetadata": type("DPMetadata", (), {}),
            "ForwardContext": type("ForwardContext", (), {}),
            "get_forward_context": lambda: None,
        },
        "vllm.logger": {
            "init_logger": lambda *args, **kwargs: logging.getLogger("test")
        },
    }

    # Only the modules that are genuinely absent are installed: overwriting one that
    # another test imported would survive this context, because the cleanup below
    # only removes the names it added.
    for name in missing:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        # find_spec("vllm") runs at import time; a spec-less module makes it raise.
        module.__spec__ = ModuleSpec(name, None)
        for attribute, value in builders[name].items():
            setattr(module, attribute, value)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)


with _vllm_stub():
    from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate
    from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
        local_hash_input_ids,
    )


def test_returns_ids_aligned_with_router_tokens():
    ids = torch.tensor([5, 6, 7], dtype=torch.int32)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=3,
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [5, 6, 7]


def test_flattens_multi_dimensional_ids():
    ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=4,
    )

    assert result.tolist() == [5, 6, 7, 8]


def test_rejects_missing_ids():
    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        local_hash_input_ids(
            input_ids=None,
            router_tokens=3,
        )


def test_rejects_unalignable_token_count():
    """A count mismatch must fail here, before any cross-role transfer."""
    ids = torch.tensor([5, 6], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        local_hash_input_ids(
            input_ids=ids,
            router_tokens=3,
        )


@pytest.mark.parametrize(
    "scoring_func,selector",
    [
        ("sqrtsoftplus", "_compute_sqrtsoftplus_topk"),
        ("softmax", "_compute_standard_topk"),
    ],
)
@pytest.mark.parametrize("in_profile_run", [False, True])
@pytest.mark.parametrize("precomputed_fp32", [False, True])
def test_dsv4_router_uses_model_gate(
    monkeypatch, scoring_func, selector, in_profile_run, precomputed_fp32
):
    extra_context = types.ModuleType("vllm_ascend.ascend_forward_context")
    extra_context._EXTRA_CTX = types.SimpleNamespace(  # type: ignore[attr-defined]
        in_profile_run=in_profile_run
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ascend_forward_context", extra_context
    )
    monkeypatch.setenv("AFD_FORCE_BALANCED_TOPK_IDS", "1")
    hidden = torch.tensor([[1.0, 0.25]], dtype=torch.bfloat16)
    router_input = torch.tensor([[1.0, 1.001953125]], dtype=torch.float32)
    router_logits = router_input if precomputed_fp32 else hidden.float()
    moe = types.SimpleNamespace(
        gate=types.SimpleNamespace(weight_fp32=torch.eye(2)),
        scoring_func=scoring_func,
    )
    captured = []

    def select(_moe, logits, **_kwargs):
        captured.append(logits)
        ids = logits.argmax(dim=-1, keepdim=True)
        return torch.ones_like(ids, dtype=torch.float32), ids

    def unexpected_selector(*_args):
        raise AssertionError("incorrect selector for scoring_func")

    for name in ("_compute_sqrtsoftplus_topk", "_compute_standard_topk"):
        monkeypatch.setattr(deepseek_v4_attention_gate, name, unexpected_selector)
    monkeypatch.setattr(deepseek_v4_attention_gate, selector, select)
    weights, ids = deepseek_v4_attention_gate.compute_attention_gate_topk(
        moe,
        hidden,
        hidden_states_fp32=router_input if precomputed_fp32 else None,
    )

    assert torch.equal(captured[0], router_logits)
    assert ids.tolist() == [[int(precomputed_fp32 and not in_profile_run)]]
    assert weights.tolist() == [[1.0]]
    assert weights.dtype == torch.float32
