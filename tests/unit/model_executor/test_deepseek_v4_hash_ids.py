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

import ast
import contextlib
import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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

    saved = {name: sys.modules.get(name) for name in missing}

    def make(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module

    make("vllm")
    make(
        "vllm.forward_context",
        DPMetadata=type("DPMetadata", (), {}),
        ForwardContext=type("ForwardContext", (), {}),
        get_forward_context=lambda: None,
    )
    make(
        "vllm.logger",
        init_logger=lambda *args, **kwargs: logging.getLogger("test"),
    )
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


with _vllm_stub():
    from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate
    from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
        local_hash_input_ids,
    )


def _make_model(**attributes: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(**attributes)


def _forward_context(**overrides: object) -> types.SimpleNamespace:
    """Build a forward context carrying the fields the helper reads.

    A real Ascend forward context always defines all three, so a fixture that
    omitted one would exercise an ``AttributeError`` rather than a routing
    condition.
    """

    fields: dict[str, object] = {
        "input_ids": None,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


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


def test_model_sharded_ids_are_not_split_again():
    ids = torch.tensor([8, -1], dtype=torch.int64)
    assert local_hash_input_ids(input_ids=ids, router_tokens=2).tolist() == [8, -1]


def test_rejects_global_ids_for_local_router():
    with pytest.raises(RuntimeError, match="cannot align"):
        local_hash_input_ids(input_ids=torch.arange(7), router_tokens=2)


_DSV4_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v4.py"
)
_REMOTE_MOE_CLASS = "AFDDeepseekV4RemoteMoE"


class _RecordingProxy:
    """Stand-in base class recording the transfer the shell requests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def _send_and_receive(self, hidden_states: Any, **send_kwargs: Any) -> str:
        self.sent.append({"hidden_states": hidden_states, **send_kwargs})
        return "ffn-output"


def _load_remote_moe_class(forward_context: Any) -> Any:
    """Execute the real ``AFDDeepseekV4RemoteMoE`` from the module source.

    Importing the DSV4 NPU module pulls in the whole ``vllm``/``vllm_ascend``
    model surface, so the class body is executed against a parameter-free base
    class instead. The body under test is read from the file unchanged.

    Raises:
        AssertionError: If the module no longer defines the class, which means
            this test needs to be repointed rather than silently pass.
    """

    tree = ast.parse(_DSV4_MODULE_PATH.read_text(encoding="utf-8"))
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == _REMOTE_MOE_CLASS
        ),
        None,
    )
    assert node is not None, (
        f"{_DSV4_MODULE_PATH} no longer defines {_REMOTE_MOE_CLASS}"
    )

    namespace: dict[str, Any] = {
        "RemoteFFNProxy": _RecordingProxy,
        "torch": torch,
        "get_forward_context": lambda: forward_context,
    }
    code = compile(ast.Module(body=[node], type_ignores=[]), "<remote-moe>", "exec")
    exec(code, namespace)  # noqa: S102 - source is this repository's own file
    return namespace[_REMOTE_MOE_CLASS]


def test_remote_moe_sends_the_context_ids_alongside_the_activations():
    """The gate-on-FFN shell must transport ids, since FFN cannot route without them."""

    forward_context = _forward_context(
        input_ids=torch.tensor([5, 6, 7], dtype=torch.int32),
    )
    layer = _load_remote_moe_class(forward_context)()

    output = layer.forward(torch.zeros(3, 8))

    assert output == "ffn-output"
    assert layer.sent[0]["input_ids"].tolist() == [5, 6, 7]


def test_remote_moe_uses_explicit_model_sharded_ids():
    # Native model forward shards IDs before passing them to the remote MoE;
    # the runner's context still carries the full input vector.
    layer = _load_remote_moe_class(
        _forward_context(input_ids=torch.arange(7)),
    )()
    local_ids = torch.tensor([6, 0])
    layer.forward(torch.zeros(2, 8), input_ids=local_ids)
    assert layer.sent[0]["input_ids"].tolist() == [6, 0]


def test_remote_moe_without_context_ids_raises_instead_of_sending_activations_only():
    """A quiet activations-only fallback would leave FFN reading an unwritten slot."""

    layer = _load_remote_moe_class(_forward_context(input_ids=None))()

    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        layer.forward(torch.zeros(3, 8))

    assert layer.sent == []


@pytest.mark.parametrize(
    "scoring_func,selector",
    [
        ("sqrtsoftplus", "_compute_sqrtsoftplus_topk"),
        ("softmax", "_compute_standard_topk"),
    ],
)
def test_dsv4_router_uses_model_gate(monkeypatch, scoring_func, selector):
    hidden = torch.tensor([[1.0, 0.25]], dtype=torch.bfloat16)
    router_input = torch.tensor([[1.0, 1.001953125]], dtype=torch.float32)
    router_logits = router_input
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
        hidden_states_fp32=router_input,
    )

    assert torch.equal(captured[0], router_logits)
    assert ids.tolist() == [[1]]
    assert weights.dtype == torch.float32
