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
    from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
        hash_input_ids_from_context,
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
        "flash_comm_v1_enabled": False,
        "pad_size": 0,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_returns_global_ids_when_no_flash_comm_split_is_needed():
    ids = torch.tensor([5, 6, 7], dtype=torch.int32)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=3,
        flash_comm_v1_enabled=False,
        pad_size=0,
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [5, 6, 7]


def test_flattens_multi_dimensional_ids():
    ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=4,
        flash_comm_v1_enabled=False,
        pad_size=0,
    )

    assert result.tolist() == [5, 6, 7, 8]


def test_rejects_missing_ids():
    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        local_hash_input_ids(
            input_ids=None,
            router_tokens=3,
            flash_comm_v1_enabled=False,
            pad_size=0,
        )


def test_rejects_unalignable_token_count():
    """A count mismatch must fail here, before any cross-role transfer."""
    ids = torch.tensor([5, 6], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        local_hash_input_ids(
            input_ids=ids,
            router_tokens=3,
            flash_comm_v1_enabled=False,
            pad_size=0,
        )


def test_applies_flash_comm_padding_and_tp_split(monkeypatch):
    """FlashComm v1 must slice ids exactly like the router logits it mirrors."""
    captured: dict[str, object] = {}

    def fake_split(tensor: Any, *, num_partitions: int, **kwargs: object):
        captured["num_partitions"] = num_partitions
        captured["kwargs"] = kwargs
        return list(torch.chunk(tensor, num_partitions))

    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: _make_model(  # type: ignore[attr-defined]
        world_size=2,
        rank_in_group=1,
    )
    ascend_distributed = types.ModuleType("vllm_ascend.distributed")
    ascend_utils = types.ModuleType("vllm_ascend.distributed.utils")
    ascend_utils.split_tensor_along_first_dim = fake_split  # type: ignore[attr-defined]
    ascend_distributed.utils = ascend_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed", ascend_distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.utils", ascend_utils)

    # 4 global ids plus one padded slot gives 5. torch.chunk(5, 2) splits that
    # into [3, 2] rather than evenly, so rank 1 receives the trailing pair.
    result = local_hash_input_ids(
        input_ids=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        router_tokens=2,
        flash_comm_v1_enabled=True,
        pad_size=1,
    )

    assert captured["num_partitions"] == 2
    assert captured["kwargs"] == {"contiguous_split_chunks": True}
    assert result.tolist() == [8, 0]


def test_context_without_ids_is_an_error_not_a_quiet_fallback():
    """A context with no ids must fail here, not downgrade the transfer.

    The FFN role decides once per run that it expects ids and cannot tell Hash
    layers from non-Hash ones, so "this layer does not need them" is not a case
    that can be answered by sending activations alone.
    """

    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        hash_input_ids_from_context(
            forward_context=_forward_context(input_ids=None),
            router_tokens=3,
        )


def test_context_ids_are_returned_as_router_aligned_tokens():
    result = hash_input_ids_from_context(
        forward_context=_forward_context(
            input_ids=torch.tensor([5, 6, 7], dtype=torch.int32),
        ),
        router_tokens=3,
    )

    assert result.tolist() == [5, 6, 7]


def test_context_ids_are_still_validated_against_the_local_token_count():
    """Carrying ids opts into validation; a mismatch must not pass silently."""
    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        hash_input_ids_from_context(
            forward_context=_forward_context(
                input_ids=torch.tensor([5, 6], dtype=torch.int32),
            ),
            router_tokens=3,
        )


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


def test_remote_moe_without_context_ids_raises_instead_of_sending_activations_only():
    """A quiet activations-only fallback would leave FFN reading an unwritten slot."""

    layer = _load_remote_moe_class(_forward_context(input_ids=None))()

    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        layer.forward(torch.zeros(3, 8))

    assert layer.sent == []
