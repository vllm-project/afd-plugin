# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The profile flag must be read inside the custom op, not traced."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

NUM_EXPERTS = 8
TOKENS = 4
TOPK = 2
OP_NAME = "afd_balance_routing_ids_"


def _ids(offset: int) -> torch.Tensor:
    flat = torch.arange(offset, offset + TOKENS * TOPK, dtype=torch.int32)
    return (flat % NUM_EXPERTS).reshape(TOKENS, TOPK)


def _balanced() -> list[int]:
    return [i % NUM_EXPERTS for i in range(TOKENS * TOPK)]


def test_balances_in_profile_and_is_inert_otherwise(
    registered_routing_ids_op, monkeypatch
):
    balance_op, _op = registered_routing_ids_op
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    profile_ids = _ids(3)
    _op(profile_ids, NUM_EXPERTS, False)
    assert profile_ids.flatten().tolist() == _balanced()

    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live_ids = _ids(3)
    expected = live_ids.clone()
    _op(live_ids, NUM_EXPERTS, False)
    assert torch.equal(live_ids, expected)

    second = _ids(21)
    expected_second = second.clone()
    _op(second, NUM_EXPERTS, False)
    assert torch.equal(second, expected_second)


def test_explicit_force_balances_outside_profile(
    registered_routing_ids_op, monkeypatch
):
    balance_op, _op = registered_routing_ids_op
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    forced = _ids(5)
    _op(forced, NUM_EXPERTS, True)
    assert forced.flatten().tolist() == _balanced()


def test_opcheck_contract(registered_routing_ids_op, monkeypatch):
    balance_op, _op = registered_routing_ids_op
    # Profile mode is the branch that actually mutates topk_ids, so the declared
    # mutation is exercised; a schema claiming no mutation must fail here.
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    torch.library.opcheck(
        _op,
        (_ids(1), NUM_EXPERTS, False),
        test_utils=("test_schema", "test_faketensor"),
    )


class _ThroughOp(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, ids):
        self.op(ids, NUM_EXPERTS, False)
        return ids.clone()


def _compile_once(module):
    compiled = []

    def backend(gm, example_inputs):
        fn = torch._dynamo.backends.registry.lookup_backend("eager")(gm, example_inputs)
        compiled.append((fn, example_inputs))
        return fn

    torch.compile(module, backend=backend)(_ids(9))
    assert len(compiled) == 1
    return compiled[0]


def test_reuse_without_guards_keeps_live_routing(
    registered_routing_ids_op, monkeypatch
):
    balance_op, _op = registered_routing_ids_op
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    module = _ThroughOp(_op)
    compiled_fn, example_inputs = _compile_once(module)

    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live = _ids(11)
    expected = live.clone()
    compiled_fn(live, *example_inputs[1:])
    assert torch.equal(live, expected)


def _inductor_available() -> bool:
    try:
        import torch._inductor  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _inductor_available(), reason="inductor unavailable on this CPU host"
)
def test_inductor_backend_keeps_live_routing(registered_routing_ids_op, monkeypatch):
    balance_op, _op = registered_routing_ids_op
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    module = torch.compile(_ThroughOp(_op), backend="inductor")
    module(_ids(9))

    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live = _ids(13)
    expected = live.clone()
    module(live)
    assert torch.equal(live, expected)
