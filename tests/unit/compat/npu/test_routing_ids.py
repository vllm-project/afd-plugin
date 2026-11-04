"""The profile flag must be read inside the custom op, not traced."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

NUM_EXPERTS = 8
TOKENS = 4
TOPK = 2
OP_NAME = "afd_balance_routing_ids_"


def _ids(offset: int) -> torch.Tensor:
    flat = torch.arange(offset, offset + TOKENS * TOPK, dtype=torch.int32)
    return (flat % NUM_EXPERTS).reshape(TOKENS, TOPK)


def _balanced() -> list[int]:
    return [i % NUM_EXPERTS for i in range(TOKENS * TOPK)]


def _op():
    return getattr(torch.ops.vllm, OP_NAME)


@pytest.fixture
def balance_op(monkeypatch):
    """Load the real op through a CPU registration of the production code."""
    from torch.library import _scoped_library, infer_schema

    stub = types.ModuleType("vllm.utils.torch_utils")

    with contextlib.ExitStack() as stack:
        lib = stack.enter_context(_scoped_library("vllm", "FRAGMENT"))

        def registering_library(op_name, op_func, mutates_args, fake_impl):
            schema = infer_schema(op_func, mutates_args=mutates_args)
            lib.define(f"{op_name}{schema}")
            lib.impl(op_name, op_func, "CPU")
            if fake_impl is not None:
                lib.impl(op_name, fake_impl, "Meta")

        stub.direct_register_custom_op = registering_library
        for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.utils", types.ModuleType("vllm.utils")),
            ("vllm.utils.torch_utils", stub),
        ):
            monkeypatch.setitem(sys.modules, name, module)

        import afd_plugin.compat.npu as afd_npu_package

        module_name = "afd_plugin.compat.npu.routing_ids"
        package = afd_npu_package
        saved_module = sys.modules.get(module_name)
        had_package_attr = hasattr(package, "routing_ids")
        saved_package_attr = getattr(package, "routing_ids", None)

        def restore():
            if saved_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = saved_module
            if True:
                if had_package_attr:
                    setattr(package, "routing_ids", saved_package_attr)
                else:
                    package.__dict__.pop("routing_ids", None)

        stack.callback(restore)
        sys.modules.pop(module_name, None)
        package.__dict__.pop("routing_ids", None)

        from afd_plugin.compat.npu import routing_ids

        yield routing_ids


def test_balances_in_profile_and_is_inert_otherwise(balance_op, monkeypatch):
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    profile_ids = _ids(3)
    _op()(profile_ids, NUM_EXPERTS, False)
    assert profile_ids.flatten().tolist() == _balanced()

    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live_ids = _ids(3)
    expected = live_ids.clone()
    _op()(live_ids, NUM_EXPERTS, False)
    assert torch.equal(live_ids, expected)

    second = _ids(21)
    expected_second = second.clone()
    _op()(second, NUM_EXPERTS, False)
    assert torch.equal(second, expected_second)


def test_explicit_force_balances_outside_profile(balance_op, monkeypatch):
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    forced = _ids(5)
    _op()(forced, NUM_EXPERTS, True)
    assert forced.flatten().tolist() == _balanced()


def test_opcheck_contract(balance_op, monkeypatch):
    # Profile mode is the branch that actually mutates topk_ids, so the declared
    # mutation is exercised; a schema claiming no mutation must fail here.
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    torch.library.opcheck(_op(), (_ids(1), NUM_EXPERTS, False),
                          test_utils=("test_schema", "test_faketensor"))


class _ThroughOp(torch.nn.Module):
    def forward(self, ids):
        _op()(ids, NUM_EXPERTS, False)
        return ids.clone()


class _PlainPython(torch.nn.Module):
    def forward(self, ids):
        if _PLAIN_FLAG["profile"]:
            from afd_plugin.routing_balance import _force_balanced_topk_ids

            _force_balanced_topk_ids(ids, num_logical_experts=NUM_EXPERTS)
        return ids.clone()


_PLAIN_FLAG = {"profile": False}


def _compile_once(module):
    compiled = []

    def backend(gm, example_inputs):
        fn = torch._dynamo.backends.registry.lookup_backend("eager")(gm, example_inputs)
        compiled.append((fn, example_inputs))
        return fn

    torch.compile(module, backend=backend)(_ids(9))
    assert len(compiled) == 1
    return compiled[0]


@pytest.mark.parametrize("module_cls", [_ThroughOp, _PlainPython])
def test_reuse_without_guards_keeps_live_routing(balance_op, monkeypatch, module_cls):
    _PLAIN_FLAG["profile"] = True
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    compiled_fn, example_inputs = _compile_once(module_cls())

    _PLAIN_FLAG["profile"] = False
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live = _ids(11)
    expected = live.clone()
    compiled_fn(live, *example_inputs[1:])
    if module_cls is _ThroughOp:
        assert torch.equal(live, expected)
    else:
        assert live.flatten().tolist() == _balanced()


def _inductor_available() -> bool:
    try:
        import torch._inductor  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _inductor_available(), reason="inductor unavailable on this CPU host")
def test_inductor_backend_keeps_live_routing(balance_op, monkeypatch):
    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: True)
    module = torch.compile(_ThroughOp(), backend="inductor")
    module(_ids(9))

    monkeypatch.setattr(balance_op, "_afd_in_profile_run", lambda: False)
    live = _ids(13)
    expected = live.clone()
    module(live)
    assert torch.equal(live, expected)
