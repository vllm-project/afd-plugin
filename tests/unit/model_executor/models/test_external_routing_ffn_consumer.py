# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The sync FFN consumer: what it uses, what it returns, and what it leaves behind.

Three properties matter here and none of them is "each collaborator was called
once":

- the values handed to ``apply`` are the ones ``prepare`` aligned, not the raw
  transport values, so a missing alignment step cannot pass unnoticed;
- ``finalize``'s result is what the caller receives;
- every exit path clears the stash and the experts module's per-call state, and a
  later call cannot inherit routing from an earlier one.

Skipped on a CPU-only host: the consumer needs vLLM-Ascend symbols at call time.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm.forward_context")

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402

TOP_K = 2
HIDDEN = 4
RAW_ID = 7
ALIGNED_ID = 99


class _AllGatherCommImpl:
    """Marker type the consumer accepts."""


class _Mc2CommImpl:
    """Marker type for the other wired communication method."""


class _PrepareFinalize:
    """Carries the stash the prepare patch would read and replace."""

    def __init__(self) -> None:
        self.afd_external_routing = None
        self.prepare_calls = 0


class _CommBehaviour:
    """The flow both wired methods share: record, serve aligned routing, apply."""

    def __init__(self, *, fail_in: str | None = None) -> None:
        self.prepare_finalize = _PrepareFinalize()
        self.fail_in = fail_in
        self.applied: dict[str, object] = {}
        self.events: list[str] = []

    def prepare(self, *, hidden_states, router_logits, replace_allreduce, quant_type):
        del router_logits, replace_allreduce, quant_type
        self.events.append("prepare")
        if self.fail_in == "prepare":
            raise RuntimeError("prepare exploded")
        stash = self.prepare_finalize.afd_external_routing
        assert stash is not None, "the consumer must publish routing before prepare"
        # Deliberately different from the transport values: anything downstream that
        # kept using the raw routing is now visible in the result.
        self.prepare_finalize.afd_external_routing = type(stash)(
            topk_ids=torch.full_like(stash.topk_ids, ALIGNED_ID),
            topk_weights=stash.topk_weights,
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            mc2_mask=None,
            pertoken_scale=None,
            padded_hidden_states_shape=hidden_states.shape,
        )

    def finalize(self, *, hidden_states, reduce_results, padded_hidden_states_shape):
        del reduce_results, padded_hidden_states_shape
        self.events.append("finalize")
        if self.fail_in == "finalize":
            raise RuntimeError("finalize exploded")
        return hidden_states


class _Comm(_CommBehaviour, _AllGatherCommImpl):
    """The ALLGATHER case: same behaviour, ALLGATHER marker."""


def _experts(comm: _Comm):
    def apply(*, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        del layer, shared_experts, shared_experts_input
        comm.events.append("apply")
        if comm.fail_in == "apply":
            raise RuntimeError("apply exploded")
        comm.applied["topk_ids"] = topk_ids
        comm.applied["topk_weights"] = topk_weights
        return SimpleNamespace(routed_out=x + 1.0)

    return SimpleNamespace(
        moe_config=SimpleNamespace(is_sequence_parallel=False),
        quant_type="quant",
        quant_method=SimpleNamespace(apply=apply),
        ascend_pertoken_scale="leaked",
        ascend_mc2_mask="leaked",
    )


def _layer(comm, experts):
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    layer.mlp = SimpleNamespace(experts=SimpleNamespace(routed_experts=experts))
    return layer


def _install_vllm_ascend(monkeypatch, comm):
    modules = {}
    for name in (
        "vllm_ascend",
        "vllm_ascend.ascend_forward_context",
        "vllm_ascend.ops",
        "vllm_ascend.ops.fused_moe",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    modules["vllm_ascend.ascend_forward_context"]._EXTRA_CTX = SimpleNamespace(
        moe_comm_method=comm,
    )
    method_module = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    # Both wired methods have to exist: the consumer imports them to test membership,
    # so a fake without MC2CommImpl fails at import rather than at the assertion.
    method_module.AllGatherCommImpl = _AllGatherCommImpl
    method_module.MC2CommImpl = _Mc2CommImpl
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        method_module,
    )


def _raw_routing():
    return (
        torch.full((2, TOP_K), RAW_ID, dtype=torch.int32),
        torch.full((2, TOP_K), 0.5, dtype=torch.float32),
    )


def test_apply_consumes_the_aligned_routing_not_the_transport_values(monkeypatch):
    """A missing alignment step must change the result, not just the call order."""

    comm = _Comm()
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    output = layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert comm.events == ["prepare", "apply", "finalize"]
    assert torch.equal(comm.applied["topk_ids"], torch.full_like(topk_ids, ALIGNED_ID))
    assert torch.equal(comm.applied["topk_weights"], topk_weights)
    # finalize's result is returned unchanged.
    assert torch.equal(output, torch.ones(2, HIDDEN))
    # Per-call state is cleared after a clean run too.
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


@pytest.mark.parametrize("fail_in", ["prepare", "apply", "finalize"])
def test_every_failure_stage_clears_state(monkeypatch, fail_in):
    comm = _Comm(fail_in=fail_in)
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match=f"{fail_in} exploded"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


def test_a_later_call_does_not_inherit_the_previous_routing(monkeypatch):
    failing = _Comm(fail_in="prepare")
    _install_vllm_ascend(monkeypatch, failing)
    experts = _experts(failing)
    layer = _layer(failing, experts)
    topk_ids, topk_weights = _raw_routing()
    with pytest.raises(RuntimeError):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    healthy = _Comm()
    _install_vllm_ascend(monkeypatch, healthy)
    layer.mlp = SimpleNamespace(experts=SimpleNamespace(routed_experts=_experts(healthy)))

    layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert healthy.applied["topk_ids"] is not None
    assert healthy.prepare_finalize.afd_external_routing is None


def test_an_unwired_comm_method_is_rejected(monkeypatch):
    """MC2 is wired; the all-to-all paths are not, and must not be guessed at."""

    comm = SimpleNamespace(prepare_finalize=_PrepareFinalize())
    _install_vllm_ascend(monkeypatch, comm)
    layer = _layer(comm, _experts(_Comm()))
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="ALLGATHER and MC2"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


class _Mc2Comm(_CommBehaviour, _Mc2CommImpl):
    """The MC2 case: same behaviour, MC2 marker, so only the reduce decision differs."""


@pytest.mark.parametrize("comm_type", [_Comm, _Mc2Comm])
def test_both_wired_comm_methods_run_the_same_flow(monkeypatch, comm_type):
    """MC2 shares the consumer flow, so it must apply and finalize the same way."""

    comm = comm_type()
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    output = layer._compute_external_routing_ffn(
        torch.zeros(2, HIDDEN),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    assert comm.events == ["prepare", "apply", "finalize"]
    assert torch.equal(comm.applied["topk_ids"], torch.full_like(topk_ids, ALIGNED_ID))
    assert torch.equal(output, torch.ones(2, HIDDEN))
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None


@pytest.mark.parametrize("comm_type", [_Comm, _Mc2Comm])
def test_both_wired_comm_methods_clear_state_after_a_failure(monkeypatch, comm_type):
    comm = comm_type(fail_in="apply")
    _install_vllm_ascend(monkeypatch, comm)
    experts = _experts(comm)
    layer = _layer(comm, experts)
    topk_ids, topk_weights = _raw_routing()

    with pytest.raises(RuntimeError, match="apply exploded"):
        layer._compute_external_routing_ffn(
            torch.zeros(2, HIDDEN),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None
