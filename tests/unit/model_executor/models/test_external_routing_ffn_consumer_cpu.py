# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the FFN side consuming Attention-side routing.

``deepseek_v4`` cannot be imported on a host without vLLM, so the consumer method is
compiled out of the real source with ``ast`` and run against stand-ins for the two
upstream modules it imports at call time. The routing payload itself is the real
type, loaded from its file.

What this covers: the prepared (padded/gathered/split) routing reaches ``apply``, the
ALLGATHER and MC2 paths differ only in whether finalize reduces across ranks, and a
failure in prepare or apply leaves no per-call state behind. A failure inside finalize
is not covered here.

Native model construction, the real operators and graph execution stay with the
green-zone suite.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_PATH = (
    _REPO_ROOT
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v4.py"
)
_ROUTING_PATH = (
    _REPO_ROOT / "afd_plugin" / "connectors" / "npu" / "external_routing.py"
)
_CONSUMER = "_compute_external_routing_ffn"


def _load_routing_inputs():
    spec = importlib.util.spec_from_file_location(
        "afd_external_routing_under_test",
        _ROUTING_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AFDExternalRoutingInputs


def _compile_consumer(inputs_type):
    """Compile the real method out of the model module, stubbing only its imports."""

    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"))
    # The consumer is a method, so it is nested inside a class body, not top level.
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == _CONSUMER
        ),
        None,
    )
    assert node is not None, f"{_CONSUMER} is missing from {_MODEL_PATH}"

    namespace = {
        "__builtins__": __builtins__,
        "torch": torch,
        # Plumbing, not behaviour: the test hands the routed experts in directly.
        "_routed_experts_of": lambda mlp: mlp,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_MODEL_PATH), "exec"),
         namespace)
    return namespace[_CONSUMER]


class _Recorder:
    """Records the calls the consumer makes and the state it leaves behind."""

    def __init__(self, inputs_type):
        self.inputs_type = inputs_type
        self.apply_kwargs = None
        self.finalize_kwargs = None
        self.prepare_error = None
        self.apply_error = None

    def build(self, comm_base, *, padded_rows: int = 3, mc2_mask=None):
        recorder = self

        class FakeComm(comm_base):
            def __init__(self):
                self.prepare_finalize = types.SimpleNamespace(
                    afd_external_routing=None,
                )
                self.mc2_mask = mc2_mask

            def prepare(self, **kwargs):
                if recorder.prepare_error is not None:
                    raise recorder.prepare_error
                published = self.prepare_finalize.afd_external_routing
                # Stand-in for the prepare patch: the aligned routing is the padded one.
                self.prepare_finalize.afd_external_routing = published.padded(
                    padded_rows - published.token_count,
                )
                return types.SimpleNamespace(
                    hidden_states=torch.zeros(padded_rows, 4),
                    # BF16 carries no activation scale; the mask is MC2's own state.
                    pertoken_scale=None,
                    mc2_mask=self.mc2_mask,
                    padded_hidden_states_shape=None,
                )

            def finalize(self, **kwargs):
                recorder.finalize_kwargs = kwargs
                return torch.zeros(1, 4)

        recorder.comm = FakeComm()
        return recorder.comm

    def routed_experts(self):
        recorder = self

        class FakeQuantMethod:
            def apply(self, **kwargs):
                if recorder.apply_error is not None:
                    raise recorder.apply_error
                recorder.apply_kwargs = kwargs
                return types.SimpleNamespace(routed_out=torch.zeros(1, 4))

        return types.SimpleNamespace(
            quant_type="fake-quant-type",
            moe_config=types.SimpleNamespace(is_sequence_parallel=False),
            quant_method=FakeQuantMethod(),
            ascend_pertoken_scale=None,
            ascend_mc2_mask=None,
        )


@pytest.fixture
def consumer(monkeypatch):
    inputs_type = _load_routing_inputs()
    recorder = _Recorder(inputs_type)

    class AllGatherCommImpl:
        pass

    class MC2CommImpl:
        pass

    comm_module = types.ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    comm_module.AllGatherCommImpl = AllGatherCommImpl
    comm_module.MC2CommImpl = MC2CommImpl
    context_module = types.ModuleType("vllm_ascend.ascend_forward_context")
    context_module._EXTRA_CTX = types.SimpleNamespace(moe_comm_method=None)

    for module in (
        types.ModuleType("vllm_ascend"),
        types.ModuleType("vllm_ascend.ops"),
        types.ModuleType("vllm_ascend.ops.fused_moe"),
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        comm_module,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context", context_module)

    # The real payload type, registered so the consumer's deferred import finds it
    # without importing the connectors package (which pulls vLLM in).
    routing_module = types.ModuleType("afd_plugin.connectors.npu.external_routing")
    routing_module.AFDExternalRoutingInputs = inputs_type
    for name in ("afd_plugin", "afd_plugin.connectors", "afd_plugin.connectors.npu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "afd_plugin.connectors.npu.external_routing",
        routing_module,
    )

    return recorder, _compile_consumer(inputs_type), AllGatherCommImpl, MC2CommImpl, context_module


def _routing_rows(rows: int = 2):
    values = torch.arange(rows * 2)
    return (
        values.to(torch.int32).reshape(rows, 2),
        (values.to(torch.float32) / 10).reshape(rows, 2),
    )


def test_ffn_consumes_the_prepared_routing_and_reduces_across_ranks(consumer):
    recorder, consumer_fn, AllGather, _, context_module = consumer
    subject = types.SimpleNamespace(mlp=recorder.routed_experts())
    comm = recorder.build(AllGather, padded_rows=3)
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    consumer_fn(subject, torch.zeros(2, 4), topk_ids=topk_ids, topk_weights=topk_weights)

    # apply sees the padded routing the prepare step published, not the raw pair.
    assert recorder.apply_kwargs["topk_ids"].shape[0] == 3
    assert recorder.apply_kwargs["topk_ids"][:2].tolist() == topk_ids.tolist()
    assert recorder.apply_kwargs["topk_weights"][:2].tolist() == topk_weights.tolist()
    assert recorder.apply_kwargs["shared_experts"] is None
    assert recorder.finalize_kwargs["reduce_results"] is True


def test_mc2_path_skips_the_cross_rank_reduce(consumer):
    recorder, consumer_fn, _, MC2, context_module = consumer
    subject = types.SimpleNamespace(mlp=recorder.routed_experts())
    comm = recorder.build(MC2, padded_rows=2)
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    consumer_fn(subject, torch.zeros(2, 4), topk_ids=topk_ids, topk_weights=topk_weights)

    assert recorder.finalize_kwargs["reduce_results"] is False


def test_an_unwired_comm_method_is_rejected(consumer):
    recorder, consumer_fn, _, _, context_module = consumer
    subject = types.SimpleNamespace(mlp=recorder.routed_experts())
    context_module._EXTRA_CTX.moe_comm_method = object()

    with pytest.raises(RuntimeError, match="ALLGATHER and MC2"):
        consumer_fn(
            subject,
            torch.zeros(2, 4),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            topk_weights=torch.zeros(2, 2, dtype=torch.float32),
        )


def test_a_failed_prepare_leaves_no_published_routing(consumer):
    recorder, consumer_fn, AllGather, _, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    comm = recorder.build(AllGather, padded_rows=3)
    recorder.prepare_error = ValueError("prepare exploded")
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    with pytest.raises(ValueError, match="prepare exploded"):
        consumer_fn(
            subject,
            torch.zeros(2, 4),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    # The next call must not start from another call's routing or per-token state.
    assert comm.prepare_finalize.afd_external_routing is None
    assert routed_experts.ascend_pertoken_scale is None
    assert routed_experts.ascend_mc2_mask is None


def test_a_failed_apply_clears_the_per_token_state(consumer):
    recorder, consumer_fn, AllGather, _, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    comm = recorder.build(AllGather, padded_rows=3)
    recorder.apply_error = RuntimeError("apply exploded")
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    with pytest.raises(RuntimeError, match="apply exploded"):
        consumer_fn(
            subject,
            torch.zeros(2, 4),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert routed_experts.ascend_pertoken_scale is None
    assert routed_experts.ascend_mc2_mask is None
    assert comm.prepare_finalize.afd_external_routing is None

def test_mc2_mask_is_cleared_when_apply_fails(consumer):
    """A stale valid-token mask would silently route the next call's tokens."""

    recorder, consumer_fn, _, MC2, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    mask = torch.tensor([True, False, True])
    comm = recorder.build(MC2, padded_rows=3, mc2_mask=mask)
    recorder.apply_error = RuntimeError("apply exploded")
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    with pytest.raises(RuntimeError, match="apply exploded"):
        consumer_fn(
            subject,
            torch.zeros(2, 4),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    # The mask was non-empty before the failure, so only the cleanup can clear it.
    assert routed_experts.ascend_mc2_mask is None
    assert routed_experts.ascend_pertoken_scale is None
