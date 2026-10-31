# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the FFN side consuming Attention-side routing.

``deepseek_v4`` cannot be imported on a host without vLLM, so the consumer method is
compiled out of the real source with ``ast`` and run against stand-ins for the two
upstream modules it imports at call time. The routing payload itself is the real
type, loaded from its file.

What this covers: the prepared (padded/gathered/split) routing reaches ``apply``, the
ALLGATHER and MC2 paths differ only in whether finalize reduces across ranks, a failure
in prepare, apply or finalize leaves no per-call state behind, and a later call starts
from its own routing rather than the previous call's.

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
    return module


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
        self.finalize_error = None

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
                aligned = self.prepare_finalize.afd_external_routing
                received = kwargs["hidden_states"]
                missing = padded_rows - received.shape[0]
                prepared = (
                    received
                    if missing <= 0
                    else torch.cat(
                        (
                            received,
                            received.new_zeros((missing, received.shape[1])),
                        ),
                    )
                )
                return types.SimpleNamespace(
                    hidden_states=prepared,
                    # Derived from what prepare received: if the consumer never put the scale
                    # on the routing, this stays None and the assertions below fail.
                    pertoken_scale=aligned.scale,
                    mc2_mask=self.mc2_mask,
                    padded_hidden_states_shape=None,
                )

            def finalize(self, **kwargs):
                if recorder.finalize_error is not None:
                    raise recorder.finalize_error
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
                # The consumer clears this in its finally, so capture it here.
                scale = getattr(kwargs.get("layer"), "ascend_pertoken_scale", None)
                recorder.applied_scale = None if scale is None else scale.clone()
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
    routing_source = _load_routing_inputs()
    inputs_type = routing_source.AFDExternalRoutingInputs
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
    moe_utils = types.ModuleType("vllm_ascend.ops.fused_moe.moe_utils")

    def maybe_normalize_mxfp_scale_layout(scale):
        """Upstream's 2-D -> ``(rows, cols // 2, 2)`` conversion, as the kernels want."""

        if scale.dim() != 2:
            return scale
        return scale.view(scale.shape[0], scale.shape[1] // 2, 2)

    moe_utils.maybe_normalize_mxfp_scale_layout = maybe_normalize_mxfp_scale_layout
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.fused_moe.moe_utils", moe_utils
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context", context_module)

    # The real payload type, registered so the consumer's deferred import finds it
    # without importing the connectors package (which pulls vLLM in).
    routing_module = types.ModuleType("afd_plugin.connectors.npu.external_routing")
    routing_module.AFDExternalRoutingInputs = inputs_type
    routing_module.require_matching_activation_scale = (
        routing_source.require_matching_activation_scale
    )
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


def test_a_later_call_uses_its_own_routing(consumer):
    """A second forward must publish its own pair instead of reusing the first."""

    recorder, consumer_fn, AllGather, _, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    comm = recorder.build(AllGather, padded_rows=2)
    context_module._EXTRA_CTX.moe_comm_method = comm

    first_ids, first_weights = _routing_rows()
    consumer_fn(
        subject,
        torch.zeros(2, 4),
        topk_ids=first_ids,
        topk_weights=first_weights,
    )
    assert recorder.apply_kwargs["topk_ids"][:2].tolist() == first_ids.tolist()

    second_ids = first_ids + 50
    second_weights = first_weights + 5
    consumer_fn(
        subject,
        torch.zeros(2, 4),
        topk_ids=second_ids,
        topk_weights=second_weights,
    )

    assert recorder.apply_kwargs["topk_ids"][:2].tolist() == second_ids.tolist()
    assert recorder.apply_kwargs["topk_weights"][:2].tolist() == second_weights.tolist()


@pytest.mark.parametrize(
    ("comm_name", "reduce_results"),
    [("allgather", True), ("mc2", False)],
)
def test_the_routing_reaches_apply_and_only_allgather_reduces(
    consumer, comm_name, reduce_results
):
    """Both paths run the same flow; they differ in whether finalize reduces."""

    recorder, consumer_fn, AllGather, MC2, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    comm = recorder.build(
        AllGather if comm_name == "allgather" else MC2, padded_rows=3
    )
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    consumer_fn(subject, torch.zeros(2, 4), topk_ids=topk_ids, topk_weights=topk_weights)

    # apply sees the padded routing the prepare step published, not the raw pair.
    assert recorder.apply_kwargs["topk_ids"].shape[0] == 3
    assert recorder.apply_kwargs["topk_ids"][:2].tolist() == topk_ids.tolist()
    assert recorder.apply_kwargs["topk_weights"][:2].tolist() == topk_weights.tolist()
    assert recorder.apply_kwargs["shared_experts"] is None
    assert recorder.finalize_kwargs["reduce_results"] is reduce_results


@pytest.mark.parametrize("stage", ["prepare", "apply", "finalize"])
def test_a_failure_at_any_stage_leaves_no_per_call_state(consumer, stage):
    """A stale mask or published routing would route the next call's tokens."""

    recorder, consumer_fn, _, MC2, context_module = consumer
    routed_experts = recorder.routed_experts()
    subject = types.SimpleNamespace(mlp=routed_experts)
    # MC2 carries a valid-token mask, which is per-call state as well.
    comm = recorder.build(MC2, padded_rows=3, mc2_mask=torch.tensor([True, False, True]))
    setattr(recorder, f"{stage}_error", RuntimeError(f"{stage} exploded"))
    context_module._EXTRA_CTX.moe_comm_method = comm
    topk_ids, topk_weights = _routing_rows()

    with pytest.raises(RuntimeError, match=f"{stage} exploded"):
        consumer_fn(
            subject,
            torch.zeros(2, 4),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    assert comm.prepare_finalize.afd_external_routing is None
    assert routed_experts.ascend_pertoken_scale is None
    assert routed_experts.ascend_mc2_mask is None

_HIDDEN = 64  # 64 / 32 = 2 blocks, so a two-column scale is the right width


def _counted_scale(rows: int, columns: int = 2) -> torch.Tensor:
    """Bytes that encode row and column, so a re-layout that reorders them is visible."""

    values = [[10 * (row + 1) + column for column in range(columns)] for row in range(rows)]
    return torch.tensor(values, dtype=torch.uint8).view(torch.float8_e8m0fnu)


@pytest.mark.parametrize(
    ("path", "expected_shape"),
    [("allgather", (2, 1, 2)), ("mc2", (2, 2))],
)
def test_the_scale_layout_follows_the_moe_path(consumer, path, expected_shape):
    """ALLGATHER's GMM takes three dimensions; MC2's dispatcher takes two."""

    recorder, consumer_fn, AllGather, MC2, context_module = consumer
    subject = types.SimpleNamespace(mlp=recorder.routed_experts())
    comm_cls = AllGather if path == "allgather" else MC2
    context_module._EXTRA_CTX.moe_comm_method = recorder.build(comm_cls, padded_rows=2)
    ids, weights = _routing_rows()

    consumer_fn(
        subject,
        torch.zeros(2, _HIDDEN, dtype=torch.float8_e4m3fn),
        topk_ids=ids,
        topk_weights=weights,
        activation_scale=_counted_scale(2),
    )

    observed = recorder.applied_scale
    assert observed is not None, "the experts must see the transported scale"
    assert tuple(observed.shape) == expected_shape
    # Bytes, not just the count: a re-layout that dropped, duplicated or reordered a block
    # still has four values.
    assert observed.view(torch.uint8).reshape(-1).tolist() == [10, 11, 20, 21]

    prepared = recorder.apply_kwargs["x"]
    assert prepared.dtype == torch.float8_e4m3fn, "the experts receive the quantized form"
    assert prepared.shape == (2, _HIDDEN)


@pytest.mark.parametrize(
    ("dtype", "with_scale", "match"),
    [
        (torch.float8_e4m3fn, False, "together"),
        (torch.bfloat16, True, "together"),
        (torch.float8_e5m2, True, "E4M3"),
    ],
)
def test_a_mismatched_activation_and_scale_are_rejected(consumer, dtype, with_scale, match):
    recorder, consumer_fn, AllGather, _, context_module = consumer
    subject = types.SimpleNamespace(mlp=recorder.routed_experts())
    context_module._EXTRA_CTX.moe_comm_method = recorder.build(AllGather, padded_rows=2)
    ids, weights = _routing_rows()
    kwargs: dict[str, object] = {"topk_ids": ids, "topk_weights": weights}
    if with_scale:
        kwargs["activation_scale"] = _counted_scale(2)

    with pytest.raises(RuntimeError, match=match):
        consumer_fn(subject, torch.zeros(2, _HIDDEN, dtype=dtype), **kwargs)
