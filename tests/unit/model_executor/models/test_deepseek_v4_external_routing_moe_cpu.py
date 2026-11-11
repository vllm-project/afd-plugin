# mypy: disable-error-code="attr-defined, name-defined"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the AFD-owned external-routing MoE component.

The component imports vLLM-Ascend types at module scope, so the tests
install stand-ins for those modules, load the real component from its file, and
drive it with fakes for the comm instance and the routed experts.

What this covers: the routing reaches ``apply`` unchanged, only the ALLGATHER path
reduces during finalize, an unsupported comm method is rejected, comm instances are
reused across calls, and a failure in any stage leaves no per-call state behind.

The real operators and graph execution stay with the green-zone suite.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = (
    _REPO_ROOT
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v4_external_routing_moe.py"
)
_ROUTING_PATH = (
    _REPO_ROOT / "afd_plugin" / "connectors" / "npu" / "camp2p" / "external_routing.py"
)


class _MoECommType:
    ALLGATHER = "ALLGATHER"
    MC2 = "MC2"
    ALLTOALL = "ALLTOALL"
    FUSED_MC2 = "FUSED_MC2"


@dataclass(frozen=True)
class _PrepareOutput:
    hidden_states: torch.Tensor
    router_logits: torch.Tensor | None
    mc2_mask: torch.Tensor | None
    padded_hidden_states_shape: torch.Size | None
    pertoken_scale: torch.Tensor | None = None


class _PrepareFinalizeBase:
    def __init__(self, moe_config):
        self.moe_config = moe_config
        self.tp_rank = 0
        self.tp_size = 1

    def prepare_external(self, hidden_states, routing, **kwargs):
        return (
            _PrepareOutput(
                hidden_states=hidden_states,
                router_logits=None,
                mc2_mask=torch.ones(8, dtype=torch.bool),
                padded_hidden_states_shape=None,
                pertoken_scale=None,
            ),
            routing,
        )


class _CommBase:
    def __init__(self, moe_config):
        self.moe_config = moe_config
        self.prepare_finalize = self._get_prepare_finalize()
        self.finalized: list[dict] = []

    def _get_prepare_finalize(self):
        return _PrepareFinalizeBase(self.moe_config)

    def finalize(self, **kwargs):
        self.finalized.append(kwargs)
        return kwargs["hidden_states"]


class _AllGatherComm(_CommBase):
    pass


class _MC2Comm(_CommBase):
    pass


class _TokenDispatchInput:
    """Stand-in for the upstream dispatch input the dispatcher reads."""


class _UpstreamMC2Dispatcher:
    """Stand-in for ``TokenDispatcherWithMC2`` with upstream's kwarg shape.

    The body mirrors upstream's resolution rule so the subclass is exercised
    against the kwargs it actually receives: ``scales`` is ``None`` here and the
    quant mode comes from the quant feature.
    """

    def get_dispatch_mc2_kwargs(self, token_dispatch_input):
        quant = token_dispatch_input.quant
        if quant.comm_quant_mode is not None:
            quant_mode = quant.comm_quant_mode
        elif quant.dispatch_with_quant:
            quant_mode = 4 if quant.is_mxfp else 2
        else:
            quant_mode = 0
        return {
            "x": token_dispatch_input.hidden_states,
            "expert_ids": token_dispatch_input.topk_ids,
            "scales": None,
            "quant_mode": quant_mode,
        }


def _moe_config():
    return types.SimpleNamespace(dp_size=1, pcp_size=1, tp_size=1, dp_group=None)


_STUBBED_MODULE_NAMES = (
    "vllm",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
    "vllm_ascend",
    "vllm_ascend.ops",
    "vllm_ascend.ops.fused_moe",
    "vllm_ascend.ops.fused_moe.dataclass",
    "vllm_ascend.ops.fused_moe.dataclass.prepare_finalize",
    "vllm_ascend.ops.fused_moe.prepare_finalize",
    "vllm_ascend.ops.fused_moe.moe_comm_method",
    "vllm_ascend.ops.fused_moe.moe_utils",
    "vllm_ascend.quantization",
    "vllm_ascend.quantization.quant_type",
    "vllm_ascend.ascend_forward_context",
    "afd_plugin",
    "afd_plugin.connectors",
    "afd_plugin.connectors.npu",
    "afd_plugin.connectors.npu.camp2p.external_routing",
)


def _install_stub_modules():
    """Register the upstream modules the component imports from."""
    routing_spec = importlib.util.spec_from_file_location(
        "afd_external_routing_under_test", _ROUTING_PATH
    )
    assert routing_spec is not None and routing_spec.loader is not None
    routing_module = importlib.util.module_from_spec(routing_spec)
    routing_spec.loader.exec_module(routing_module)

    quant = types.ModuleType("vllm_ascend.quantization.quant_type")
    quant.QuantType = type("QuantType", (), {"NONE": 0})

    dataclass_module = types.ModuleType(
        "vllm_ascend.ops.fused_moe.dataclass.prepare_finalize"
    )
    dataclass_module.MoEPrepareOutput = _PrepareOutput

    prepare_module = types.ModuleType("vllm_ascend.ops.fused_moe.prepare_finalize")
    prepare_module.PrepareAndFinalizeWithAllGather = _PrepareFinalizeBase
    prepare_module.PrepareAndFinalizeWithMC2 = _PrepareFinalizeBase

    comm_module = types.ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    comm_module.AllGatherCommImpl = _AllGatherComm
    comm_module.MC2CommImpl = _MC2Comm
    # The native selector is keyed by MoECommType and returns None when this
    # configuration was not initialised with that method.
    comm_by_type = {
        _MoECommType.ALLGATHER: _AllGatherComm,
        _MoECommType.MC2: _MC2Comm,
    }

    def get_moe_comm_method(comm_type, moe_config=None):
        impl = comm_by_type.get(comm_type)
        return None if impl is None else impl(moe_config)

    comm_module.get_moe_comm_method = get_moe_comm_method

    moe_utils = types.ModuleType("vllm_ascend.ops.fused_moe.moe_utils")
    moe_utils._pad_tokens_with_cat = lambda tensor, length: tensor

    def maybe_normalize_mxfp_scale_layout(scale):
        """Mirror upstream: 2-D MXFP scales become ``(rows, cols // 2, 2)``."""

        if scale is None or scale.ndim != 2:
            return scale
        return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)

    moe_utils.maybe_normalize_mxfp_scale_layout = maybe_normalize_mxfp_scale_layout

    dispatch_input_module = types.ModuleType(
        "vllm_ascend.ops.fused_moe.dataclass.token_dispatcher"
    )
    dispatch_input_module.MoETokenDispatchInput = _TokenDispatchInput

    dispatcher_module = types.ModuleType("vllm_ascend.ops.fused_moe.token_dispatcher")
    dispatcher_module.TokenDispatcherWithMC2 = _UpstreamMC2Dispatcher

    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.get_pcp_group = lambda: None

    enum_module = types.ModuleType("vllm_ascend.ascend_forward_context")
    enum_module.MoECommType = _MoECommType

    context = enum_module
    context._EXTRA_CTX = types.SimpleNamespace(
        moe_comm_method=_AllGatherComm(_moe_config()),
        moe_comm_type=_MoECommType.ALLGATHER,
        mc2_mask=torch.ones(8, dtype=torch.bool),
        padded_num_tokens=0,
        max_tokens_across_dp=0,
        max_tokens_across_pcp=0,
    )

    afd_routing = types.ModuleType("afd_plugin.connectors.npu.camp2p.external_routing")
    afd_routing.AFDExternalRoutingInputs = routing_module.AFDExternalRoutingInputs

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.parallel_state": parallel,
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.ops": types.ModuleType("vllm_ascend.ops"),
        "vllm_ascend.ops.fused_moe": types.ModuleType("vllm_ascend.ops.fused_moe"),
        "vllm_ascend.ops.fused_moe.dataclass": types.ModuleType(
            "vllm_ascend.ops.fused_moe.dataclass"
        ),
        "vllm_ascend.ops.fused_moe.dataclass.prepare_finalize": dataclass_module,
        "vllm_ascend.ops.fused_moe.dataclass.token_dispatcher": dispatch_input_module,
        "vllm_ascend.ops.fused_moe.token_dispatcher": dispatcher_module,
        "vllm_ascend.ops.fused_moe.prepare_finalize": prepare_module,
        "vllm_ascend.ops.fused_moe.moe_comm_method": comm_module,
        "vllm_ascend.ops.fused_moe.moe_utils": moe_utils,
        "vllm_ascend.quantization": types.ModuleType("vllm_ascend.quantization"),
        "vllm_ascend.quantization.quant_type": quant,
        "vllm_ascend.ascend_forward_context": context,
        "afd_plugin": types.ModuleType("afd_plugin"),
        "afd_plugin.connectors": types.ModuleType("afd_plugin.connectors"),
        "afd_plugin.connectors.npu": types.ModuleType("afd_plugin.connectors.npu"),
        "afd_plugin.connectors.npu.camp2p.external_routing": afd_routing,
    }
    assert set(modules) == set(_STUBBED_MODULE_NAMES)
    for name, module in modules.items():
        # find_spec() is called on these names at import time and raises on a
        # module without a spec, so the stand-ins carry one.
        module.__spec__ = ModuleSpec(name, None)
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        "afd_deepseek_v4_external_routing_moe", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    component_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component_module)
    return component_module, context, comm_module


def _restore_modules(written: dict[str, types.ModuleType | None]) -> None:
    """Put back only the names this fixture replaced, and drop the ones it added."""

    for name, original in written.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(scope="module")
def loaded():
    written = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}
    try:
        module, context, comm_module = _install_stub_modules()
    except BaseException:
        _restore_modules(written)
        raise
    try:
        yield module, context, comm_module
    finally:
        _restore_modules(written)


class _Experts(torch.nn.Module):
    def __init__(self, fail_at=None):
        super().__init__()
        self.moe_config = types.SimpleNamespace(
            is_sequence_parallel=False, dp_size=1, pcp_size=1, tp_size=1, dp_group=None
        )
        self.quant_type = 0
        self.ascend_pertoken_scale = "stale"
        self.ascend_mc2_mask = "stale"
        self.seen = []
        self.scales_seen = []
        self.fail_at = fail_at
        self.quant_method = self

    def apply(
        self, *, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.fail_at == "apply":
            raise RuntimeError("apply failed")
        self.seen.append((topk_ids.clone(), topk_weights.clone()))
        self.scales_seen.append(self.ascend_pertoken_scale)
        # Identity, not arithmetic: the activation may be FP8 and the tests only
        # need to see that finalize received the experts' output.
        return types.SimpleNamespace(routed_out=x)


def _e8m0_rows(rows, columns):
    """Build a scale whose every row carries different E8M0 bytes.

    Row ``r`` starts at byte ``130 + 2 * r`` so both the value and the row order
    are visible in the assertion: a transform that reorders, pads or re-lays the
    scale cannot keep these bytes in place by accident.
    """

    values = [
        [130 + 2 * row + column for column in range(columns)] for row in range(rows)
    ]
    return torch.tensor(values, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _e8m0_bytes(tensor):
    return tensor.view(torch.uint8).reshape(-1).tolist()


def _routing(component_module, offset, tokens=4):
    ids = torch.arange(offset, offset + tokens, dtype=torch.int32).reshape(tokens, 1)
    return component_module.AFDExternalRoutingInputs(
        topk_ids=ids, topk_weights=torch.ones(tokens, 1)
    )


def test_the_routing_reaches_apply_unchanged(loaded):
    module, context, _ = loaded
    experts = _Experts()
    component = module.AFDExternalRoutingMoE(experts)
    context._EXTRA_CTX.moe_comm_method = _AllGatherComm(_moe_config())

    hidden = torch.zeros(4, 2, dtype=torch.bfloat16)
    ids = torch.arange(4, dtype=torch.int32).reshape(4, 1)
    weights = torch.ones(4, 1)
    component.forward(hidden, topk_ids=ids, topk_weights=weights)

    assert torch.equal(experts.seen[-1][0], ids)
    assert torch.equal(experts.seen[-1][1], weights)
    # A model-dtype activation publishes no scale, which is the BF16 transport.
    assert experts.scales_seen[-1] is None


@pytest.mark.parametrize(
    "comm_cls,reduces", [(_AllGatherComm, True), (_MC2Comm, False)]
)
def test_only_allgather_reduces_during_finalize(loaded, comm_cls, reduces):
    module, context, _ = loaded
    comm = comm_cls(_moe_config())
    context._EXTRA_CTX.moe_comm_method = comm
    experts = _Experts()
    component = module.AFDExternalRoutingMoE(experts)

    component.forward(
        torch.zeros(4, 2),
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        topk_weights=torch.ones(4, 1),
    )
    afd_comm = component._comm_for(context._EXTRA_CTX.moe_comm_type, comm)
    assert afd_comm.finalized[-1]["reduce_results"] is reduces

    # finalize runs after the per-token state is cleared; record what it saw
    seen = {}
    original_finalize = afd_comm.finalize

    def recording_finalize(**kwargs):
        seen["mask"] = experts.ascend_mc2_mask
        seen["scale"] = experts.ascend_pertoken_scale
        return original_finalize(**kwargs)

    afd_comm.finalize = recording_finalize
    component.forward(
        torch.zeros(4, 2),
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        topk_weights=torch.ones(4, 1),
    )
    assert seen["mask"] is None, "finalize ran with per-token state still set"
    assert seen["scale"] is None, "finalize ran with per-token state still set"
    assert context._EXTRA_CTX.moe_comm_method is comm, "the comm was not restored"
    assert experts.ascend_mc2_mask is None, "per-call state leaked"
    assert experts.ascend_pertoken_scale is None, "per-call state leaked"


def test_comm_is_reused_across_calls(loaded):
    module, context, _ = loaded
    native = _AllGatherComm(_moe_config())
    context._EXTRA_CTX.moe_comm_method = native
    component = module.AFDExternalRoutingMoE(_Experts())

    before = dict(component._candidates)
    for _ in range(2):
        component.forward(
            torch.zeros(4, 2),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
        )
    assert component._candidates == before, "a forward built a new comm"


def test_an_unsupported_comm_method_is_rejected(loaded):
    module, context, _ = loaded
    context._EXTRA_CTX.moe_comm_method = types.SimpleNamespace(moe_config=None)
    component = module.AFDExternalRoutingMoE(_Experts())

    with pytest.raises(RuntimeError, match="ALLGATHER and MC2"):
        component.forward(
            torch.zeros(4, 2),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
        )


def test_a_failure_in_apply_leaves_no_per_call_state(loaded):
    module, context, _ = loaded
    experts = _Experts(fail_at="apply")
    native = _AllGatherComm(_moe_config())
    context._EXTRA_CTX.moe_comm_method = native
    component = module.AFDExternalRoutingMoE(experts)

    with pytest.raises(RuntimeError, match="apply failed"):
        component.forward(
            torch.zeros(4, 2),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
        )
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert context._EXTRA_CTX.moe_comm_method is native, "the comm was not restored"


class _GatherGroup:
    def __init__(self, ranks):
        self.ranks = ranks

    def all_gather(self, tensor, dim=0):
        return torch.cat([tensor] * self.ranks, dim=dim)


@pytest.mark.parametrize(
    "activation_dtype,scale_present,scale_rows,message",
    [
        pytest.param(
            torch.float8_e4m3fn, False, 0, "without its MX scale", id="no-scale"
        ),
        pytest.param(torch.bfloat16, True, 4, "E4M3 activation", id="scale-on-bf16"),
        pytest.param(torch.float8_e5m2, True, 4, "E4M3 activation", id="e5m2"),
    ],
)
def test_only_a_matching_e4m3_scale_pair_reaches_the_experts(
    loaded, activation_dtype, scale_present, scale_rows, message
):
    """A half pair or a wrongly sized one would be silently wrong, so neither starts."""
    module, context, _ = loaded
    context._EXTRA_CTX.moe_comm_method = _AllGatherComm(_moe_config())
    experts = _Experts()
    component = module.AFDExternalRoutingMoE(experts)

    with pytest.raises(RuntimeError, match=message):
        component.forward(
            torch.zeros(4, 64, dtype=activation_dtype),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
            activation_scale=_e8m0_rows(scale_rows, 2) if scale_present else None,
        )
    assert experts.seen == [], "expert execution ran on a broken pair"


def test_the_dp_gather_pads_the_scale_with_the_neutral_exponent(loaded):
    """Padding rows get byte 127, and each rank's rows keep their order."""
    module, context, _ = loaded
    cfg = types.SimpleNamespace(
        is_sequence_parallel=False,
        dp_size=2,
        pcp_size=1,
        tp_size=1,
        dp_group=_GatherGroup(2),
    )
    experts = _Experts()
    experts.moe_config = cfg
    context._EXTRA_CTX.max_tokens_across_dp = 6
    context._EXTRA_CTX.moe_comm_method = _AllGatherComm(cfg)
    component = module.AFDExternalRoutingMoE(experts)

    scale = _e8m0_rows(4, 2)
    component.forward(
        torch.zeros(4, 64, dtype=torch.float8_e4m3fn),
        topk_ids=torch.arange(4, dtype=torch.int32).reshape(4, 1),
        topk_weights=torch.ones(4, 1),
        activation_scale=scale,
    )

    # Each rank pads its 4 rows to the DP maximum of 6 before the gather, so the
    # gathered scale is [r0..r3, 127, 127] twice, two bytes (one MX block pair)
    # per row.
    expected = [130, 131, 132, 133, 134, 135, 136, 137, 127, 127, 127, 127] * 2
    consumed = experts.scales_seen[-1]
    # The gather ran on the ALLGATHER path, so the experts' layout is the
    # three-dimensional one the fused GMM consumes.
    assert consumed.shape == (12, 1, 2)
    assert _e8m0_bytes(consumed) == expected


def test_the_dp_gather_and_padding_align_the_routing(loaded):
    """Padding then gathering must move the routing with the activation."""
    module, context, _ = loaded
    group = _GatherGroup(2)
    cfg = types.SimpleNamespace(
        is_sequence_parallel=False, dp_size=2, pcp_size=1, tp_size=1, dp_group=group
    )
    experts = _Experts()
    experts.moe_config = cfg
    context._EXTRA_CTX.max_tokens_across_dp = 6
    context._EXTRA_CTX.moe_comm_method = _AllGatherComm(cfg)

    component = module.AFDExternalRoutingMoE(experts)
    ids = torch.arange(4, dtype=torch.int32).reshape(4, 1)
    component.forward(
        torch.zeros(4, 2),
        topk_ids=ids,
        topk_weights=torch.ones(4, 1),
    )
    # padded from 4 to 6 rows, then gathered across 2 ranks
    assert experts.seen[-1][0].shape[0] == 12


def test_a_failure_during_prepare_clears_state_and_restores_comm(loaded):
    module, context, _ = loaded
    native = _AllGatherComm(_moe_config())
    context._EXTRA_CTX.moe_comm_method = native
    experts = _Experts()
    component = module.AFDExternalRoutingMoE(experts)

    def exploding_prepare(*args, **kwargs):
        raise RuntimeError("prepare failed")

    component._comm_for(
        context._EXTRA_CTX.moe_comm_type, native
    ).prepare_finalize.prepare_external = exploding_prepare
    with pytest.raises(RuntimeError, match="prepare failed"):
        component.forward(
            torch.zeros(4, 2),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
        )
    assert experts.ascend_pertoken_scale is None
    assert experts.ascend_mc2_mask is None
    assert context._EXTRA_CTX.moe_comm_method is native, "the comm was not restored"


def test_a_missing_mc2_candidate_still_serves_allgather(loaded):
    """A configuration initialised without MC2 must not need an MC2 candidate."""
    module, context, comm_module = loaded
    original = comm_module.get_moe_comm_method
    comm_module.get_moe_comm_method = lambda comm_type, moe_config=None: (
        None if comm_type == _MoECommType.MC2 else original(comm_type, moe_config)
    )
    try:
        experts = _Experts()
        native = _AllGatherComm(experts.moe_config)
        context._EXTRA_CTX.moe_comm_method = native
        component = module.AFDExternalRoutingMoE(experts)
        assert module.AFDMC2CommImpl not in component._candidates
        component.forward(
            torch.zeros(4, 2),
            topk_ids=torch.zeros(4, 1, dtype=torch.int32),
            topk_weights=torch.ones(4, 1),
        )
        assert experts.seen, "the allgather candidate was not used"
    finally:
        comm_module.get_moe_comm_method = original


def test_the_comm_type_resolves_when_the_context_instance_is_unset(loaded):
    """MRV2 can leave moe_comm_method unset; the comm type is always present."""
    module, context, _ = loaded
    experts = _Experts()
    context._EXTRA_CTX.moe_comm_type = _MoECommType.ALLGATHER
    context._EXTRA_CTX.moe_comm_method = None

    component = module.AFDExternalRoutingMoE(experts)
    component.forward(
        torch.zeros(4, 2),
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        topk_weights=torch.ones(4, 1),
    )
    assert experts.seen, "expert execution did not run"
    assert context._EXTRA_CTX.moe_comm_method is None, (
        "the unset instance was not restored"
    )


_DSV4_LAYER_PATH = (
    Path(__file__).resolve().parents[4]
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v4.py"
)
_LAYER_CLASS = "AFDDeepseekV4DecoderLayer"
_LAYER_METHODS = ("compute_ffn_output", "_compute_external_routing_ffn")


def _load_decoder_layer_methods():
    """Execute the real FFN entry points from the DSV4 layer source.

    Importing the module pulls in the whole ``vllm``/``vllm_ascend`` model
    surface, so the two methods are executed against a stand-in base. Their
    bodies are read from the file unchanged.
    """

    tree = ast.parse(_DSV4_LAYER_PATH.read_text(encoding="utf-8"))
    layer = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == _LAYER_CLASS
        ),
        None,
    )
    assert layer is not None, f"{_DSV4_LAYER_PATH} no longer defines {_LAYER_CLASS}"
    methods: list[ast.stmt] = [
        item
        for item in layer.body
        if isinstance(item, ast.FunctionDef) and item.name in _LAYER_METHODS
    ]
    assert len(methods) == len(_LAYER_METHODS), (
        f"{_LAYER_CLASS} no longer defines {_LAYER_METHODS}"
    )

    class _RoutedExperts:
        """Stand-in for the upstream MoE module the layer type-checks."""

    namespace: dict[str, Any] = {
        "native": types.SimpleNamespace(DeepseekV4MoE=_RoutedExperts),
        "torch": torch,
    }
    exec(  # noqa: S102 - source is this repository's own file
        compile(ast.Module(body=methods, type_ignores=[]), "<dsv4-layer>", "exec"),
        namespace,
    )
    return namespace


def _dispatch_input(
    *,
    scale=None,
    comm_quant_mode=None,
    is_mxfp=True,
    dispatch_with_quant=True,
    activation_dtype=torch.float8_e4m3fn,
):
    """Build the fields the MC2 dispatch kwargs are read from."""
    return types.SimpleNamespace(
        hidden_states=torch.zeros(4, 64, dtype=activation_dtype),
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        quant=types.SimpleNamespace(
            is_mxfp=is_mxfp,
            dispatch_with_quant=dispatch_with_quant,
            comm_quant_mode=comm_quant_mode,
        ),
        routing=types.SimpleNamespace(pertoken_scale=scale, mc2_mask=None),
    )


@pytest.mark.parametrize("comm_quant_mode", [None, 0], ids=["unset", "explicit-0"])
def test_mc2_dispatch_consumes_the_transported_scale(loaded, comm_quant_mode):
    """The operator must read the Attention-side scale instead of making one."""
    module, _, _ = loaded
    scale = _e8m0_rows(4, 2)

    kwargs = module.AFDTokenDispatcherWithMC2().get_dispatch_mc2_kwargs(
        _dispatch_input(scale=scale, comm_quant_mode=comm_quant_mode)
    )

    assert kwargs["scales"] is scale
    assert kwargs["quant_mode"] == 0, "the activation would be quantized twice"


def test_mc2_dispatch_without_a_scale_keeps_the_native_kwargs(loaded):
    """The FFN-side gate still quantizes, so its resolution must not change."""
    module, _, _ = loaded

    kwargs = module.AFDTokenDispatcherWithMC2().get_dispatch_mc2_kwargs(
        _dispatch_input()
    )

    assert kwargs["scales"] is None
    assert kwargs["quant_mode"] == 4, "upstream resolves the MXFP communication mode"


def test_mc2_dispatch_rejects_a_conflicting_quant_mode(loaded):
    """Two quantisation schemes at once is a configuration error, not a merge."""
    module, _, _ = loaded

    with pytest.raises(RuntimeError, match="conflicts with a pre-quantized activation"):
        module.AFDTokenDispatcherWithMC2().get_dispatch_mc2_kwargs(
            _dispatch_input(scale=_e8m0_rows(4, 2), comm_quant_mode=2)
        )


@pytest.mark.parametrize(
    "scale,activation_dtype,is_mxfp,message",
    [
        pytest.param(
            _e8m0_rows(4, 2).reshape(4, 1, 2),
            torch.float8_e4m3fn,
            True,
            "scale dim 3",
            id="scale-dim",
        ),
        pytest.param(
            _e8m0_rows(4, 2),
            torch.float8_e4m3fn,
            False,
            "MXFP quant feature",
            id="not-mxfp",
        ),
    ],
)
def test_mc2_dispatch_only_accepts_the_smoked_combination(
    loaded, scale, activation_dtype, is_mxfp, message
):
    module, _, _ = loaded

    with pytest.raises(RuntimeError, match=message):
        module.AFDTokenDispatcherWithMC2().get_dispatch_mc2_kwargs(
            _dispatch_input(
                scale=scale,
                is_mxfp=is_mxfp,
                activation_dtype=activation_dtype,
            )
        )


_FFN_RUNNER_PATH = (
    Path(__file__).resolve().parents[4]
    / "afd_plugin"
    / "v1"
    / "worker"
    / "npu"
    / "ffn_model_runner.py"
)


def _load_ffn_forward():
    """Execute the real ``_ffn_forward`` from the NPU FFN runner source.

    Importing the runner pulls in vLLM and the Ascend runtime, so the method is
    executed against stand-ins for the six helpers it calls. Everything that
    decides which keywords reach the model -- the payload read and the
    ``recv_routing`` branch -- is the code from the file.
    """
    tree = ast.parse(_FFN_RUNNER_PATH.read_text(encoding="utf-8"))
    runner = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "AFDNPUFFNModelRunner"
        ),
        None,
    )
    assert runner is not None, f"{_FFN_RUNNER_PATH} no longer defines the runner"
    forward = next(
        (
            item
            for item in runner.body
            if isinstance(item, ast.FunctionDef) and item.name == "_ffn_forward"
        ),
        None,
    )
    assert forward is not None, "the runner no longer defines _ffn_forward"

    @contextlib.contextmanager
    def ascend_forward_context(**_kwargs):
        yield types.SimpleNamespace(additional_kwargs={})

    namespace: dict[str, Any] = {
        "torch": torch,
        "AFDForwardContextMetadata": types.SimpleNamespace,
        "_make_dp_metadata_payload": lambda *args, **kwargs: None,
        "_ffn_token_counts_across_ranks": lambda *args, **kwargs: torch.tensor([4]),
        "_ffn_token_count_for_rank": lambda *args, **kwargs: 4,
        "_to_dp_level_token_counts": lambda counts, **kwargs: counts,
        "_ffn_layer_indices": lambda _runner: [0],
        "ascend_forward_context": ascend_forward_context,
        "_set_moe_layer_index": lambda *args, **kwargs: None,
        "_send_ffn_output": lambda connector, output, context, **kwargs: (
            connector.send_ffn_output(output, context)
        ),
    }
    exec(  # noqa: S102 - source is this repository's own file
        compile(ast.Module(body=[forward], type_ignores=[]), "<ffn-forward>", "exec"),
        namespace,
    )
    return namespace["_ffn_forward"]


class _StubConnector:
    """Serve one payload and record what the runner sends back."""

    def __init__(self, payload, control_plane=True):
        self._payload = payload
        self.control_plane = object() if control_plane else None
        self.recv_kwargs: list[dict[str, Any]] = []
        self.sent: list[Any] = []

    def recv_attn_output(self, **kwargs):
        self.recv_kwargs.append(kwargs)
        return self._payload

    def send_ffn_output(self, ffn_output, context):
        self.sent.append(ffn_output)


def _scale_payload(scale, *, tokens=4, hidden=64, routing=True):
    ids = torch.arange(tokens, dtype=torch.int32).reshape(tokens, 1)
    weights = torch.ones(tokens, 1)
    return types.SimpleNamespace(
        hidden_states=torch.zeros(tokens, hidden, dtype=torch.float8_e4m3fn),
        context=types.SimpleNamespace(
            metadata=types.SimpleNamespace(layer_idx=0, stage_idx=0),
            states=object(),
        ),
        input_ids=None,
        topk_ids=ids if routing else None,
        topk_weights=weights if routing else None,
        activation_scale=scale,
    )


def test_the_ffn_runner_carries_the_received_scale_to_the_experts(loaded):
    """Deleting the runner's scale forwarding must fail this test.

    The whole chain from the runner's payload read to the expert call is real:
    the runner method comes from the source file, and the model entry point and
    MoE component are the ones under test elsewhere in this module.
    """
    module, context, _ = loaded
    # MC2 keeps the transported scale two-dimensionally; the DP-gather case
    # above already asserts the ALLGATHER layout, its row order and padding.
    context._EXTRA_CTX.moe_comm_method = _MC2Comm(_moe_config())
    experts = _Experts()
    namespace = _load_decoder_layer_methods()
    layer = types.SimpleNamespace(
        uses_external_routing=True,
        mlp=namespace["native"].DeepseekV4MoE(),
        afd_external_routing_moe=module.AFDExternalRoutingMoE(experts),
    )
    layer._compute_external_routing_ffn = types.MethodType(
        namespace["_compute_external_routing_ffn"], layer
    )

    class _Model:
        afd_requires_attention_routing = True
        mlp = layer.mlp

        def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
            return namespace["compute_ffn_output"](layer, hidden_states, **kwargs)

    scale = _e8m0_rows(4, 2)
    connector = _StubConnector(_scale_payload(scale))
    runner = types.SimpleNamespace(
        model=_Model(),
        connector=connector,
        max_num_tokens=4,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(data_parallel_size=1),
        ),
    )

    _load_ffn_forward()(
        runner,
        dp_metadata_list={0: object()},
        update_connector_state=False,
    )

    assert connector.recv_kwargs[0]["recv_routing"] is True
    consumed = experts.scales_seen[-1]
    assert consumed.shape == (4, 2)
    assert _e8m0_bytes(consumed) == _e8m0_bytes(scale)
    assert connector.sent, "the runner did not send the FFN output back"
