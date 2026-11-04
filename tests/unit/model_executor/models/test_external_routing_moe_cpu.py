# mypy: disable-error-code="attr-defined, name-defined"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the AFD-owned external-routing MoE component.

``external_routing_moe`` imports vLLM-Ascend types at module scope, so the tests
install stand-ins for those modules, load the real component from its file, and
drive it with fakes for the comm instance and the routed experts.

What this covers: the routing reaches ``apply`` unchanged, only the ALLGATHER path
reduces during finalize, an unsupported comm method is rejected, comm instances are
reused across calls, and a failure in any stage leaves no per-call state behind.

The real operators and graph execution stay with the green-zone suite.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = (
    _REPO_ROOT
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "external_routing_moe.py"
)
_ROUTING_PATH = _REPO_ROOT / "afd_plugin" / "connectors" / "npu" / "external_routing.py"


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


def _moe_config():
    return types.SimpleNamespace(dp_size=1, pcp_size=1, tp_size=1, dp_group=None)


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

    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.get_pcp_group = lambda: None

    enum_module = types.ModuleType("vllm_ascend.ascend_forward_context")
    enum_module.MoECommType = _MoECommType

    context = enum_module
    context._EXTRA_CTX = types.SimpleNamespace(
        moe_comm_method=_AllGatherComm(_moe_config()),
        mc2_mask=torch.ones(8, dtype=torch.bool),
        padded_num_tokens=0,
        max_tokens_across_dp=0,
        max_tokens_across_pcp=0,
    )

    afd_routing = types.ModuleType("afd_plugin.connectors.npu.external_routing")
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
        "vllm_ascend.ops.fused_moe.prepare_finalize": prepare_module,
        "vllm_ascend.ops.fused_moe.moe_comm_method": comm_module,
        "vllm_ascend.ops.fused_moe.moe_utils": moe_utils,
        "vllm_ascend.quantization": types.ModuleType("vllm_ascend.quantization"),
        "vllm_ascend.quantization.quant_type": quant,
        "vllm_ascend.ascend_forward_context": context,
        "afd_plugin": types.ModuleType("afd_plugin"),
        "afd_plugin.connectors": types.ModuleType("afd_plugin.connectors"),
        "afd_plugin.connectors.npu": types.ModuleType("afd_plugin.connectors.npu"),
        "afd_plugin.connectors.npu.external_routing": afd_routing,
    }
    for name, module in modules.items():
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        "afd_external_routing_moe", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    component_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component_module)
    return component_module, context, comm_module


@pytest.fixture(scope="module")
def loaded():
    before = dict(sys.modules)
    module, context, comm_module = _install_stub_modules()
    yield module, context, comm_module
    sys.modules.clear()
    sys.modules.update(before)


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
        self.fail_at = fail_at
        self.quant_method = self

    def apply(
        self, *, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.fail_at == "apply":
            raise RuntimeError("apply failed")
        self.seen.append((topk_ids.clone(), topk_weights.clone()))
        return types.SimpleNamespace(routed_out=x + 1)


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

    hidden = torch.zeros(4, 2)
    ids = torch.arange(4, dtype=torch.int32).reshape(4, 1)
    weights = torch.ones(4, 1)
    component.forward(hidden, topk_ids=ids, topk_weights=weights)

    assert torch.equal(experts.seen[-1][0], ids)
    assert torch.equal(experts.seen[-1][1], weights)


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
    afd_comm = component._comm_for(comm)
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


def test_dp_gather_and_padding_align_the_routing(loaded):
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

    component._comm_for(native).prepare_finalize.prepare_external = exploding_prepare
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
