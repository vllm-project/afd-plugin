# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""What the MC2 prepare patch puts on the wire, observed from its output.

MC2 pads to ``padded_num_tokens``, shards across TP ranks instead of gathering,
and carries ``mc2_mask``. These tests drive the patched function with a fake
forward context and assert on ``MoEPrepareOutput``: rows, mask alignment, and
whether a missing ``router_logits`` is tolerated in each branch.

They load the patch module with a minimal fake ``vllm`` / ``vllm_ascend`` surface
so they run on a CPU host.
"""

import importlib
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_PATCH_MODULE = "afd_plugin.compat.patches.npu.external_routing_mc2_prepare"
_ROUTING_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "afd_plugin"
    / "connectors"
    / "npu"
    / "external_routing.py"
)

TOP_K = 2
HIDDEN = 4

def _routing_inputs_type():
    spec = importlib.util.spec_from_file_location(
        "afd_external_routing_for_mc2_test",
        _ROUTING_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AFDExternalRoutingInputs


def _routing(rows):
    inputs_type = _routing_inputs_type()
    ids = torch.arange(rows * TOP_K, dtype=torch.int32).reshape(rows, TOP_K)
    weights = torch.full((rows, TOP_K), 0.5, dtype=torch.float32)
    return inputs_type(topk_ids=ids, topk_weights=weights)


def _install_fake_upstream(monkeypatch):
    """Install the smallest ``vllm`` / ``vllm_ascend`` surface the patch imports."""

    class QuantType(Enum):
        NONE = 0

    class MoEPrepareOutput:
        def __init__(self, hidden_states, router_logits, mc2_mask,
                     padded_hidden_states_shape, pertoken_scale):
            self.hidden_states = hidden_states
            self.router_logits = router_logits
            self.mc2_mask = mc2_mask
            self.padded_hidden_states_shape = padded_hidden_states_shape
            self.pertoken_scale = pertoken_scale

    class PrepareAndFinalizeWithMC2:
        """Stands in for the real class; the patch replaces one method on it."""

    def pad_tokens_with_cat(tensor, padded_len):
        pad_rows = padded_len - tensor.shape[0]
        assert pad_rows >= 0
        return torch.cat(
            [tensor, torch.zeros(pad_rows, *tensor.shape[1:], dtype=tensor.dtype)],
            dim=0,
        )

    context = SimpleNamespace(
        max_tokens_across_dp=0,
        max_tokens_across_pcp=0,
        padded_num_tokens=0,
        mc2_mask=torch.zeros(0, dtype=torch.bool),
    )

    def module(name):
        created = ModuleType(name)
        created.__path__ = []
        monkeypatch.setitem(sys.modules, name, created)
        return created

    def bind(parent, name, child):
        setattr(parent, name, child)
        monkeypatch.setitem(sys.modules, f"{parent.__name__}.{name}", child)

    vllm = module("vllm")
    vllm_ascend = module("vllm_ascend")
    ascend_fc = module("vllm_ascend.ascend_forward_context")
    ascend_fc._EXTRA_CTX = context
    ascend_ops = module("vllm_ascend.ops")
    ascend_fused = module("vllm_ascend.ops.fused_moe")
    ascend_dataclass = module("vllm_ascend.ops.fused_moe.dataclass")
    ascend_pf_dataclass = module(
        "vllm_ascend.ops.fused_moe.dataclass.prepare_finalize",
    )
    ascend_pf_dataclass.MoEPrepareOutput = MoEPrepareOutput
    ascend_moe_utils = module("vllm_ascend.ops.fused_moe.moe_utils")
    ascend_moe_utils._pad_tokens_with_cat = pad_tokens_with_cat
    ascend_pf = module("vllm_ascend.ops.fused_moe.prepare_finalize")
    ascend_pf.PrepareAndFinalizeWithMC2 = PrepareAndFinalizeWithMC2
    ascend_quant = module("vllm_ascend.quantization")
    ascend_quant_type = module("vllm_ascend.quantization.quant_type")
    ascend_quant_type.QuantType = QuantType

    bind(vllm, "vllm_ascend", None)  # placeholder, rebound below
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    bind(vllm_ascend, "ascend_forward_context", ascend_fc)
    bind(vllm_ascend, "ops", ascend_ops)
    bind(ascend_ops, "fused_moe", ascend_fused)
    bind(ascend_fused, "dataclass", ascend_dataclass)
    bind(ascend_dataclass, "prepare_finalize", ascend_pf_dataclass)
    bind(ascend_fused, "moe_utils", ascend_moe_utils)
    bind(ascend_fused, "prepare_finalize", ascend_pf)
    bind(vllm_ascend, "quantization", ascend_quant)
    bind(ascend_quant, "quant_type", ascend_quant_type)

    return PrepareAndFinalizeWithMC2, context


def _load_patch_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _PATCH_MODULE, raising=False)
    return importlib.import_module(_PATCH_MODULE)


def _subject(*, tp_size=1, tp_rank=0, external=None):
    return SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_tokens=0,
        afd_external_routing=external,
    )


def test_non_sp_branch_pads_activation_and_routing_together(monkeypatch):
    """The default branch pads both to ``padded_num_tokens``."""

    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.padded_num_tokens = 3
    subject = _subject(external=_routing(rows=2))

    output = patch.prepare(
        subject,
        hidden_states=torch.zeros(2, HIDDEN, dtype=torch.bfloat16),
        router_logits=None,
    )

    # This stage carries BF16 with no activation scale, and none was invented.
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.hidden_states.shape[0] == 3
    assert subject.afd_external_routing.token_count == 3
    # The appended row is inert.
    assert subject.afd_external_routing.topk_ids[2].tolist() == [0, 0]
    assert output.pertoken_scale is None


def test_tp_sharding_takes_the_same_slice_for_routing(monkeypatch):
    """With TP > 1 the routing follows the activation's slice."""

    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.padded_num_tokens = 4
    # Explicit valid bits, so a wrong slice is visible: rank 1 must keep the
    # true bit from column 2 and the false bit from column 3.
    context.mc2_mask = torch.tensor([True, True, True, False])
    subject = _subject(tp_size=2, tp_rank=1, external=_routing(rows=4))

    output = patch.prepare(
        subject,
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        router_logits=None,
    )

    # rank 1 keeps rows 2..3 of the padded tensor.
    assert output.hidden_states.shape[0] == 2
    assert output.hidden_states[0, 0].item() == 4.0
    assert subject.afd_external_routing.token_count == 2
    assert subject.afd_external_routing.topk_ids[0].tolist() == [4, 5]
    # The mask keeps its own bits, not just its length.
    assert output.mc2_mask.tolist() == [True, False]


def test_replace_allreduce_pads_the_local_shard(monkeypatch):
    """The SP branch pads only its own shard, and the routing follows."""

    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.padded_num_tokens = 4
    # Explicit bits, so a wrong shard or a lost padding shows up in the contents.
    context.mc2_mask = torch.tensor([True, False, True, True])
    subject = _subject(tp_size=2, tp_rank=0, external=_routing(rows=1))

    output = patch.prepare(
        subject,
        hidden_states=torch.zeros(1, HIDDEN),
        router_logits=None,
        replace_allreduce=True,
    )

    # pad target is padded_num_tokens // tp_size == 2, from one local row.
    assert output.hidden_states.shape[0] == 2
    assert subject.afd_external_routing.token_count == 2
    assert subject.afd_external_routing.topk_ids[1].tolist() == [0, 0]
    # Shard 0 owns the first bit; the padding row is False.
    assert output.mc2_mask.tolist() == [True, False]


def test_absent_routing_keeps_upstream_behaviour(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.padded_num_tokens = 3
    subject = _subject(external=None)

    output = patch.prepare(
        subject,
        hidden_states=torch.zeros(2, HIDDEN),
        router_logits=torch.zeros(2, 8),
    )

    assert output.hidden_states.shape[0] == 3
    assert output.router_logits.shape[0] == 3
    assert output.pertoken_scale is None
    assert subject.afd_external_routing is None


def test_routing_with_a_different_token_count_is_rejected(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.padded_num_tokens = 3
    subject = _subject(external=_routing(rows=1))

    with pytest.raises(ValueError, match="different token count"):
        patch.prepare(
            subject,
            hidden_states=torch.zeros(2, HIDDEN),
            router_logits=None,
        )
