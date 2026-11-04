# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""What the ALLGATHER prepare patch puts on the wire, observed from its output.

The patch copies upstream and injects routing handling into the two places where
the activation is padded and gathered. These tests drive the patched function with
a fake DP group and assert on ``MoEPrepareOutput``: which rows came back, what the
padding rows carry, and whether a missing ``router_logits`` is tolerated.

They load the patch module with a minimal fake ``vllm`` / ``vllm_ascend`` surface,
so they run on a CPU host. They do not assert how the patch reaches that output.
"""

import importlib
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_PATCH_MODULE = "afd_plugin.compat.patches.npu.external_routing_prepare"
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
    """Load the routing type by file path, bypassing the connectors __init__."""

    spec = importlib.util.spec_from_file_location(
        "afd_external_routing_for_prepare_test",
        _ROUTING_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AFDExternalRoutingInputs


def _install_fake_upstream(monkeypatch):
    """Install the smallest ``vllm`` / ``vllm_ascend`` surface the patch imports."""

    class QuantType(Enum):
        NONE = 0
        W8A8 = 1

    class MoEPrepareOutput:
        def __init__(self, hidden_states, router_logits, mc2_mask,
                     padded_hidden_states_shape, pertoken_scale):
            self.hidden_states = hidden_states
            self.router_logits = router_logits
            self.mc2_mask = mc2_mask
            self.padded_hidden_states_shape = padded_hidden_states_shape
            self.pertoken_scale = pertoken_scale

    class PrepareAndFinalizeWithAllGather:
        """Stands in for the real class; the patch replaces one method on it."""

        @staticmethod
        def _prepare_with_dp_group(self, *, hidden_states, router_logits):
            """The upstream implementation the installer captures."""

            return "upstream"

    context = SimpleNamespace(max_tokens_across_dp=0, max_tokens_across_pcp=0)

    def module(name):
        created = ModuleType(name)
        created.__path__ = []
        monkeypatch.setitem(sys.modules, name, created)
        return created

    def bind(parent, name, child):
        setattr(parent, name, child)
        monkeypatch.setitem(sys.modules, f"{parent.__name__}.{name}", child)

    vllm = module("vllm")
    vllm_distributed = module("vllm.distributed")
    vllm_parallel = module("vllm.distributed.parallel_state")
    vllm_parallel.get_pcp_group = lambda: None
    bind(vllm, "distributed", vllm_distributed)
    bind(vllm_distributed, "parallel_state", vllm_parallel)

    ascend = module("vllm_ascend")
    ascend_fc = module("vllm_ascend.ascend_forward_context")
    ascend_fc._EXTRA_CTX = context
    ascend_ops = module("vllm_ascend.ops")
    ascend_fused = module("vllm_ascend.ops.fused_moe")
    ascend_dataclass = module("vllm_ascend.ops.fused_moe.dataclass")
    ascend_pf_dataclass = module(
        "vllm_ascend.ops.fused_moe.dataclass.prepare_finalize",
    )
    ascend_pf_dataclass.MoEPrepareOutput = MoEPrepareOutput
    ascend_pf = module("vllm_ascend.ops.fused_moe.prepare_finalize")
    ascend_pf.PrepareAndFinalizeWithAllGather = PrepareAndFinalizeWithAllGather
    ascend_quant = module("vllm_ascend.quantization")
    ascend_quant_type = module("vllm_ascend.quantization.quant_type")
    ascend_quant_type.QuantType = QuantType

    bind(ascend, "ascend_forward_context", ascend_fc)
    bind(ascend, "ops", ascend_ops)
    bind(ascend_ops, "fused_moe", ascend_fused)
    bind(ascend_fused, "dataclass", ascend_dataclass)
    bind(ascend_dataclass, "prepare_finalize", ascend_pf_dataclass)
    bind(ascend_fused, "prepare_finalize", ascend_pf)
    bind(ascend, "quantization", ascend_quant)
    bind(ascend_quant, "quant_type", ascend_quant_type)

    return PrepareAndFinalizeWithAllGather, context


def _load_patch_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _PATCH_MODULE, raising=False)
    return importlib.import_module(_PATCH_MODULE)


def _routing(rows):
    """Routing whose ids encode the row index."""

    inputs_type = _routing_inputs_type()
    ids = torch.arange(rows * TOP_K, dtype=torch.int32).reshape(rows, TOP_K)
    weights = torch.full((rows, TOP_K), 0.5, dtype=torch.float32)
    return inputs_type(
        topk_ids=ids,
        topk_weights=weights,
    )


class _FakeDpGroup:
    """Collective that concatenates the shard arriving from each rank."""

    def __init__(self, dp_size):
        self.dp_size = dp_size
        self.calls = []

    def all_gather(self, tensor, dim):
        assert dim == 0
        self.calls.append(tuple(tensor.shape))
        return tensor.repeat(self.dp_size, *([1] * (tensor.dim() - 1)))


def _prepared_self(dp_group, external):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            dp_size=dp_group.dp_size,
            pcp_size=1,
            dp_group=dp_group,
        ),
        num_tokens=0,
        afd_external_routing=external,
    )


def test_routing_is_padded_and_gathered_with_the_activation(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 3
    dp_group = _FakeDpGroup(dp_size=2)
    subject = _prepared_self(dp_group, _routing(rows=2))

    output = patch.prepare_with_dp_group(
        subject,
        hidden_states=torch.zeros(2, HIDDEN),
        router_logits=None,
    )

    # Two local rows padded to three, gathered from two ranks.
    assert output.hidden_states.shape[0] == 6
    assert subject.afd_external_routing.token_count == 6
    # The gathered routing keeps the ids the Attention side produced.
    assert output.hidden_states.dtype == torch.float32
    assert subject.afd_external_routing.topk_ids[0].tolist() == [0, 1]
    # Row 1 of the routing still describes row 1 of the activation.
    assert subject.afd_external_routing.topk_ids[1].tolist() == [2, 3]


def test_padding_rows_are_inert(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 3
    # dp_size > 1 is what makes upstream enter the pad-then-gather block.
    dp_group = _FakeDpGroup(dp_size=2)
    subject = _prepared_self(dp_group, _routing(rows=2))

    output = patch.prepare_with_dp_group(
        subject,
        hidden_states=torch.zeros(2, HIDDEN),
        router_logits=None,
    )
    gathered = subject.afd_external_routing

    # Two local rows padded to three, then gathered from two ranks.
    assert gathered.token_count == 6
    # The appended row is inert.
    assert gathered.topk_ids[2].tolist() == [0, 0]
    assert gathered.topk_weights[2].tolist() == [0.0, 0.0]
    assert output.pertoken_scale is None


def test_absent_routing_leaves_upstream_behaviour(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 3
    dp_group = _FakeDpGroup(dp_size=2)
    subject = _prepared_self(dp_group, external=None)

    output = patch.prepare_with_dp_group(
        subject,
        hidden_states=torch.zeros(2, HIDDEN),
        router_logits=torch.zeros(2, 8),
    )

    assert output.hidden_states.shape[0] == 6
    assert output.router_logits.shape[0] == 6
    assert output.pertoken_scale is None
    assert subject.afd_external_routing is None


def test_missing_router_logits_and_dp_size_one_is_a_noop(monkeypatch):
    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 0
    dp_group = _FakeDpGroup(dp_size=1)
    subject = _prepared_self(dp_group, _routing(rows=2))

    output = patch.prepare_with_dp_group(
        subject,
        hidden_states=torch.zeros(2, HIDDEN),
        router_logits=None,
    )

    # dp_size == 1 skips the whole block, so nothing is padded or gathered.
    assert output.hidden_states.shape[0] == 2
    assert output.router_logits is None
    assert dp_group.calls == []
    assert subject.afd_external_routing.topk_ids.shape[0] == 2


def test_bf16_transfer_keeps_rows_paired(monkeypatch):
    """The next stage runs BF16 with no scale; rows must still line up.

    Every activation row carries its own marker, so a padding or gather bug shows
    up as a mismatched marker instead of passing on shapes alone.
    """

    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 3
    dp_group = _FakeDpGroup(dp_size=2)

    inputs_type = _routing_inputs_type()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    weights = torch.full((2, TOP_K), 0.5, dtype=torch.float32)
    subject = _prepared_self(
        dp_group,
        inputs_type(topk_ids=ids, topk_weights=weights),
    )

    # Two local rows: row 0 filled with 10, row 1 with 20.
    hidden_states = torch.stack(
        [torch.full((HIDDEN,), 10.0), torch.full((HIDDEN,), 20.0)],
    ).to(torch.bfloat16)
    output = patch.prepare_with_dp_group(
        subject,
        hidden_states=hidden_states,
        router_logits=None,
    )

    gathered = subject.afd_external_routing
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.hidden_states.shape[0] == 6
    assert gathered.token_count == 6
    # This stage carries no activation scale, and none was invented.
    assert output.pertoken_scale is None
    # Row r of the routing still names the row-r marker, and the padding row
    # lines up with a zeroed activation row.
    for row, marker, expected_ids in (
        (0, 10.0, [0, 1]),
        (1, 20.0, [2, 3]),
        (2, 0.0, [0, 0]),
    ):
        assert output.hidden_states[row].float().max().item() == marker
        assert gathered.topk_ids[row].tolist() == expected_ids


def test_routing_with_a_different_token_count_is_rejected(monkeypatch):
    """Different row counts pad to different totals, misaligning the result."""

    _, context = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    context.max_tokens_across_dp = 3
    dp_group = _FakeDpGroup(dp_size=2)
    subject = _prepared_self(dp_group, _routing(rows=1))

    with pytest.raises(ValueError, match="different token count"):
        patch.prepare_with_dp_group(
            subject,
            hidden_states=torch.zeros(2, HIDDEN),
            router_logits=None,
        )

    # The check runs before any collective, so nothing was gathered.
    assert dp_group.calls == []


def test_install_captures_the_upstream_implementation(monkeypatch):
    """The reference needs the unpatched function, so the first install keeps it."""

    base, _ = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    upstream = base._prepare_with_dp_group

    assert patch.apply_afd_external_routing_prepare_patch() is True

    assert patch.ORIGINAL_PREPARE_WITH_DP_GROUP is upstream
    assert base._prepare_with_dp_group is patch.prepare_with_dp_group


def test_repeated_install_keeps_the_first_capture(monkeypatch):
    """A second install must not replace the control with the patch."""

    base, _ = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    upstream = base._prepare_with_dp_group
    patch.apply_afd_external_routing_prepare_patch()

    assert patch.apply_afd_external_routing_prepare_patch() is False

    assert patch.ORIGINAL_PREPARE_WITH_DP_GROUP is upstream


def test_install_after_the_patch_already_landed_is_rejected(monkeypatch):
    """Capturing late would hand the reference the patch itself."""

    base, _ = _install_fake_upstream(monkeypatch)
    patch = _load_patch_module(monkeypatch)
    # The class already points at the patch, as if an install ran elsewhere first
    # without capturing anything.
    base._prepare_with_dp_group = patch.prepare_with_dp_group

    with pytest.raises(RuntimeError, match="before the upstream implementation"):
        patch.apply_afd_external_routing_prepare_patch()

    assert patch.ORIGINAL_PREPARE_WITH_DP_GROUP is None
