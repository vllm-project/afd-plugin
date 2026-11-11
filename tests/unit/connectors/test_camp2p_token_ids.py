# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for CAMP2P token-id transport helpers.

These helpers are pure tensor logic, so they are testable without an Ascend
device. The surrounding connector needs ``torch_npu`` and is covered by
``test_camp2p_connector.py`` instead.

The module-level ``vllm`` stub mirrors the pattern used by the compat patch
tests: importing any connector module pulls in ``vllm`` through
``afd_plugin.connectors``, so the import is satisfied with the minimal surface
the helpers' module needs.

The mode tests at the end drive the real ``send_attn_output`` and
``recv_attn_output`` against recorded operator calls. The operator's
``compute_gate`` mode is selected independently on each side and the operator
only writes its ids slot in that mode, so both sides deriving it from the same
run-level decision is the invariant worth pinning here.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from collections.abc import Iterator
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.fakes.afd_config import camp2p_config as _afd_config
from tests.unit.fakes.vllm_config import camp2p_config as _vllm_config

torch = pytest.importorskip("torch")

from tests.unit.fakes.dp_metadata import DPMetadata as _FakeDPMetadata  # noqa: E402

_STUB_MODULES = (
    "vllm",
    "vllm.forward_context",
    "vllm.logger",
    "vllm.utils",
    "vllm.utils.torch_utils",
    "vllm.distributed",
    "vllm.distributed.parallel_state",
)


def _drop_stub_bound_modules(before: set[str], stub_objects: set[int]) -> None:
    """Forget modules the stub block loaded that still hold one of its objects.

    A cached module keeps the doubles in its globals, and the parent package keeps the
    child module as an attribute, so both have to be dropped for a later import to build
    the real thing.
    """

    for name in [
        name
        for name in sys.modules
        if name not in before and name.startswith("afd_plugin")
    ]:
        module = sys.modules[name]
        if not any(
            value is not None and id(value) in stub_objects
            for value in vars(module).values()
        ):
            continue
        sys.modules.pop(name, None)
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, leaf, None) is module:
            delattr(parent, leaf)


@contextlib.contextmanager
def _vllm_stub() -> Iterator[None]:
    """Expose a minimal ``vllm`` surface for the duration of one import.

    Importing any connector module pulls in ``vllm`` through
    ``afd_plugin.connectors``. This stub supplies only the names the helpers'
    module needs at import time.

    The stub is removed again immediately afterwards. Leaving a partial
    ``vllm`` in ``sys.modules`` would mask the real "vllm is not installed"
    failure for every other test module in the same pytest session, turning a
    clear error into a confusing one.
    """

    missing = [name for name in _STUB_MODULES if name not in sys.modules]
    if not missing:
        yield
        return

    builders: dict[str, dict[str, object]] = {
        "vllm": {},
        "vllm.forward_context": {
            "DPMetadata": type("DPMetadata", (), {}),
            "get_forward_context": lambda: None,
        },
        "vllm.logger": {
            "init_logger": lambda *args, **kwargs: logging.getLogger("test")
        },
        "vllm.utils": {},
        "vllm.utils.torch_utils": {
            "direct_register_custom_op": lambda **kwargs: None,
            "is_torch_equal_or_newer": lambda *args, **kwargs: True,
        },
        "vllm.distributed": {},
        "vllm.distributed.parallel_state": {
            "get_pcp_group": None,
            "get_tensor_model_parallel_rank": None,
        },
    }

    # Only the modules that are genuinely absent are installed. Overwriting a module
    # that another test already imported would outlive this context: the restore
    # below only knows about the names it added.
    _imported_before = set(sys.modules)
    for name in missing:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        # find_spec("vllm") runs at import time; a spec-less module makes it raise.
        module.__spec__ = ModuleSpec(name, None)
        for attribute, value in builders[name].items():
            setattr(module, attribute, value)
        sys.modules[name] = module

    _stub_objects = {
        id(value)
        for name in missing
        for attribute, value in vars(sys.modules[name]).items()
        if not attribute.startswith("__") and value is not None
    }
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        # A module imported while the stub was installed keeps the doubles in its
        # globals, so it must not stay cached. Only modules the stub block loaded, and
        # only those that bind one of the stub objects, are dropped; evicting their
        # packages would rebuild the connector factory and leave two registries.
        _drop_stub_bound_modules(_imported_before, _stub_objects)


with _vllm_stub():
    from afd_plugin.connectors.metadata import (
        AFDTransferContext,
        AFDTransferMetadata,
    )
    from afd_plugin.connectors.npu.camp2p import (
        CAMP2pAFDConnector,
        prepare_token_id_transfer,
        received_token_ids,
    )
    from afd_plugin.connectors.npu.camp2p import connector as camp2p_module


class _CpuTorch:
    """``torch`` with the NPU device mapped to CPU for host-side tests.

    The connector hands the operator empty device tensors, which only exist on an
    Ascend host. Only ``tensor()`` and ``empty()`` are adapted; every other
    attribute, including ``ops``, stays the real one so the recorded operator
    calls are the calls the connector makes.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def tensor(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return torch.tensor(*args, **kwargs)

    def empty(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return torch.empty(*args, **kwargs)


def _connector(*, role: str, rank: int) -> CAMP2pAFDConnector:
    """Build a connector whose communication groups are not needed yet."""

    connector = CAMP2pAFDConnector(
        rank,
        rank,
        _vllm_config(),
        _afd_config(role=role),
        rank,
    )
    connector._initialized = True
    connector.hccl_comm_name = "hccl0"
    connector.hccl_comm_name2 = "hccl1"
    connector.hccl_comm_name3 = ""
    connector.hccl_comm_name1 = "moe"
    return connector


def test_prepare_token_id_transfer_replicates_ids_across_columns():
    input_ids = torch.tensor([7, 11, 13], dtype=torch.int64)

    ids, scales = prepare_token_id_transfer(
        input_ids,
        topk=2,
        expected_tokens=3,
    )

    assert ids.dtype == torch.int32
    assert tuple(ids.shape) == (3, 2)
    assert ids[:, 0].tolist() == [7, 11, 13]
    assert ids[:, 1].tolist() == [7, 11, 13]
    assert scales.dtype == torch.float32
    assert tuple(scales.shape) == (3, 2)
    # Scales accompany token identity, not routing weights, so they are inert.
    assert torch.count_nonzero(scales) == 0


def test_prepare_token_id_transfer_rejects_token_count_mismatch():
    input_ids = torch.tensor([7, 11], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not match the AFD transfer"):
        prepare_token_id_transfer(input_ids, topk=2, expected_tokens=3)


def test_received_token_ids_collapses_replicated_columns():
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11, 13], dtype=torch.int32),
        topk=2,
        expected_tokens=3,
    )

    received = received_token_ids(sent_ids, expected_tokens=3)

    assert received.dtype == torch.int32
    assert received.tolist() == [7, 11, 13]


def test_received_token_ids_trims_operator_padded_capacity():
    # The operator works on a padded capacity, so extra trailing rows are normal.
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11, 13, 99, 99], dtype=torch.int32),
        topk=2,
        expected_tokens=5,
    )

    received = received_token_ids(sent_ids, expected_tokens=3)

    assert received.tolist() == [7, 11, 13]


def test_received_token_ids_rejects_alignment_shortfall():
    """Misaligned ids must fail loudly rather than route the wrong tokens."""
    sent_ids, _ = prepare_token_id_transfer(
        torch.tensor([7, 11], dtype=torch.int32),
        topk=2,
        expected_tokens=2,
    )

    with pytest.raises(ValueError, match="not aligned with the FFN token layout"):
        received_token_ids(sent_ids, expected_tokens=5)


def test_send_attn_output_selects_the_operator_ids_mode(monkeypatch):
    """Sending ids must raise ``compute_gate`` and fill the operator's id slot.

    The receiving rank selects the same mode from its own declaration, so the
    sending rank has to derive it from the presence of ids alone: an
    activations-only send must leave the slot untouched.
    """

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda *args: calls.append(args),
        raising=False,
    )
    forward_context = SimpleNamespace()
    monkeypatch.setattr(
        camp2p_module,
        "get_forward_context",
        lambda: forward_context,
    )
    connector = _connector(role="attention", rank=0)
    hidden_states = torch.zeros(3, connector.hidden_size)
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=3,
        ),
    )

    connector.send_attn_output(
        hidden_states,
        context,
        input_ids=torch.tensor([7, 11, 13], dtype=torch.int64),
    )
    connector.send_attn_output(hidden_states, context)

    with_ids, without_ids = calls
    # Trailing operator arguments: aiv_num, compute_gate, ids, scales, mode.
    assert with_ids[-4] == 1
    assert with_ids[-3].dtype == torch.int32
    assert with_ids[-3][:, 0].tolist() == [7, 11, 13]
    assert with_ids[-2].dtype == torch.float32
    assert torch.count_nonzero(with_ids[-2]) == 0
    assert with_ids[-1] == camp2p_module.A2E_QUANT_MODE_BF16
    assert without_ids[-4] == 0
    assert without_ids[-3] is None
    assert without_ids[-2] is None
    assert without_ids[-1] == camp2p_module.A2E_QUANT_MODE_BF16


def test_recv_attn_output_mode_and_ids_follow_the_receiver_declaration(
    monkeypatch,
):  # long-test: paired IDs/no-IDs receive ABI and returned tensor values
    """The receiving rank declares the mode and gets ids only in that mode.

    ``recv_input_ids`` is the FFN side's half of the run-level decision: it
    selects the operator's mode and decides whether the id slot may be read. The
    ids are model-specific tensors, so they travel on the payload rather than in
    the backend transfer state.
    """

    calls: list[tuple[Any, ...]] = []

    def fake_a2e(*args: Any) -> tuple[Any, ...]:
        calls.append(args)
        tokens, topk = int(args[3]), int(args[5])
        ids = (
            torch.arange(tokens, dtype=torch.int32)
            .mul(10)
            .unsqueeze(1)
            .expand(tokens, topk)
            .contiguous()
        )
        return ("hidden", ids, None, "atten-batch", "active-mask")

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    # FFN rank 1 owns Attention ranks 1 and 3: 3 + 7 tokens.
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    with_ids = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_input_ids=True,
    )
    without_ids = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_input_ids=False,
    )

    assert calls[0][-1] == 1
    assert with_ids.input_ids is not None
    assert with_ids.input_ids.tolist() == [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]
    assert calls[1][-1] == 0
    assert without_ids.input_ids is None


def _routing(connector, rows, *, columns=None):
    """Routing sized the way the operator is configured."""

    topk = connector.num_experts_per_tok if columns is None else columns
    ids = torch.arange(rows * topk, dtype=torch.int64).reshape(rows, topk)
    weights = torch.full((rows, topk), 0.25, dtype=torch.bfloat16)
    return ids, weights


def test_prepare_routing_transfer_packs_int32_ids_and_float32_weights():
    """The ids channel takes int32 ids and float32 weights, not the gate dtypes."""

    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    weights = torch.full((2, 2), 0.25, dtype=torch.bfloat16)

    wire_ids, wire_weights = camp2p_module.prepare_routing_transfer(
        ids,
        weights,
        expected_tokens=2,
        top_k=2,
    )

    assert wire_ids.dtype == torch.int32
    assert wire_ids.tolist() == [[0, 1], [2, 3]]
    assert wire_weights.dtype == torch.float32
    assert wire_weights.tolist() == [[0.25, 0.25], [0.25, 0.25]]


def test_prepare_routing_transfer_rejects_expert_count_mismatch():
    """The wire declares ``k``, so a narrower routing tensor must not be packed."""

    ids = torch.zeros(3, 1, dtype=torch.int32)
    weights = torch.zeros(3, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="k=2"):
        camp2p_module.prepare_routing_transfer(
            ids,
            weights,
            expected_tokens=3,
            top_k=2,
        )


def test_prepare_routing_transfer_rejects_token_count_mismatch():
    ids = torch.zeros(2, 2, dtype=torch.int32)
    weights = torch.zeros(2, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="declares 3"):
        camp2p_module.prepare_routing_transfer(
            ids,
            weights,
            expected_tokens=3,
            top_k=2,
        )


def test_received_routing_trims_operator_padded_capacity():
    """Both channel tensors are cut together, so ids and weights stay paired."""

    ids = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    weights = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    topk_ids, topk_weights = camp2p_module.received_routing(
        ids,
        weights,
        expected_tokens=2,
    )

    assert topk_ids.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert topk_weights.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_received_routing_rejects_alignment_shortfall():
    ids = torch.zeros(2, 3, dtype=torch.int32)
    weights = torch.zeros(2, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="routing rows"):
        camp2p_module.received_routing(ids, weights, expected_tokens=3)


def test_send_attn_output_puts_routing_on_the_operator_ids_channel(monkeypatch):
    """Attention-side routing must select the operator's ids mode and fill it."""

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda *args: calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(camp2p_module, "get_forward_context", SimpleNamespace)
    connector = _connector(role="attention", rank=0)
    hidden_states = torch.zeros(3, connector.hidden_size)
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=3,
        ),
    )
    ids, weights = _routing(connector, 3)

    connector.send_attn_output(hidden_states, context, routing=(ids, weights))

    sent = calls[0]
    # Trailing operator arguments: aiv_num, compute_gate, ids, scales, mode.
    assert sent[-4] == 1
    assert sent[-3].dtype == torch.int32
    assert sent[-3].tolist() == ids.tolist()
    assert sent[-2].dtype == torch.float32
    assert sent[-2].tolist() == weights.to(torch.float32).tolist()
    assert sent[-1] == camp2p_module.A2E_QUANT_MODE_BF16


def test_send_attn_output_rejects_ids_and_routing_together(monkeypatch):
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(camp2p_module, "get_forward_context", SimpleNamespace)
    connector = _connector(role="attention", rank=0)
    hidden_states = torch.zeros(3, connector.hidden_size)
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=3,
        ),
    )
    ids, weights = _routing(connector, 3)

    with pytest.raises(ValueError, match="both token ids"):
        connector.send_attn_output(
            hidden_states,
            context,
            input_ids=torch.tensor([7, 11, 13], dtype=torch.int64),
            routing=(ids, weights),
        )


def test_recv_attn_output_reads_the_routing_slot(monkeypatch):
    """The routing mode exposes the ids slot as paired top-k ids and weights."""

    calls: list[tuple[Any, ...]] = []

    def fake_a2e(*args: Any) -> tuple[Any, ...]:
        calls.append(args)
        tokens, topk = int(args[3]), int(args[5])
        ids = torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk)
        weights = torch.full((tokens, topk), 0.5, dtype=torch.float32)
        return ("hidden", ids, weights, "atten-batch", "active-mask")

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    payload = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_routing=True,
    )

    # 10 tokens reach this FFN rank, in the same column order the gate produced.
    assert calls[0][-1] == 1
    assert payload.input_ids is None
    assert payload.topk_ids.shape == (10, connector.num_experts_per_tok)
    assert payload.topk_ids[0].tolist() == list(range(connector.num_experts_per_tok))
    assert payload.topk_weights.shape == (10, connector.num_experts_per_tok)
    assert payload.topk_weights[0].tolist() == [0.5] * connector.num_experts_per_tok


def test_recv_attn_output_rejects_two_wire_slot_modes(monkeypatch):
    connector = _connector(role="ffn", rank=1)

    with pytest.raises(ValueError, match="exactly one"):
        connector.recv_attn_output(
            ubatch_idx=0,
            layer_idx=0,
            recv_input_ids=True,
            recv_routing=True,
        )


# ### MXFP8 transport (quant_mode=4) ###########################################


def _mxfp8_config(*, attention=4, ffn=4, gate=True):
    from afd_plugin.config import AFDConfig

    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role="attention",
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
        compute_gate_on_attention=gate,
    )


def _mxfp8_connector(*, attention=4, ffn=4, gate=True):
    return CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(extra_config={"quant_mode": 4}),
        _mxfp8_config(attention=attention, ffn=ffn, gate=gate),
        0,
    )


def test_mxfp8_construction_accepts_a_supported_topology(monkeypatch):
    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: True)

    connector = _mxfp8_connector(attention=4, ffn=4)

    assert connector.quant_mode == camp2p_module.A2E_QUANT_MODE_MXFP8
    assert connector.attn_size == connector.ffn_size == 4


@pytest.mark.parametrize(
    "attention,ffn,gate,match",
    [
        pytest.param(4, 2, True, "equal Attention and FFN rank counts", id="4a2f"),
        pytest.param(4, 4, False, "compute_gate_on_attention=true", id="ffn-gate"),
        pytest.param(40, 40, True, "1..32", id="too-many-ranks"),
    ],
)
def test_mxfp8_construction_refuses_a_deployment_that_cannot_carry_it(
    monkeypatch, attention, ffn, gate, match
):
    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: True)

    with pytest.raises(RuntimeError, match=match):
        _mxfp8_connector(attention=attention, ffn=ffn, gate=gate)


@pytest.mark.parametrize(
    "probe,match",
    [
        pytest.param(False, "this build does not", id="unsupported"),
        pytest.param(None, "cannot confirm", id="unreadable"),
    ],
)
def test_mxfp8_construction_reports_what_the_operator_probe_said(
    monkeypatch, probe, match
):
    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: probe)

    with pytest.raises(RuntimeError, match=match):
        _mxfp8_connector()


def test_send_attn_output_forwards_the_configured_quant_mode(monkeypatch):
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda *args: calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(camp2p_module, "get_forward_context", SimpleNamespace)
    connector = _connector(role="attention", rank=0)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8

    connector.send_attn_output(
        torch.zeros(3, connector.hidden_size),
        AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_len=3,
            ),
        ),
    )

    assert calls[0][-1] == camp2p_module.A2E_QUANT_MODE_MXFP8


def test_the_mxfp8_send_calls_the_operator_with_the_scale(monkeypatch):
    """The opaque op quantizes, and the fourteen-argument call carries the pair."""
    sent: list[tuple[Any, ...]] = []
    quantized = torch.zeros(3, 16, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[130], [131], [132]], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    quantize_calls = []

    def fake_quantize(hidden_states):
        quantize_calls.append(hidden_states)
        return quantized, scale

    methods: dict[str, Any] = {}
    monkeypatch.setattr(camp2p_module, "_CAMP2P_CUSTOM_OPS_REGISTERED", False)
    monkeypatch.setattr(
        camp2p_module,
        "direct_register_custom_op",
        lambda **kwargs: methods.update({kwargs["op_name"]: kwargs["op_func"]}),
    )
    monkeypatch.setattr(camp2p_module, "mx_quantize_a2e_activation", fake_quantize)

    def fake_a2e(*args: Any) -> tuple[Any, ...]:
        sent.append(args)
        return (None, None, None, "atten-batch", None)

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    forward_context = SimpleNamespace(ubatch_idx=0, cam_afdtransfer_state=None)
    monkeypatch.setattr(camp2p_module, "get_forward_context", lambda: forward_context)
    camp2p_module._register_camp2p_custom_ops()

    hidden = torch.zeros(3, 16)
    methods["afd_camp2p_send_attn_output"](
        hidden,
        "n0",
        "n1",
        "",
        3,
        16,
        2,
        4,
        4,
        0,
        8,
        0,
        None,
        None,
        camp2p_module.A2E_QUANT_MODE_MXFP8,
    )

    call = sent[0]
    assert len(call) == 14
    assert quantize_calls == [hidden], "the operator must quantize the wire tensor"
    assert call[0] is quantized
    assert call[12] is scale
    assert call[13] == camp2p_module.A2E_QUANT_MODE_MXFP8


def test_the_mxfp8_receive_reads_the_sixth_return_value(monkeypatch):
    received_activation = torch.zeros(10, 16, dtype=torch.float8_e4m3fn)
    received_scale = torch.tensor([[127]] * 10, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    calls: list[tuple[Any, ...]] = []

    def fake_a2e(*args: Any) -> tuple[Any, ...]:
        calls.append(args)
        tokens, topk = int(args[3]), int(args[5])
        ids = torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk)
        weights = torch.full((tokens, topk), 0.5, dtype=torch.float32)
        return (
            received_activation,
            ids,
            weights,
            "atten-batch",
            "active-mask",
            received_scale,
        )

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    payload = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_routing=True,
    )

    call = calls[0]
    assert len(call) == 14
    assert call[0].dtype == torch.float8_e4m3fn
    assert tuple(call[0].shape) == (0, connector.hidden_size)
    assert call[12] is None
    assert call[13] == camp2p_module.A2E_QUANT_MODE_MXFP8
    assert payload.hidden_states is received_activation
    assert payload.activation_scale is received_scale


def test_the_mxfp8_receive_names_a_build_that_returns_five_values(monkeypatch):
    """A pre-MXFP8 operator returns five values; report that, not IndexError."""
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: (
            torch.zeros(10, 16, dtype=torch.float8_e4m3fn),
            torch.zeros(10, 2, dtype=torch.int32),
            torch.zeros(10, 2, dtype=torch.float32),
            "atten-batch",
            "active-mask",
        ),
        raising=False,
    )
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    with pytest.raises(RuntimeError, match="returned 5"):
        connector.recv_attn_output(ubatch_idx=0, layer_idx=0, recv_routing=True)


def test_the_mxfp8_receive_refuses_the_token_ids_slot():
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8

    with pytest.raises(ValueError, match="requires recv_routing"):
        connector.recv_attn_output(
            ubatch_idx=0,
            layer_idx=0,
            recv_input_ids=True,
        )
