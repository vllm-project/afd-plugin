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
from types import ModuleType, SimpleNamespace
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
        AFDControlPayload,
        AFDDPMetadata,
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


@pytest.fixture
def on_a5(monkeypatch):
    """Stand in for the device check so the remaining gate rules stay covered."""

    monkeypatch.setattr(camp2p_module, "_require_a5_device", lambda: None)


def test_mxfp8_requires_an_a5_device(monkeypatch):
    import enum

    class _DeviceType(enum.Enum):
        A3 = 1
        A5 = 3

    utils = ModuleType("vllm_ascend.utils")
    vars(utils)["AscendDeviceType"] = _DeviceType
    vars(utils)["get_ascend_device_type"] = lambda: _DeviceType.A3
    monkeypatch.setitem(sys.modules, "vllm_ascend", ModuleType("vllm_ascend"))
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)

    with pytest.raises(RuntimeError, match="requires an Ascend A5 device"):
        camp2p_module._require_a5_device()

    vars(utils)["get_ascend_device_type"] = lambda: _DeviceType.A5
    camp2p_module._require_a5_device()


def _mxfp8_connector(*, attention=4, ffn=4, gate=True):
    return CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(extra_config={"quant_mode": 4}),
        _mxfp8_config(attention=attention, ffn=ffn, gate=gate),
        0,
    )


def test_mxfp8_construction_accepts_a_supported_topology(monkeypatch, on_a5):
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
    monkeypatch, on_a5, attention, ffn, gate, match
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
    monkeypatch, on_a5, probe, match
):
    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: probe)

    with pytest.raises(RuntimeError, match=match):
        _mxfp8_connector()


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


def test_the_control_plane_declares_the_local_quant_mode(monkeypatch):
    sent: list[AFDControlPayload] = []
    monkeypatch.setattr(
        camp2p_module,
        "send_control_payload",
        lambda payload, **kwargs: sent.append(payload),
    )
    connector = _connector(role="attention", rank=0)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8
    connector.p2p_pg = object()
    connector.dst_list = [1]
    connector.topology = SimpleNamespace(is_attn_top_min_size_rank=True)

    connector.control_plane.send_dp_metadata_list(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata(torch.tensor([2], dtype=torch.int32))},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )

    assert sent[0].quant_mode == camp2p_module.A2E_QUANT_MODE_MXFP8


@pytest.mark.parametrize(
    "peer_mode,local_mode,refused",
    [
        pytest.param(4, 0, True, id="attention-4-ffn-0"),
        pytest.param(0, 4, True, id="attention-0-ffn-4"),
        pytest.param(4, 4, False, id="both-4"),
    ],
)
def test_the_control_payload_mode_is_checked_against_the_local_one(
    monkeypatch, peer_mode, local_mode, refused
):
    """A mixed pair must fail at the handshake; a matching one passes through."""
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata(torch.tensor([2], dtype=torch.int32))},
        is_graph_capturing=False,
        is_warmup=False,
        quant_mode=peer_mode,
    )
    monkeypatch.setattr(camp2p_module, "recv_control_payload", lambda **kwargs: payload)
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = local_mode
    connector.p2p_pg = object()
    connector.p2p_rank = 0
    connector.min_size = 1
    connector.ffn_size = 2

    if refused:
        with pytest.raises(RuntimeError, match=f"quant_mode={peer_mode}"):
            connector.control_plane.recv_dp_metadata_list()
    else:
        assert connector.control_plane.recv_dp_metadata_list() is payload


class _FakeGathered:
    """Stand-in for the int32 mode tensor the handshake writes back."""

    def __init__(self, modes):
        self._modes = list(modes)

    def tolist(self):
        return list(self._modes)


def _init_pair(
    monkeypatch,
    *,
    role,
    mode,
    bound=0,
    gathered_modes,
    verdicts=None,
    selection="ALLGATHER",
    selector=None,
    topk=2,
):
    """Drive the real startup handshake for a 1A1F pair and record every event."""
    import types as _types

    events = []
    published = []

    def fake_group(**kwargs):
        name = kwargs["group_name"]
        events.append(name)
        backend = SimpleNamespace(get_hccl_comm_name=lambda rank: f"hccl-{name}")
        return SimpleNamespace(_get_backend=lambda device: backend, name=name)

    def fake_gather(output, local, group=None):
        width = int(local.numel())
        events.append("mode_gather" if width == 2 else "verdict_gather")
        published.append([int(value) for value in local.tolist()])
        source = gathered_modes if width == 2 else verdicts
        assert source is not None, "no preset for this collective"
        target = output.view(-1, width)
        target.copy_(torch.as_tensor(source, dtype=torch.int32).reshape_as(target))
        # Only the peers come from the preset: this rank's slot carries what it
        # actually published, so a wrong verdict cannot be masked by the preset.
        local_slot = 0 if role == "ffn" else 1
        target[local_slot].copy_(local)

    class _FakeTorch:
        int32 = torch.int32
        distributed = SimpleNamespace(all_gather_into_tensor=fake_gather)

        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def _cpu(kwargs):
            if kwargs.get("device") == "npu":
                kwargs["device"] = "cpu"
            return kwargs

        def tensor(self, *args, **kwargs):
            return torch.tensor(*args, **self._cpu(kwargs))

        def empty(self, *args, **kwargs):
            return torch.empty(*args, **self._cpu(kwargs))

        @staticmethod
        def device(name):
            return name

    if selector is not None or mode == 4:
        module = _types.ModuleType("vllm_ascend.ascend_forward_context")

        class _MoECommType:
            ALLGATHER = "ALLGATHER"
            MC2 = "MC2"
            ALLTOALL = "ALLTOALL"
            FUSED_MC2 = "FUSED_MC2"

        chosen = selector if selector is not None else lambda *args, **kwargs: selection
        vars(module)["MoECommType"] = _MoECommType
        vars(module)["select_moe_comm_method"] = chosen
        monkeypatch.setitem(
            sys.modules, "vllm_ascend", _types.ModuleType("vllm_ascend")
        )
        monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context", module)

    from afd_plugin.config import AFDConfig

    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: True)
    vllm_config = _vllm_config(extra_config={"quant_mode": mode})
    vllm_config.scheduler_config.max_num_batched_tokens = bound
    vllm_config.model_config.hf_config.num_experts_per_tok = topk
    try:
        connector = CAMP2pAFDConnector(
            0,
            0,
            vllm_config,
            AFDConfig(
                connector="CAMP2pAFDConnector",
                role=role,
                num_attention_ranks=1,
                num_ffn_ranks=1,
                compute_gate_on_attention=True,
            ),
            0,
        )
    except RuntimeError as exc:
        return events, published, exc, None
    connector._initialized = False
    monkeypatch.setitem(sys.modules, "torch_npu", _types.ModuleType("torch_npu"))
    monkeypatch.setattr(camp2p_module, "ensure_cam_p2p_ops_available", lambda: None)
    monkeypatch.setattr(camp2p_module, "_register_camp2p_custom_ops", lambda: None)
    monkeypatch.setattr(camp2p_module, "init_afd_process_group", fake_group)
    monkeypatch.setattr(
        camp2p_module, "create_hccl_process_group_options", lambda *args: None
    )
    monkeypatch.setattr(camp2p_module, "a2e_accepts_mxfp8_arguments", lambda: True)
    monkeypatch.setattr(camp2p_module, "torch", _FakeTorch())
    error = None
    try:
        connector.init_afd_connector()
    except RuntimeError as exc:
        error = exc
    return events, published, error, connector


@pytest.mark.parametrize(
    "role,mode,gathered_modes,verdicts,selection,events,error_match",
    [
        pytest.param(
            "ffn",
            0,
            [[0, 0], [0, 0]],
            None,
            "ALLGATHER",
            ["afd", "mode_gather", "afd_moe", "p2p"],
            None,
            id="ffn-bf16",
        ),
        pytest.param(
            "attention",
            0,
            [[0, 0], [0, 0]],
            None,
            "ALLGATHER",
            ["afd", "mode_gather", "p2p"],
            None,
            id="attn-bf16",
        ),
        pytest.param(
            "ffn",
            4,
            [[4, 0], [4, 1024]],
            [1, 1],
            "MC2",
            ["afd", "mode_gather", "verdict_gather", "afd_moe", "p2p"],
            None,
            id="ffn-mode4",
        ),
        pytest.param(
            "attention",
            4,
            [[4, 0], [4, 1024]],
            [1, 1],
            "ALLGATHER",
            ["afd", "mode_gather", "verdict_gather", "p2p"],
            None,
            id="attn-mode4",
        ),
        pytest.param(
            "ffn",
            0,
            [[0, 0], [4, 1024]],
            None,
            "ALLGATHER",
            ["afd", "mode_gather"],
            "mismatch across ranks",
            id="ffn-mixed",
        ),
        pytest.param(
            "attention",
            4,
            [[0, 0], [4, 1024]],
            None,
            "ALLGATHER",
            ["afd", "mode_gather"],
            "mismatch across ranks",
            id="attn-mixed",
        ),
        pytest.param(
            "ffn",
            4,
            [[4, 0], [4, 1024]],
            [0, 1],
            "ALLTOALL",
            ["afd", "mode_gather", "verdict_gather"],
            "chose another path",
            id="ffn-alltoall",
        ),
        pytest.param(
            "attention",
            4,
            [[4, 0], [4, 1024]],
            [0, 1],
            "ALLTOALL",
            ["afd", "mode_gather", "verdict_gather"],
            "refused by the FFN role",
            id="attn-peer-refused",
        ),
    ],
)
def test_startup_handshake_stops_where_the_pair_is_refused(
    monkeypatch,
    on_a5,
    role,
    mode,
    gathered_modes,
    verdicts,
    selection,
    events,
    error_match,
):
    """Each case records the collectives it reached, so the stop point is proved."""
    seen, published, error, _ = _init_pair(
        monkeypatch,
        role=role,
        mode=mode,
        gathered_modes=gathered_modes,
        verdicts=verdicts,
        selection=selection,
    )

    assert seen == events
    if error_match is None:
        assert error is None
    else:
        assert error is not None and error_match in str(error)
    if mode == 4 and role == "ffn":
        assert published[0] == [4, 0], "the FFN must publish no batch bound"


def test_startup_handshake_asks_the_selector_about_attentions_bound(monkeypatch, on_a5):
    """FFN's own limit is irrelevant: the selector sees what Attention can send."""
    asked = []

    def selector(max_rows, vllm_config, **kwargs):
        asked.append(max_rows)
        return "MC2"

    seen, published, error, _ = _init_pair(
        monkeypatch,
        role="ffn",
        mode=4,
        bound=512,
        gathered_modes=[[4, 0], [4, 1024]],
        verdicts=[1, 1],
        selector=selector,
    )

    assert error is None
    assert asked == [1024]
    assert published[0] == [4, 0]


def test_startup_handshake_reports_a_selector_failure_to_both_roles(monkeypatch, on_a5):
    """A raising selector must still publish its verdict, or Attention waits."""

    class _NoCapacityError(RuntimeError):
        pass

    def explode(*args, **kwargs):
        raise _NoCapacityError("mc2 capacity is not set")

    seen, published, error, _ = _init_pair(
        monkeypatch,
        role="ffn",
        mode=4,
        gathered_modes=[[4, 0], [4, 1024]],
        verdicts=[0, 1],
        selector=explode,
    )

    assert seen == ["afd", "mode_gather", "verdict_gather"]
    assert published[-1] == [0], "the FFN published a passing verdict"
    assert isinstance(error, RuntimeError)
    assert "could not resolve" in str(error)
    assert isinstance(error.__cause__, _NoCapacityError), "the original cause is lost"


def test_send_attn_output_reaches_the_operator_as_a_quantized_pair(monkeypatch):
    """Through the public send: one call, fourteen arguments, the pair and the mode."""
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

    def fake_a2e(*args):
        sent.append(args)
        return (None, None, None, "atten-batch", None)

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    values = None
    # attn_size > ffn_size in this fake config, so the padding branch is reached;
    # one stage keeps it inert and out of the way of what this test asserts.
    forward_context = SimpleNamespace(
        ubatch_idx=0,
        cam_afdtransfer_state=None,
        additional_kwargs={
            "afd_metadata": SimpleNamespace(
                connector=SimpleNamespace(dp_metadata_list=[{}])
            )
        },
    )
    monkeypatch.setattr(camp2p_module, "get_forward_context", lambda: forward_context)
    camp2p_module._register_camp2p_custom_ops()

    def registered(*args):
        if values is not None:
            values.append(args)
        return methods["afd_camp2p_send_attn_output"](*args)

    monkeypatch.setattr(
        torch.ops.vllm, "afd_camp2p_send_attn_output", registered, raising=False
    )
    connector = _connector(role="attention", rank=0)
    connector.quant_mode = camp2p_module.A2E_QUANT_MODE_MXFP8

    hidden = torch.zeros(3, 16)
    connector.send_attn_output(
        hidden,
        AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_len=3,
            ),
        ),
    )

    call = sent[0]
    assert len(call) == 14
    assert quantize_calls == [hidden], "the operator must quantize the wire tensor"
    assert call[0] is quantized
    assert call[12] is scale
    assert call[13] == camp2p_module.A2E_QUANT_MODE_MXFP8


def test_the_routing_id_bound_is_judged_from_the_attention_role(monkeypatch, on_a5):
    """FFN's own limit is irrelevant; Attention's published bound decides."""
    seen, published, error, _ = _init_pair(
        monkeypatch,
        role="ffn",
        mode=4,
        bound=8192,
        gathered_modes=[[4, 0], [4, 4096]],
        verdicts=[1, 1],
        selector=lambda *args, **kwargs: "MC2",
        topk=6,
    )

    assert error is None, "the FFN rejected a valid pair on its own limit"
    assert seen == ["afd", "mode_gather", "verdict_gather", "afd_moe", "p2p"]
    assert published[0] == [4, 0]


def test_the_handshake_rejects_a_bound_the_ids_cannot_fit(monkeypatch, on_a5):
    """The sender is rejected at construction; the receiver on the handshake."""
    seen, published, error, _ = _init_pair(
        monkeypatch,
        role="ffn",
        mode=4,
        bound=8192,
        gathered_modes=[[4, 0], [4, 8150]],
        verdicts=[1, 1],
        selector=lambda *args, **kwargs: "MC2",
        topk=6,
    )
    assert seen == ["afd", "mode_gather"], seen
    assert error is not None and "publishes 8150" in str(error)

    # The Attention role cannot publish an over-limit bound: its own check runs
    # before any group exists, so the pair never reaches a transfer.
    seen, published, error, _ = _init_pair(
        monkeypatch,
        role="attention",
        mode=4,
        bound=8150,
        gathered_modes=[[4, 0], [4, 8150]],
        verdicts=[1, 1],
        selector=lambda *args, **kwargs: "MC2",
        topk=6,
    )
    assert seen == ["afd", "mode_gather"], seen
    assert error is not None and "publishes 8150" in str(error)
