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

    saved = {name: sys.modules.get(name) for name in missing}

    def make(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module

    make("vllm")
    make(
        "vllm.forward_context",
        DPMetadata=type("DPMetadata", (), {}),
        get_forward_context=lambda: None,
    )
    make("vllm.logger", init_logger=lambda *args, **kwargs: logging.getLogger("test"))
    make("vllm.utils")
    make(
        "vllm.utils.torch_utils",
        direct_register_custom_op=lambda **kwargs: None,
        is_torch_equal_or_newer=lambda *args, **kwargs: True,
    )
    make("vllm.distributed")
    make(
        "vllm.distributed.parallel_state",
        get_pcp_group=None,
        get_tensor_model_parallel_rank=None,
    )
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


with _vllm_stub():
    from afd_plugin.connectors.metadata import (
        AFDTransferContext,
        AFDTransferMetadata,
    )
    from afd_plugin.connectors.npu import camp2p as camp2p_module
    from afd_plugin.connectors.npu.camp2p import (
        CAMP2pAFDConnector,
        prepare_token_id_transfer,
        received_token_ids,
    )


class _CpuTorch:
    """``torch`` with the NPU device mapped to CPU for host-side tests.

    The connector hands the operator empty device tensors, which only exist on an
    Ascend host. Only ``tensor()`` is adapted; every other attribute, including
    ``ops``, stays the real one so the recorded operator calls are the calls the
    connector makes.
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
    # Trailing operator arguments: compute_gate, ids, scales, quant_mode.
    assert with_ids[-4] == 1
    assert with_ids[-3].dtype == torch.int32
    assert with_ids[-3][:, 0].tolist() == [7, 11, 13]
    assert with_ids[-2].dtype == torch.float32
    assert torch.count_nonzero(with_ids[-2]) == 0
    assert with_ids[-1] == 0, "the BF16 transport stays on quant_mode 0"
    assert without_ids[-4] == 0
    assert without_ids[-3] is None
    assert without_ids[-2] is None
    assert without_ids[-1] == 0


def test_recv_attn_output_mode_and_ids_follow_the_receiver_declaration(monkeypatch):
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
    # Trailing operator arguments: compute_gate, ids, scales, quant_mode.
    assert sent[-4] == 1
    assert sent[-3].dtype == torch.int32
    assert sent[-3].tolist() == ids.tolist()
    assert sent[-2].dtype == torch.float32
    assert sent[-2].tolist() == weights.to(torch.float32).tolist()
    assert sent[-1] == 0, "the BF16 transport stays on quant_mode 0"


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
    assert payload.topk_ids[0].tolist() == list(
        range(connector.num_experts_per_tok)
    )
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

def test_recv_attn_output_returns_the_mxfp8_pair_together(monkeypatch):
    """Mode 4 receives an E4M3 activation and its E8M0 scale as one pair.

    The pair is what makes the transport a pass-through: the FFN side hands both to the MoE
    instead of restoring the model dtype and letting the MoE quantize again.
    """

    calls: list[tuple[Any, ...]] = []
    activation = torch.zeros(10, 8, dtype=torch.float8_e4m3fn)
    scale = torch.full((10, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = 4
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}
    ids, weights = _routing(connector, 10)

    def fake_a2e(*args: Any) -> tuple[Any, ...]:
        calls.append(args)
        return (activation, ids, weights, 0, None, scale)

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)

    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=0, recv_routing=True)

    assert calls[0][-1] == 4, "the receive declares the quantized transport"
    assert calls[0][-2] is None, "the receiving side supplies no x_scale"
    assert payload.hidden_states is activation
    assert payload.activation_scale is scale
    assert payload.hidden_states.shape[0] == payload.activation_scale.shape[0]


def test_mxfp8_receive_requires_routing(monkeypatch):
    """The token-ids path runs the router here, so it cannot take an FP8 activation."""

    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: (torch.zeros(10, 8), None, None, 0, None, torch.zeros(10, 1)),
        raising=False,
    )
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = 4
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    with pytest.raises(ValueError, match="requires recv_routing"):
        connector.recv_attn_output(ubatch_idx=0, layer_idx=0, recv_input_ids=True)

def test_send_op_annotations_cover_every_parameter(monkeypatch):
    """The manual table replaces the real annotations, so it must list every parameter.

    PyTorch derives the schema from those annotations; a parameter missing from the table
    fails registration with "must have a type annotation", and the registration call itself
    is replaced by a stand-in in these host tests, so nothing else would notice.
    """

    import inspect

    registered: dict[str, object] = {}
    # Registration is process-global, so the stand-in must not leave the "already done" flag
    # behind: the real initialisation would then skip it, and a second run of this test would
    # find an empty capture.
    monkeypatch.setattr(camp2p_module, "_CAMP2P_CUSTOM_OPS_REGISTERED", False)
    # Registration is process-global, so the stand-in must not leave the "already done" flag
    # behind: the real initialisation would then skip it, and a second run of this test would
    # find an empty capture.

    def fake_register(**kwargs: object) -> None:
        registered[str(kwargs["op_name"])] = kwargs["op_func"]

    monkeypatch.setattr(camp2p_module, "direct_register_custom_op", fake_register)
    monkeypatch.setattr(torch.ops.afd_ascend, "_", SimpleNamespace(), raising=False)
    camp2p_module._register_camp2p_custom_ops()

    send = registered["afd_camp2p_send_attn_output"]
    parameters = set(inspect.signature(send).parameters)
    assert parameters == set(send.__annotations__) - {"return"}
    assert send.__annotations__["quant_mode"] is int


def test_wire_padding_keeps_the_quantized_activation_and_pads_with_neutral_bytes():
    """Padding must not put the model-dtype tensor back next to an E8M0 scale."""

    activation = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.full((2, 1), 100, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    padded_activation, padded_scale = camp2p_module._pad_wire_activation(
        activation, scale, 1, 8
    )

    assert padded_activation.dtype == torch.float8_e4m3fn
    assert padded_activation.shape == (3, 8)
    assert padded_scale.view(torch.uint8).tolist() == [[100], [100], [127]]


def test_mxfp8_receive_refuses_before_communicating(monkeypatch):
    """An unsupported mode must not start a receive it cannot finish."""

    def unreachable(*args: object) -> None:
        raise AssertionError("the operator must not be called for an unsupported mode")

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", unreachable, raising=False)
    monkeypatch.setattr(camp2p_module, "torch", _CpuTorch())
    connector = _connector(role="ffn", rank=1)
    connector.quant_mode = 4
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    with pytest.raises(ValueError, match="requires recv_routing"):
        connector.recv_attn_output(ubatch_idx=0, layer_idx=0, recv_input_ids=True)
