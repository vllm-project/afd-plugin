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

torch = pytest.importorskip("torch")

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
    from afd_plugin.config import AFDConfig
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


class _FakeDPMetadata:
    def __init__(self, values: list[int]) -> None:
        # The connector counts tokens with .flatten().tolist(), so this has to be
        # a tensor like the real DP metadata rather than a plain list.
        self.num_tokens_across_dp_cpu = torch.tensor(values, dtype=torch.int32)


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=0,
            ),
        ),
    )


def _afd_config(*, role: str) -> AFDConfig:
    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )


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
    # Trailing operator arguments: aiv_num, compute_gate, ids, scales.
    assert with_ids[-3] == 1
    assert with_ids[-2].dtype == torch.int32
    assert with_ids[-2][:, 0].tolist() == [7, 11, 13]
    assert with_ids[-1].dtype == torch.float32
    assert torch.count_nonzero(with_ids[-1]) == 0
    assert without_ids[-3] == 0
    assert without_ids[-2] is None
    assert without_ids[-1] is None


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
    # FFN rank 1 owns attention ranks 2 and 3, so it computes on 5 + 7 tokens.
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
        100,
        110,
    ]
    assert calls[1][-1] == 0
    assert without_ids.input_ids is None
