# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the Attention-side routing payload.

The payload is pure tensor logic, so it is testable without an Ascend device. It is
loaded from its file because importing the ``afd_plugin.connectors`` package pulls
vLLM in through its ``__init__``.

Rows carry rank- and position-specific values instead of a constant fill: a gather
or split that drops, reorders or mismatches a row has to change the observed
contents, not just the shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_TOP_K = 2


def _module_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "afd_plugin"
        / "connectors"
        / "npu"
        / "external_routing.py"
    )


@pytest.fixture(scope="module")
def routing_module():
    spec = importlib.util.spec_from_file_location(
        "afd_external_routing_under_test",
        _module_path(),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inputs_type(routing_module):
    return routing_module.AFDExternalRoutingInputs


def _inputs(inputs_type, rows: int, rank: int = 0):
    """Rows encode rank and position, so an ordering bug changes the values."""

    base = rank * 100
    values = torch.arange(base, base + rows * _TOP_K)
    return inputs_type(
        topk_ids=values.to(torch.int32).reshape(rows, _TOP_K),
        topk_weights=(values.to(torch.float32) / 10).reshape(rows, _TOP_K),
    )


_RANK_STRIDE = 100


def _two_rank_gather(tensor):
    """Stand-in collective whose ranks carry *different* rows.

    A real DP group gathers each rank's own tokens, so rank 1's shard is offset from
    rank 0's. Repeating one shard would hide an implementation that reorders or drops
    a rank.
    """

    return torch.cat([tensor, tensor + _RANK_STRIDE], dim=0)


def test_padding_appends_inert_rows_and_keeps_valid_ones(inputs_type):
    inputs = _inputs(inputs_type, 2)

    padded = inputs.padded(2)

    assert padded.topk_ids[:2].tolist() == inputs.topk_ids.tolist()
    assert padded.topk_weights[:2].tolist() == inputs.topk_weights.tolist()
    assert padded.topk_ids[2:].tolist() == [[0, 0], [0, 0]]
    assert padded.topk_weights[2:].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert padded.token_count == 4


def test_gathering_keeps_rank_order_and_contents(inputs_type):
    """Rank 0's rows come first, then rank 1's, with their own values intact."""

    inputs = _inputs(inputs_type, 2, rank=0)

    gathered = inputs.gathered(_two_rank_gather)

    assert gathered.topk_ids.tolist() == (
        inputs.topk_ids.tolist() + (inputs.topk_ids + _RANK_STRIDE).tolist()
    )
    assert gathered.topk_weights.tolist() == (
        inputs.topk_weights.tolist() + (inputs.topk_weights + _RANK_STRIDE).tolist()
    )


def test_split_rows_takes_one_rank_slice(inputs_type):
    inputs = _inputs(inputs_type, 4)

    taken = inputs.split_rows(lambda tensor: torch.tensor_split(tensor, 2, dim=0)[1])

    assert taken.topk_ids.tolist() == inputs.topk_ids.tolist()[2:]
    assert taken.topk_weights.tolist() == inputs.topk_weights.tolist()[2:]


def test_non_matrix_routing_is_rejected(inputs_type):
    with pytest.raises(ValueError, match="2-D"):
        inputs_type(
            topk_ids=torch.zeros(3, dtype=torch.int32),
            topk_weights=torch.zeros(3, dtype=torch.float32),
        )


def test_mismatched_topk_shapes_are_rejected(inputs_type):
    with pytest.raises(ValueError, match="same shape"):
        inputs_type(
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            topk_weights=torch.zeros(2, 3, dtype=torch.float32),
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expert_indices_accept_both_index_dtypes(inputs_type, dtype):
    """Both index widths are in the operator contract; neither may be rejected."""

    inputs = inputs_type(
        topk_ids=torch.zeros(2, 2, dtype=dtype),
        topk_weights=torch.zeros(2, 2, dtype=torch.float32),
    )

    assert inputs.topk_ids.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_expert_indices_reject_non_index_dtypes(inputs_type, dtype):
    with pytest.raises(ValueError, match="expert indices"):
        inputs_type(
            topk_ids=torch.zeros(2, 2, dtype=dtype),
            topk_weights=torch.zeros(2, 2, dtype=torch.float32),
        )


def test_non_float_weights_are_rejected(inputs_type):
    with pytest.raises(ValueError, match="gate weights"):
        inputs_type(
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            topk_weights=torch.zeros(2, 2, dtype=torch.int32),
        )
