# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the Attention-side routing payload.

The payload is pure tensor logic, so it is testable without an Ascend device. The
package resolves its connector lazily, so importing this module does not pull vLLM in.

Rows carry rank- and position-specific values instead of a constant fill: a gather
or split that drops, reorders or mismatches a row has to change the observed
contents, not just the shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

_TOP_K = 2


@pytest.fixture(scope="module")
def routing_module():
    from afd_plugin.connectors.npu.camp2p import external_routing

    return external_routing


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


def _scale(rows: int, blocks: int, rank: int = 0):
    """Distinct byte per rank and row, so a row mixed up by a transform shows."""

    values = torch.arange(rank * 100, rank * 100 + rows * blocks, dtype=torch.uint8)
    return values.reshape(rows, blocks).view(torch.float8_e8m0fnu)


def _scale_bytes(scale):
    return scale.view(torch.uint8).tolist()


def test_scale_follows_padding_with_the_neutral_byte(routing_module, inputs_type):
    """Padding rows carry 2**0, not 127 as a number: the byte view is the contract."""

    routing = inputs_type(
        topk_ids=torch.zeros(3, _TOP_K, dtype=torch.int32),
        topk_weights=torch.ones(3, _TOP_K),
        scale=_scale(3, 2),
    )
    padded = routing.padded(2).scale
    rows = _scale_bytes(padded)
    assert len(rows) == 5
    assert rows[:3] == _scale_bytes(_scale(3, 2))
    assert rows[3:] == [[127] * 2] * 2


def test_scale_gathers_in_rank_order(inputs_type):
    """Each rank contributes its own bytes, so an ordering bug is visible."""

    left = inputs_type(
        topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
        topk_weights=torch.ones(2, _TOP_K),
        scale=_scale(2, 2, rank=1),
    )
    right = inputs_type(
        topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
        topk_weights=torch.ones(2, _TOP_K),
        scale=_scale(2, 2, rank=2),
    )

    # The collective is applied once per tensor, so the fake has to answer for the
    # ids, the weights and the scale separately; only the scale assertion matters.
    def gather(tensor):
        # Every tensor the payload gathers has to grow by the same row count, or the
        # rebuilt payload would pair rows with another rank's tokens. E8M0 is
        # rejected on purpose: the payload has to move the exponent bytes.
        if tensor.dtype == torch.float8_e8m0fnu:
            raise AssertionError("the collective received the E8M0 dtype")
        if tensor.shape[1] == right.scale.shape[1] and tensor.dtype == torch.uint8:
            return torch.cat([tensor, right.scale.view(torch.uint8)], dim=0)
        return torch.cat([tensor, tensor], dim=0)

    gathered = left.gathered(gather)
    assert _scale_bytes(gathered.scale) == _scale_bytes(
        _scale(2, 2, rank=1)
    ) + _scale_bytes(_scale(2, 2, rank=2))


def test_scale_takes_the_same_split_as_the_activation(inputs_type):
    routing = inputs_type(
        topk_ids=torch.zeros(4, _TOP_K, dtype=torch.int32),
        topk_weights=torch.ones(4, _TOP_K),
        scale=_scale(4, 2),
    )

    def take_second_rank(tensor):
        if tensor.dtype == torch.float8_e8m0fnu:
            raise AssertionError("the split received the E8M0 dtype")
        return torch.tensor_split(tensor, 2, dim=0)[1]

    split = routing.split_rows(take_second_rank)
    assert _scale_bytes(split.scale) == _scale_bytes(_scale(4, 2))[2:]


def test_a_payload_without_a_scale_keeps_none(inputs_type):
    routing = inputs_type(
        topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
        topk_weights=torch.ones(2, _TOP_K),
    )
    assert routing.padded(1).scale is None
    assert routing.gathered(lambda tensor: tensor).scale is None
    assert routing.split_rows(lambda tensor: tensor).scale is None


def test_scale_rows_must_match_the_token_rows(inputs_type):
    with pytest.raises(ValueError, match="one row per token"):
        inputs_type(
            topk_ids=torch.zeros(3, _TOP_K, dtype=torch.int32),
            topk_weights=torch.ones(3, _TOP_K),
            scale=_scale(2, 2),
        )


def test_scale_dtype_must_be_e8m0(inputs_type):
    with pytest.raises(ValueError, match="E8M0"):
        inputs_type(
            topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
            topk_weights=torch.ones(2, _TOP_K),
            scale=torch.ones(2, 2, dtype=torch.float32),
        )


def test_scale_must_keep_at_least_one_block(inputs_type):
    with pytest.raises(ValueError, match="at least one MX block"):
        inputs_type(
            topk_ids=torch.zeros(2, _TOP_K, dtype=torch.int32),
            topk_weights=torch.ones(2, _TOP_K),
            scale=torch.zeros(2, 0, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        )
