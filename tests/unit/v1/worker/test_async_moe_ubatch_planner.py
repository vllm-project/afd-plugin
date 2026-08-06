# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from afd_plugin.async_moe import plan_async_moe_stages


@pytest.mark.parametrize(
    (
        "scheduled_tokens",
        "use_sequence_parallel",
        "tensor_parallel_size",
        "expected_actual_tokens",
        "expected_physical_tokens",
    ),
    [
        ([8, 8], False, 2, (8, 8), (8, 8)),
        ([1099], True, 2, (550, 549), (550, 550)),
        ([40], True, 2, (20, 20), (20, 20)),
        ([70], True, 2, (35, 35), (36, 36)),
        ([5], True, 2, (3, 2), (4, 2)),
        ([1, 1, 100, 2], False, 2, (52, 52), (52, 52)),
    ],
)
def test_token_stage_plan_balances_real_tokens_and_preserves_coverage(
    scheduled_tokens,
    use_sequence_parallel,
    tensor_parallel_size,
    expected_actual_tokens,
    expected_physical_tokens,
):
    stage_plan = plan_async_moe_stages(
        scheduled_tokens,
        split="token",
        use_sequence_parallel=use_sequence_parallel,
        tensor_parallel_size=tensor_parallel_size,
    )

    assert stage_plan is not None
    stages = stage_plan
    assert tuple(stage.actual_tokens for stage in stages) == expected_actual_tokens
    assert tuple(stage.input_tokens for stage in stages) == expected_physical_tokens
    assert stages[0].token_slice.start == 0
    assert stages[0].token_slice.stop == stages[1].token_slice.start
    assert stages[1].token_slice.stop == sum(scheduled_tokens)
    assert sum(stage.actual_tokens for stage in stages) == sum(scheduled_tokens)
    assert abs(stages[0].actual_tokens - stages[1].actual_tokens) <= 1
    if use_sequence_parallel:
        assert all(stage.input_tokens % tensor_parallel_size == 0 for stage in stages)


def test_token_stage_plan_rebuilds_request_ranges_when_split_inside_request():
    stage_plan = plan_async_moe_stages(
        [1, 1, 100, 2],
        split="token",
        use_sequence_parallel=False,
        tensor_parallel_size=2,
    )

    assert stage_plan is not None
    stages = stage_plan
    assert stages[0].request_slice == slice(0, 3)
    assert stages[0].token_slice == slice(0, 52)
    assert stages[1].request_slice == slice(2, 4)
    assert stages[1].token_slice == slice(52, 104)


@pytest.mark.parametrize(
    (
        "scheduled_tokens",
        "use_sequence_parallel",
        "tensor_parallel_size",
        "expected_request_slices",
        "expected_token_slices",
        "expected_actual_tokens",
        "expected_physical_tokens",
    ),
    [
        (
            [824, 846, 16],
            False,
            1,
            [slice(0, 1), slice(1, 3)],
            [slice(0, 824), slice(824, 1686)],
            (824, 862),
            (824, 862),
        ),
        (
            [5, 6, 7],
            True,
            2,
            [slice(0, 2), slice(2, 3)],
            [slice(0, 11), slice(11, 18)],
            (11, 7),
            (12, 8),
        ),
        ([18], True, 2, None, None, None, None),
    ],
    ids=("request-boundary", "flashcomm-aligned", "single-request"),
)
def test_request_stage_plan(
    scheduled_tokens,
    use_sequence_parallel,
    tensor_parallel_size,
    expected_request_slices,
    expected_token_slices,
    expected_actual_tokens,
    expected_physical_tokens,
):
    stage_plan = plan_async_moe_stages(
        scheduled_tokens,
        split="request",
        use_sequence_parallel=use_sequence_parallel,
        tensor_parallel_size=tensor_parallel_size,
    )

    if expected_request_slices is None:
        assert stage_plan is None
        return
    assert stage_plan is not None
    stages = stage_plan
    assert [stage.request_slice for stage in stages] == expected_request_slices
    assert [stage.token_slice for stage in stages] == expected_token_slices
    assert tuple(stage.actual_tokens for stage in stages) == expected_actual_tokens
    assert tuple(stage.input_tokens for stage in stages) == expected_physical_tokens


@pytest.mark.parametrize("scheduled_tokens", ([], [4, 0]))
def test_stage_plan_rejects_nonpositive_token_counts(scheduled_tokens):
    with pytest.raises(ValueError, match="must all be positive"):
        plan_async_moe_stages(
            scheduled_tokens,
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )


def test_token_stage_plan_handles_minimal_batch():
    assert (
        plan_async_moe_stages(
            [2],
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )
        is not None
    )
