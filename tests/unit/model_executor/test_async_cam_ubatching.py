# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from afd_plugin.model_executor.npu.async_cam_ubatching import (
    plan_async_moe_stages,
)


@pytest.mark.parametrize(
    ("scheduled_tokens", "use_sp", "tp_size", "expected"),
    [
        pytest.param(
            [1099],
            True,
            2,
            (
                (slice(0, 1), slice(0, 550), 550),
                (slice(0, 1), slice(550, 1099), 550),
            ),
            id="single-request-with-padding",
        ),
        pytest.param(
            [1, 1, 100, 2],
            False,
            2,
            (
                (slice(0, 3), slice(0, 52), 52),
                (slice(2, 4), slice(52, 104), 52),
            ),
            id="split-inside-request",
        ),
        pytest.param(
            [5],
            True,
            2,
            (
                (slice(0, 1), slice(0, 3), 4),
                (slice(0, 1), slice(3, 5), 2),
            ),
            id="uneven-minimal-stages",
        ),
        pytest.param([1], True, 2, None, id="single-token"),
    ],
)
def test_token_stage_plan(scheduled_tokens, use_sp, tp_size, expected):
    stages = plan_async_moe_stages(
        scheduled_tokens,
        split="token",
        use_sequence_parallel=use_sp,
        tensor_parallel_size=tp_size,
    )

    if expected is None:
        assert stages is None
        return

    assert stages is not None
    assert (
        tuple(
            (stage.request_slice, stage.token_slice, stage.input_tokens)
            for stage in stages
        )
        == expected
    )
    assert sum(stage.actual_tokens for stage in stages) == sum(scheduled_tokens)
    assert abs(stages[0].actual_tokens - stages[1].actual_tokens) <= 1


@pytest.mark.parametrize(
    ("scheduled_tokens", "use_sp", "tp_size", "expected"),
    [
        pytest.param(
            [5, 6, 7],
            True,
            2,
            (
                (slice(0, 2), slice(0, 11), 12),
                (slice(2, 3), slice(11, 18), 8),
            ),
            id="request-boundary-with-padding",
        ),
        pytest.param([18], True, 2, None, id="single-request"),
    ],
)
def test_request_stage_plan(scheduled_tokens, use_sp, tp_size, expected):
    stages = plan_async_moe_stages(
        scheduled_tokens,
        split="request",
        use_sequence_parallel=use_sp,
        tensor_parallel_size=tp_size,
    )

    if expected is None:
        assert stages is None
    else:
        assert stages is not None
        assert (
            tuple(
                (stage.request_slice, stage.token_slice, stage.input_tokens)
                for stage in stages
            )
            == expected
        )


@pytest.mark.parametrize("scheduled_tokens", ([], [4, 0]))
def test_stage_plan_rejects_nonpositive_token_counts(scheduled_tokens):
    with pytest.raises(ValueError, match="must all be positive"):
        plan_async_moe_stages(
            scheduled_tokens,
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )
