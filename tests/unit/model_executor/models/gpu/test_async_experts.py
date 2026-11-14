# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host contracts for already-dispatched expert rows; no device-kernel proof."""

from types import SimpleNamespace
from typing import cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402

from afd_plugin.model_executor.models.gpu.async_experts import (  # noqa: E402
    AsyncGpuRoutedExperts,
    compute_local_experts,
    use_local_expert_factory,
)
from tests.unit.fakes.deepseek_v2 import LocalExpertRows  # noqa: E402


@pytest.mark.parametrize("routed_rows,shared_rows", [(3, 2), (0, 2), (3, 0), (0, 0)])
def test_local_expert_payload_keeps_row_order_and_shared_slice(
    routed_rows, shared_rows
):
    hidden = torch.ones(routed_rows, 2)
    shared = torch.full((shared_rows, 2), 4.0)
    output = compute_local_experts(
        cast(AsyncGpuRoutedExperts, LocalExpertRows()),
        hidden_states=hidden,
        group_list=torch.tensor([min(routed_rows, 2), max(routed_rows - 2, 0)]),
        shared_experts=nn.Identity(),
        expand_x_shared=shared,
        shared_scale=0.5,
    )
    expected = torch.tensor([[3.0, 3.0], [3.0, 3.0], [7.0, 7.0]])[:routed_rows]
    torch.testing.assert_close(output.routed_output, expected)
    if shared_rows:
        torch.testing.assert_close(
            output.shared_output, torch.full((shared_rows, 2), 2.0)
        )
    else:
        assert output.shared_output is None


def test_local_expert_factory_restores_native_binding_after_failure():
    original = lambda **kwargs: kwargs  # noqa: E731
    native = SimpleNamespace(FusedMoEFactory=original)
    with (
        pytest.raises(ValueError, match="construction failed"),
        use_local_expert_factory(native),
    ):
        raise ValueError("construction failed")
    assert native.FusedMoEFactory is original
