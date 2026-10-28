# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU token-count contracts for the shared AFD DP metadata payload."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.metadata import AFDDPMetadata  # noqa: E402


@pytest.mark.parametrize(
    "sp_size, context_options, expected",
    [
        pytest.param(2, {}, [2, 2, 3, 3], id="legacy-one-argument"),
        pytest.param(
            2, {"pcp_size": 1, "use_ep": True}, [2, 2, 3, 3], id="gpu-pcp1-ep"
        ),
        pytest.param(
            2, {"pcp_size": 2, "use_ep": False}, [3, 3, 5, 5], id="pcp-without-ep"
        ),
        pytest.param(
            2,
            {"pcp_size": 2, "use_ep": True},
            [2, 2, 2, 2, 3, 3, 3, 3],
            id="pcp-with-ep",
        ),
    ],
)
def test_sp_local_sizes_preserves_native_token_layout(
    sp_size, context_options, expected
):
    metadata = AFDDPMetadata(torch.tensor([3, 5]))

    with metadata.sp_local_sizes(sp_size, **context_options) as local_sizes:
        assert local_sizes == expected
        assert metadata.get_chunk_sizes_across_dp_rank() == expected

    assert metadata.get_chunk_sizes_across_dp_rank() is None
    assert metadata.num_tokens_across_dp_cpu.tolist() == [3, 5]


def test_sp_local_sizes_clears_chunk_sizes_after_forward_failure():
    metadata = AFDDPMetadata(torch.tensor([3, 5]))

    with (
        pytest.raises(RuntimeError, match="forward failed"),
        metadata.sp_local_sizes(2, pcp_size=1, use_ep=True),
    ):
        raise RuntimeError("forward failed")

    assert metadata.get_chunk_sizes_across_dp_rank() is None
