# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host selection contract; native MLP arithmetic is not emulated or validated."""

import pytest

from tests.unit.fakes.async_ffn import load_mxfp_executor

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("layer, rows, multiplier", [(3, 2, 2), (7, 3, 4), (3, 0, 2)])
def test_mxfp_returns_selected_layer_valid_rows(monkeypatch, layer, rows, multiplier):
    executor = load_mxfp_executor(monkeypatch, torch)
    states = torch.ones((4, 2), dtype=torch.bfloat16)
    result = executor(states, None, torch.tensor([rows]), torch.tensor([0, 0, layer]))
    assert torch.equal(result, states[:rows] * multiplier)


@pytest.mark.parametrize("rows", [-1, 5])
def test_mxfp_rejects_counts_outside_receive_capacity(monkeypatch, rows):
    executor = load_mxfp_executor(monkeypatch, torch)
    with pytest.raises(ValueError, match="receive capacity"):
        executor(
            torch.ones((4, 2), dtype=torch.bfloat16),
            None,
            torch.tensor([rows]),
            torch.tensor([0, 0, 3]),
        )


def test_mxfp_rejects_receive_scales(monkeypatch):
    executor = load_mxfp_executor(monkeypatch, torch)
    with pytest.raises(ValueError, match="BF16 transport"):
        executor(
            torch.ones((4, 2), dtype=torch.bfloat16),
            torch.ones(4),
            torch.tensor([2]),
            torch.tensor([0, 0, 3]),
        )
