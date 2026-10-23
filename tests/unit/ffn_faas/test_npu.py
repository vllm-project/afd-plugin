# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual compiled control-op Meta contract; no device execution."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded  # noqa: E402
from afd_plugin.ffn_faas.protocol import Mode, SessionSpec  # noqa: E402


@pytest.fixture
def control_inputs():
    ensure_afd_ascend_ops_loaded()
    spec = SessionSpec(
        model_identity="compiled-meta-contract",
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=256,
        top_k=6,
        model_layers=43,
        layer_indices=tuple(range(43)),
        attention_ranks=12,
        attention_tp=4,
        ffn_ranks=4,
        capacity=8,
        slots=2,
    )
    # The fused BF16 entry borrows the executor-owned weights for every mode.
    # Control modes never project with them, so empty placeholders suffice.
    inputs = (
        torch.empty((2, 4096), dtype=torch.bfloat16, device="meta"),
        torch.empty((2, 6), dtype=torch.int32, device="meta"),
        torch.empty((2, 6), dtype=torch.float32, device="meta"),
        torch.empty((0,), dtype=torch.bfloat16, device="meta"),
        torch.empty((0,), dtype=torch.bfloat16, device="meta"),
        torch.empty((43,), dtype=torch.int32, device="meta"),
        torch.empty((3,), dtype=torch.int64, device="meta"),
        torch.empty((4,), dtype=torch.int64, device="meta"),
        torch.empty((1,), dtype=torch.uint8, device="meta"),
    )
    return inputs, spec.native_config(rank=12, poll_budget=16)


@pytest.mark.parametrize("mode", [Mode.INITIALIZE, Mode.PREPARE, Mode.FINISH])
def test_control_op_preserves_result_shapes_with_borrowed_weights(control_inputs, mode):
    inputs, config = control_inputs
    output, status = torch.ops.afd_ascend.ffn_faas(
        *inputs, int(mode), config, "meta-group"
    )
    assert tuple(output.shape) == (2, 4096)
    assert output.dtype == torch.bfloat16
    assert tuple(status.shape) == (8,)
    assert status.dtype == torch.int64


def test_compute_modes_validate_borrowed_weight_layout(control_inputs):
    inputs, config = control_inputs
    with pytest.raises(RuntimeError, match="weights do not match"):
        torch.ops.afd_ascend.ffn_faas(*inputs, int(Mode.EXECUTE), config, "meta-group")
