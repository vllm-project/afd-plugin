# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in numerical checks; requires a 910C and the compiled AFD extension.

Run with AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c via pytest.
This exercises the existing kernels, not the CAM distributed completion protocol.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("counts", [(0, 0), (1, 0), (0, 7), (16, 16)])
@pytest.mark.parametrize("layer_ids", [(0, 1), (2, 5)])
def test_layered_w4a8_valid_rows_and_capacity_tail(per_channel, counts, layer_ids):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
    from afd_plugin.model_executor.npu.async_cam_w4a8 import (
        AsyncCAMW4A8Executor,
        W4A8LayerWeights,
    )

    ensure_cam_async_ops_available()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(7)
    experts, hidden, intermediate, capacity = 2, 256, 256, 32
    layers, reference = [], []

    def make_weight(k, n, factor, squeeze):
        values = torch.randint(-7, 8, (experts, k, n), dtype=torch.int32)
        groups = 1 if per_channel else 2
        scale = torch.full((experts, groups, n), factor, dtype=torch.float32)
        if not per_channel:
            scale[:, 1] *= 2
        scale = scale.half().float()
        dequant = values.float() * scale.repeat_interleave(k // groups, dim=1)
        compensation = 8 * dequant.sum(dim=1)
        # Match the current upstream W4A8 loader: two signed INT4 values per
        # INT8, followed by NZ conversion and an INT32 view.
        pairs = values.to(torch.int8).reshape(-1, 2)
        packed8 = (
            torch.bitwise_or(
                torch.bitwise_left_shift(pairs[:, 1], 4),
                torch.bitwise_and(pairs[:, 0], 0x0F),
            )
            .reshape(experts, k, n // 2)
            .clone()
        )
        weight_nz = torch_npu.npu_format_cast(packed8.npu(), 29)
        weight = weight_nz.view(torch.int32).contiguous()
        # Preserve FP32 scale bits in the INT64 operator encoding.
        encoded_scale = scale.view(torch.int32).to(torch.int64)
        if per_channel and squeeze:
            encoded_scale = encoded_scale.squeeze(1)
        return weight, encoded_scale.npu(), compensation.npu(), dequant

    for idx, factor in zip(layer_ids, (0.01, 0.02), strict=True):
        w13, s13, b13, ref13 = make_weight(hidden, 2 * intermediate, factor, True)
        w2, s2, b2, ref2 = make_weight(intermediate, hidden, factor, False)
        layers.append(
            W4A8LayerWeights(idx, w13, w2, s13, s2, b13, b2, per_channel, 0.0, 1.0)
        )
        reference.append((ref13, ref2))
    executor = AsyncCAMW4A8Executor(layers)
    x = torch.randint(-100, 101, (capacity, hidden), dtype=torch.int8)
    x_scale = torch.full((capacity,), 0.005)
    valid_rows = sum(counts)
    device_counts = torch.tensor(counts, dtype=torch.int64, device="npu")
    for slot in (1, 0, 1):
        metadata = torch.tensor(
            [capacity * 2, 0, layers[slot].layer_idx, valid_rows],
            dtype=torch.int64,
            device="npu",
        )
        output = executor(x.npu(), x_scale.npu(), device_counts, metadata)
        torch.npu.synchronize()
        actual = output[:valid_rows].cpu().float()
        if valid_rows == 0:
            assert output.shape == (capacity, hidden)
            continue
        expected = []
        offset = 0
        w13, w2 = reference[slot]
        for expert, count in enumerate(counts):
            activation = (
                x[offset : offset + count].float()
                * x_scale[offset : offset + count, None]
            )
            gate, up = (activation @ w13[expert]).chunk(2, dim=-1)
            swiglu = torch.nn.functional.silu(gate) * up
            scale = swiglu.abs().amax(dim=-1).clamp_min(1e-12) / 127
            quantized = (swiglu / scale[:, None]).round().clamp(-127, 127)
            expected.append((quantized * scale[:, None]) @ w2[expert])
            offset += count
        expected = torch.cat(expected)
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.05)
        # Tail perturbations must keep valid rows within the golden tolerance.
        # Use independent inputs because W2 MSD may modify its activation buffer;
        # identical invocations can also differ slightly on this kernel path.
        perturbed = x.clone()
        perturbed[valid_rows:] = 127
        changed = executor(perturbed.npu(), x_scale.npu(), device_counts, metadata)
        torch.npu.synchronize()
        torch.testing.assert_close(
            changed[:valid_rows].cpu().float(), expected, rtol=0.04, atol=0.05
        )
