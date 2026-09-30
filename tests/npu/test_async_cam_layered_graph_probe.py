# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in device probe for reusing one layered W4A8 computation graph.

This isolates the device-controlled layer and row-count inputs. Full AsyncCam
DR/GMM/CS graph capture requires a running Attention/FFN service and is tested
separately on the complete deployment.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.npu


def test_layered_w4a8_graph_reuses_device_metadata():
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
    torch.manual_seed(19)

    experts = 2
    hidden = 256
    intermediate = 256
    capacity = 32

    def make_weight(k: int, n: int, factor: float, squeeze: bool):
        values = torch.randint(-7, 8, (experts, k, n), dtype=torch.int8)
        pairs = values.reshape(-1, 2)
        packed = (
            torch.bitwise_or(
                torch.bitwise_left_shift(pairs[:, 1], 4),
                torch.bitwise_and(pairs[:, 0], 0x0F),
            )
            .reshape(experts, k, n // 2)
            .clone()
        )
        weight = torch_npu.npu_format_cast(packed.npu(), 29).view(torch.int32)
        scale = torch.full((experts, 1, n), factor, dtype=torch.float16).float()
        compensation = 8 * (values.float() * scale).sum(dim=1)
        encoded_scale = scale.view(torch.int32).to(torch.int64)
        if squeeze:
            encoded_scale = encoded_scale.squeeze(1)
        return weight.contiguous(), encoded_scale.npu(), compensation.npu()

    layers = []
    for layer_idx, factor in ((2, 0.01), (5, 0.04)):
        w13, s13, b13 = make_weight(hidden, 2 * intermediate, factor, True)
        w2, s2, b2 = make_weight(intermediate, hidden, factor, False)
        layers.append(
            W4A8LayerWeights(
                layer_idx, w13, w2, s13, s2, b13, b2, True, 0.0, 1.0
            )
        )
    executor = AsyncCAMW4A8Executor(layers)

    static_hidden = torch.zeros((capacity, hidden), dtype=torch.int8, device="npu")
    static_scales = torch.full((capacity,), 0.005, device="npu")
    static_counts = torch.tensor([1, 0], dtype=torch.int64, device="npu")
    static_info = torch.tensor([capacity * 2, 0, 2, 1], dtype=torch.int64, device="npu")

    # Initialize the custom operators before graph capture without changing
    # the addresses used by the graph itself.
    executor(static_hidden, static_scales, static_counts, static_info)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_output = executor(
            static_hidden, static_scales, static_counts, static_info
        )

    shared_hidden = torch.randint(-100, 101, (capacity, hidden), dtype=torch.int8)
    layer_outputs = {}
    for layer_idx, counts in (
        (2, (1, 0)),
        (5, (1, 0)),
        (2, (0, 7)),
        (5, (16, 16)),
        (2, (0, 0)),
    ):
        valid_rows = sum(counts)
        static_hidden.copy_(shared_hidden.npu())
        static_counts.copy_(torch.tensor(counts, dtype=torch.int64).npu())
        static_info.copy_(
            torch.tensor([capacity * 2, 0, layer_idx, valid_rows], dtype=torch.int64).npu()
        )
        graph.replay()
        torch.npu.synchronize()
        assert captured_output.shape == (capacity, hidden)
        graphed = captured_output[:valid_rows].cpu().float()
        eager = executor(
            static_hidden, static_scales, static_counts, static_info
        )[:valid_rows].cpu().float()
        if valid_rows:
            difference = (graphed - eager).abs()
            print(
                f"layer={layer_idx} counts={counts} "
                f"max_abs={difference.max().item():.6f} "
                f"mean_abs={difference.mean().item():.6f}"
            )
        torch.testing.assert_close(graphed, eager, rtol=0.04, atol=0.05)
        if counts == (1, 0):
            layer_outputs[layer_idx] = eager

    assert not torch.allclose(
        layer_outputs[2], layer_outputs[5], rtol=0.04, atol=0.05
    ), "synthetic layers must have distinguishable outputs for the same input"
