# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""W8A8 executor versus CANN W13 and ordinary GMM W2, with FULL replay."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests.npu._gmm_native_reference import (
    builtin_a8w8_references,
    capture_a8w8_request,
)

pytestmark = pytest.mark.npu

if TYPE_CHECKING:
    from torch import Tensor


@pytest.fixture(params=[0.0, 10.0])
def w8a8_runtime(request):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
    from afd_plugin.model_executor.npu.async_cam_layered import (
        AsyncCAMLayeredExecutor,
        LayeredFFNWeights,
    )

    ensure_cam_async_ops_available()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.config.allow_internal_format = True
    generator = torch.Generator().manual_seed(524)
    experts, hidden, intermediate, capacity = 2, 256, 256, 32
    layers = []
    for layer_id in (2, 5):
        w13 = torch.randint(
            -8,
            8,
            (experts, hidden, 2 * intermediate),
            dtype=torch.int8,
            generator=generator,
        )
        w2 = torch.randint(
            -8,
            8,
            (experts, intermediate, hidden),
            dtype=torch.int8,
            generator=generator,
        )
        layers.append(
            LayeredFFNWeights(
                layer_idx=layer_id,
                w13=torch_npu.npu_format_cast(w13.npu(), 29),
                w2=torch_npu.npu_format_cast(w2.npu(), 29),
                w13_scale=(
                    torch.rand(experts, 2 * intermediate, generator=generator) * 0.01
                    + 0.005
                ).npu(),
                w2_scale=(
                    torch.rand(experts, hidden, generator=generator) * 0.01 + 0.005
                )
                .to(torch.bfloat16)
                .npu(),
                w13_bias=None,
                w2_bias=None,
                per_channel=True,
                swiglu_limit=request.param,
                routed_scaling_factor=1.0,
                quantization="W8A8",
            )
        )
    executor = AsyncCAMLayeredExecutor(layers)
    assert executor.applied_swiglu_limit == 0.0
    x = torch.randint(
        -8, 8, (capacity, hidden), dtype=torch.int8, generator=generator
    ).npu()
    scales = torch.full((capacity,), 0.01 if request.param == 0 else 2.0, device="npu")
    return torch, torch_npu, layers, executor, x, scales


@pytest.mark.parametrize("counts", [(0, 0), (1, 0), (0, 7), (16, 16)])
def test_w8a8_layer_selection_eager_and_graph(
    counts, tmp_path, w8a8_runtime
):  # long-test: real operator and graph differential
    torch, torch_npu, layers, executor, x, scales = w8a8_runtime
    capacity, hidden = x.shape
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    metadata = torch.tensor(
        [capacity, 0, 2, sum(counts)], dtype=torch.int64, device="npu"
    )
    executor(x, scales, groups, metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    retained: list[Tensor] = []
    with torch.npu.graph(graph):
        captured = executor(x, scales, groups, metadata, retained_tensors=retained)
    requests, observed = [], []
    valid = sum(counts)
    for slot in (1, 0, 1):
        metadata[2:3].fill_(layers[slot].layer_idx)
        eager = executor(x, scales, groups, metadata)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured[:valid], eager[:valid], rtol=0, atol=0)
        assert captured.shape == (capacity, hidden) and captured.dtype == torch.bfloat16
        # Reused receive buffers may contain arbitrary values outside valid rows.
        x[valid:].fill_(127)
        poisoned = executor(x, scales, groups, metadata)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(poisoned[:valid], eager[:valid], rtol=0, atol=0)
        torch.testing.assert_close(captured[:valid], eager[:valid], rtol=0, atol=0)
        if valid:
            requests.append(
                capture_a8w8_request(
                    torch_npu,
                    x,
                    scales,
                    layers[slot].w13,
                    layers[slot].w13_scale,
                    groups,
                    1,
                )
            )
            observed.append(eager[:valid].cpu())
    if not valid:
        return
    references = builtin_a8w8_references(torch, requests, tmp_path, 29)
    for slot, actual, (activation, activation_scale) in zip(
        (1, 0, 1), observed, references, strict=True
    ):
        expected = torch_npu.npu_grouped_matmul(
            x=[activation.npu()],
            weight=[layers[slot].w2],
            scale=[layers[slot].w2_scale],
            per_token_scale=[activation_scale.npu()],
            bias=None,
            group_list=groups,
            group_list_type=1,
            group_type=0,
            split_item=2,
            output_dtype=torch.bfloat16,
        )[0][:valid].cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
