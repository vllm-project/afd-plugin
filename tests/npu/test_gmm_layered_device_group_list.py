# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""910C regression for a device-produced layered GMM group list.

Run after building the Ascend extension::

    SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 \
        pytest tests/npu/test_gmm_layered_device_group_list.py

Set ``AFD_GMM_PROFILE_TRACE=/path/to/trace`` to export one Chrome trace per
group-list representation. Only the layered operator invocation is profiled;
the built-in reference and comparison run afterward.

The built-in, non-layered grouped matmul is the numerical reference. Both
implementations of A8W4 MSD may modify x in place, so each call gets a clone.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu

_M = 128
_K = 256
_N = 256
_E = 4
_LAYERS = 2
_INT4_PER_INT32 = 8
_NZ_FORMAT = 29
_COUNT_BASE = 20
_COUNT_STEP = 8
_RTOL = 0.02
_ATOL = 0.02


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_device_group_list_matches_builtin(group_list_type: int) -> None:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in 910C runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

    ensure_afd_ascend_ops_loaded()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.config.allow_internal_format = True
    generator = torch.Generator(device="cpu").manual_seed(381)

    x = torch.randint(-64, 64, (_M, _K), generator=generator, dtype=torch.int8).npu()
    per_token_scale = torch.full((_M,), 0.01, dtype=torch.float32, device="npu")
    weights = []
    scales = []
    biases = []
    for _ in range(_LAYERS):
        nibbles = torch.randint(
            -8,
            8,
            (_E, _K, _N // _INT4_PER_INT32, _INT4_PER_INT32),
            generator=generator,
            dtype=torch.int32,
        )
        packed = sum(
            (nibbles[..., index] & 0xF) << (4 * index)
            for index in range(_INT4_PER_INT32)
        )
        weights.append(
            torch_npu.npu_format_cast(packed.to(torch.int32).npu(), _NZ_FORMAT)
        )
        scale_values = torch.rand((_E, _N), generator=generator) * 1e-3 + 1e-4
        scale_bits = scale_values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        scales.append(((scale_bits << 32) | scale_bits).npu())
        biases.append((torch.rand((_E, _N), generator=generator) - 0.5).float().npu())

    # The counts and cumulative form are calculated on the NPU. No test-side
    # conversion to a Python list is needed by the layered operator.
    counts = (
        torch.arange(_E, device="npu", dtype=torch.int64) * _COUNT_STEP + _COUNT_BASE
    )
    group_list = counts.cumsum(0) if group_list_type == 0 else counts
    trace_base = os.environ.get("AFD_GMM_PROFILE_TRACE")

    for layer in range(_LAYERS):
        layer_index = torch.full((1,), layer, device="npu", dtype=torch.int64)
        for repeat in range(3):
            # Keep clone/copy activity outside the measured operator path.
            activation = x.clone()

            run_layered = partial(
                torch.ops.afd_ascend.grouped_matmul_layered,
                x=[activation],
                all_weight=weights,
                all_bias=biases,
                all_scale=scales,
                layer_index=layer_index,
                group_list=group_list,
                per_token_scale=per_token_scale,
                group_list_type=group_list_type,
                split_item=3,
                output_dtype=torch.bfloat16,
            )

            if trace_base and layer == 0 and repeat == 0:
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ]
                ) as profile:
                    actual = run_layered()[0]
                trace_path = f"{Path(trace_base)}.{group_list_type}.json"
                profile.export_chrome_trace(trace_path)
            else:
                actual = run_layered()[0]
            expected = torch_npu.npu_grouped_matmul(
                x=[x.clone()],
                weight=[weights[layer]],
                bias=[biases[layer]],
                scale=[scales[layer].unsqueeze(1)],
                per_token_scale=[per_token_scale],
                group_list=counts,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
            torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
