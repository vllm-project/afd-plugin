# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5-only bit-exact check of the real FP4 descriptor and physical storage."""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")


def test_native_fp4_weight_round_trip_on_a5():
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    from afd_plugin.connectors.npu.mega_weights import (
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_ND,
        PACKED_BYTE_NZ_C0_16,
        _packed_weight,
    )

    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires Ascend950")
    shape = (2, 128, 192)
    original = torch.randint(0, 256, shape, dtype=torch.uint8, device="npu")
    fp4 = torch_npu.float4_e2m1fn_x2
    native = torch_npu.npu_format_cast(
        original,
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=fp4,
    )
    parameter = torch.nn.Parameter(native.transpose(1, 2), requires_grad=False)
    packed = _packed_weight(parameter, shape)
    assert torch_npu.get_npu_format(packed) == PACKED_BYTE_NZ_C0_16
    restored = torch_npu.npu_format_cast(packed, ACL_FORMAT_ND, input_dtype=fp4)
    assert torch.equal(restored.cpu(), original.cpu())
