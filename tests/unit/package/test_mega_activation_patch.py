# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check that unsupported SDK quantization is rejected instead of left unpatched."""

import runpy
from pathlib import Path

import pytest

PATCH = (
    Path(__file__).resolve().parents[3]
    / "csrc/npu/ascend_kernels/afd_mega_moe_pull_native/op_kernel"
    / "afd_mega_moe_patch_activation.py"
)
patched_activation = runpy.run_path(str(PATCH))["patched_activation"]


@pytest.mark.parametrize(
    ("scale", "tensor", "length"),
    [
        ("127.0f / rowAbsMax", "ubGateFp32", "tileLength"),
        ("127.f / GMubDequantScale", "ubOutputTmp", "ChunkTileLen"),
    ],
)
def test_layered_rounding_is_applied(scale, tensor, length):
    source = f"""
        auto ubQuantS32 = ubGateFp32.template ReinterpretCast<int32_t>();
        auto scale = {scale};
        AscendC::Cast(ubQuantS32, {tensor}, AscendC::RoundMode::CAST_RINT, {length});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale(static_cast<half>(1.0));
        AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, {length});
    """
    result = patched_activation(source)
    assert "1.0f / (" in result
    cast = f"Cast(ubQuantF16, {tensor}, AscendC::RoundMode::CAST_RINT, {length})"
    assert cast in result
    assert "ubQuantS32" not in result


def test_unknown_sdk_header_is_rejected():
    with pytest.raises(ValueError, match="unsupported CANN activation"):
        patched_activation("unrecognized SDK implementation")
