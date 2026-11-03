# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Generate a Layered-compatible MC2 activation from the installed CANN header."""

import re
import sys
from pathlib import Path


def patched_activation(source: str) -> str:
    source = source.replace(
        '"../template_linear_algebra_v2/', '"template_linear_algebra_v2/'
    )
    for header in ("get_tensor_addr.hpp", "gated_activation.hpp"):
        source = source.replace(f'"{header}"', f'"utils/{header}"')

    source, multipliers = re.subn(
        r"127\.0f / rowAbsMax",
        "1.0f / (rowAbsMax / 127.0f)",
        source,
    )
    source, legacy_multipliers = re.subn(
        r"127\.f / GMubDequantScale",
        "(GMubDequantScale > 0.0f ? 1.0f / (GMubDequantScale / 127.f) : 0.0f)",
        source,
    )
    if multipliers + legacy_multipliers != 1:
        raise ValueError("unsupported CANN activation scale calculation")

    pattern = (
        r"AscendC::Cast\(ubQuantS32, (\w+), AscendC::RoundMode::CAST_RINT, (\w+)\);"
        r"\s*AscendC::PipeBarrier<PIPE_V>\(\);"
        r"\s*AscendC::SetDeqScale\(static_cast<half>\(1(?:\.0)?\)\);"
        r"\s*AscendC::Cast\(ubQuantF16, ubQuantS32, "
        r"AscendC::RoundMode::CAST_RINT, \2\);"
    )
    source, casts = re.subn(
        pattern,
        r"AscendC::Cast(ubQuantF16, \1, AscendC::RoundMode::CAST_RINT, \2);",
        source,
    )
    if casts != 1:
        raise ValueError("unsupported CANN activation quantization sequence")
    source = re.sub(r"^\s*auto ubQuantS32 = .*?;\n", "\n", source, flags=re.MULTILINE)
    return source


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(patched_activation(Path(sys.argv[1]).read_text()))
