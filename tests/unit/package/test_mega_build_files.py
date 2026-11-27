# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validate the split MegaMoE extension selection.

Why: the Mega extension is declared only for 950 SoCs, and it shares the
``csrc/npu/pybind`` CMake project with the Ascend extension, so the declared
extension list changes with ``SOC_VERSION``.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest
import setuptools


def _extension_names(
    monkeypatch: pytest.MonkeyPatch,
    *,
    soc_version: str | None = None,
) -> list[str]:
    root = Path(__file__).resolve().parents[3]
    captured: dict[str, object] = {}

    def fake_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setenv("AFD_BUILD_ASCEND_OPS", "1")
    monkeypatch.setenv("AFD_BUILD_GPU_ASYNC_OPS", "0")
    monkeypatch.delenv("SOC_VERSION", raising=False)
    if soc_version is not None:
        monkeypatch.setenv("SOC_VERSION", soc_version)

    runpy.run_path(str(root / "setup.py"))

    ext_modules = captured["ext_modules"]
    return [ext.name for ext in ext_modules]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        (None, ["afd_plugin._C_ascend"]),
        ("910c", ["afd_plugin._C_ascend"]),
        ("950", ["afd_plugin._C_ascend", "afd_plugin._C_afd_async_mega"]),
        (
            "ascend950dt_9582",
            ["afd_plugin._C_ascend", "afd_plugin._C_afd_async_mega"],
        ),
    ],
)
def test_split_mega_extension_follows_the_soc(
    monkeypatch: pytest.MonkeyPatch,
    soc_version: str | None,
    expected: list[str],
):
    assert _extension_names(monkeypatch, soc_version=soc_version) == expected


@pytest.mark.parametrize("suffix", ["atten_send", "mega_ffn", "atten_recv"])
def test_mega_native_registration_uses_one_op_type(suffix: str):
    """All native registration keys must match the generated ACLNN op type."""
    root = Path(__file__).resolve().parents[3] / "csrc/npu/ascend_kernels"
    op = root / f"afd_async_{suffix}_arch35"
    definition = (op / f"op_host/{suffix}_def.cpp").read_text()
    op_type = re.search(r"OP_ADD\((\w+)\)", definition).group(1)
    declarations = [
        (op / f"op_host/{suffix}_infershape.cpp", "IMPL_OP_INFERSHAPE"),
        (op / f"op_host/op_tiling/arch35/{suffix}_tiling.cpp", "IMPL_OP_OPTILING"),
        (op / f"op_kernel/afd_async_{suffix}_tiling_key.h", "ASCENDC_TPL_ARGS_DECL"),
    ]
    for path, macro in declarations:
        registered = re.search(rf"{macro}\(\s*(\w+)", path.read_text()).group(1)
        assert registered == op_type, (
            f"{path.name}: {macro} registers {registered}, expected {op_type}"
        )
    api = (op / f"op_api/aclnn_afd_async_{suffix}.cpp").read_text()
    assert f"aclnnInner{op_type}GetWorkspaceSize(" in api
