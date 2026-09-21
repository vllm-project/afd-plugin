# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checks for the routed-only CAM sources and build selection contract."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import FileList, manifest_maker

from tests.unit.package.test_ascend_build_files import _run_select_ops

CAM_OPERATORS = (
    "afd_async_dispatch_send",
    "afd_async_dispatch_recv",
    "afd_async_combine_send",
    "afd_async_combine_recv",
)

CAM_MIT_NOTICE = """\
The MIT License

Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


@pytest.mark.parametrize("operator", CAM_OPERATORS)
def test_cam_operator_selection_rejects_unqualified_soc(operator: str):
    result = _run_select_ops("--soc", "ascend950", "--shmem", "1", "--ops", operator)

    assert result.returncode != 0
    assert operator in result.stderr
    assert "not in SOC support list" in result.stderr
    assert not result.stdout


def test_cam_selection_preserves_registry_order_with_legacy_operators():
    requested = [CAM_OPERATORS[-1], "e2a", CAM_OPERATORS[0]]
    result = _run_select_ops(
        "--soc", "ascend910_93", "--shmem", "0", "--ops", ";".join(requested)
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["e2a", CAM_OPERATORS[0], CAM_OPERATORS[-1]]


def test_cam_sources_survive_source_distribution_manifest(monkeypatch):
    """Apply setuptools' real MANIFEST rules to the vendored build inputs."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    files = FileList()
    files.allfiles = [
        str(path.relative_to(root))
        for path in (root / "csrc").rglob("*")
        if path.is_file()
    ]
    for line in (root / "MANIFEST.in").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            files.process_template_line(line)
    distribution = setuptools.Distribution()
    distribution.parse_config_files()
    manifest = manifest_maker(distribution)
    manifest.filelist = files
    manifest.add_license_files()
    packaged = {Path(path).as_posix() for path in files.files}

    required = {
        "csrc/npu/pybind/torch_binding_cam_async.cpp",
        "csrc/npu/ascend_kernels/operator_registry.json",
        "LICENSE",
    }
    for operator in CAM_OPERATORS:
        source_dir = root / "csrc/npu/ascend_kernels" / operator
        for part in ("op_api", "op_host", "op_kernel"):
            sources = tuple((source_dir / part).glob("*"))
            assert any(path.suffix == ".cpp" for path in sources), f"{operator}/{part}"
            if part != "op_host":
                assert any(path.suffix == ".h" for path in sources), (
                    f"{operator}/{part}"
                )
            required.update(path.relative_to(root).as_posix() for path in sources)
    shared_headers = root / "csrc/npu/ascend_kernels/utils"
    required.update(
        path.relative_to(root).as_posix() for path in shared_headers.rglob("*.h")
    )

    assert required <= packaged, sorted(required - packaged)
    for path in required:
        assert (root / path).is_file(), path
    license_text = (root / "LICENSE").read_text()
    assert " ".join(CAM_MIT_NOTICE.split()) in " ".join(license_text.split())


def test_selected_cam_sources_do_not_overwrite_other_operators_when_staged():
    """The build driver flattens each source category into one directory."""
    root = Path(__file__).resolve().parents[3] / "csrc/npu/ascend_kernels"
    result = _run_select_ops("--soc", "ascend910_93", "--shmem", "0")
    assert result.returncode == 0, result.stderr

    for part in ("op_api", "op_host", "op_kernel"):
        staged: dict[str, Path] = {}
        for operator in ("utils", *result.stdout.split()):
            part_dir = root / operator / part
            for path in part_dir.rglob("*"):
                if not path.is_file():
                    # Nested subdirectories (e.g. op_host/op_tiling/) are copied
                    # recursively by the build driver; only file collisions
                    # matter for the staging check.
                    continue
                rel = path.relative_to(part_dir).as_posix()
                previous = staged.setdefault(rel, path)
                assert previous.read_bytes() == path.read_bytes(), (
                    f"{path} overwrites different contents from {previous}"
                )


@pytest.mark.parametrize("soc", [None, "ascend910_93", "950"])
def test_skipping_aclnn_build_still_passes_soc_to_binding_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, soc: str | None
):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("AFD_BUILD_ASCEND_OPS", "1")
    monkeypatch.setenv("AFD_SKIP_ACLNN_BUILD", "1")
    monkeypatch.delenv("SOC_VERSION", raising=False)
    if soc is not None:
        monkeypatch.setenv("SOC_VERSION", soc)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup_namespace = runpy.run_path(str(root / "setup.py"))
    extension = setup_namespace["CMakeExtension"](
        "afd_plugin._C_ascend", "csrc/npu/pybind"
    )
    command = setup_namespace["BuildAscendOps"](setuptools.Distribution())
    command.extensions = [extension]
    command.build_temp = str(tmp_path / "build")
    command.build_lib = str(tmp_path / "lib")
    calls: list[list[str]] = []

    def record_call(args: list[str], *, cwd: Path) -> None:
        assert cwd.is_relative_to(tmp_path)
        calls.append(args)

    monkeypatch.setattr(subprocess, "check_call", record_call)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "/pybind11/cmake"
    )
    # Keep setuptools' platform compiler detection out of this CMake driver test.
    monkeypatch.setattr(build_ext, "run", lambda self: self.build_extension(extension))

    command.run()

    assert len(calls) == 3
    assert calls[0][:2] == ["cmake", str(root / "csrc/npu/pybind")]
    assert f"-DAFD_SOC_VERSION={soc or '910c'}" in calls[0]
    assert calls[1][:3] == ["cmake", "--build", "."]
    assert calls[2] == ["cmake", "--install", "."]
