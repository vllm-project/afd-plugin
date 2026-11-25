# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: CPU checks for the routed-only CAM sources and build selection
#              contract.
# Create: 2026-09-21
# Note: Checks operator selection, source staging and packaged license notices.
# History: 2026-09-21 create
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

AFD_ASYNC_OPERATORS = (
    "afd_async_dispatch_send",
    "afd_async_dispatch_recv",
    "afd_async_combine_send",
    "afd_async_combine_recv",
)

AFD_ASYNC_MIT_NOTICE = """\
The MIT License

Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
Copyright contributors to the AFD plugin project

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


@pytest.mark.parametrize("operator", AFD_ASYNC_OPERATORS)
@pytest.mark.parametrize("soc", ["ascend910_93", "ascend950"])
def test_afd_async_operator_selection_uses_soc_implementation(operator: str, soc: str):
    result = _run_select_ops("--soc", soc, "--shmem", "0", "--ops", operator)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"{operator}_arch35" if soc == "ascend950" else operator
    ]


def test_afd_async_arch35_selector_is_specific_to_950():
    directory = "afd_async_dispatch_send_arch35"
    args = ("--shmem", "0", "--ops", directory)
    result = _run_select_ops("--soc", "ascend950", *args)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [directory]
    assert _run_select_ops("--soc", "ascend910_93", *args).returncode != 0


def test_afd_async_selection_preserves_registry_order_with_legacy_operators():
    requested = [AFD_ASYNC_OPERATORS[-1], "e2a", AFD_ASYNC_OPERATORS[0]]
    result = _run_select_ops(
        "--soc", "ascend910_93", "--shmem", "0", "--ops", ";".join(requested)
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "e2a",
        AFD_ASYNC_OPERATORS[0],
        AFD_ASYNC_OPERATORS[-1],
    ]


def test_native_sources_survive_source_distribution_manifest(monkeypatch):
    """Apply setuptools' real MANIFEST rules to the vendored build inputs."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    files = FileList()
    # setuptools works on manifest paths, which are always posix; native
    # separators would make every required entry miss on Windows.
    files.allfiles = [
        path.relative_to(root).as_posix()
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

    required = set(files.allfiles) | {"LICENSE"}
    assert required <= packaged, sorted(required - packaged)
    license_text = (root / "LICENSE").read_text()
    assert " ".join(AFD_ASYNC_MIT_NOTICE.split()) in " ".join(license_text.split())


@pytest.mark.parametrize("soc", ["ascend910_93", "ascend950"])
def test_selected_afd_async_sources_do_not_overwrite_other_operators_when_staged(soc):
    """The build driver flattens each source category into one directory."""
    root = Path(__file__).resolve().parents[3] / "csrc/npu/ascend_kernels"
    result = _run_select_ops("--soc", soc, "--shmem", "0")
    assert result.returncode == 0, result.stderr

    for part in ("op_api", "op_host", "op_kernel"):
        staged: dict[str, Path] = {}
        for operator in ("utils", *result.stdout.split()):
            part_dir = root / operator / part
            if (
                part == "op_api"
                and operator.endswith("_arch35")
                and not part_dir.is_dir()
            ):
                part_dir = root / operator.removesuffix("_arch35") / part
                assert part_dir.is_dir(), part_dir
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
    setup_options = {}
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: setup_options.update(kwargs)
    )
    setup_namespace = runpy.run_path(str(root / "setup.py"))
    extension = setup_namespace["CMakeExtension"](
        "afd_plugin._C_ascend", "csrc/npu/pybind"
    )
    command = setup_options["cmdclass"]["build_ext"](setuptools.Distribution())
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
