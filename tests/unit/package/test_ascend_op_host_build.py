# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU check that the op_host build consumes every staged host source.

``csrc/npu/scripts/compile_ascend_proj.sh`` stages each selected operator's
``op_host/`` tree recursively, so ``grouped_matmul_swiglu_quant_v2_layered``
arrives with its tiling implementation in a nested ``op_host/op_tiling/``
directory. The host CMakeLists must hand those sources to ``npu_op_code_gen``
and to libcust_optiling.so; collecting only the top level silently drops that
operator's tiling registration (``IMPL_OP_OPTILING`` /
``REGISTER_OPS_TILING_TEMPLATE``) from the run package, and neither the registry
tests nor the Meta dispatcher tests can observe the omission.

The probe replays the staging into a temporary project, configures the real
``cmake_files/op_host/CMakeLists.txt`` against a stub ``npu_op_*`` API, and
records the source list given to each build step, so it needs neither a CANN
toolchain nor a compiler. On an Ascend host the equivalent check on the linked
artifact is that the installed ``libcust_optiling.so`` still carries the tiling
registration, for example::

    strings .../vendors/afd-plugin/.../libcust_optiling.so | grep -i <OpName>
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.package.test_ascend_build_files import _run_select_ops

_ASCEND_KERNELS = Path("csrc/npu/ascend_kernels")

# The operator whose tiling implementation lives in a nested op_tiling/ dir.
_LAYERED_OP = "grouped_matmul_swiglu_quant_v2_layered"
_NESTED_TILING_SOURCES = frozenset(
    f"op_tiling/{_LAYERED_OP}{suffix}_tiling.cpp" for suffix in ("_base", "_fusion", "")
)
# Macros that place a host source into the tiling registry consumed at runtime.
_REGISTRATION_MARKERS = ("IMPL_OP_OPTILING", "REGISTER_OPS_TILING_TEMPLATE")

# Stub npu_op_* build API: records the sources each step receives instead of
# compiling. The npu_op_library()/npu_op_package() helpers decorate targets, so
# their target_* commands are shadowed as well, keeping the probe free of both
# the CANN toolchain and any C++ compiler.
_PROBE_CMAKE_LISTS = """\
cmake_minimum_required(VERSION 3.16)
project(afd_op_host_build_probe NONE)

set(package_name probe_vendor)
set(ASCEND_AUTOGEN_PATH "${CMAKE_BINARY_DIR}/autogen")
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}")
# Stand in for the files opbuild emits, so the autogen globs are non-empty
# exactly as they are in a real build.
file(WRITE "${ASCEND_AUTOGEN_PATH}/aclnn_probe.cpp" "// probe\\n")
file(WRITE "${ASCEND_AUTOGEN_PATH}/op_proto.cc" "// probe\\n")
file(WRITE "${ASCEND_AUTOGEN_PATH}/fallback_probe.cpp" "// probe\\n")

function(npu_op_code_gen)
  cmake_parse_arguments(GEN "" "PACKAGE;OUT_DIR" "SRC;COMPILE_OPTIONS" "${ARGN}")
  string(REPLACE ";" "\\n" gen_srcs "${GEN_SRC}")
  file(WRITE "${CMAKE_BINARY_DIR}/code_gen_srcs.txt" "${gen_srcs}\\n")
endfunction()

function(npu_op_library target kind)
  string(REPLACE ";" "\\n" lib_srcs "${ARGN}")
  file(WRITE "${CMAKE_BINARY_DIR}/lib_${target}_srcs.txt" "${lib_srcs}\\n")
endfunction()

function(npu_op_package_add package)
  string(REPLACE ";" "\\n" libs "${ARGN}")
  file(WRITE "${CMAKE_BINARY_DIR}/package_libs.txt" "${libs}\\n")
endfunction()

function(target_link_libraries)
endfunction()
function(target_compile_options)
endfunction()
function(target_include_directories)
endfunction()

add_subdirectory(op_host)
"""


def _stage_project(project: Path, root: Path, selected_ops: list[str]) -> Path:
    """Replay copy_ops_include() from compile_ascend_proj.sh into ``project``.

    Returns the staged ``op_host`` directory.
    """
    kernels = root / _ASCEND_KERNELS
    op_host = project / "op_host"
    autogen = project / "pregen" / "build_out" / "autogen"
    op_host.mkdir(parents=True, exist_ok=True)
    autogen.mkdir(parents=True, exist_ok=True)

    def copy_tree(source: Path, destination: Path) -> None:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)

    # The shared header directory is always staged; the op_host tree of every
    # selected operator is staged recursively (nested op_tiling/ included).
    copy_tree(kernels / "utils" / "op_host", op_host)
    for op in selected_ops:
        copy_tree(kernels / op / "op_host", op_host)
        op_api = kernels / op / "op_api"
        if op_api.is_dir():
            copy_tree(op_api, autogen)

    shutil.copyfile(
        kernels / "cmake_files" / "op_host" / "CMakeLists.txt",
        op_host / "CMakeLists.txt",
    )
    return op_host


def _cmake_generator_args() -> list[str]:
    """Return -G arguments for a project-mode configure on this platform."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the op_host build probe")
    if os.name == "nt":
        for generator, tool in (
            ("Ninja", "ninja"),
            ("MinGW Makefiles", "mingw32-make"),
            ("NMake Makefiles", "nmake"),
        ):
            if shutil.which(tool):
                return ["-G", generator]
        pytest.skip("no CMake generator is available for the op_host probe")
    if shutil.which("make"):
        return []
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    pytest.skip("no make program is available for the op_host probe")


def _configure_probe(project: Path, tmp_path: Path) -> dict[str, set[str]]:
    """Configure ``project`` and return the sources recorded per build step."""
    (project / "CMakeLists.txt").write_text(_PROBE_CMAKE_LISTS, encoding="utf-8")
    # The host CMakeLists only needs the CANN platform header to exist; the
    # include directories it derives from it are added when present.
    cann_root = tmp_path / "cann_root"
    (cann_root / "include" / "platform").mkdir(parents=True)
    (cann_root / "include" / "platform" / "platform_infos_def.h").write_text(
        "", encoding="utf-8"
    )
    build_dir = tmp_path / "build_out"
    env = dict(os.environ, ASCEND_CANN_PACKAGE_PATH=str(cann_root))

    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(project),
            "-B",
            str(build_dir),
            *_cmake_generator_args(),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    recorded: dict[str, set[str]] = {}
    for step, name in (
        ("code_gen", "code_gen_srcs.txt"),
        ("cust_opapi", "lib_cust_opapi_srcs.txt"),
        ("cust_op_proto", "lib_cust_op_proto_srcs.txt"),
        ("cust_optiling", "lib_cust_optiling_srcs.txt"),
        ("package", "package_libs.txt"),
    ):
        lines = (build_dir / name).read_text(encoding="utf-8").splitlines()
        recorded[step] = {line.strip() for line in lines if line.strip()}
    return recorded


def _collects(entries: set[str], relative_path: str) -> bool:
    """Whether a recorded source list covers ``relative_path``."""
    return any(
        entry == relative_path or entry.endswith("/" + relative_path)
        for entry in entries
    )


def test_op_host_build_collects_nested_tiling_registration(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    selection = _run_select_ops("--soc", "ascend910_93", "--shmem", "0")
    assert selection.returncode == 0, selection.stderr
    selected_ops = selection.stdout.split()
    assert _LAYERED_OP in selected_ops

    op_host = _stage_project(tmp_path / "project", root, selected_ops)
    recorded = _configure_probe(tmp_path / "project", tmp_path)

    # The staging step keeps the nested sources, so the probe must have them.
    for relative_path in sorted(_NESTED_TILING_SOURCES):
        assert (op_host / relative_path).is_file(), relative_path

    # The registration lives in one of those nested sources
    # (IMPL_OP_OPTILING / REGISTER_OPS_TILING_TEMPLATE); find it from the staged
    # content so the check follows the registration rather than a hard-coded
    # file, and fail if the operator ever stops registering tiling at all.
    registrations = {
        path.relative_to(op_host).as_posix()
        for path in op_host.rglob("*.cpp")
        if any(
            marker in path.read_text(encoding="utf-8")
            for marker in _REGISTRATION_MARKERS
        )
    }
    assert registrations & _NESTED_TILING_SOURCES, (
        "no tiling registration is staged under op_tiling/"
    )

    # Both the code generator (which derives the op_proto/fallback inputs) and
    # the tiling library, which is what the run package installs as this
    # operator's tiling implementation, must receive them.
    for step in ("code_gen", "cust_optiling"):
        missing = sorted(
            relative_path
            for relative_path in _NESTED_TILING_SOURCES | registrations
            if not _collects(recorded[step], relative_path)
        )
        assert not missing, f"{step} drops tiling sources: {missing}"

    # The registration only reaches the run package if the tiling library is
    # packaged, which is what the probe records from npu_op_package_add().
    assert "cust_optiling" in recorded["package"]
