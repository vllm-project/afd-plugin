# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()

_ASCEND_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "TORCH_NPU_PATH",
)
_DEFAULT_ASCEND_TOOLKIT_PATH = Path("/usr/local/Ascend/ascend-toolkit/latest")
_FASTAFD_SOURCE_DIR = Path("csrc") / "gpu" / "fastafd"
# The FastAFD kernels are JIT-compiled at first use from <kernel root>/include.
_FASTAFD_KERNEL_ROOT = Path("afd_plugin") / "_fastafd"
# Header-only dependencies of the FastAFD kernels: name, pinned archive,
# its sha256, and the archive's top-level directory.
_FASTAFD_HEADER_DEPENDENCIES = (
    (
        "cutlass",
        "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.2.1.tar.gz",
        "a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df",
        "cutlass-4.2.1",
    ),
    (
        "fmt",
        "https://github.com/fmtlib/fmt/archive/refs/tags/11.2.0.tar.gz",
        "bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af",
        "fmt-11.2.0",
    ),
)


def _env_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_disabled(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no", "off"}


def _running_on_ascend_platform() -> bool:
    if importlib.util.find_spec("torch_npu") is not None:
        return True
    if any(os.environ.get(name) for name in _ASCEND_ENV_VARS):
        return True
    return _DEFAULT_ASCEND_TOOLKIT_PATH.exists()


def _should_build_ascend_ops() -> bool:
    requested = os.environ.get("AFD_BUILD_ASCEND_OPS")
    if requested is not None and requested.strip() != "":
        if _env_enabled(requested):
            return True
        if _env_disabled(requested):
            return False
        raise RuntimeError(
            "AFD_BUILD_ASCEND_OPS must be one of: 1, 0, true, false, yes, no, on, off",
        )
    return _running_on_ascend_platform()


def _should_build_fastafd_ops() -> bool:
    requested = os.environ.get("AFD_BUILD_FASTAFD_OPS", "")
    if requested.strip() == "":
        return False
    if _env_enabled(requested):
        return True
    if _env_disabled(requested):
        return False
    raise RuntimeError(
        "AFD_BUILD_FASTAFD_OPS must be one of: 1, 0, true, false, yes, no, on, off",
    )


def _fastafd_header_dependency(
    name: str,
    url: str,
    sha256: str,
    top_dir: str,
) -> Path:
    """Return a header-only dependency tree, fetching the pinned archive once."""
    override = os.environ.get(f"AFD_FASTAFD_{name.upper()}_DIR")
    if override:
        return Path(override)
    cache_dir = ROOT / "build" / "fastafd-deps"
    target = cache_dir / top_dir
    if target.is_dir():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{top_dir}.tar.gz"
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{url} has sha256 {digest}, expected {sha256}")
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir, filter="data")
    return target


def _fastafd_build(cutlass_dir: Path, fmt_dir: Path) -> tuple[Extension, type]:
    """Extension and build command for the FastAFD M2N MegaMoE kernels."""
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is None:
        raise RuntimeError("Building FastAFD ops requires the CUDA toolkit")
    cuda_include = Path(CUDA_HOME) / "include"
    # CUDA library headers (cuSPARSE and others pulled in by ATen) may come
    # from the nvidia-* wheels instead of the toolkit.
    nvidia_spec = importlib.util.find_spec("nvidia")
    wheel_includes = (
        []
        if nvidia_spec is None or nvidia_spec.submodule_search_locations is None
        else sorted(
            str(include)
            for location in nvidia_spec.submodule_search_locations
            for include in Path(location).glob("*/include")
        )
    )
    extension = CUDAExtension(
        "afd_plugin._C_fastafd",
        sources=[str(_FASTAFD_SOURCE_DIR / "csrc" / "python_api.cpp")],
        include_dirs=[
            str(cuda_include),
            str(cuda_include / "cccl"),
            *wheel_includes,
            str(ROOT / _FASTAFD_SOURCE_DIR / "csrc"),
            str(ROOT / _FASTAFD_SOURCE_DIR / "include"),
            str(cutlass_dir / "include"),
            str(fmt_dir / "include"),
        ],
        libraries=["cudart", "nvrtc"],
        extra_compile_args={
            "cxx": ["-std=c++17", "-O3", "-Wno-psabi", "-Wno-deprecated-declarations"],
        },
    )
    include_trees = (
        ("deep_gemm", ROOT / _FASTAFD_SOURCE_DIR / "include" / "deep_gemm"),
        ("cute", cutlass_dir / "include" / "cute"),
        ("cutlass", cutlass_dir / "include" / "cutlass"),
    )
    # Notices for the code shipped with the package: the vendored kernels, the
    # CUTLASS/CuTe headers, and fmt, which is compiled into the extension.
    license_files = (
        ("LICENSE.FastAFD", ROOT / _FASTAFD_SOURCE_DIR / "LICENSE.FastAFD"),
        ("LICENSE.DeepGEMM", ROOT / _FASTAFD_SOURCE_DIR / "LICENSE.DeepGEMM"),
        ("LICENSE.CUTLASS", cutlass_dir / "LICENSE.txt"),
        ("LICENSE.fmt", fmt_dir / "LICENSE"),
    )

    class BuildFastAFDOps(BuildExtension):
        def run(self) -> None:
            super().run()
            package_root = ROOT if self.inplace else Path(self.build_lib)
            include_dir = package_root / _FASTAFD_KERNEL_ROOT / "include"
            if include_dir.exists():
                shutil.rmtree(include_dir)
            for name, source in include_trees:
                shutil.copytree(source, include_dir / name)
            license_dir = include_dir.parent / "licenses"
            license_dir.mkdir(parents=True, exist_ok=True)
            for name, source in license_files:
                shutil.copyfile(source, license_dir / name)

    return extension, BuildFastAFDOps


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str) -> None:
        super().__init__(name, sources=[])
        self.source_dir = str((ROOT / source_dir).resolve())


class BuildAscendOps(build_ext):
    def run(self) -> None:
        if not self.extensions:
            return super().run()
        if os.environ.get("AFD_SKIP_ACLNN_BUILD", "0") != "1":
            soc_version = os.environ.get("SOC_VERSION", "910c")
            subprocess.check_call(
                ["bash", "csrc/npu/build_aclnn.sh", str(ROOT), soc_version],
                cwd=ROOT,
            )
        return super().run()

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        install_prefix = Path(self.build_lib).resolve()
        cmake_args = [
            "cmake",
            ext.source_dir,
            f"-DCMAKE_BUILD_TYPE={os.environ.get('CMAKE_BUILD_TYPE', 'Release')}",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DAFD_SOC_VERSION={os.environ.get('SOC_VERSION', '910c')}",
        ]
        try:
            pybind11_cmake_dir = subprocess.check_output(
                [sys.executable, "-m", "pybind11", "--cmakedir"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("pybind11 is required to build Ascend ops") from exc
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={pybind11_cmake_dir}")
        if os.environ.get("ASCEND_HOME_PATH"):
            cmake_args.append(f"-DASCEND_HOME_PATH={os.environ['ASCEND_HOME_PATH']}")
        if os.environ.get("TORCH_NPU_PATH"):
            cmake_args.append(f"-DTORCH_NPU_PATH={os.environ['TORCH_NPU_PATH']}")

        subprocess.check_call(cmake_args, cwd=build_temp)
        jobs = os.environ.get("MAX_JOBS") or str(os.cpu_count() or 1)
        subprocess.check_call(["cmake", "--build", ".", f"-j={jobs}"], cwd=build_temp)
        subprocess.check_call(["cmake", "--install", "."], cwd=build_temp)

        src_cann_ops = ROOT / "afd_plugin" / "_cann_ops_custom"
        dst_cann_ops = Path(self.build_lib) / "afd_plugin" / "_cann_ops_custom"
        if src_cann_ops.exists():
            if dst_cann_ops.exists():
                shutil.rmtree(dst_cann_ops)
            shutil.copytree(src_cann_ops, dst_cann_ops)


ext_modules: list[Extension] = []
build_ext_command: type = BuildAscendOps
if _should_build_ascend_ops():
    ext_modules.append(
        CMakeExtension("afd_plugin._C_ascend", "csrc/npu/pybind"),
    )
if _should_build_fastafd_ops():
    if ext_modules:
        raise RuntimeError("Build Ascend ops and FastAFD ops in separate environments")
    fastafd_extension, build_ext_command = _fastafd_build(
        *(
            _fastafd_header_dependency(*dependency)
            for dependency in _FASTAFD_HEADER_DEPENDENCIES
        ),
    )
    ext_modules.append(fastafd_extension)


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_command},
)
