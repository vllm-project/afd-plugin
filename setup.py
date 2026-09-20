from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
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


ext_modules = []
if _should_build_ascend_ops():
    ext_modules.append(
        CMakeExtension("afd_plugin._C_ascend", "csrc/npu/pybind"),
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildAscendOps},
)
