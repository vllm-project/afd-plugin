from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import setuptools

_ASCEND_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "TORCH_NPU_PATH",
)


def _run_setup_py(
    monkeypatch: pytest.MonkeyPatch,
    *,
    afd_build_ascend_ops: str | None = None,
    has_torch_npu: bool = False,
    has_ascend_toolkit: bool = False,
    ascend_env_var: str | None = None,
) -> list[str]:
    root = Path(__file__).resolve().parents[3]
    captured: dict[str, object] = {}

    def fake_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object | None:
        if name == "torch_npu":
            return object() if has_torch_npu else None
        return real_find_spec(name, *args, **kwargs)

    real_path_exists = Path.exists

    def fake_path_exists(path: Path) -> bool:
        if path.as_posix() == "/usr/local/Ascend/ascend-toolkit/latest":
            return has_ascend_toolkit
        return real_path_exists(path)

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(Path, "exists", fake_path_exists)
    monkeypatch.delenv("AFD_BUILD_ASCEND_OPS", raising=False)
    for name in _ASCEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    if afd_build_ascend_ops is not None:
        monkeypatch.setenv("AFD_BUILD_ASCEND_OPS", afd_build_ascend_ops)
    if ascend_env_var is not None:
        monkeypatch.setenv(ascend_env_var, "/opt/ascend")

    runpy.run_path(str(root / "setup.py"))

    ext_modules = captured["ext_modules"]
    return [ext.name for ext in ext_modules]  # type: ignore[attr-defined]


def test_ascend_a2e_e2a_sources_are_vendored():
    root = Path(__file__).resolve().parents[3]
    required = [
        # ACLNN operator run package (npu_op_* build system).
        "csrc/npu/ascend_kernels/CMakeLists.txt",
        "csrc/npu/ascend_kernels/AddCustom.json",
        "csrc/npu/ascend_kernels/operator_registry.json",
        "csrc/npu/ascend_kernels/cmake_files/cmake/func.cmake",
        "csrc/npu/ascend_kernels/cmake_files/op_host/CMakeLists.txt",
        "csrc/npu/ascend_kernels/cmake_files/op_kernel/CMakeLists.txt",
        "csrc/npu/ascend_kernels/a2e/op_api/aclnn_a2e.cpp",
        "csrc/npu/ascend_kernels/a2e/op_host/a2e.cpp",
        "csrc/npu/ascend_kernels/a2e/op_kernel/a2e.cpp",
        "csrc/npu/ascend_kernels/e2a/op_api/aclnn_e2a.cpp",
        "csrc/npu/ascend_kernels/e2a/op_host/e2a.cpp",
        "csrc/npu/ascend_kernels/e2a/op_kernel/e2a.cpp",
        # Shared headers, deduplicated out of the per-operator directories.
        "csrc/npu/ascend_kernels/utils/op_kernel/comm_args.h",
        "csrc/npu/ascend_kernels/utils/op_kernel/data_copy.h",
        "csrc/npu/ascend_kernels/utils/op_kernel/moe_distribute_base.h",
        # Build drivers.
        "csrc/npu/build_aclnn.sh",
        "csrc/npu/scripts/compile_ascend_proj.sh",
        "csrc/npu/scripts/select_ops.py",
        "csrc/npu/scripts/set_conf.py",
        # PyTorch extension.
        "csrc/npu/pybind/CMakeLists.txt",
        "csrc/npu/pybind/torch_binding.cpp",
        "csrc/npu/pybind/torch_binding_meta.cpp",
    ]

    for relpath in required:
        assert (root / relpath).is_file(), relpath


def test_ascend_shared_kernel_headers_are_not_duplicated():
    """comm_args.h/data_copy.h/moe_distribute_base.h live only under utils/."""
    root = Path(__file__).resolve().parents[3]
    shared = ("comm_args.h", "data_copy.h", "moe_distribute_base.h")

    for op in ("a2e", "e2a"):
        for name in shared:
            stale = root / "csrc/npu/ascend_kernels" / op / "op_kernel" / name
            assert not stale.exists(), stale


def test_ascend_ops_build_is_disabled_by_default_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _run_setup_py(monkeypatch) == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"has_torch_npu": True}, ["afd_plugin._C_ascend"]),
        ({"has_ascend_toolkit": True}, ["afd_plugin._C_ascend"]),
        ({"ascend_env_var": "ASCEND_HOME_PATH"}, ["afd_plugin._C_ascend"]),
    ],
)
def test_ascend_ops_build_is_enabled_by_default_on_npu(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    expected: list[str],
):
    assert _run_setup_py(monkeypatch, **kwargs) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", ["afd_plugin._C_ascend"]),
        ("true", ["afd_plugin._C_ascend"]),
        (" yes ", ["afd_plugin._C_ascend"]),
        ("yes", ["afd_plugin._C_ascend"]),
        ("on", ["afd_plugin._C_ascend"]),
        ("0", []),
        ("false", []),
        (" no ", []),
        ("no", []),
        ("off", []),
    ],
)
def test_ascend_ops_build_env_overrides_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: list[str],
):
    assert (
        _run_setup_py(
            monkeypatch,
            afd_build_ascend_ops=value,
            has_torch_npu=value in {"0", "false", "no", "off"},
        )
        == expected
    )


def test_ascend_ops_build_env_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(RuntimeError, match="AFD_BUILD_ASCEND_OPS"):
        _run_setup_py(monkeypatch, afd_build_ascend_ops="maybe")


def test_empty_ascend_ops_build_env_uses_platform_default(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _run_setup_py(monkeypatch, afd_build_ascend_ops="") == []
    assert _run_setup_py(
        monkeypatch,
        afd_build_ascend_ops=" ",
        has_torch_npu=True,
    ) == ["afd_plugin._C_ascend"]


def test_ascend_ops_use_isolated_namespace_and_vendor_path():
    root = Path(__file__).resolve().parents[3]
    torch_binding = (root / "csrc/npu/pybind/torch_binding.cpp").read_text()
    torch_binding_meta = (root / "csrc/npu/pybind/torch_binding_meta.cpp").read_text()
    torch_cmake = (root / "csrc/npu/pybind/CMakeLists.txt").read_text()
    gen_script = (root / "csrc/npu/scripts/compile_ascend_proj.sh").read_text()
    op_api_common = (
        root / "csrc/npu/pybind/pytorch_extension/op_api_common.h"
    ).read_text()

    assert "TORCH_LIBRARY(afd_ascend" in torch_binding
    assert "TORCH_LIBRARY(_C_ascend" not in torch_binding
    assert "TORCH_LIBRARY_IMPL(afd_ascend, Meta" in torch_binding_meta
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta" not in torch_binding_meta
    assert "vendors/afd-plugin/op_api/lib" in torch_cmake
    assert "vendors/vllm-ascend/op_api/lib" not in torch_cmake
    # The vendor name is now passed to the generated CMakePresets.json by the
    # build driver instead of a checked-in CMakeLists.txt cache variable.
    assert "afd-plugin" in gen_script
    assert "vllm-ascend" not in gen_script
    assert "AFD_CUST_OPAPI_LIB_PATH" in op_api_common
    assert 'return "libcust_opapi.so"' not in op_api_common


def test_ascend_operator_registry_covers_both_soc_generations():
    """select_ops.py resolves operators from the registry, not shell logic."""
    root = Path(__file__).resolve().parents[3]
    registry = json.loads(
        (root / "csrc/npu/ascend_kernels/operator_registry.json").read_text()
    )

    assert set(registry["soc_versions"]) == {"ascend910_93", "ascend950"}
    for soc, ops in registry["soc_versions"].items():
        assert ops, soc
        for op in ops:
            op_dir = root / "csrc/npu/ascend_kernels" / op
            assert (op_dir / "op_host").is_dir(), f"{soc}:{op} op_host"
            assert (op_dir / "op_kernel").is_dir(), f"{soc}:{op} op_kernel"
            # Every registered operator must carry build metadata.
            assert op in registry["operator_meta"], f"{soc}:{op} missing meta"

    assert "utils" not in {
        op for ops in registry["soc_versions"].values() for op in ops
    }, "utils is a shared header directory and must not be a registry operator"


def _load_select_ops():
    """Import csrc/npu/scripts/select_ops.py, which is not an importable package."""
    root = Path(__file__).resolve().parents[3]
    path = root / "csrc/npu/scripts/select_ops.py"
    spec = importlib.util.spec_from_file_location("afd_select_ops", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_select_ops(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable,
            str(root / "csrc/npu/scripts/select_ops.py"),
            "--registry",
            str(root / "csrc/npu/ascend_kernels/operator_registry.json"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_select_ops_resolves_registry_order():
    result = _run_select_ops("--soc", "ascend910_93", "--shmem", "0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["a2e", "e2a"]


def test_select_ops_accepts_explicit_operator_subset():
    result = _run_select_ops("--soc", "ascend950", "--shmem", "0", "--ops", "e2a")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["e2a"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--soc", "ascend310p", "--shmem", "0"), "not registered"),
        (("--soc", "ascend910_93", "--shmem", "0", "--ops", "nope"), "not in SOC"),
        (("--soc", "ascend910_93", "--shmem", "0", "--ops", "a2e;"), "empty entry"),
    ],
)
def test_select_ops_rejects_invalid_selection(args: tuple[str, ...], expected: str):
    """Registry validation replaces the previous unvalidated shell selection."""
    result = _run_select_ops(*args)

    assert result.returncode != 0
    assert expected in result.stderr


def test_select_ops_drops_shmem_operators_when_shmem_missing():
    select_ops = _load_select_ops()
    registry = {
        "soc_versions": {"ascend910_93": ["a2e", "e2a"]},
        "operator_meta": {"a2e": {"requires_shmem": True}, "e2a": {}},
    }

    assert select_ops.resolve(registry, ["a2e", "e2a"], False, None) == ["e2a"]
    assert select_ops.resolve(registry, ["a2e", "e2a"], True, None) == ["a2e", "e2a"]
