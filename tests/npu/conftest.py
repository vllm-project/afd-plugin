# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Optional runtime setup shared by the compiled Ascend dispatcher tests.
# Create: 2026-11-05
# Note: The 950 fixture selects the compact dispatcher tests for that build.
# History: 2026-11-05 create
"""Optional runtime setup shared by the compiled Ascend dispatcher tests."""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded


def _load_compiled_ops() -> ModuleType:
    for module in ("torch", "torch_npu", "afd_plugin._C_ascend"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for compiled Ascend dispatcher tests")
    # Import only when a dispatcher test requests this fixture; host-side
    # harness tests and CPU collection must not load the optional runtime.
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    ensure_afd_ascend_ops_loaded()
    return torch


@pytest.fixture(scope="module")
def ascend_ops_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc != "910c" and not soc.startswith("ascend910_93"):
        pytest.skip("Set SOC_VERSION=910c to test the compiled 910C operators")
    return _load_compiled_ops()


@pytest.fixture(scope="module")
def afd_async_ops_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc not in {"910c", "950"} and not soc.startswith(("ascend910_93", "ascend950")):
        pytest.skip("Set SOC_VERSION to 910c or ascend950 to test compact CAM")
    return _load_compiled_ops()
