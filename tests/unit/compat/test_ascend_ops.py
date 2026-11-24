# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Test the Ascend custom operator loader.
# Create: 2026-05-25
# Note: Tests cover vendor search order, compact schemas and missing native symbols.
# History: 2026-05-25 create

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import afd_plugin
from afd_plugin.compat.npu import ops
from afd_plugin.compat.npu.ops import (
    ensure_afd_async_ops_available,
    ensure_cam_p2p_ops_available,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
    has_afd_ascend_ops,
)


def test_ascend_ops_loader_fails_clearly_without_extension():
    if has_afd_ascend_ops():
        pytest.skip("AFD Ascend extension is installed in this environment")

    with pytest.raises(RuntimeError, match="CAMP2P Ascend custom ops"):
        ensure_cam_p2p_ops_available()


def test_ascend_ops_paths_are_plugin_owned():
    vendor_path = get_afd_cann_vendor_path()
    cust_opapi_path = get_afd_cust_opapi_path()

    assert vendor_path == (
        Path(afd_plugin.__file__).resolve().parent
        / "_cann_ops_custom/vendors/afd-plugin"
    )
    assert cust_opapi_path.parent == vendor_path / "op_api" / "lib"
    assert cust_opapi_path.name in {"libcust_opapi.so", "libafd_cust_opapi.so"}


@pytest.mark.parametrize("private_a5_library", [False, True])
def test_ascend_ops_loader_sets_explicit_cust_opapi_path(
    tmp_path, monkeypatch, private_a5_library
):
    vendor_path = (
        tmp_path / "afd_plugin" / "_cann_ops_custom" / "vendors" / "afd-plugin"
    )
    op_api_lib = vendor_path / "op_api" / "lib"
    op_api_lib.mkdir(parents=True)
    cust_opapi_path = op_api_lib / "libcust_opapi.so"
    cust_opapi_path.touch()
    if private_a5_library:
        cust_opapi_path = op_api_lib / "libafd_cust_opapi.so"
        cust_opapi_path.touch()
    monkeypatch.setattr(ops, "get_afd_cann_vendor_path", lambda: vendor_path)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/native/vendor")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/native/lib")
    monkeypatch.delenv("AFD_CUST_OPAPI_LIB_PATH", raising=False)

    ops.ensure_afd_custom_opp_env()
    ops.ensure_afd_custom_opp_env()

    if private_a5_library:
        assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [
            "/native/vendor",
            str(vendor_path),
        ]
        assert os.environ["LD_LIBRARY_PATH"] == "/native/lib"
    else:
        assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep) == [
            str(vendor_path),
            "/native/vendor",
        ]
        assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [
            str(op_api_lib),
            "/native/lib",
        ]
    assert get_afd_cust_opapi_path() == cust_opapi_path
    assert os.environ["AFD_CUST_OPAPI_LIB_PATH"] == str(cust_opapi_path)


@pytest.fixture
def compact_dispatcher(monkeypatch):
    names = (
        "afd_async_dispatch_send",
        "afd_async_dispatch_recv",
        "afd_async_combine_send",
        "afd_async_combine_recv",
    )
    installed = {}
    for name in names:
        schema = SimpleNamespace(
            name=f"afd_ascend::{name}",
            arguments=[SimpleNamespace(name=a) for a in ("comm_args", "group_name")],
        )
        installed[name] = SimpleNamespace(default=SimpleNamespace(_schema=schema))
    namespace = SimpleNamespace(**installed)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(afd_ascend=namespace),
        ),
    )
    monkeypatch.setattr(ops, "ensure_cam_p2p_ops_available", lambda: None)
    ensure_afd_async_ops_available.cache_clear()
    yield namespace
    ensure_afd_async_ops_available.cache_clear()


@pytest.mark.parametrize(
    "missing",
    [
        "afd_async_dispatch_send",
        "afd_async_dispatch_recv",
        "afd_async_combine_send",
        "afd_async_combine_recv",
    ],
)
def test_async_loader_requires_all_source_ops(compact_dispatcher, missing):
    delattr(compact_dispatcher, missing)
    with pytest.raises(RuntimeError, match="afd_ascend"):
        ensure_afd_async_ops_available()


def test_async_loader_rejects_registered_buffer_abi(compact_dispatcher):
    schema = compact_dispatcher.afd_async_combine_recv.default._schema
    schema.arguments = [SimpleNamespace(name=a) for a in ("context", "group_ep")]
    with pytest.raises(RuntimeError, match="compact comm_args/group_name"):
        ensure_afd_async_ops_available()


_CAM_EXPORTS = tuple(
    f"aclnnAfdAsync{op}{suffix}"
    for op in ("DispatchSend", "DispatchRecv", "CombineSend", "CombineRecv")
    for suffix in ("", "GetWorkspaceSize")
)


@pytest.mark.parametrize("missing", [None, *_CAM_EXPORTS])
def test_async_loader_requires_each_vendor_export(
    compact_dispatcher, monkeypatch, tmp_path, missing
):
    path = str(tmp_path / "libafd_cust_opapi.so")
    monkeypatch.setenv("AFD_CUST_OPAPI_LIB_PATH", path)
    library = SimpleNamespace(
        **{name: object() for name in _CAM_EXPORTS if name != missing}
    )
    monkeypatch.setattr(ctypes, "CDLL", {path: library}.__getitem__)
    if missing is None:
        ensure_afd_async_ops_available()
    else:
        with pytest.raises(RuntimeError, match=missing + r"\b") as error:
            ensure_afd_async_ops_available()
        assert path in str(error.value)


def test_async_loader_reports_unloadable_vendor(
    compact_dispatcher, monkeypatch, tmp_path
):
    path = str(tmp_path / "missing-opapi.so")
    monkeypatch.setenv("AFD_CUST_OPAPI_LIB_PATH", path)
    with pytest.raises(RuntimeError) as error:
        ensure_afd_async_ops_available()
    assert path in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
