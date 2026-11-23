# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from afd_plugin import _cann_vendor
from afd_plugin.compat.npu import ops
from afd_plugin.compat.npu.ops import (
    ensure_cam_async_ops_available,
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

    assert vendor_path.parts[-3:] == ("_cann_ops_custom", "vendors", "afd-plugin")
    assert "vllm-ascend" not in vendor_path.parts
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
    monkeypatch.setattr(_cann_vendor, "get_afd_cann_vendor_path", lambda: vendor_path)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/native/vendor")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/native/lib")
    monkeypatch.delenv("AFD_CUST_OPAPI_LIB_PATH", raising=False)

    _cann_vendor.ensure_afd_custom_opp_env()
    _cann_vendor.ensure_afd_custom_opp_env()

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


@pytest.mark.parametrize(
    "missing",
    [
        "afd_async_dispatch_send",
        "afd_async_dispatch_recv",
        "afd_async_combine_send",
        "afd_async_combine_recv",
        "registered_buffer_abi",
        None,
    ],
)
def test_async_loader_requires_all_source_ops(monkeypatch, missing):
    names = (
        "afd_async_dispatch_send",
        "afd_async_dispatch_recv",
        "afd_async_combine_send",
        "afd_async_combine_recv",
    )
    installed = {}
    for name in names:
        if name == missing:
            continue
        arguments = ("comm_args", "group_name")
        if missing == "registered_buffer_abi" and name == "afd_async_combine_recv":
            arguments = ("context", "group_ep")
        schema = SimpleNamespace(
            name=f"afd_ascend::{name}",
            arguments=[SimpleNamespace(name=argument) for argument in arguments],
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
    ensure_cam_async_ops_available.cache_clear()
    try:
        if missing == "registered_buffer_abi":
            with pytest.raises(RuntimeError, match="compact comm_args/group_name"):
                ensure_cam_async_ops_available()
        elif missing:
            with pytest.raises(RuntimeError, match="afd_ascend"):
                ensure_cam_async_ops_available()
        else:
            ensure_cam_async_ops_available()
    finally:
        ensure_cam_async_ops_available.cache_clear()
