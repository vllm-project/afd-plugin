# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Publish the packaged CANN vendor before importing accelerator runtimes."""

from __future__ import annotations

import os
from pathlib import Path

AFD_ASCEND_VENDOR_NAME = "afd-plugin"
AFD_CUST_OPAPI_ENV = "AFD_CUST_OPAPI_LIB_PATH"
_A5_CUST_OPAPI_NAME = "libafd_cust_opapi.so"


def get_afd_cann_vendor_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "_cann_ops_custom"
        / "vendors"
        / AFD_ASCEND_VENDOR_NAME
    )


def get_afd_cust_opapi_path() -> Path:
    lib_dir = get_afd_cann_vendor_path() / "op_api" / "lib"
    private_api = lib_dir / _A5_CUST_OPAPI_NAME
    return private_api if private_api.is_file() else lib_dir / "libcust_opapi.so"


def prepare_afd_ascend950_vendor() -> bool:
    """Publish the A5 wheel's private API before Ascend loads custom operators."""
    library = get_afd_cust_opapi_path()
    if library.name != _A5_CUST_OPAPI_NAME:
        return False
    vendor = str(get_afd_cann_vendor_path())
    # Native vendors provide operators such as SwigluGroupQuant and must keep
    # their lookup priority. The AFD dispatcher opens its own API by full path.
    entries = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    entries = [entry for entry in entries if entry and entry != vendor]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([*entries, vendor])
    os.environ[AFD_CUST_OPAPI_ENV] = str(library)
    return True


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if path_str in entries:
        return
    os.environ[name] = os.pathsep.join([path_str, *entries])


def ensure_afd_custom_opp_env() -> None:
    if prepare_afd_ascend950_vendor():
        return
    vendor_dir = get_afd_cann_vendor_path()
    if not vendor_dir.exists():
        return

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    op_api_lib = vendor_dir / "op_api" / "lib"
    if op_api_lib.exists():
        _prepend_env_path("LD_LIBRARY_PATH", op_api_lib)
    cust_opapi = get_afd_cust_opapi_path()
    if cust_opapi.exists():
        os.environ[AFD_CUST_OPAPI_ENV] = str(cust_opapi)
