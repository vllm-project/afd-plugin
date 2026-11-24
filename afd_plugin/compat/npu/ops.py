# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Loader for plugin-owned Ascend custom operators.
# Create: 2026-05-25
# Note: Async startup resolves the ACLNN execution and workspace symbols before HCCL
#       initialization.
# History: 2026-05-25 create
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Final

AFD_ASCEND_OPS_NAMESPACE = "afd_ascend"
AFD_ASCEND_VENDOR_NAME = "afd-plugin"
AFD_CUST_OPAPI_ENV = "AFD_CUST_OPAPI_LIB_PATH"
_A5_CUST_OPAPI_NAME = "libafd_cust_opapi.so"
AFD_ASYNC_OP_NAMESPACE = AFD_ASCEND_OPS_NAMESPACE
AFD_ASYNC_DISPATCH_SEND = "afd_async_dispatch_send"
AFD_ASYNC_DISPATCH_RECV = "afd_async_dispatch_recv"
AFD_ASYNC_COMBINE_SEND = "afd_async_combine_send"
AFD_ASYNC_COMBINE_RECV = "afd_async_combine_recv"


def get_afd_cann_vendor_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
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


def _assert_afd_namespace_registered(torch: object) -> None:
    ops = vars(torch)["ops"]
    _ = ops.afd_ascend.a2e
    _ = ops.afd_ascend.e2a


#: Argument count of the operator once the MXFP8 arguments are present. The transport
#: calls it positionally, so a shorter schema means those names live elsewhere.
A2E_MXFP8_ARGUMENT_COUNT: Final = 14


def a2e_accepts_mxfp8_arguments() -> bool | None:
    """Whether the loaded ``a2e`` operator can be called with the MXFP8 pair.

    Three answers rather than two, because the callers need different messages:

    ``False``
        The operators could not be loaded, or the schema does not have the expected
        argument count and trailing pair by name, type and position. This build cannot
        run the quantized transport.
    ``None``
        The operators loaded but no schema is exposed. Treating that as "unsupported"
        would hide a working build; treating it as "supported" would enable a transport
        nobody verified. The caller refuses and says it cannot confirm.
    ``True``
        The schema matches the input interface this transport calls. It says nothing
        about the **output** surface: the sixth return value cannot be seen in a schema
        whose returns are declared as ``Tensor[]``, so the receive path checks it at run
        time.

    The check reads ``a2e.default._schema.arguments`` rather than the rendered schema
    text: the text spells the optional as ``Tensor?`` while the argument's own type
    spells it ``Optional[Tensor]``, and the text does not mark keyword-only arguments.
    """

    try:
        ensure_cam_p2p_ops_available()
        import torch
    except (ImportError, OSError, RuntimeError):
        # Missing extension, missing torch, or a torch that cannot load its libraries.
        return False
    packet = getattr(getattr(torch, "ops", None), "afd_ascend", None)
    packet = getattr(packet, "a2e", None) if packet is not None else None
    if packet is None:
        return False
    default = getattr(packet, "default", None)
    schema = getattr(default, "_schema", None) if default is not None else None
    if schema is None:
        return None
    arguments = list(getattr(schema, "arguments", ()))
    if len(arguments) != A2E_MXFP8_ARGUMENT_COUNT:
        return False
    scale_argument, mode_argument = arguments[-2:]
    if scale_argument.name != "x_scale" or mode_argument.name != "quant_mode":
        return False
    if scale_argument.kwarg_only or mode_argument.kwarg_only:
        # The transport passes both positionally, so keyword-only fails the call.
        return False
    # ``List[Tensor]`` also has ``getElementType()``; only the optional wrapper lets the
    # FFN side pass ``None``.
    if not isinstance(scale_argument.type, torch.OptionalType):
        return False
    if not isinstance(scale_argument.type.getElementType(), torch.TensorType):
        return False
    return isinstance(mode_argument.type, torch.IntType)


def _assert_afd_async_namespace_registered(torch: object) -> None:
    ops = vars(torch)["ops"]
    operators = (
        ops.afd_ascend.afd_async_dispatch_send,
        ops.afd_ascend.afd_async_dispatch_recv,
        ops.afd_ascend.afd_async_combine_send,
        ops.afd_ascend.afd_async_combine_recv,
    )
    for operator in operators:
        schema = operator.default._schema
        arguments = {arg.name for arg in schema.arguments}
        if not {"comm_args", "group_name"} <= arguments:
            raise RuntimeError(
                f"{schema.name} has an incompatible ABI. CAMAsyncAFDConnector "
                "requires the compact comm_args/group_name operators."
            )


@lru_cache(maxsize=1)
def ensure_cam_p2p_ops_available() -> None:
    """Import the custom operators used by ``CAMP2pAFDConnector``.

    The extension is optional at package import time.  It is built by default
    in an Ascend environment unless ``AFD_BUILD_ASCEND_OPS=0`` is set.
    """

    ensure_afd_custom_opp_env()
    try:
        import torch

        import afd_plugin._C_ascend  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CAMP2P Ascend custom ops are not available. Build the package with "
            "Ascend ops enabled in a torch-npu/CANN environment.",
        ) from exc
    _assert_afd_namespace_registered(torch)


def ensure_afd_ascend_ops_loaded() -> None:
    """Backward-compatible alias for the CAMP2P custom-op loader."""

    ensure_cam_p2p_ops_available()


def has_afd_ascend_ops() -> bool:
    try:
        ensure_cam_p2p_ops_available()
    except RuntimeError:
        return False
    return True


@lru_cache(maxsize=1)
def ensure_afd_async_ops_available() -> None:
    """Check the compact schemas and their native entry points before CAM startup."""
    ensure_cam_p2p_ops_available()
    import torch

    try:
        _assert_afd_async_namespace_registered(torch)
    except AttributeError as exc:
        raise RuntimeError(
            "Async CAM requires all four torch.ops.afd_ascend.afd_async_* "
            "operators. Rebuild AFD with AFD_BUILD_ASCEND_OPS=1."
        ) from exc

    # The bindings open the configured vendor by path, or libopapi.so when
    # none is configured. A rebuilt extension alone cannot verify its vendor.
    path = os.environ.get(AFD_CUST_OPAPI_ENV) or "libopapi.so"
    try:
        library = ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError(
            f"Async CAM could not load the ACLNN library {path}: {exc}"
        ) from exc
    missing = [
        symbol
        for op in (
            "aclnnAfdAsyncDispatchSend",
            "aclnnAfdAsyncDispatchRecv",
            "aclnnAfdAsyncCombineSend",
            "aclnnAfdAsyncCombineRecv",
        )
        for symbol in (op, op + "GetWorkspaceSize")
        if not hasattr(library, symbol)
    ]
    if missing:
        raise RuntimeError(
            f"Async CAM ACLNN library {path} is missing {', '.join(missing)}. "
            "Rebuild the matching AFD vendor package with AFD_SKIP_ACLNN_BUILD=0."
        )


# Public spellings used by older connector integrations.
ensure_cam_async_ops_available = ensure_afd_async_ops_available
CAM_COMBINE_RECV = AFD_ASYNC_COMBINE_RECV
CAM_COMBINE_SEND = AFD_ASYNC_COMBINE_SEND
CAM_DISPATCH_RECV = AFD_ASYNC_DISPATCH_RECV
CAM_DISPATCH_SEND = AFD_ASYNC_DISPATCH_SEND
CAM_OP_NAMESPACE = AFD_ASYNC_OP_NAMESPACE


__all__ = [
    "a2e_accepts_mxfp8_arguments",
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "AFD_ASYNC_COMBINE_RECV",
    "AFD_ASYNC_COMBINE_SEND",
    "AFD_ASYNC_DISPATCH_RECV",
    "AFD_ASYNC_DISPATCH_SEND",
    "AFD_ASYNC_OP_NAMESPACE",
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_afd_ascend_ops_loaded",
    "ensure_afd_async_ops_available",
    "ensure_cam_async_ops_available",
    "ensure_cam_p2p_ops_available",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
    "prepare_afd_ascend950_vendor",
]
