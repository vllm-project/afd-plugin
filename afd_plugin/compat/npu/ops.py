# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

AFD_ASCEND_OPS_NAMESPACE = "afd_ascend"
AFD_ASCEND_VENDOR_NAME = "afd-plugin"
AFD_CUST_OPAPI_ENV = "AFD_CUST_OPAPI_LIB_PATH"
CAM_OP_NAMESPACE = AFD_ASCEND_OPS_NAMESPACE
CAM_DISPATCH_SEND = "afd_async_dispatch_send"
CAM_DISPATCH_RECV = "afd_async_dispatch_recv"
CAM_COMBINE_SEND = "afd_async_combine_send"
CAM_COMBINE_RECV = "afd_async_combine_recv"


def get_afd_cann_vendor_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "_cann_ops_custom"
        / "vendors"
        / AFD_ASCEND_VENDOR_NAME
    )


def get_afd_cust_opapi_path() -> Path:
    return get_afd_cann_vendor_path() / "op_api" / "lib" / "libcust_opapi.so"


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(":") if entry]
    if path_str in entries:
        return
    os.environ[name] = ":".join([path_str, *entries])


def _ensure_afd_custom_opp_env() -> None:
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


#: Trailing ``a2e`` arguments the MXFP8 transport needs, in order. A build that predates
#: them still accepts the call shape at the Python level and fails inside the kernel, so
#: the connector reads the schema instead of trusting the call to raise.
A2E_MXFP8_ARGUMENT_NAMES: Final = ("x_scale", "quant_mode")
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


def _assert_cam_namespace_registered(torch: object) -> None:
    ops = vars(torch)["ops"]
    _ = (
        ops.afd_ascend.afd_async_dispatch_send,
        ops.afd_ascend.afd_async_dispatch_recv,
        ops.afd_ascend.afd_async_combine_send,
        ops.afd_ascend.afd_async_combine_recv,
    )


@lru_cache(maxsize=1)
def ensure_cam_p2p_ops_available() -> None:
    """Import the custom operators used by ``CAMP2pAFDConnector``.

    The extension is optional at package import time.  It is built by default
    in an Ascend environment unless ``AFD_BUILD_ASCEND_OPS=0`` is set.
    """

    _ensure_afd_custom_opp_env()
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
def ensure_cam_async_ops_available() -> None:
    """Load the source-built extension and require its routed-only protocol."""
    ensure_cam_p2p_ops_available()
    import torch

    try:
        _assert_cam_namespace_registered(torch)
    except AttributeError as exc:
        raise RuntimeError(
            "Async CAM requires all four torch.ops.afd_ascend.afd_async_* "
            "operators. Rebuild AFD with AFD_BUILD_ASCEND_OPS=1."
        ) from exc


__all__ = [
    "a2e_accepts_mxfp8_arguments",
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_afd_ascend_ops_loaded",
    "ensure_cam_async_ops_available",
    "ensure_cam_p2p_ops_available",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
]
