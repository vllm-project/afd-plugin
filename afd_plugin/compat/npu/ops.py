# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

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


@lru_cache(maxsize=1)
def ensure_mega_ops_available() -> torch.library.Library:
    """Load Mega and retain its torch registrations for the process lifetime."""
    _ensure_afd_custom_opp_env()
    if not get_afd_cust_opapi_path().is_file():
        raise RuntimeError(
            "AFD Mega custom operators are missing. Build with "
            "AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=950 in a torch-npu/CANN environment."
        )
    try:
        import torch_npu  # noqa: F401

        import torch

        from afd_plugin import _C_afd_async_mega
    except ImportError as exc:
        raise RuntimeError(
            "AFD Mega bindings are unavailable. Rebuild with "
            "AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=950."
        ) from exc

    # CAM registers its operators in C++ via TORCH_LIBRARY macros; the CANN-derived
    # Mega bindings only export pybind functions, so register their schemas,
    # Meta and PrivateUse1 impls here. FRAGMENT coexists with CAM; the cached
    # return value keeps the Library alive so the operators stay registered.
    library = torch.library.Library(AFD_ASCEND_OPS_NAMESPACE, "FRAGMENT")

    # Mega schema/Meta adapters below retain their source license:
    # Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.

    library.define('afd_async_atten_send(Tensor(a!) context, Tensor x, Tensor topk_ids, Tensor topk_weights, int[] configuration, int[] dp_ranks, float[] activation_parameters, int layer_index=0) -> (Tensor, Tensor, Tensor)')

    def atten_send_meta(context, x, topk_ids, topk_weights, configuration, dp_ranks, activation_parameters, layer_index=0):
        tokens = configuration[13] if configuration[12] else 0
        return (context.new_empty((tokens, configuration[4]), dtype=torch.uint8),
                context.new_empty((tokens, (configuration[4] + 63) // 64 * 2), dtype=torch.uint8), context.new_empty((1,), dtype=torch.int32))

    library.impl("afd_async_atten_send", atten_send_meta, "Meta")
    library.impl("afd_async_atten_send", _C_afd_async_mega.afd_async_atten_send, "PrivateUse1")

    library.define('afd_async_atten_recv(Tensor(a!) context, Tensor topk_ids, Tensor topk_weights, Tensor snapshot_data, Tensor snapshot_scales, Tensor[] shared_weight1, Tensor[] shared_weight2, Tensor[] shared_scales1, Tensor[] shared_scales2, int[] configuration, int[] dp_ranks, float[] activation_parameters) -> (Tensor, Tensor)')

    def atten_recv_meta(context, topk_ids, topk_weights, snapshot_data, snapshot_scales, shared_weight1, shared_weight2, shared_scales1, shared_scales2, configuration, dp_ranks, activation_parameters):
        return (context.new_empty((configuration[13], configuration[4]), dtype=torch.bfloat16),
                context.new_empty((1,), dtype=torch.int32))

    library.impl("afd_async_atten_recv", atten_recv_meta, "Meta")
    library.impl("afd_async_atten_recv", _C_afd_async_mega.afd_async_atten_recv, "PrivateUse1")

    library.define('afd_async_mega_ffn(Tensor(a!) context, Tensor[] weight1, Tensor[] weight2, Tensor[] weight_scales1, Tensor[] weight_scales2, int[] configuration, float[] activation_parameters) -> Tensor')

    def mega_ffn_meta(context, weight1, weight2, weight_scales1, weight_scales2, configuration, activation_parameters):
        return context.new_empty((configuration[2] // configuration[1],), dtype=torch.int32)

    library.impl("afd_async_mega_ffn", mega_ffn_meta, "Meta")
    library.impl("afd_async_mega_ffn", _C_afd_async_mega.afd_async_mega_ffn, "PrivateUse1")

    return library


__all__ = [
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
    "ensure_mega_ops_available",
    "ensure_cam_p2p_ops_available",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
]
