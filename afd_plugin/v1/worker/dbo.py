# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from collections.abc import Callable

import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_switch_to_comm_sync,
    dbo_yield,
    dbo_yield_and_switch_from_comm_to_compute,
)

_AFD_DBO_YIELD_OP_REGISTERED = False


def maybe_apply_dbo_yield(
    tensor: torch.Tensor,
    *,
    role: str,
) -> torch.Tensor:
    """Yield to the peer ubatch thread when vLLM DBO is active."""
    try:
        register_dbo_yield_custom_op()
    except ImportError:
        return tensor

    torch.ops.vllm.manual_dbo_yield(tensor)
    return tensor


def begin_gpu_dbo_transfer(tensor: torch.Tensor) -> None:
    """Queue the FFN round trip on DBO's communication stream."""
    register_dbo_yield_custom_op()
    torch.ops.vllm.afd_dbo_transfer_begin(tensor)


def end_gpu_dbo_transfer(tensor: torch.Tensor) -> None:
    """Overlap the peer's compute, then wait for this FFN result."""
    torch.ops.vllm.afd_dbo_transfer_end(tensor)


def register_dbo_yield_custom_op() -> None:
    global _AFD_DBO_YIELD_OP_REGISTERED

    if _AFD_DBO_YIELD_OP_REGISTERED:
        return

    def afd_manual_dbo_yield_op(x: torch.Tensor) -> None:
        _yield_if_dbo_enabled()

    def afd_manual_dbo_yield_fake(x: torch.Tensor) -> None:
        return None

    def afd_dbo_transfer_begin(x: torch.Tensor) -> None:
        dbo_switch_to_comm_sync()

    def afd_dbo_transfer_end(x: torch.Tensor) -> None:
        dbo_yield_and_switch_from_comm_to_compute()

    for name, implementation in (
        ("manual_dbo_yield", afd_manual_dbo_yield_op),
        ("afd_dbo_transfer_begin", afd_dbo_transfer_begin),
        ("afd_dbo_transfer_end", afd_dbo_transfer_end),
    ):
        try:
            direct_register_custom_op(
                op_name=name,
                op_func=implementation,
                fake_impl=afd_manual_dbo_yield_fake,
                mutates_args=["x"],
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
    _AFD_DBO_YIELD_OP_REGISTERED = True


def _yield_if_dbo_enabled() -> None:
    ascend_dbo_enabled: Callable[[], bool] | None
    ascend_dbo_yield: Callable[[], None] | None
    try:
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_enabled as ascend_dbo_enabled,
        )
        from afd_plugin.v1.worker.npu.ubatching import (
            dbo_yield as ascend_dbo_yield,
        )
    except ImportError:
        ascend_dbo_enabled = None
        ascend_dbo_yield = None

    if (
        ascend_dbo_enabled is not None
        and ascend_dbo_yield is not None
        and ascend_dbo_enabled()
    ):
        ascend_dbo_yield()
        return

    if dbo_enabled():
        dbo_yield()


__all__ = [
    "begin_gpu_dbo_transfer",
    "end_gpu_dbo_transfer",
    "maybe_apply_dbo_yield",
    "register_dbo_yield_custom_op",
]
