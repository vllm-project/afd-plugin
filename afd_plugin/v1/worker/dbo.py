# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.ubatching import dbo_enabled, dbo_yield

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


def register_dbo_yield_custom_op() -> None:
    global _AFD_DBO_YIELD_OP_REGISTERED

    if _AFD_DBO_YIELD_OP_REGISTERED:
        return

    def afd_manual_dbo_yield_op(x: torch.Tensor) -> None:
        _yield_if_dbo_enabled()

    def afd_manual_dbo_yield_fake(x: torch.Tensor) -> None:
        return None

    try:
        direct_register_custom_op(
            op_name="manual_dbo_yield",
            op_func=afd_manual_dbo_yield_op,
            fake_impl=afd_manual_dbo_yield_fake,
            mutates_args=["x"],
        )
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise
    _AFD_DBO_YIELD_OP_REGISTERED = True


def _yield_if_dbo_enabled() -> None:
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


__all__ = ["maybe_apply_dbo_yield", "register_dbo_yield_custom_op"]
