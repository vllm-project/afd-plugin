# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DBO helpers used by AFD runtime/model wrappers."""

from collections.abc import Callable

import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_yield,
)

# Resolve the Ascend yield once. This used to be imported inside the op body,
# which runs once per MoE layer: on a CUDA build the module is absent, Python
# does not cache a failed import, and so every call re-walked the import
# machinery. A profile of an Attention rank put that at 833us per call and 258ms
# of a 1403ms window -- the single largest host cost on the layer path, for an
# import that can never succeed there.
_ascend_dbo_enabled: Callable[[], bool] | None
_ascend_dbo_yield: Callable[[], None] | None
try:
    from afd_plugin.v1.worker.npu.ubatching import (
        dbo_enabled as _ascend_dbo_enabled,
    )
    from afd_plugin.v1.worker.npu.ubatching import (
        dbo_yield as _ascend_dbo_yield,
    )
except ImportError:  # not an Ascend build
    _ascend_dbo_enabled = None
    _ascend_dbo_yield = None

_AFD_DBO_YIELD_OP_REGISTERED = False


def current_dbo_ubatch_id() -> int | None:
    """Which ubatch this thread is running, or ``None`` when DBO is off.

    The forward context does not carry it. vLLM tracks the ubatch by thread
    rather than on the context, so reading ``forward_context.ubatch_idx``
    silently finds nothing and both DBO halves look like stage 0 -- and so
    claim the same connector stage, the same window slot, and the same flag,
    where the second dispatch overwrites the first and the reply the first is
    waiting for never comes.
    """
    if not dbo_enabled():
        return None
    # During Dynamo tracing this whole read is off limits: vLLM marks
    # dbo_current_ubatch_id as skipped, so tracing it raises Unsupported
    # before any real code runs. There is no ubatch to report while tracing
    # anyway -- the caller keeps the stage it already had.
    if torch.compiler.is_compiling():
        return None
    try:
        return int(dbo_current_ubatch_id())
    except KeyError:
        # This thread registered no ubatch, yet some thread did -- the model
        # is running on a side thread while the two ubatch threads hold the
        # map. There is no ubatch to report; the caller keeps the stage it
        # already had.
        return None


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
    if (
        _ascend_dbo_enabled is not None
        and _ascend_dbo_yield is not None
        and _ascend_dbo_enabled()
    ):
        _ascend_dbo_yield()
        return

    if dbo_enabled():
        dbo_yield()


__all__ = [
    "current_dbo_ubatch_id",
    "maybe_apply_dbo_yield",
    "register_dbo_yield_custom_op",
]
