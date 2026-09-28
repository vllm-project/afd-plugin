# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Install the A5 MXFP dispatcher patch after an Ascend worker is imported.

vLLM loads general plugins from ``WorkerWrapperBase.init_worker`` immediately
before resolving the configured worker class. Hooking that resolver lets the
AFD plugin wait until vLLM-Ascend's worker and ops packages have completed
their imports, while still patching the dispatcher before model loading.
"""

from __future__ import annotations

import importlib
from typing import Any

worker_base_module = importlib.import_module("vllm.v1.worker.worker_base")

_WORKER_RESOLVER_PATCH_ATTR = "_afd_plugin_mxfp_worker_resolver_patch_state"
_TARGET_NPU_WORKERS = frozenset(
    {
        "vllm_ascend.worker.worker.NPUWorker",
        "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker",
        "afd_plugin.v1.worker.npu.AFDNPUFFNWorker",
    }
)


def apply_afd_mxfp_worker_resolution_patch() -> bool:
    """Apply the MXFP patch lazily for standard and AFD Ascend workers.

    The resolver wrapper preserves the exact upstream one-argument contract.
    Non-Ascend workers delegate without importing vLLM-Ascend. The dispatcher
    patch itself checks the pinned 11ee4565 method signature, so other Ascend
    revisions are left unchanged.
    """

    if hasattr(worker_base_module, _WORKER_RESOLVER_PATCH_ATTR):
        return True

    original_resolver = worker_base_module.resolve_obj_by_qualname

    # Patch reason: importing the Ascend MoE dispatcher directly from the
    # general-plugin hook can observe DeviceOperator during a partial import,
    # while installing only in AFDNPUFFNWorker misses standard Prefill workers.
    # Patch functionality: delegates worker resolution first, then installs the
    # signature-gated dispatcher patch for standard and AFD Ascend workers.
    # Signature: matches the worker_base module's imported resolver; no added
    # parameters.
    def resolve_obj_by_qualname(qualname: str) -> Any:
        resolved = original_resolver(qualname)
        # ### PATCH START: lazy A5 MXFP AllToAllV compatibility
        if qualname in _TARGET_NPU_WORKERS:
            from afd_plugin.compat.patches.npu.mxfp_alltoallv import (
                apply_afd_mxfp_alltoallv_patch,
            )

            apply_afd_mxfp_alltoallv_patch()
        # ### PATCH END: lazy A5 MXFP AllToAllV compatibility
        return resolved

    worker_base_module.resolve_obj_by_qualname = resolve_obj_by_qualname
    setattr(
        worker_base_module,
        _WORKER_RESOLVER_PATCH_ATTR,
        original_resolver,
    )
    return True


__all__ = ["apply_afd_mxfp_worker_resolution_patch"]
