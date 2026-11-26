# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Select an implemented adapter without importing its tensor/runtime stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.gpu import GPU_DEEPEP_M2N_BACKEND

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNPlatformAdapter

ASCEND_CAM_BACKEND = "ascend-cam"
SUPPORTED_BACKENDS = (ASCEND_CAM_BACKEND, GPU_DEEPEP_M2N_BACKEND)


def get_platform_adapter(name: str) -> FFNPlatformAdapter:
    """Import and construct the CPU-only adapter of one backend by name.

    Each adapter is imported only when selected, so choosing one platform
    never imports another platform's modules.
    """
    if name == ASCEND_CAM_BACKEND:
        from afd_plugin.ffn_faas.platforms.npu.adapter import AscendCAMAdapter

        return AscendCAMAdapter()
    if name == GPU_DEEPEP_M2N_BACKEND:
        from afd_plugin.ffn_faas.platforms.gpu.adapter import GPUDeepEPM2NAdapter

        return GPUDeepEPM2NAdapter()
    raise ValueError(f"unsupported standalone FFN backend: {name}")
