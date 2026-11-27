# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Platform helpers for host-side vLLM tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def use_cpu_moe_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build CPU test runners with vLLM's factory even after Ascend patches it."""
    ascend_patch = sys.modules.get("vllm_ascend.patch.platform.patch_fused_moe")
    if ascend_patch is not None:
        from afd_plugin.model_executor.ffn import remote

        factory = SimpleNamespace(FusedMoEFactory=ascend_patch._original_FusedMoE)
        monkeypatch.setattr(remote, "fused_moe", factory)


def cuda_like_platform(monkeypatch: pytest.MonkeyPatch) -> type:
    """A platform that maps device ids like vLLM's CUDA platform, starting with
    no assigned physical ids and no ``CUDA_VISIBLE_DEVICES``."""
    # Deferred: this fakes module must import on CPU-only hosts without vLLM.
    from vllm.platforms import interface

    class CudaLikePlatform(interface.Platform):
        device_control_env_var = "CUDA_VISIBLE_DEVICES"

    # vLLM keeps the assigned ids process-wide; start every case unset.
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return CudaLikePlatform
