# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A CUDA-like vLLM platform for device-selection tests on hosts without GPUs."""

from __future__ import annotations

import pytest


def use_cuda_like_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map device ids like vLLM's CUDA platform: no assigned physical ids and
    no ``CUDA_VISIBLE_DEVICES`` until a test sets them."""
    # Deferred: this fakes module must import on CPU-only hosts without vLLM.
    from vllm.platforms import interface

    class CudaLikePlatform(interface.Platform):
        device_control_env_var = "CUDA_VISIBLE_DEVICES"

    monkeypatch.setattr("vllm.platforms.current_platform", CudaLikePlatform)
    # vLLM keeps the assigned ids process-wide; start every case unset.
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
