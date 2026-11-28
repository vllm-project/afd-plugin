# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Minimal native NPU worker construction for runtime tests."""

import pytest


def require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")


def new_ffn_worker():
    require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

    worker = object.__new__(AFDNPUFFNWorker)
    worker._ffn_loop_error = None
    # Most tests exercise the daemon loop rather than CPU placement. Tests for
    # the startup binding path explicitly reset this guard.
    worker._cpu_binding_attempted = True
    return worker
