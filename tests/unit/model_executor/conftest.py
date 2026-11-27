# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU construction helpers for model tests running inside an Ascend install.

vLLM-Ascend's 0.34 FusedMoE patch reads the process-wide Ascend config while
allocating and post-processing expert parameters. Production initializes that
config from the platform/worker setup; these tests build MoE layers directly
and never run it, so the factory raises "Ascend config is not initialized".
Install the Ascend defaults, scoped to this directory. Tests that exercise the
real Ascend config install their own fixtures.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest


@pytest.fixture(autouse=True)
def default_ascend_config(monkeypatch):
    if find_spec("vllm_ascend") is None:
        return
    try:
        from vllm_ascend import ascend_config
        from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
    except ImportError:
        return

    is_initialized = getattr(ascend_config, "_is_ascend_config_initialized", None)
    current = getattr(ascend_config, "_ASCEND_CONFIG", None)
    if is_initialized is not None and is_initialized(current):
        return

    # Only sparse_kv_offload_config has no default; the rest are production
    # defaults, so tests read the same switches the runtime would.
    config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig())
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", config, raising=False)
