# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""CPU-only coverage for the ParallelConfig seam.

The patch replaces ``ParallelConfig.__post_init__`` for *every* process that
loads the plugin, not only AFD ones, so its auto-selected defaults are
user-visible behaviour. These tests pin the EPLB communicator choice against
the upstream ordering (vLLM 0.32.0, ``5df63a72``: XPU -> torch_xccl, then NIXL,
then pynccl for elastic EP, then torch_gloo).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_PATCH = "afd_plugin/compat/patches/elastic_parallel.py"


def _module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load(monkeypatch, *, is_xpu: bool, nixl: bool):
    """Load the seam with a stub vLLM, then return its __post_init__."""

    class _Platform:
        def is_xpu(self) -> bool:
            return is_xpu

        def is_cuda_alike(self) -> bool:
            return not is_xpu

        def get_device_count(self) -> int:
            return 8

        def __getattr__(self, name):  # any other probe reads as unsupported
            return lambda *a, **k: False

    class ParallelConfig:
        def __post_init__(self) -> None:  # pragma: no cover - replaced by the patch
            pass

    distributed = _module("vllm.distributed")
    distributed.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.nixl_utils",
        _module("vllm.distributed.nixl_utils", is_nixl_available=lambda: nixl),
    )
    utils = _module("vllm.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    for name, mod in {
        "vllm": _module("vllm"),
        "vllm.envs": _module("vllm.envs"),
        "vllm.config": _module("vllm.config"),
        "vllm.config.parallel": _module(
            "vllm.config.parallel",
            ParallelConfig=ParallelConfig,
            DistributedExecutorBackend=str,
        ),
        "vllm.logger": _module("vllm.logger", init_logger=logging.getLogger),
        "vllm.platforms": _module("vllm.platforms", current_platform=_Platform()),
        "vllm.utils.network_utils": _module(
            "vllm.utils.network_utils", get_open_ports_list=lambda n: list(range(n))
        ),
        "vllm.distributed": distributed,
        "vllm.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    path = Path(__file__).parents[3] / _PATCH
    spec = importlib.util.spec_from_file_location("_afd_parallel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__post_init__


def _config(*, enable_elastic_ep: bool) -> MagicMock:
    """A ParallelConfig double: numeric and boolean fields set, the rest inert."""
    self = MagicMock(name="ParallelConfig")
    self.tensor_parallel_size = 1
    self.pipeline_parallel_size = 1
    self.prefill_context_parallel_size = 1
    self.decode_context_parallel_size = 1
    self.data_parallel_size = 2
    self.data_parallel_size_local = 2
    self.data_parallel_rank = 0
    self.elastic_ep_max_dp_size = 8
    self._data_parallel_master_port_list = [1234, 1235]
    self.nnodes = 1
    self.distributed_executor_backend = "uni"
    self.data_parallel_external_lb = False
    self.data_parallel_hybrid_lb = False
    self.enable_elastic_ep = enable_elastic_ep
    self.enable_eplb = True
    self.eplb_config = SimpleNamespace(communicator=None, use_async=False)
    return self


@pytest.mark.parametrize(
    ("is_xpu", "nixl", "elastic", "expected"),
    [
        # XPU wins over NIXL: dropping this branch would move every non-AFD XPU
        # deployment off the device-native communicator.
        (True, True, False, "torch_xccl"),
        (True, False, True, "torch_xccl"),
        (False, True, False, "nixl"),
        (False, False, True, "pynccl"),
        (False, False, False, "torch_gloo"),
    ],
)
def test_eplb_communicator_default(
    monkeypatch, is_xpu: bool, nixl: bool, elastic: bool, expected: str
) -> None:
    post_init = _load(monkeypatch, is_xpu=is_xpu, nixl=nixl)
    self = _config(enable_elastic_ep=elastic)
    post_init(self)
    assert self.eplb_config.communicator == expected
