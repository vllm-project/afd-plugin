# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import importlib.util
import logging
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _config(*, connector: str = "CAMAsyncAFDConnector", async_dp: bool = True):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": connector,
                "role": "attention",
                "async": async_dp,
            },
        },
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_all_moe_layers=None,
            static_forward_context={},
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            is_moe_model=True,
        ),
    )


def _install_fake_vllm_forward_context(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    config_module = types.ModuleType("vllm.config")
    forward_module = types.ModuleType("vllm.forward_context")
    vllm_v1_module = types.ModuleType("vllm.v1")
    worker_module = types.ModuleType("vllm.v1.worker")
    dp_utils_module = types.ModuleType("vllm.v1.worker.dp_utils")

    class CUDAGraphMode:
        NONE = "NONE"

    vars(config_module)["CUDAGraphMode"] = CUDAGraphMode

    class DPMetadata:
        @staticmethod
        def make(parallel_config, num_tokens, num_tokens_across_dp):
            return SimpleNamespace(
                parallel_config=parallel_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            )

    class BatchDescriptor:
        def __init__(self, num_tokens):
            self.num_tokens = num_tokens

    @contextmanager
    def original_set_forward_context(*_args, **_kwargs):
        vars(forward_module)["original_set_forward_context_called"] = True
        yield

    def original_coordinate_batch_across_dp(*_args, **_kwargs):
        vars(forward_module)["original_coordinate_called"] = True
        return True, "tokens", CUDAGraphMode.NONE

    @contextmanager
    def override_forward_context(forward_context):
        previous = vars(forward_module)["current_forward_context"]
        vars(forward_module)["current_forward_context"] = forward_context
        try:
            yield
        finally:
            vars(forward_module)["current_forward_context"] = previous

    def create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding=None,
    ):
        return SimpleNamespace(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            dp_metadata=dp_metadata,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
            slot_mapping=slot_mapping,
            additional_kwargs=additional_kwargs,
            skip_compiled=skip_compiled,
            is_padding=is_padding,
        )

    vars(forward_module).update(
        DPMetadata=DPMetadata,
        BatchDescriptor=BatchDescriptor,
        set_forward_context=original_set_forward_context,
        coordinate_batch_across_dp=original_coordinate_batch_across_dp,
        override_forward_context=override_forward_context,
        create_forward_context=create_forward_context,
        current_forward_context=None,
        original_set_forward_context_called=False,
        original_coordinate_called=False,
        track_batchsize=False,
        forward_start_time=0.0,
        last_logging_time=0.0,
        batchsize_logging_interval=1000.0,
        batchsize_forward_time={},
        time=time,
        current_platform=SimpleNamespace(
            set_additional_forward_context=lambda **kwargs: {"platform": kwargs},
            synchronize=None,
        ),
        logger=logging.getLogger("fake-forward-context"),
    )
    vars(dp_utils_module)["coordinate_batch_across_dp"] = (
        original_coordinate_batch_across_dp
    )

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.dp_utils", dp_utils_module)
    return forward_module, dp_utils_module, CUDAGraphMode


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm_forward_context(monkeypatch)
    # Keep fake forward contexts out of real consumers and the adapter cache.
    for name in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.kv_connector_model_runner_mixin",
        "vllm_ascend.ascend_forward_context",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    source = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/compat/patches/async_dp_forward_context.py"
    )
    spec = importlib.util.spec_from_file_location("_afd_test_forward_context", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_async_dp_forward_context_skips_dp_metadata(monkeypatch):
    _load_patch_module(monkeypatch)
    forward_module = sys.modules["vllm.forward_context"]

    with forward_module.set_forward_context(
        attn_metadata=object(),
        vllm_config=_config(),
        num_tokens=4,
    ):
        context = forward_module.current_forward_context
        assert context.dp_metadata is None
        assert context.additional_kwargs["platform"]["dp_metadata"] is None

    assert forward_module.original_set_forward_context_called is False


def test_async_dp_forward_context_preserves_non_async_path(monkeypatch):
    _load_patch_module(monkeypatch)
    forward_module = sys.modules["vllm.forward_context"]

    with forward_module.set_forward_context(
        attn_metadata=object(),
        vllm_config=_config(connector="CAMP2pAFDConnector", async_dp=False),
        num_tokens=4,
    ):
        context = forward_module.current_forward_context
        assert context.dp_metadata is not None
        assert context.dp_metadata.num_tokens == 4

    assert forward_module.original_set_forward_context_called is False
    assert forward_module.original_coordinate_called is True
