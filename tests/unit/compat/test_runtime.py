# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.npu import runtime as ascend_runtime
from afd_plugin.compat.npu.runtime import fix_all2all_backend_for_afd


def _vllm_config(*, enable_sp=False, all2all_backend="allgather_reducescatter"):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
        ),
        parallel_config=SimpleNamespace(
            all2all_backend=all2all_backend,
        ),
    )


def test_fix_all2all_backend_overrides_to_flashinfer_when_sp_disabled():
    config = _vllm_config(enable_sp=False, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_fix_all2all_backend_skips_when_sp_enabled():
    config = _vllm_config(enable_sp=True, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_fix_all2all_backend_skips_when_already_flashinfer():
    config = _vllm_config(enable_sp=False, all2all_backend="flashinfer_all2allv")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


@pytest.mark.parametrize("skip_mc2_mask", [False, True])
def test_ascend_forward_context_installs_afd_metadata(monkeypatch, skip_mc2_mask):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")
    fake_forward_context_module = ModuleType("vllm.forward_context")
    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_ascend_forward_context = ModuleType(
        "vllm_ascend.ascend_forward_context",
    )
    original_mask = object()
    forward_context = SimpleNamespace(additional_kwargs=None, mc2_mask=original_mask)
    calls = []

    class CUDAGraphMode:
        NONE = "none"

    @contextmanager
    def set_ascend_forward_context(
        attn_metadata,
        vllm_config,
        *,
        batch_descriptor,
        aclgraph_runtime_mode,
        model_instance,
        num_tokens,
        num_tokens_across_dp,
        in_profile_run,
    ):
        calls.append(
            {
                "attn_metadata": attn_metadata,
                "vllm_config": vllm_config,
                "batch_descriptor": batch_descriptor,
                "aclgraph_runtime_mode": aclgraph_runtime_mode,
                "model_instance": model_instance,
                "num_tokens": num_tokens,
                "num_tokens_across_dp": num_tokens_across_dp,
                "in_profile_run": in_profile_run,
            },
        )
        yield

    fake_config.__dict__["CUDAGraphMode"] = CUDAGraphMode
    fake_forward_context_module.__dict__["get_forward_context"] = lambda: (
        forward_context
    )
    fake_ascend_forward_context.__dict__["set_ascend_forward_context"] = (
        set_ascend_forward_context
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        fake_forward_context_module,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_ascend_forward_context,
    )

    vllm_config = SimpleNamespace(use_v2_model_runner=False)
    afd_metadata = SimpleNamespace()
    model_instance = SimpleNamespace()
    with ascend_runtime.ascend_forward_context(
        vllm_config=vllm_config,
        afd_metadata=afd_metadata,
        model_instance=model_instance,
        num_tokens=3,
        in_profile_run=True,
        skip_mc2_mask=skip_mc2_mask,
    ) as current_forward_context:
        assert current_forward_context is forward_context
        assert forward_context.additional_kwargs["afd_metadata"] is afd_metadata
        assert forward_context.mc2_mask is (None if skip_mc2_mask else original_mask)

    assert calls == [
        {
            "attn_metadata": None,
            "vllm_config": vllm_config,
            "batch_descriptor": None,
            "aclgraph_runtime_mode": CUDAGraphMode.NONE,
            "model_instance": model_instance,
            "num_tokens": 3,
            "num_tokens_across_dp": None,
            "in_profile_run": True,
        },
    ]


def test_ascend_forward_context_uses_native_mrv2_layout(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")
    fake_forward_context_module = ModuleType("vllm.forward_context")
    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_ascend_forward_context = ModuleType(
        "vllm_ascend.ascend_forward_context",
    )
    forward_context = SimpleNamespace(
        additional_kwargs={"in_profile_run": True},
    )
    context_calls = []
    profile_calls = []

    class CUDAGraphMode:
        NONE = "none"

    @contextmanager
    def set_forward_context(
        attn_metadata,
        vllm_config,
        *,
        num_tokens,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
        batch_descriptor,
    ):
        context_calls.append(
            {
                "attn_metadata": attn_metadata,
                "vllm_config": vllm_config,
                "num_tokens": num_tokens,
                "num_tokens_across_dp": num_tokens_across_dp,
                "cudagraph_runtime_mode": cudagraph_runtime_mode,
                "batch_descriptor": batch_descriptor,
            },
        )
        yield

    @contextmanager
    def override_mrv2_in_profile_run(enabled):
        profile_calls.append(enabled)
        yield

    def unexpected_legacy_context(*args, **kwargs):
        raise AssertionError("MRv2 must not use set_ascend_forward_context")

    fake_config.__dict__["CUDAGraphMode"] = CUDAGraphMode
    fake_forward_context_module.__dict__["get_forward_context"] = lambda: (
        forward_context
    )
    fake_forward_context_module.__dict__["set_forward_context"] = set_forward_context
    fake_ascend_forward_context.__dict__["override_mrv2_in_profile_run"] = (
        override_mrv2_in_profile_run
    )
    fake_ascend_forward_context.__dict__["set_ascend_forward_context"] = (
        unexpected_legacy_context
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        fake_forward_context_module,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_ascend_forward_context,
    )

    vllm_config = SimpleNamespace(use_v2_model_runner=True)
    afd_metadata = SimpleNamespace()
    model_instance = SimpleNamespace()
    num_tokens_across_dp = SimpleNamespace()
    with ascend_runtime.ascend_forward_context(
        vllm_config=vllm_config,
        afd_metadata=afd_metadata,
        model_instance=model_instance,
        num_tokens=7,
        num_tokens_across_dp=num_tokens_across_dp,
        in_profile_run=True,
        skip_mc2_mask=True,
    ) as current_forward_context:
        assert current_forward_context is forward_context
        assert forward_context.additional_kwargs["afd_metadata"] is afd_metadata
        assert forward_context.additional_kwargs["model_instance"] is model_instance

    assert profile_calls == [True]
    assert context_calls == [
        {
            "attn_metadata": None,
            "vllm_config": vllm_config,
            "num_tokens": 7,
            "num_tokens_across_dp": num_tokens_across_dp,
            "cudagraph_runtime_mode": CUDAGraphMode.NONE,
            "batch_descriptor": None,
        },
    ]


def test_npu_afd_config_patch_restores_dbo_for_afd(monkeypatch):
    from afd_plugin.compat.patches.npu import mla_graph

    fake_package = ModuleType("vllm_ascend")
    fake_package.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")

    class FakeParallelConfig:
        def __init__(self, *, enable_dbo, ubatch_size):
            self.enable_dbo = enable_dbo
            self.ubatch_size = ubatch_size
            self.all2all_backend = "deepep_low_latency"

        @property
        def use_ubatching(self):
            return self.enable_dbo or self.ubatch_size > 1

    class NPUPlatform:
        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            parallel_config.enable_dbo = False
            parallel_config.ubatch_size = 0

        @classmethod
        def check_and_update_config(cls, vllm_config):
            cls._fix_incompatible_config(vllm_config)
            parallel_config = vllm_config.parallel_config
            parallel_config.all2all_backend = "flashinfer_all2allv"
            if getattr(vllm_config, "fail_update", False):
                raise RuntimeError("upstream config failure")

    def afd_vllm_config(*, active=True):
        config = _vllm_config()
        config.additional_config = (
            {
                "afd": {
                    "role": "attention",
                    "connector": "CAMP2pAFDConnector",
                },
            }
            if active
            else {}
        )
        config.parallel_config = FakeParallelConfig(enable_dbo=True, ubatch_size=4)
        config.fail_update = False
        return config

    fake_platform.__dict__["NPUPlatform"] = NPUPlatform
    monkeypatch.setattr(mla_graph, "apply_afd_mla_graph_patch", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    config = afd_vllm_config()
    assert NPUPlatform.check_and_update_config(config) is None
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.use_ubatching is True
    assert config.parallel_config.ubatch_size == 4
    assert config.parallel_config.all2all_backend == "deepep_low_latency"

    failing_config = afd_vllm_config()
    failing_config.fail_update = True
    with pytest.raises(RuntimeError, match="upstream config failure"):
        NPUPlatform.check_and_update_config(failing_config)
    assert failing_config.parallel_config.enable_dbo is True
    assert failing_config.parallel_config.ubatch_size == 4
    assert failing_config.parallel_config.all2all_backend == "deepep_low_latency"

    inactive_config = afd_vllm_config(active=False)
    assert NPUPlatform.check_and_update_config(inactive_config) is None
    assert inactive_config.parallel_config.enable_dbo is False
    assert inactive_config.parallel_config.use_ubatching is False
    assert inactive_config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_npu_afd_config_patch_raises_and_retries_after_import_error(monkeypatch):
    from afd_plugin.compat.patches.npu import mla_graph

    fake_package = ModuleType("vllm_ascend")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", None)
    monkeypatch.setattr(mla_graph, "apply_afd_mla_graph_patch", lambda: True)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    with pytest.raises(RuntimeError, match="DBO config patch"):
        ascend_runtime.apply_afd_ascend_patches_if_needed()

    assert ascend_runtime._PATCHES_APPLIED is False

    fake_platform = ModuleType("vllm_ascend.platform")

    class NPUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            del cls, vllm_config

    fake_platform.__dict__["NPUPlatform"] = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    assert ascend_runtime._PATCHES_APPLIED is True
    assert hasattr(NPUPlatform, "_afd_plugin_ascend_platform_patch_state")


def test_npu_patches_reject_missing_mla_resolver(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_forward_context = ModuleType("vllm.forward_context")
    fake_forward_context.__dict__["get_forward_context"] = lambda: None
    fake_forward_context.__dict__["is_forward_context_available"] = lambda: False
    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention.__path__ = []

    class NPUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            del cls, vllm_config

    fake_platform.__dict__["NPUPlatform"] = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", fake_forward_context)
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", fake_attention)
    monkeypatch.delitem(
        sys.modules,
        "vllm_ascend.attention.mla_v1",
        raising=False,
    )
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    with pytest.raises(RuntimeError, match="MLA graph patch"):
        ascend_runtime.apply_afd_ascend_patches_if_needed()
    assert ascend_runtime._PATCHES_APPLIED is False


def test_npu_patches_route_mla_graph_params_from_forward_context(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_forward_context = ModuleType("vllm.forward_context")
    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention.__path__ = []
    fake_mla = ModuleType("vllm_ascend.attention.mla_v1")

    class NPUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            del cls, vllm_config

    upstream_registry = object()
    afd_registry = object()
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_mla_graph_params": afd_registry},
    )
    context_available = True

    def get_forward_context():
        return forward_context

    def is_forward_context_available():
        return context_available

    def get_graph_params():
        return upstream_registry

    fake_forward_context.__dict__["get_forward_context"] = get_forward_context
    fake_forward_context.__dict__["is_forward_context_available"] = (
        is_forward_context_available
    )
    fake_platform.__dict__["NPUPlatform"] = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        fake_forward_context,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", fake_attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.mla_v1",
        fake_mla,
    )
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    with pytest.raises(AttributeError, match="get_graph_params"):
        ascend_runtime.apply_afd_ascend_patches_if_needed()
    assert ascend_runtime._PATCHES_APPLIED is False

    fake_mla.__dict__["get_graph_params"] = get_graph_params
    ascend_runtime.apply_afd_ascend_patches_if_needed()
    patched_get_graph_params = fake_mla.get_graph_params

    assert fake_mla.get_graph_params() is afd_registry

    forward_context.additional_kwargs = {}
    assert fake_mla.get_graph_params() is upstream_registry

    context_available = False
    assert fake_mla.get_graph_params() is upstream_registry

    ascend_runtime.apply_afd_ascend_patches_if_needed()
    assert fake_mla.get_graph_params is patched_get_graph_params
