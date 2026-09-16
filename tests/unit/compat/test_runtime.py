# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.npu import runtime as ascend_runtime
from afd_plugin.compat.npu.runtime import fix_all2all_backend_for_afd


@pytest.fixture(autouse=True)
def stub_ascend_namespace_patch(monkeypatch):
    # Namespace behavior is covered separately with a strict native config stub.
    patch_module = ModuleType("afd_plugin.compat.patches.npu.ascend_config")
    calls = []
    patch_module.apply_afd_ascend_config_patch = lambda: calls.append("namespace")
    monkeypatch.setitem(sys.modules, patch_module.__name__, patch_module)
    return calls


def test_config_namespace_patch_installed_after_dbo_patch(
    monkeypatch, stub_ascend_namespace_patch
):
    platform_patch = ModuleType("afd_plugin.compat.patches.npu.ascend_platform")

    def install_dbo():
        stub_ascend_namespace_patch.append("dbo")
        return True

    platform_patch.apply_afd_ascend_dbo_config_patch = install_dbo
    monkeypatch.setitem(sys.modules, platform_patch.__name__, platform_patch)
    ascend_runtime.apply_afd_ascend_config_patch_if_needed()
    assert stub_ascend_namespace_patch == ["dbo", "namespace"]


@pytest.fixture
def backend_config_env(monkeypatch):
    native_config = ModuleType("vllm_ascend.ascend_config")
    native_config.validate_additional_config_bool = lambda value, name: (
        value if isinstance(value, bool) else str(value).lower() in {"1", "true"}
    )
    monkeypatch.setitem(sys.modules, native_config.__name__, native_config)
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    return native_config


def _vllm_config(*, enable_sp=False, all2all_backend="allgather_reducescatter"):
    return SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
        ),
        parallel_config=SimpleNamespace(
            all2all_backend=all2all_backend,
        ),
    )


def test_fix_all2all_backend_overrides_to_flashinfer_when_sp_disabled(
    backend_config_env,
):
    config = _vllm_config(enable_sp=False, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_fix_all2all_backend_ignores_compile_pass_sp(backend_config_env):
    config = _vllm_config(enable_sp=True, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_fix_all2all_backend_skips_when_already_flashinfer(backend_config_env):
    config = _vllm_config(enable_sp=False, all2all_backend="flashinfer_all2allv")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_ascend_forward_context_installs_afd_metadata(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")
    fake_forward_context_module = ModuleType("vllm.forward_context")
    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_ascend_forward_context = ModuleType(
        "vllm_ascend.ascend_forward_context",
    )
    forward_context = SimpleNamespace(additional_kwargs=None)
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

    fake_config.CUDAGraphMode = CUDAGraphMode
    fake_forward_context_module.get_forward_context = lambda: forward_context
    fake_ascend_forward_context.set_ascend_forward_context = set_ascend_forward_context
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
    ) as current_forward_context:
        assert current_forward_context is forward_context
        assert forward_context.additional_kwargs["afd_metadata"] is afd_metadata

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

    fake_config.CUDAGraphMode = CUDAGraphMode
    fake_forward_context_module.get_forward_context = lambda: forward_context
    fake_forward_context_module.set_forward_context = set_forward_context
    fake_ascend_forward_context.override_mrv2_in_profile_run = (
        override_mrv2_in_profile_run
    )
    fake_ascend_forward_context.set_ascend_forward_context = unexpected_legacy_context
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

    fake_platform.NPUPlatform = NPUPlatform
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

    fake_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    assert ascend_runtime._PATCHES_APPLIED is True
    assert hasattr(NPUPlatform, "_afd_plugin_ascend_platform_patch_state")


def test_npu_patches_reject_missing_mla_resolver(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_forward_context = ModuleType("vllm.forward_context")
    fake_forward_context.get_forward_context = lambda: None
    fake_forward_context.is_forward_context_available = lambda: False
    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention.__path__ = []

    class NPUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            del cls, vllm_config

    fake_platform.NPUPlatform = NPUPlatform
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

    fake_forward_context.get_forward_context = get_forward_context
    fake_forward_context.is_forward_context_available = is_forward_context_available
    fake_platform.NPUPlatform = NPUPlatform
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

    fake_mla.get_graph_params = get_graph_params
    ascend_runtime.apply_afd_ascend_patches_if_needed()
    patched_get_graph_params = fake_mla.get_graph_params

    assert fake_mla.get_graph_params() is afd_registry

    forward_context.additional_kwargs = {}
    assert fake_mla.get_graph_params() is upstream_registry

    context_available = False
    assert fake_mla.get_graph_params() is upstream_registry

    ascend_runtime.apply_afd_ascend_patches_if_needed()
    assert fake_mla.get_graph_params is patched_get_graph_params


@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize(
    ("options", "environment", "preserved"),
    [
        ({"enable_flashcomm1": True}, None, True),
        ({"enable_flashcomm1": "true"}, None, True),
        ({"enable_flashcomm1": "false"}, None, False),
        ({}, " TRUE ", True),
        ({}, " 1 ", True),
        ({}, "false", False),
        ({"enable_dsa_cp": True}, None, True),
        ({"enable_dsa_cp": "true"}, None, True),
        ({"enable_dsa_cp": "false"}, None, False),
    ],
)
def test_backend_follows_native_raw_flashcomm_selector(
    monkeypatch, backend_config_env, enable_sp, options, environment, preserved
):
    config = _vllm_config(enable_sp=enable_sp)
    config.additional_config = options
    # Finalized native DSA-CP can be false despite the original true input.
    backend_config_env.get_ascend_config = lambda: SimpleNamespace(enable_dsa_cp=False)
    if environment is not None:
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", environment)
    fix_all2all_backend_for_afd(config)
    backend = config.parallel_config.all2all_backend
    assert backend == (
        "allgather_reducescatter" if preserved else "flashinfer_all2allv"
    )
    fix_all2all_backend_for_afd(config)
    assert config.parallel_config.all2all_backend == backend
