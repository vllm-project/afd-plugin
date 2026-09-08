from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat import npu as npu_compat
from afd_plugin.compat.npu import runtime as ascend_runtime
from afd_plugin.compat.patches.npu import mla_graph
from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    VLLM_ASCEND_310P_WORKER_FQCN,
    VLLM_ASCEND_NPU_WORKER_FQCN,
    VLLM_ASCEND_XLITE_WORKER_FQCN,
    VLLM_GPU_WORKER_FQCN,
)


def _install_fake_vllm_config(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.28.0"
    config_package = types.ModuleType("vllm.config")
    config_module = types.ModuleType("vllm.config.vllm")
    engine_package = types.ModuleType("vllm.engine")
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(
        is_cuda=lambda: True,
        device_type="cuda",
    )

    class VllmConfig:
        platform_worker_cls = VLLM_GPU_WORKER_FQCN

        def __post_init__(self):
            if self.parallel_config.use_ubatching:
                assert self.parallel_config.all2all_backend in {
                    "deepep_low_latency",
                    "deepep_high_throughput",
                }, "native all2all backend assertion"
            if self.parallel_config.worker_cls == "auto":
                self.parallel_config.worker_cls = self.platform_worker_cls
            self.post_init_backend = self.parallel_config.all2all_backend

    class EngineArgs:
        def create_engine_config(self, usage_context=None, headless=False):
            del usage_context, headless
            if self.enable_dbo:
                assert self.all2all_backend in {
                    "deepep_low_latency",
                    "deepep_high_throughput",
                }, "native all2all backend assertion"
            cfg = VllmConfig()
            cfg.additional_config = self.additional_config
            cfg.parallel_config = SimpleNamespace(
                use_ubatching=self.enable_dbo or self.ubatch_size > 1,
                all2all_backend=self.all2all_backend,
                worker_cls=self.worker_cls,
            )
            cfg.__post_init__()
            return cfg

    config_module.VllmConfig = VllmConfig
    config_module.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    arg_utils_module.EngineArgs = EngineArgs
    arg_utils_module.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_package)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", config_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_package)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    return arg_utils_module, config_module


def _load_patch_module():
    module_name = "afd_plugin.compat.patches.config_validation"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _engine_args(*, active, role="attention", worker_cls="auto"):
    args = sys.modules["vllm.engine.arg_utils"].EngineArgs()
    args.additional_config = {"afd": {"role": role}} if active else {}
    args.enable_dbo = True
    args.ubatch_size = 0
    args.all2all_backend = "allgather_reducescatter"
    args.worker_cls = worker_cls
    return args


def _set_fake_platform(*, is_cuda, device_type):
    sys.modules["vllm.platforms"].current_platform = SimpleNamespace(
        is_cuda=lambda: is_cuda,
        device_type=device_type,
    )


def _install_fake_npu_config(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    events = []

    class FakeParallelConfig:
        def __init__(
            self,
            *,
            enable_dbo,
            ubatch_size,
            all2all_backend,
            worker_cls,
        ):
            self.enable_dbo = enable_dbo
            self.ubatch_size = ubatch_size
            self.all2all_backend = all2all_backend
            self.worker_cls = worker_cls

        @property
        def use_ubatching(self):
            return self.enable_dbo or self.ubatch_size > 1

    class NPUPlatform:
        device_type = "npu"
        last_config = None

        @staticmethod
        def is_cuda():
            return False

        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            events.append(
                (
                    "fix_incompatible_config",
                    parallel_config.enable_dbo,
                    parallel_config.ubatch_size,
                ),
            )
            parallel_config.enable_dbo = False
            parallel_config.ubatch_size = 0

        @classmethod
        def check_and_update_config(cls, vllm_config):
            cls.last_config = vllm_config
            cls._fix_incompatible_config(vllm_config)
            parallel_config = vllm_config.parallel_config
            if (
                parallel_config.worker_cls == "auto"
                and not vllm_config.compilation_config.pass_config.enable_sp
            ):
                parallel_config.all2all_backend = "flashinfer_all2allv"
                parallel_config.worker_cls = VLLM_ASCEND_NPU_WORKER_FQCN
            events.append(
                (
                    "ascend_normalization",
                    parallel_config.all2all_backend,
                    parallel_config.worker_cls,
                ),
            )
            if vllm_config.fail_update:
                raise RuntimeError("upstream config failure")

    def post_init(vllm_config):
        NPUPlatform.check_and_update_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_ubatching:
            events.append(
                ("native_dbo_validation", parallel_config.all2all_backend),
            )
            assert parallel_config.all2all_backend in {
                "deepep_low_latency",
                "deepep_high_throughput",
            }, "native all2all backend assertion"
        vllm_config.post_init_backend = parallel_config.all2all_backend

    def create_engine_config(engine_args, usage_context=None, headless=False):
        del usage_context, headless
        config = config_module.VllmConfig()
        config.additional_config = engine_args.additional_config
        config.parallel_config = FakeParallelConfig(
            enable_dbo=engine_args.enable_dbo,
            ubatch_size=engine_args.ubatch_size,
            all2all_backend=engine_args.all2all_backend,
            worker_cls=engine_args.worker_cls,
        )
        config.compilation_config = SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=engine_args.enable_sp),
        )
        config.fail_update = engine_args.fail_update
        config.__post_init__()
        return config

    config_module.VllmConfig.__post_init__ = post_init
    arg_utils_module.EngineArgs.create_engine_config = create_engine_config

    fake_package = types.ModuleType("vllm_ascend")
    fake_package.__path__ = []
    fake_platform = types.ModuleType("vllm_ascend.platform")
    fake_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    sys.modules["vllm.platforms"].current_platform = NPUPlatform
    monkeypatch.setattr(mla_graph, "apply_afd_mla_graph_patch", lambda: True)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)
    return arg_utils_module, NPUPlatform, events


def test_config_validation_patch_relaxes_backend_for_afd_ubatching(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert args.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_preserves_non_afd_validation(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=False)

    try:
        arg_utils_module.EngineArgs.create_engine_config(args)
    except AssertionError as exc:
        assert "native all2all" in str(exc)
    else:
        raise AssertionError("expected native all2all backend assertion")


def test_config_validation_patch_allows_vllm_dev_checkout(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    sys.modules["vllm"].__version__ = "0.1.dev14230+g68b0c3135"
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_selects_worker_after_upstream_post_init(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.worker_cls == ATTENTION_WORKER_FQCN


def test_config_validation_patch_relaxes_explicit_post_init_revalidation(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    cfg.__post_init__()

    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


@pytest.mark.parametrize(
    ("role", "expected_worker_cls"),
    [
        ("attention", NPU_ATTENTION_WORKER_FQCN),
        ("ffn", NPU_FFN_WORKER_FQCN),
    ],
)
def test_config_validation_preserves_npu_dbo_through_auto_worker_normalization(
    monkeypatch,
    role,
    expected_worker_cls,
):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True, role=role)
    args.ubatch_size = 2
    args.enable_sp = False
    args.fail_update = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert (
        "ascend_normalization",
        "flashinfer_all2allv",
        VLLM_ASCEND_NPU_WORKER_FQCN,
    ) in events
    assert ("native_dbo_validation", "deepep_low_latency") in events
    assert cfg.post_init_backend == "deepep_low_latency"
    assert args.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.enable_dbo is True
    assert cfg.parallel_config.ubatch_size == 2
    assert cfg.parallel_config.use_ubatching is True
    assert cfg.parallel_config.worker_cls == expected_worker_cls


def test_config_validation_revalidates_npu_dbo_and_restores_backend(monkeypatch):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)
    args.ubatch_size = 2
    args.enable_sp = False
    args.fail_update = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    events.clear()
    cfg.__post_init__()

    assert ("native_dbo_validation", "deepep_low_latency") in events
    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.enable_dbo is True
    assert cfg.parallel_config.ubatch_size == 2
    assert cfg.parallel_config.worker_cls == NPU_ATTENTION_WORKER_FQCN


def test_config_validation_restores_npu_snapshot_when_platform_update_fails(
    monkeypatch,
):
    arg_utils_module, npu_platform, _events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)
    args.ubatch_size = 2
    args.enable_sp = False
    args.fail_update = True

    with pytest.raises(RuntimeError, match="upstream config failure"):
        arg_utils_module.EngineArgs.create_engine_config(args)

    cfg = npu_platform.last_config
    assert cfg.parallel_config.enable_dbo is True
    assert cfg.parallel_config.ubatch_size == 2
    assert cfg.parallel_config.all2all_backend == "deepep_low_latency"
    assert args.all2all_backend == "allgather_reducescatter"


def test_config_validation_preserves_explicit_npu_worker(monkeypatch):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(
        active=True,
        worker_cls=NPU_ATTENTION_WORKER_FQCN,
    )
    args.ubatch_size = 2
    args.enable_sp = False
    args.fail_update = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert ("native_dbo_validation", "deepep_low_latency") in events
    assert cfg.parallel_config.worker_cls == NPU_ATTENTION_WORKER_FQCN
    assert cfg.parallel_config.enable_dbo is True
    assert cfg.parallel_config.ubatch_size == 2
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_preserves_non_afd_npu_upstream_behavior(monkeypatch):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=False)
    args.ubatch_size = 2
    args.enable_sp = False
    args.fail_update = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert not any(event[0] == "native_dbo_validation" for event in events)
    assert cfg.parallel_config.enable_dbo is False
    assert cfg.parallel_config.ubatch_size == 0
    assert cfg.parallel_config.use_ubatching is False
    assert cfg.parallel_config.all2all_backend == "flashinfer_all2allv"
    assert cfg.parallel_config.worker_cls == VLLM_ASCEND_NPU_WORKER_FQCN


def test_config_validation_preserves_npu_dbo_off_behavior(monkeypatch):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True, role="ffn")
    args.enable_dbo = False
    args.ubatch_size = 0
    args.enable_sp = False
    args.fail_update = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert not any(event[0] == "native_dbo_validation" for event in events)
    assert cfg.parallel_config.enable_dbo is False
    assert cfg.parallel_config.ubatch_size == 0
    assert cfg.parallel_config.all2all_backend == "flashinfer_all2allv"
    assert cfg.parallel_config.worker_cls == NPU_FFN_WORKER_FQCN


@pytest.mark.parametrize(
    (
        "role",
        "platform_worker_cls",
        "is_cuda",
        "device_type",
        "expected_worker_cls",
    ),
    [
        (
            "attention",
            VLLM_GPU_WORKER_FQCN,
            True,
            "cuda",
            ATTENTION_WORKER_FQCN,
        ),
        ("ffn", VLLM_GPU_WORKER_FQCN, True, "cuda", FFN_WORKER_FQCN),
        (
            "attention",
            VLLM_ASCEND_NPU_WORKER_FQCN,
            False,
            "npu",
            NPU_ATTENTION_WORKER_FQCN,
        ),
        (
            "ffn",
            VLLM_ASCEND_NPU_WORKER_FQCN,
            False,
            "npu",
            NPU_FFN_WORKER_FQCN,
        ),
    ],
)
def test_config_validation_patch_auto_selects_afd_worker(
    monkeypatch,
    role,
    platform_worker_cls,
    is_cuda,
    device_type,
    expected_worker_cls,
):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: None,
    )
    config_module.VllmConfig.platform_worker_cls = platform_worker_cls
    _set_fake_platform(is_cuda=is_cuda, device_type=device_type)
    _load_patch_module()
    args = _engine_args(active=True, role=role)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == expected_worker_cls


def test_config_validation_patch_preserves_explicit_worker(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(
        active=True,
        role="attention",
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == ATTENTION_WORKER_FQCN


def test_config_validation_patch_auto_selects_without_ubatching(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(active=True, role="ffn")
    args.enable_dbo = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == FFN_WORKER_FQCN


def test_config_validation_installs_ascend_patch_only_on_npu(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)

    calls = []
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: calls.append("npu"),
    )
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    cuda_args = _engine_args(active=True)
    arg_utils_module.EngineArgs.create_engine_config(cuda_args)
    assert calls == []

    config_module.VllmConfig.platform_worker_cls = VLLM_ASCEND_NPU_WORKER_FQCN
    _set_fake_platform(is_cuda=False, device_type="npu")
    npu_args = _engine_args(active=True)
    arg_utils_module.EngineArgs.create_engine_config(npu_args)
    assert calls == ["npu"]


def test_config_validation_finalizes_async_attention_patch_after_ascend(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_ASCEND_NPU_WORKER_FQCN
    _set_fake_platform(is_cuda=False, device_type="npu")

    config_patch_calls = []
    engine_patch_configs = []
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: config_patch_calls.append("config"),
    )
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_async_dp_engine_patch_if_needed",
        engine_patch_configs.append,
    )
    patch_module = _load_patch_module()
    importlib.reload(patch_module)
    args = _engine_args(active=True)
    args.additional_config["afd"].update(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
    )

    config = arg_utils_module.EngineArgs.create_engine_config(args)

    assert config_patch_calls == ["config"]
    assert engine_patch_configs == [config]


def test_config_validation_patch_preserves_non_afd_platform_default(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(active=False)
    args.enable_dbo = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == VLLM_GPU_WORKER_FQCN


@pytest.mark.parametrize(
    ("platform_worker_cls", "is_cuda", "device_type"),
    [
        (VLLM_ASCEND_310P_WORKER_FQCN, False, "npu"),
        (VLLM_ASCEND_XLITE_WORKER_FQCN, False, "npu"),
        ("other_platform.worker.Worker", False, "xpu"),
        (VLLM_GPU_WORKER_FQCN, False, "cuda"),
    ],
)
def test_config_validation_patch_rejects_unsupported_auto_platform(
    monkeypatch,
    platform_worker_cls,
    is_cuda,
    device_type,
):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: None,
    )
    config_module.VllmConfig.platform_worker_cls = platform_worker_cls
    _set_fake_platform(is_cuda=is_cuda, device_type=device_type)
    _load_patch_module()
    args = _engine_args(active=True)

    with pytest.raises(ValueError, match="automatic worker selection"):
        arg_utils_module.EngineArgs.create_engine_config(args)
