# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import importlib.util
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
    vllm_module.__dict__["__version__"] = "0.30.0"
    config_package = types.ModuleType("vllm.config")
    config_module = types.ModuleType("vllm.config.vllm")
    engine_package = types.ModuleType("vllm.engine")
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.__dict__["current_platform"] = SimpleNamespace(
        is_cuda=lambda: True,
        device_type="cuda",
    )

    class VllmConfig:
        model_config: SimpleNamespace
        parallel_config: SimpleNamespace
        additional_config: dict
        platform_worker_cls = VLLM_GPU_WORKER_FQCN

        def _get_dbo_unsupported_features(self):
            if self.model_config.is_hybrid:
                return ["dual batch overlap with hybrid models"]
            return []

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
        enable_dbo: bool
        ubatch_size: int
        all2all_backend: str
        worker_cls: str
        additional_config: dict

        def create_engine_config(self, usage_context=None, headless=False):
            del usage_context, headless
            if (
                self.enable_dbo
                and platforms_module.__dict__["current_platform"].device_type != "npu"
            ):
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

    config_module.__dict__["VllmConfig"] = VllmConfig
    config_module.__dict__["logger"] = SimpleNamespace(
        debug=lambda *args, **kwargs: None
    )
    arg_utils_module.__dict__["EngineArgs"] = EngineArgs
    arg_utils_module.__dict__["logger"] = SimpleNamespace(
        debug=lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_package)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", config_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_package)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    monkeypatch.setattr(npu_compat, "fix_all2all_backend_for_afd", lambda config: None)
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_engine_core_config_patch_if_needed",
        lambda config: False,
    )
    return arg_utils_module, config_module


def _load_patch_module():
    # Keep fake-vLLM globals off the cached production module and its package
    # attribute: later runtime tests must retain their real imported bindings.
    spec = importlib.util.find_spec("afd_plugin.compat.patches.config_validation")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Execute twice: installing the patch must be idempotent.
    spec.loader.exec_module(module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("is_hybrid", [False, True])
def test_patch_preserves_native_dbo_model_restrictions(monkeypatch, is_hybrid):
    _, config_module = _install_fake_vllm_config(monkeypatch)
    native_restrictions = config_module.VllmConfig._get_dbo_unsupported_features

    _load_patch_module()

    assert config_module.VllmConfig._get_dbo_unsupported_features is native_restrictions
    config = config_module.VllmConfig()
    config.additional_config = {"afd": {"role": "attention"}}
    config.model_config = SimpleNamespace(is_hybrid=is_hybrid)
    expected = ["dual batch overlap with hybrid models"] if is_hybrid else []
    assert config._get_dbo_unsupported_features() == expected


def _engine_args(
    *,
    active,
    role="attention",
    worker_cls="auto",
    enable_dbo=True,
    ubatch_size=0,
    fail_update=False,
):
    args = sys.modules["vllm.engine.arg_utils"].EngineArgs()
    args.additional_config = {"afd": {"role": role}} if active else {}
    args.enable_dbo = enable_dbo
    args.ubatch_size = ubatch_size
    args.all2all_backend = "allgather_reducescatter"
    args.worker_cls = worker_cls
    args.enable_sp = False
    args.fail_update = fail_update
    return args


def _set_fake_platform(*, is_cuda, device_type):
    sys.modules["vllm.platforms"].__dict__["current_platform"] = SimpleNamespace(
        is_cuda=lambda: is_cuda,
        device_type=device_type,
    )


def _install_fake_npu_config(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    events: list[tuple[str, str] | tuple[str, str, str]] = []

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
            self.enable_expert_parallel = False
            self.tensor_parallel_size = 1
            self.data_parallel_size = 1

        @property
        def use_sequence_parallel_moe(self):
            return (
                self.all2all_backend
                in {
                    "allgather_reducescatter",
                    "deepep_low_latency",
                    "deepep_high_throughput",
                    "nixl_ep",
                }
                and self.enable_expert_parallel
                and self.tensor_parallel_size > 1
                and self.data_parallel_size > 1
            )

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
    fake_platform.__dict__["NPUPlatform"] = NPUPlatform
    factory_patch = types.ModuleType("afd_plugin.compat.patches.npu.ascend_config")
    factory_patch.__dict__["apply_afd_ascend_config_patch"] = lambda: None
    monkeypatch.setitem(sys.modules, factory_patch.__name__, factory_patch)
    monkeypatch.setattr(
        npu_compat,
        "fix_all2all_backend_for_afd",
        ascend_runtime.fix_all2all_backend_for_afd,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    sys.modules["vllm.platforms"].__dict__["current_platform"] = NPUPlatform
    monkeypatch.setattr(mla_graph, "apply_afd_mla_graph_patch", lambda: True)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)
    return arg_utils_module, NPUPlatform, events


_CUDA = (VLLM_GPU_WORKER_FQCN, True, "cuda")
_NPU = (VLLM_ASCEND_NPU_WORKER_FQCN, False, "npu")
_AUTO_REJECTED = (ValueError, "automatic worker selection")


def _cuda_case(
    case_id,
    *,
    platform=_CUDA,
    version="0.30.0",
    revalidate=False,
    worker=None,
    post_init_backend="deepep_low_latency",
    raises=None,
    **engine_args,
):
    engine_args.setdefault("active", True)
    return pytest.param(
        platform,
        version,
        engine_args,
        revalidate,
        (worker, post_init_backend),
        raises,
        id=case_id,
    )


@pytest.mark.parametrize(
    ("platform", "version", "engine_args", "revalidate", "expected", "raises"),
    [
        _cuda_case(
            "afd-ubatching-relaxed-auto-attention", worker=ATTENTION_WORKER_FQCN
        ),
        _cuda_case("afd-auto-ffn", role="ffn", worker=FFN_WORKER_FQCN),
        _cuda_case(
            "vllm-dev-checkout",
            version="0.1.dev14230+g68b0c3135",
            worker=ATTENTION_WORKER_FQCN,
        ),
        _cuda_case(
            "explicit-post-init-revalidation",
            revalidate=True,
            worker=ATTENTION_WORKER_FQCN,
        ),
        _cuda_case(
            "explicit-worker-preserved",
            worker_cls=ATTENTION_WORKER_FQCN,
            worker=ATTENTION_WORKER_FQCN,
        ),
        _cuda_case(
            "auto-without-ubatching",
            role="ffn",
            enable_dbo=False,
            worker=FFN_WORKER_FQCN,
            post_init_backend="allgather_reducescatter",
        ),
        _cuda_case(
            "non-afd-platform-default",
            active=False,
            enable_dbo=False,
            worker=VLLM_GPU_WORKER_FQCN,
            post_init_backend="allgather_reducescatter",
        ),
        _cuda_case(
            "non-afd-native-validation",
            active=False,
            raises=(AssertionError, "native all2all"),
        ),
        _cuda_case(
            "npu-auto-attention", platform=_NPU, worker=NPU_ATTENTION_WORKER_FQCN
        ),
        _cuda_case(
            "npu-auto-ffn", platform=_NPU, role="ffn", worker=NPU_FFN_WORKER_FQCN
        ),
        _cuda_case(
            "reject-310p",
            platform=(VLLM_ASCEND_310P_WORKER_FQCN, False, "npu"),
            raises=_AUTO_REJECTED,
        ),
        _cuda_case(
            "reject-xlite",
            platform=(VLLM_ASCEND_XLITE_WORKER_FQCN, False, "npu"),
            raises=_AUTO_REJECTED,
        ),
        _cuda_case(
            "reject-other-platform",
            platform=("other_platform.worker.Worker", False, "xpu"),
            raises=_AUTO_REJECTED,
        ),
        _cuda_case(
            "reject-cuda-device-not-cuda",
            platform=(VLLM_GPU_WORKER_FQCN, False, "cuda"),
            raises=_AUTO_REJECTED,
        ),
    ],
)
def test_config_validation_outcome(
    monkeypatch, platform, version, engine_args, revalidate, expected, raises
):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    ascend_patch_calls = []
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: ascend_patch_calls.append("npu"),
    )
    platform_worker_cls, is_cuda, device_type = platform
    config_module.VllmConfig.platform_worker_cls = platform_worker_cls
    _set_fake_platform(is_cuda=is_cuda, device_type=device_type)
    sys.modules["vllm"].__dict__["__version__"] = version
    _load_patch_module()
    args = _engine_args(**engine_args)

    if raises is not None:
        with pytest.raises(raises[0], match=raises[1]):
            arg_utils_module.EngineArgs.create_engine_config(args)
    else:
        cfg = arg_utils_module.EngineArgs.create_engine_config(args)
        if revalidate:
            cfg.__post_init__()
        worker, post_init_backend = expected
        assert cfg.parallel_config.worker_cls == worker
        assert cfg.post_init_backend == post_init_backend
        assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert args.all2all_backend == "allgather_reducescatter"
    # The Ascend config shim is installed only for AFD configs on NPU.
    assert bool(ascend_patch_calls) == (engine_args["active"] and device_type == "npu")


def test_config_validation_finalizes_async_attention_patch_after_ascend(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_ASCEND_NPU_WORKER_FQCN
    _set_fake_platform(is_cuda=False, device_type="npu")
    calls: list = []
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_config_patch_if_needed",
        lambda: calls.append("config"),
    )
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_async_dp_engine_patch_if_needed",
        lambda config: calls.append(("engine", config)),
    )
    _load_patch_module()
    args = _engine_args(active=True)
    args.additional_config["afd"].update(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
    )

    config = arg_utils_module.EngineArgs.create_engine_config(args)

    assert calls == ["config", "config", ("engine", config)]


def _npu_case(case_id, *, worker, keeps_dbo, dbo=True, revalidate=False, **kwargs):
    kwargs.setdefault("active", True)
    kwargs.update(enable_dbo=dbo, ubatch_size=2 if dbo else 0)
    return pytest.param(kwargs, revalidate, worker, keeps_dbo, id=case_id)


@pytest.mark.parametrize(
    ("engine_args", "revalidate", "worker", "keeps_dbo"),
    [
        _npu_case("auto-attention", worker=NPU_ATTENTION_WORKER_FQCN, keeps_dbo=True),
        _npu_case("auto-ffn", role="ffn", worker=NPU_FFN_WORKER_FQCN, keeps_dbo=True),
        _npu_case(
            "revalidation-restores-backend",
            revalidate=True,
            worker=NPU_ATTENTION_WORKER_FQCN,
            keeps_dbo=True,
        ),
        _npu_case(
            "explicit-npu-worker",
            worker_cls=NPU_ATTENTION_WORKER_FQCN,
            worker=NPU_ATTENTION_WORKER_FQCN,
            keeps_dbo=True,
        ),
        _npu_case(
            "non-afd-upstream-disables-dbo",
            active=False,
            worker=VLLM_ASCEND_NPU_WORKER_FQCN,
            keeps_dbo=False,
        ),
        _npu_case(
            "afd-dbo-off",
            role="ffn",
            dbo=False,
            worker=NPU_FFN_WORKER_FQCN,
            keeps_dbo=False,
        ),
    ],
)
def test_config_validation_npu_outcome(
    monkeypatch, engine_args, revalidate, worker, keeps_dbo
):
    arg_utils_module, _npu_platform, events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(**engine_args)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    if revalidate:
        events.clear()
        cfg.__post_init__()

    parallel_config = cfg.parallel_config
    validated = ("native_dbo_validation", "deepep_low_latency") in events
    assert validated is keeps_dbo
    assert parallel_config.enable_dbo is keeps_dbo
    assert parallel_config.ubatch_size == (2 if keeps_dbo else 0)
    assert parallel_config.use_ubatching is keeps_dbo
    assert parallel_config.all2all_backend == "flashinfer_all2allv"
    assert parallel_config.worker_cls == worker
    assert args.all2all_backend == "allgather_reducescatter"
    if keeps_dbo:
        assert cfg.post_init_backend == "deepep_low_latency"


def test_config_validation_restores_npu_snapshot_when_platform_update_fails(
    monkeypatch,
):
    arg_utils_module, npu_platform, _events = _install_fake_npu_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True, ubatch_size=2, fail_update=True)

    with pytest.raises(RuntimeError, match="upstream config failure"):
        arg_utils_module.EngineArgs.create_engine_config(args)

    cfg = npu_platform.last_config
    assert cfg.parallel_config.enable_dbo is True
    assert cfg.parallel_config.ubatch_size == 2
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert args.all2all_backend == "allgather_reducescatter"
