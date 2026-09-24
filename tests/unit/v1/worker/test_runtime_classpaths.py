from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    UBATCH_WRAPPER_FQCN,
    resolve_class_from_qualname,
)

GPU_RUNTIME_CLASS_PATHS = [
    ATTENTION_WORKER_FQCN,
    ATTENTION_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    UBATCH_WRAPPER_FQCN,
    "afd_plugin.v1.worker:AFDAttentionWorker",
]

NPU_WORKER_CLASS_PATHS = [
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
]

NPU_MODEL_RUNNER_CLASS_PATHS = [
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
]

# vLLM resolves the platform and worker before the worker imports its model
# runner. Keep that production order explicit: a standalone vLLM-Ascend model
# runner import can enter device_op before the ops package finishes loading.
NPU_RUNTIME_CLASS_PATHS = NPU_WORKER_CLASS_PATHS + NPU_MODEL_RUNNER_CLASS_PATHS

_NPU_RUNTIME_RESOLUTION_SCRIPT = """
import sys

from vllm.platforms import current_platform

assert current_platform.device_type == "npu"

import afd_plugin

afd_plugin.register_afd()
force_load_balance_module = "afd_plugin.compat.patches.npu.force_load_balance"
assert force_load_balance_module not in sys.modules

from afd_plugin.validation import resolve_class_from_qualname

for qualname in sys.argv[1:]:
    cls = resolve_class_from_qualname(qualname)
    assert cls.__module__.startswith("afd_plugin.v1.worker.npu")

assert force_load_balance_module not in sys.modules
"""

V026_OVERRIDE_CONTRACTS = [
    ("attention_worker", "AFDAttentionWorker", "Worker", "__init__"),
    ("attention_worker", "AFDAttentionWorker", "Worker", "init_device"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "__init__"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "init_device"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "get_kv_cache_spec"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "initialize_from_config"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "compile_or_warm_up_model"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "execute_model"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "shutdown"),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "load_model",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_build_attention_metadata",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_determine_batch_execution_and_padding",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_model_forward",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "execute_model",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_dummy_run",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_warmup_and_capture",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "shutdown",
    ),
    ("ubatch_wrapper", "AFDUBatchWrapper", "UBatchWrapper", "__init__"),
    (
        "ubatch_wrapper",
        "AFDUBatchWrapper",
        "UBatchWrapper",
        "_make_ubatch_metadata",
    ),
]


def _call_contract(callable_obj):
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_obj).parameters.values()
    ]


@pytest.mark.parametrize(
    "qualname",
    GPU_RUNTIME_CLASS_PATHS + NPU_RUNTIME_CLASS_PATHS,
)
def test_runtime_class_paths_are_plugin_paths(qualname):
    assert qualname.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", GPU_RUNTIME_CLASS_PATHS)
def test_gpu_runtime_class_paths_resolve_when_vllm_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize(
    ("module_name", "afd_class_name", "native_class_name", "method_name"),
    V026_OVERRIDE_CONTRACTS,
)
def test_gpu_v1_overrides_match_native_call_contract(
    module_name,
    afd_class_name,
    native_class_name,
    method_name,
):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    if module_name == "attention_worker":
        from vllm.v1.worker import gpu_worker as native_module

        from afd_plugin.v1.worker import attention_worker as afd_module
    elif module_name == "ffn_worker":
        from vllm.v1.worker import gpu_worker as native_module

        from afd_plugin.v1.worker import ffn_worker as afd_module
    elif module_name == "attention_model_runner":
        from vllm.v1.worker import gpu_model_runner as native_module

        from afd_plugin.v1.worker import attention_model_runner as afd_module
    else:
        from vllm.v1.worker import gpu_ubatch_wrapper as native_module

        from afd_plugin.v1.worker import ubatch_wrapper as afd_module

    afd_method = getattr(getattr(afd_module, afd_class_name), method_name)
    native_method = getattr(getattr(native_module, native_class_name), method_name)

    assert _call_contract(afd_method) == _call_contract(native_method)


@pytest.mark.vllm_runtime
def test_npu_runtime_class_paths_resolve_when_vllm_ascend_is_available():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _NPU_RUNTIME_RESOLUTION_SCRIPT,
            *NPU_RUNTIME_CLASS_PATHS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "NPU runtime class resolution failed in an isolated process:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
