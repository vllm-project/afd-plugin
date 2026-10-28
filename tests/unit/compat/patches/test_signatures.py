# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""One table owns upstream signature checks for scoped runtime patches."""

import ast
import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

PATCHES = "afd_plugin.compat.patches"
ENGINE_CORE = "vllm.v1.engine.core"
GPU_RUNNER = "vllm.v1.worker.gpu.model_runner"
GPU_DP = "vllm.v1.worker.gpu.dp_utils"
GPU_UBATCH = "vllm.v1.worker.gpu.ubatch_utils"


def _resolved_signature(function, namespace):
    signature = inspect.signature(function)
    hints = get_type_hints(function, globalns=namespace, localns=namespace)
    return signature.replace(
        parameters=[
            parameter.replace(annotation=hints.get(name, parameter.annotation))
            for name, parameter in signature.parameters.items()
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )


def _upstream_signature(module, owner, symbol, extra_namespace):
    # Read the installed upstream definition because general-plugin import may
    # already have replaced its live binding. Never compare a patch to itself.
    tree = ast.parse(Path(module.__file__).read_text())
    nodes = tree.body
    if owner:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    function = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    )
    function.body = [ast.Pass()]
    function.decorator_list = []
    definition = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = vars(module) | extra_namespace
    exec(
        compile(ast.fix_missing_locations(definition), module.__file__, "exec"),
        namespace,
    )
    return _resolved_signature(namespace[symbol], namespace)


@pytest.mark.parametrize(
    "target_module, owner, symbol, adapter_module, scope_name",
    [
        pytest.param(
            ENGINE_CORE, "EngineCore", "__init__", "engine_core", None, id="engine-init"
        ),
        pytest.param(
            ENGINE_CORE,
            "EngineCore",
            "_initialize_kv_caches",
            "engine_core",
            None,
            id="engine-kv",
        ),
        pytest.param(
            ENGINE_CORE,
            "EngineCore",
            "shutdown",
            "engine_core",
            None,
            id="engine-shutdown",
        ),
        pytest.param(
            ENGINE_CORE,
            "EngineCoreProc",
            "run_busy_loop",
            "engine_core",
            None,
            id="engine-loop",
        ),
        pytest.param(
            ENGINE_CORE,
            "DPEngineCoreProc",
            "run_busy_loop",
            "engine_core",
            None,
            id="dp-engine-loop",
        ),
        pytest.param(
            ENGINE_CORE,
            "EngineCoreProc",
            "run_engine_core",
            "async_dp_engine",
            None,
            id="engine-process",
        ),
        pytest.param(
            "vllm.v1.engine.utils",
            None,
            "launch_core_engines",
            "async_dp_engine",
            None,
            id="engine-launch",
        ),
        pytest.param(
            "vllm.v1.engine.core_client",
            "DPAsyncMPClient",
            "add_request_async",
            "async_dp_engine",
            None,
            id="dp-request",
        ),
        pytest.param(
            "vllm.forward_context",
            None,
            "set_forward_context",
            "async_dp_forward_context",
            None,
            id="forward-context",
        ),
        pytest.param(
            "vllm.engine.arg_utils",
            "EngineArgs",
            "create_engine_config",
            "config_validation",
            None,
            id="engine-config",
        ),
        pytest.param(
            "vllm.config.vllm",
            "VllmConfig",
            "__post_init__",
            "config_validation",
            None,
            id="config-revalidation",
        ),
        pytest.param(
            GPU_UBATCH,
            None,
            "maybe_build_ubatch_runner",
            "gpu_mrv2_ubatching",
            None,
            id="dbo-factory",
        ),
        pytest.param(
            GPU_UBATCH,
            "UBatchRunner",
            "_make_forward_contexts",
            "gpu_mrv2_ubatching",
            None,
            id="dbo-contexts",
        ),
        pytest.param(
            GPU_DP,
            None,
            "dispatch_cg_and_sync_dp",
            "gpu_mrv2_ubatching",
            None,
            id="dbo-dispatch",
        ),
        pytest.param(
            "vllm.v1.worker.gpu.cudagraph_utils",
            None,
            "prepare_inputs_to_capture",
            None,
            "capture-inputs",
            id="capture-inputs",
        ),
        pytest.param(
            GPU_RUNNER,
            None,
            "dispatch_cg_and_sync_dp",
            None,
            "camp2p-dispatch",
            id="camp2p-dispatch",
        ),
        pytest.param(
            "vllm.v1.worker.workspace",
            "WorkspaceManager",
            "_get_workspace_id",
            None,
            "v4-workspace-slot",
            id="v4-workspace-slot",
        ),
        pytest.param(
            GPU_UBATCH,
            "UBatchRunner",
            "prepare",
            None,
            "ubatch-prepare",
            id="ubatch-prepare",
        ),
        pytest.param(
            "vllm.v1.worker.gpu.cudagraph_utils",
            "ModelCudaGraphManager",
            "run_fullgraph",
            None,
            "fullgraph-replay",
            id="fullgraph-replay",
        ),
    ],
)
def test_patch_preserves_upstream_signature(
    target_module, owner, symbol, adapter_module, scope_name
):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if scope_name == "camp2p-dispatch":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
    if scope_name == "capture-inputs":
        from vllm.v1.worker.gpu.pcp_manager import PCPManager  # noqa: F401
    target = importlib.import_module(target_module)
    expected_module = (
        importlib.import_module(GPU_DP) if scope_name == "camp2p-dispatch" else target
    )
    expected = _upstream_signature(expected_module, owner, symbol, locals())
    binding = target
    if scope_name == "capture-inputs":
        from afd_plugin.v1.worker import attention_model_runner_v2 as adapter

        tracker = adapter._AFDCaptureEventTracker([])
        scope = adapter._use_afd_capture_input_preparation(None, tracker)
    elif scope_name == "camp2p-dispatch":
        from afd_plugin.v1.worker.npu import attention_model_runner_v2 as adapter

        scope = adapter._use_camp2p_dp_padding(4, 2)
    elif scope_name == "v4-workspace-slot":
        from afd_plugin.model_executor.models import deepseek_v4 as adapter

        native_class = getattr(target, owner)
        binding = native_class.__new__(native_class)
        scope = adapter._use_flashmla_workspace_slot(binding, 0)
    elif scope_name in {"ubatch-prepare", "fullgraph-replay"}:
        from afd_plugin.v1.worker import attention_model_runner_v2 as adapter

        native_class = getattr(target, owner)
        binding = native_class.__new__(native_class)
        if scope_name == "ubatch-prepare":
            model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v2"))
            scope = adapter._use_afd_ubatch_preparation(
                SimpleNamespace(
                    ubatch_runner=binding,
                    vllm_config=SimpleNamespace(model_config=model),
                )
            )
        else:
            scope = adapter._use_afd_fullgraph_replay_hook(
                SimpleNamespace(cudagraph_manager=binding), 1
            )
    else:
        adapter = importlib.import_module(f"{PATCHES}.{adapter_module}")
        scope = nullcontext()
    original = getattr(binding, symbol) if scope_name else None
    with scope:
        patched = getattr(binding if scope_name else adapter, symbol)
        callable_patch = patched.__func__ if inspect.ismethod(patched) else patched
        namespace = vars(target) | vars(adapter) | locals()
        if target_module == ENGINE_CORE and symbol == "run_busy_loop":
            # The pinned native decorator has no __wrapped__; compare both its
            # exposed signature and the original function retained in its closure.
            wrapped = callable_patch
            callable_patch = inspect.getclosurevars(wrapped).nonlocals["busy_loop_func"]
            native_wrapper = target.fault_tolerant_wrapper(callable_patch)
            assert _resolved_signature(wrapped, namespace) == _resolved_signature(
                native_wrapper, namespace
            )
        assert _resolved_signature(callable_patch, namespace) == expected
        if scope_name:
            assert patched is not original
    if scope_name:
        assert getattr(binding, symbol) == original
