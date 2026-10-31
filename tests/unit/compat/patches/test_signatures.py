# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""One table owns upstream signature checks for scoped runtime patches."""

import ast
import importlib
import inspect
from contextlib import contextmanager, nullcontext
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


@contextmanager
def _use_installed_patch(binding, symbol, marker_name, install):
    """Exercise an installer while restoring its exact prior descriptor/marker."""
    original = inspect.getattr_static(binding, symbol)
    had_marker = marker_name in vars(binding)
    previous_marker = vars(binding).get(marker_name)
    if had_marker:
        delattr(binding, marker_name)
    try:
        assert install()
        yield
    finally:
        setattr(binding, symbol, original)
        if had_marker:
            setattr(binding, marker_name, previous_marker)
        elif marker_name in vars(binding):
            delattr(binding, marker_name)


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
        pytest.param(
            "vllm_ascend.ascend_config",
            None,
            "init_ascend_config",
            "npu.ascend_config",
            None,
            id="npu-config-factory",
        ),
        pytest.param(
            "vllm_ascend.patch.platform.patch_engine_core",
            None,
            "_run_engine_core_patch_func",
            "npu.ascend_config:run_afd_ascend_engine_core",
            None,
            id="npu-child-engine",
        ),
        pytest.param(
            GPU_RUNNER,
            None,
            "maybe_build_ubatch_runner",
            None,
            "npu-dbo-factory",
            id="npu-dbo-factory",
        ),
        pytest.param(
            GPU_RUNNER,
            None,
            "dispatch_cg_and_sync_dp",
            None,
            "npu-dbo-dispatch",
            id="npu-dbo-dispatch",
        ),
        pytest.param(
            GPU_UBATCH,
            None,
            "_slice_input_batch",
            None,
            "npu-dbo-slicing",
            id="npu-dbo-slicing",
        ),
        pytest.param(
            GPU_UBATCH,
            None,
            "merge_ubatch_outputs",
            None,
            "npu-dbo-output-merge",
            id="npu-dbo-output-merge",
        ),
        pytest.param(
            "vllm.v1.worker.gpu.cudagraph_utils",
            "ModelCudaGraphManager",
            "run_fullgraph",
            None,
            "npu-fullgraph-replay",
            id="npu-fullgraph-replay",
        ),
        pytest.param(
            "vllm_ascend.platform",
            "NPUPlatform",
            "check_and_update_config",
            None,
            "npu-platform-config",
            id="npu-platform-config",
        ),
        pytest.param(
            "vllm_ascend.attention.mla_v1",
            None,
            "get_graph_params",
            None,
            "npu-mla-graph-params",
            id="npu-mla-graph-params",
        ),
    ],
)
def test_patch_preserves_upstream_signature(  # long-test: scoped table bindings.
    target_module, owner, symbol, adapter_module, scope_name
):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    # Ascend's platform module replaces TYPE_CHECKING-only config names with
    # None at runtime. Resolve the source annotation against its real class.
    from vllm.config import VllmConfig  # noqa: F401

    if (
        target_module.startswith("vllm_ascend.")
        or adapter_module is not None
        and adapter_module.startswith("npu.")
        or scope_name is not None
        and scope_name.startswith("npu-")
        or scope_name == "camp2p-dispatch"
    ):
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
    if scope_name == "capture-inputs":
        from vllm.v1.worker.gpu.pcp_manager import PCPManager  # noqa: F401
    target = importlib.import_module(target_module)
    # These call sites import the native definition by value. Its defining
    # module owns the AST, even if the call-site alias has already been patched.
    if scope_name in {"camp2p-dispatch", "npu-dbo-dispatch"}:
        expected_module = importlib.import_module(GPU_DP)
    elif scope_name == "npu-dbo-factory":
        expected_module = importlib.import_module(GPU_UBATCH)
    elif scope_name == "npu-mla-graph-params":
        expected_module = importlib.import_module("vllm_ascend.compilation.acl_graph")
    else:
        expected_module = target
    expected = _upstream_signature(expected_module, owner, symbol, locals())
    binding = target
    patch_symbol = symbol
    auxiliary_bindings = []
    if scope_name == "capture-inputs":
        from afd_plugin.v1.worker import attention_model_runner_v2 as adapter

        tracker = adapter._AFDCaptureEventTracker([])
        scope = adapter._use_afd_capture_input_preparation(None, tracker)
    elif scope_name == "camp2p-dispatch":
        from afd_plugin.v1.worker.npu import attention_model_runner_v2 as adapter

        scope = adapter._use_camp2p_dp_padding(4, 2)
    elif scope_name == "v4-workspace-slot":
        from afd_plugin.compat.patches import gpu_deepseek_v4_dbo as adapter

        native_class = getattr(target, owner)
        binding = native_class.__new__(native_class)
        scope = adapter._use_flashmla_workspace_slot(binding, 0)
    elif scope_name in {"ubatch-prepare", "fullgraph-replay", "npu-fullgraph-replay"}:
        if scope_name == "npu-fullgraph-replay":
            from afd_plugin.v1.worker.npu import attention_model_runner_v2 as adapter
        else:
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
    elif scope_name in {
        "npu-dbo-factory",
        "npu-dbo-dispatch",
        "npu-dbo-slicing",
        "npu-dbo-output-merge",
    }:
        from afd_plugin.compat.patches.npu import mrv2_ubatching as adapter

        if scope_name == "npu-dbo-factory":
            scope = adapter.use_npu_ubatch_factory(SimpleNamespace())
            ubatch_module = importlib.import_module(GPU_UBATCH)
            auxiliary_bindings.append(
                (
                    ubatch_module,
                    "create_sm_control_context",
                    ubatch_module.create_sm_control_context,
                )
            )
        elif scope_name == "npu-dbo-dispatch":
            scope = adapter.use_async_cam_dbo_dispatch()
        elif scope_name == "npu-dbo-slicing":
            scope = adapter.use_npu_input_batch_slicing()
        else:
            scope = adapter.use_npu_ubatch_output_merge()
    elif scope_name == "npu-platform-config":
        from afd_plugin.compat.patches.npu import ascend_platform as adapter

        binding = target.NPUPlatform
        scope = _use_installed_patch(
            binding,
            symbol,
            adapter._ASCEND_PLATFORM_PATCH_ATTR,
            adapter.apply_afd_ascend_dbo_config_patch,
        )
    elif scope_name == "npu-mla-graph-params":
        from afd_plugin.compat.patches.npu import mla_graph as adapter

        scope = _use_installed_patch(
            binding,
            symbol,
            adapter._MLA_GRAPH_PATCH_ATTR,
            adapter.apply_afd_mla_graph_patch,
        )
    else:
        adapter_name, _, renamed_symbol = adapter_module.partition(":")
        adapter = importlib.import_module(f"{PATCHES}.{adapter_name}")
        patch_symbol = renamed_symbol or symbol
        scope = nullcontext()
    original = getattr(binding, symbol) if scope_name else None
    with scope:
        patched = getattr(binding if scope_name else adapter, patch_symbol)
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
            original_callable = (
                original.__func__ if inspect.ismethod(original) else original
            )
            assert callable_patch is not original_callable
    if scope_name:
        assert getattr(binding, symbol) == original
    for module, name, original_value in auxiliary_bindings:
        assert getattr(module, name) == original_value
