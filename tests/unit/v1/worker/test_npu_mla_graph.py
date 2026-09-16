# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

# Fake dependency modules in this file intentionally receive attributes at
# runtime before the plugin module under test imports them.
# mypy: disable-error-code="attr-defined"
import importlib
import sys
from contextlib import contextmanager
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
NPU_RUNTIME_MODULES = ("vllm", "vllm_ascend", "torch_npu")


def _reload_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> ModuleType:
    """Reload a module and restore both import caches after the test."""
    package_name, module_attribute = module_name.rsplit(".", 1)
    package = importlib.import_module(package_name)
    monkeypatch.setattr(package, module_attribute, None, raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def _preimport_real_module(module_name: str) -> None:
    """Import the real module before the fake sys.modules swap.

    ``_reload_module`` only restores state monkeypatch recorded: when the real
    module was never imported, teardown has no original sys.modules entry or
    parent-package attribute to restore, and the fresh fake-bound module leaks
    into later tests through the package attribute. Pre-importing guarantees a
    genuine original exists to restore. CPU-only environments use the fake
    path when the NPU runtime is unavailable; failures from an installed
    runtime remain visible.
    """
    if not all(find_spec(dependency) is not None for dependency in NPU_RUNTIME_MODULES):
        return
    importlib.import_module(module_name)


def test_preimport_real_module_surfaces_installed_runtime_failures(monkeypatch):
    def fail_import(_module_name):
        raise RuntimeError("ABI mismatch")

    monkeypatch.setattr(
        f"{__name__}.find_spec",
        lambda _dependency: SimpleNamespace(),
    )
    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _preimport_real_module("afd_plugin.v1.worker.npu.mla_graph")


def _load_mla_graph_module(monkeypatch):
    _preimport_real_module("afd_plugin.v1.worker.npu.mla_graph")
    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_compilation = ModuleType("vllm_ascend.compilation")
    fake_compilation.__path__ = []
    fake_acl_graph = ModuleType("vllm_ascend.compilation.acl_graph")

    class GraphParams:
        def __init__(self, events, workspaces, handles, attn_params):
            self.events = events
            self.workspaces = workspaces
            self.handles = handles
            self.attn_params = attn_params

    fake_acl_graph.GraphParams = GraphParams
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.compilation",
        fake_compilation,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.compilation.acl_graph",
        fake_acl_graph,
    )

    module_name = "afd_plugin.v1.worker.npu.mla_graph"
    return _reload_module(monkeypatch, module_name)


def _graph_params(
    num_tokens,
    workspace,
    events,
    handles,
    attn_params,
):
    return SimpleNamespace(
        events={num_tokens: list(events)},
        workspaces={num_tokens: workspace},
        handles={num_tokens: list(handles)},
        attn_params={num_tokens: list(attn_params)},
    )


def _load_forward_context_module(monkeypatch):
    _preimport_real_module("afd_plugin.v1.worker.ubatch_wrapper")
    _preimport_real_module("afd_plugin.v1.worker.npu.forward_context")
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")

    class CUDAGraphMode:
        NONE = object()
        FULL = object()

    fake_config.CUDAGraphMode = CUDAGraphMode
    fake_config.VllmConfig = object
    fake_distributed = ModuleType("vllm.distributed")
    fake_distributed.get_dp_group = lambda: SimpleNamespace(world_size=1)
    fake_distributed.get_tensor_model_parallel_world_size = lambda: 1
    fake_forward_context = ModuleType("vllm.forward_context")

    class ForwardContext(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    fake_forward_context.BatchDescriptor = object
    fake_forward_context.DPMetadata = object
    fake_forward_context.ForwardContext = ForwardContext
    fake_vllm_v1 = ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_worker = ModuleType("vllm.v1.worker")
    fake_vllm_worker.__path__ = []
    fake_ubatch_utils = ModuleType("vllm.v1.worker.ubatch_utils")
    fake_ubatch_utils.UBatchSlices = list

    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_ops = ModuleType("vllm_ascend.ops")
    fake_ops.__path__ = []
    fake_fused_moe = ModuleType("vllm_ascend.ops.fused_moe")
    fake_fused_moe.__path__ = []
    fake_comm_method = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_comm_method",
    )
    fake_comm_method.get_moe_comm_method = lambda value: value
    fake_afd_ubatch = ModuleType("afd_plugin.v1.worker.ubatch_wrapper")
    fake_afd_ubatch.build_ubatch_additional_kwargs = lambda kwargs, metadata: {
        **kwargs,
        "afd_metadata": metadata,
    }
    fake_afd_ubatch.build_ubatch_afd_metadata = lambda metadata, _slices, _index: (
        metadata
    )

    modules = {
        "vllm": fake_vllm,
        "vllm.config": fake_config,
        "vllm.distributed": fake_distributed,
        "vllm.forward_context": fake_forward_context,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.worker": fake_vllm_worker,
        "vllm.v1.worker.ubatch_utils": fake_ubatch_utils,
        "vllm_ascend": fake_ascend,
        "vllm_ascend.ops": fake_ops,
        "vllm_ascend.ops.fused_moe": fake_fused_moe,
        "vllm_ascend.ops.fused_moe.moe_comm_method": fake_comm_method,
        "afd_plugin.v1.worker.ubatch_wrapper": fake_afd_ubatch,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "afd_plugin.v1.worker.npu.forward_context"
    return _reload_module(monkeypatch, module_name)


def _load_ubatch_wrapper_module(monkeypatch):
    _preimport_real_module("afd_plugin.v1.worker.ubatch_wrapper")

    class FakeStream:
        def __init__(self, device=None):
            self.device = device

        def synchronize(self):
            return None

    class FakeNPUGraph:
        def replay(self):
            return None

    fake_npu = SimpleNamespace(
        Stream=FakeStream,
        NPUGraph=FakeNPUGraph,
        current_device=lambda: 0,
        current_stream=lambda: FakeStream(),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")

    class CUDAGraphMode:
        NONE = object()
        FULL = object()
        PIECEWISE = object()

    fake_config.CUDAGraphMode = CUDAGraphMode
    fake_config.VllmConfig = object
    fake_distributed = ModuleType("vllm.distributed")
    fake_distributed.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    fake_distributed.get_tensor_model_parallel_world_size = lambda: 1
    fake_distributed.tensor_model_parallel_all_gather = lambda value, dim=0: value
    fake_forward_context = ModuleType("vllm.forward_context")

    class DPMetadata:
        @staticmethod
        def make(*_args, **_kwargs):
            return None

    @contextmanager
    def override_forward_context(_context):
        yield

    fake_forward_context.DPMetadata = DPMetadata
    fake_forward_context.ForwardContext = object
    fake_forward_context.get_forward_context = lambda: None
    fake_forward_context.override_forward_context = override_forward_context
    fake_sequence = ModuleType("vllm.sequence")
    fake_sequence.IntermediateTensors = type("IntermediateTensors", (), {})
    fake_vllm_v1 = ModuleType("vllm.v1")
    fake_vllm_v1.__path__ = []
    fake_vllm_worker = ModuleType("vllm.v1.worker")
    fake_vllm_worker.__path__ = []
    fake_gpu_wrapper = ModuleType("vllm.v1.worker.gpu_ubatch_wrapper")
    fake_gpu_wrapper.UbatchMetadata = object
    fake_gpu_wrapper.UBatchWrapper = object

    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_compilation = ModuleType("vllm_ascend.compilation")
    fake_compilation.__path__ = []
    fake_acl_graph = ModuleType("vllm_ascend.compilation.acl_graph")

    class ACLGraphWrapper:
        pass

    class GraphParams:
        def __init__(self, events, workspaces, handles, attn_params):
            self.events = events
            self.workspaces = workspaces
            self.handles = handles
            self.attn_params = attn_params

    fake_acl_graph.ACLGraphWrapper = ACLGraphWrapper
    fake_acl_graph.GraphParams = GraphParams
    fake_acl_graph.get_graph_params = lambda: None
    fake_ascend_utils = ModuleType("vllm_ascend.utils")
    fake_ascend_utils.enable_sp = lambda: False
    fake_child_context = ModuleType(
        "afd_plugin.v1.worker.npu.forward_context",
    )
    fake_child_context.create_ascend_forward_context = lambda *_args, **_kwargs: (
        SimpleNamespace()
    )
    fake_ubatching = ModuleType("afd_plugin.v1.worker.npu.ubatching")
    fake_ubatching.AscendUBatchContext = object
    fake_ubatching.make_ubatch_contexts = lambda **_kwargs: []

    modules = {
        "vllm": fake_vllm,
        "vllm.config": fake_config,
        "vllm.distributed": fake_distributed,
        "vllm.forward_context": fake_forward_context,
        "vllm.sequence": fake_sequence,
        "vllm.v1": fake_vllm_v1,
        "vllm.v1.worker": fake_vllm_worker,
        "vllm.v1.worker.gpu_ubatch_wrapper": fake_gpu_wrapper,
        "vllm_ascend": fake_ascend,
        "vllm_ascend.compilation": fake_compilation,
        "vllm_ascend.compilation.acl_graph": fake_acl_graph,
        "vllm_ascend.utils": fake_ascend_utils,
        "afd_plugin.v1.worker.npu.forward_context": fake_child_context,
        "afd_plugin.v1.worker.npu.ubatching": fake_ubatching,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    mla_graph_name = "afd_plugin.v1.worker.npu.mla_graph"
    _reload_module(monkeypatch, mla_graph_name)
    wrapper_name = "afd_plugin.v1.worker.npu.npu_ubatch_wrapper"
    return _reload_module(monkeypatch, wrapper_name)


def _parent_forward_context():
    return SimpleNamespace(
        additional_kwargs={},
        all_moe_layers={},
        moe_comm_type="mc2",
        in_profile_run=False,
        capturing=False,
        mmrs_fusion=False,
        device_metadata_executor=None,
        is_decode_only_node=False,
        use_mega_moe=False,
        is_padding=None,
        is_first_layer=True,
        layer_idx=0,
        prefetch_mlp_gate_up_proj=False,
        prefetch_mlp_down_proj=False,
        model_instance=None,
        is_draft_model=False,
        is_draft_model_prefill=False,
        draft_attn_metadatas=None,
        max_tokens_across_pcp=None,
        sinks=None,
        input_ids=None,
        eplb_heat_collection_status=None,
        mc2_mask=None,
    )


def _two_slices(first, second):
    return [
        SimpleNamespace(
            request_slice=slice(0, 1),
            token_slice=slice(0, first),
            num_tokens=first,
        ),
        SimpleNamespace(
            request_slice=slice(1, 2),
            token_slice=slice(first, first + second),
            num_tokens=second,
        ),
    ]


def _batch_descriptor(
    num_tokens=8,
    *,
    has_lora=False,
    num_active_loras=0,
):
    return SimpleNamespace(
        num_tokens=num_tokens,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


def _new_wrapper_for_unit_test(wrapper_module, *, mla_full_graph_enabled):
    wrapper = object.__new__(wrapper_module.AscendUBatchWrapper)
    wrapper.mla_full_graph_enabled = mla_full_graph_enabled
    wrapper.cudagraphs = {}
    wrapper.cudagraph_wrapper = None
    wrapper.runnable = lambda **_kwargs: "eager"
    wrapper.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    return wrapper


class _RecordingCallable:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_ubatch_wrapper_rejects_enpu(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)

    with pytest.raises(AssertionError, match="does not support ENPU"):
        wrapper_module.AscendUBatchWrapper(
            lambda: None,
            SimpleNamespace(compilation_config=object()),
            wrapper_module.CUDAGraphMode.NONE,
            torch.device("cpu"),
            enable_enpu=True,
        )


def test_ubatch_wrapper_requires_native_two_ubatch_config(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(num_ubatches=0),
        compilation_config=object(),
    )

    with pytest.raises(AssertionError):
        wrapper_module.AscendUBatchWrapper(
            lambda: None,
            config,
            wrapper_module.CUDAGraphMode.NONE,
            torch.device("cpu"),
        )


def test_npu_graph_key_separates_stage_shapes_and_lora(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)

    keys = {
        wrapper_module.AscendNPUGraphKey((4, 4), False, 0),
        wrapper_module.AscendNPUGraphKey((3, 5), False, 0),
        wrapper_module.AscendNPUGraphKey((4, 4), True, 1),
        wrapper_module.AscendNPUGraphKey((4, 4), True, 2),
    }

    assert len(keys) == 4


def test_merge_mla_graph_params_is_layer_major_ubatch_minor(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    stage_params = (
        _graph_params(
            4,
            workspace,
            ["e00", "e10"],
            ["h00", "h10"],
            ["p00", "p10"],
        ),
        _graph_params(
            4,
            workspace,
            ["e01", "e11"],
            ["h01", "h11"],
            ["p01", "p11"],
        ),
    )
    metadata = [
        {"layer0": "m00", "layer1": "m10"},
        {"layer0": "m01", "layer1": "m11"},
    ]

    merged_metadata, merged = mla_graph.merge_mla_graph_params(
        metadata,
        stage_params,
        4,
    )

    assert list(merged_metadata) == [
        ("layer0", 0),
        ("layer0", 1),
        ("layer1", 0),
        ("layer1", 1),
    ]
    assert list(merged_metadata.values()) == ["m00", "m01", "m10", "m11"]
    assert merged.events[4] == ["e00", "e01", "e10", "e11"]
    assert merged.handles[4] == ["h00", "h01", "h10", "h11"]
    assert merged.attn_params[4] == ["p00", "p01", "p10", "p11"]
    assert merged.workspaces[4] is workspace


def test_merge_mla_graph_params_requires_two_metadata_stages(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
        _graph_params(4, workspace, ["e1"], ["h1"], ["p1"]),
    )

    with pytest.raises(RuntimeError, match="exactly two metadata stages"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_requires_matching_layer_order(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(
            4,
            workspace,
            ["e00", "e10"],
            ["h00", "h10"],
            ["p00", "p10"],
        ),
        _graph_params(
            4,
            workspace,
            ["e01", "e11"],
            ["h01", "h11"],
            ["p01", "p11"],
        ),
    )

    with pytest.raises(RuntimeError, match="layer order differs"):
        mla_graph.merge_mla_graph_params(
            [
                {"layer0": "m00", "layer1": "m10"},
                {"layer1": "m11", "layer0": "m01"},
            ],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_requires_shared_workspace(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    graph_params = (
        _graph_params(4, object(), ["e0"], ["h0"], ["p0"]),
        _graph_params(4, object(), ["e1"], ["h1"], ["p1"]),
    )

    with pytest.raises(RuntimeError, match="shared FIA workspace"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}, {"layer0": "m1"}],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_rejects_record_count_mismatch(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
        _graph_params(4, workspace, [], [], []),
    )

    with pytest.raises(RuntimeError, match="record count mismatch"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}, {"layer0": "m1"}],
            graph_params,
            4,
        )


def test_override_mla_graph_params_restores_context_after_error(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    original_metadata = [{"layer0": "old"}]
    original_kwargs = {"afd_metadata": object()}
    context = SimpleNamespace(
        attn_metadata=original_metadata,
        additional_kwargs=original_kwargs,
    )
    merged_metadata = {("layer0", 0): "new"}
    merged_params = object()

    with (
        pytest.raises(RuntimeError, match="update failed"),
        mla_graph.override_mla_graph_params(
            context,
            merged_metadata,
            merged_params,
        ),
    ):
        assert context.attn_metadata is merged_metadata
        assert context.additional_kwargs is not original_kwargs
        assert (
            context.additional_kwargs["afd_metadata"]
            is (original_kwargs["afd_metadata"])
        )
        assert context.additional_kwargs["afd_mla_graph_params"] is merged_params
        raise RuntimeError("update failed")

    assert context.attn_metadata is original_metadata
    assert context.additional_kwargs is original_kwargs


def test_child_forward_context_installs_mla_capture_registry(monkeypatch):
    forward_context = _load_forward_context_module(monkeypatch)
    registry = object()

    child = forward_context.create_ascend_forward_context(
        _parent_forward_context(),
        attn_metadata=None,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        ubatch_slices=_two_slices(4, 4),
        ubatch_num=1,
        mla_graph_params=registry,
    )

    assert child.capturing is True
    assert child.additional_kwargs["afd_mla_graph_params"] is registry
    assert child.ubatch_idx == 1
    assert child.num_ubatches == 2


def test_mla_capture_params_isolate_records_and_share_workspace(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    workspace = object()
    monkeypatch.setattr(
        wrapper_module,
        "get_graph_params",
        lambda: SimpleNamespace(workspaces={8: workspace}),
        raising=False,
    )

    registries = wrapper._new_mla_capture_params((4, 4))

    assert registries[0] is not registries[1]
    assert registries[0].events[4] is not registries[1].events[4]
    assert registries[0].handles[4] is not registries[1].handles[4]
    assert registries[0].attn_params[4] is not registries[1].attn_params[4]
    assert registries[0].workspaces[4] is workspace
    assert registries[1].workspaces[4] is workspace


def test_mla_capture_params_require_equal_stage_tokens(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_graph_params",
        lambda: SimpleNamespace(workspaces={8: object()}),
    )

    with pytest.raises(RuntimeError, match="equal padded token counts"):
        wrapper._new_mla_capture_params((2, 6))


def test_mla_capture_params_require_single_graph_workspace(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_graph_params",
        lambda: SimpleNamespace(workspaces={8: None}),
    )

    with pytest.raises(RuntimeError, match="single-batch FIA workspace"):
        wrapper._new_mla_capture_params((4, 4))


def test_single_full_graph_uses_inner_wrapper_when_ubatch_total_matches(
    monkeypatch,
):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    wrapper.cudagraphs[8] = SimpleNamespace(
        aclgraph=SimpleNamespace(replay=lambda: None),
        outputs="ubatch",
    )
    wrapper.cudagraph_wrapper = _RecordingCallable("single")
    context = SimpleNamespace(
        ubatch_slices=None,
        cudagraph_runtime_mode=wrapper_module.CUDAGraphMode.FULL,
        batch_descriptor=_batch_descriptor(),
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_forward_context",
        lambda: context,
    )

    result = wrapper(input_ids="ids")

    assert result == "single"
    assert wrapper.cudagraph_wrapper.calls == [{"input_ids": "ids"}]


def test_full_graph_capture_passes_shape_key_and_mla_registries(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    workspace = object()
    monkeypatch.setattr(
        wrapper_module,
        "get_graph_params",
        lambda: SimpleNamespace(workspaces={8: workspace}),
    )
    context = SimpleNamespace(
        ubatch_slices=_two_slices(4, 4),
        cudagraph_runtime_mode=wrapper_module.CUDAGraphMode.FULL,
        batch_descriptor=_batch_descriptor(),
        attn_metadata=[{"layer0": "m0"}, {"layer0": "m1"}],
        is_draft_model=False,
        max_tokens_across_pcp=0,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_forward_context",
        lambda: context,
    )
    captured = {}

    def make_ubatch_metadata(*_args, mla_graph_params=None, **_kwargs):
        captured["make_params"] = mla_graph_params
        return ["metadata"]

    def capture_ubatches(
        ubatch_metadata,
        model,
        *,
        graph_key,
        mla_graph_params,
    ):
        captured["metadata"] = ubatch_metadata
        captured["model"] = model
        captured["graph_key"] = graph_key
        captured["capture_params"] = mla_graph_params
        return "captured"

    wrapper._make_ubatch_metadata = make_ubatch_metadata
    wrapper._capture_ubatches = capture_ubatches

    result = wrapper(
        input_ids=object(),
        positions=object(),
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "captured"
    assert captured["graph_key"] == wrapper_module.AscendNPUGraphKey(
        (4, 4),
        False,
        0,
    )
    assert captured["make_params"] is captured["capture_params"]
    assert captured["capture_params"][0].workspaces[4] is workspace
    assert captured["capture_params"][1].workspaces[4] is workspace


def test_mla_graph_replay_updates_child_params_each_time_in_runtime_order(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    calls = []
    workspace = object()
    graph_params = (
        _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
        _graph_params(4, workspace, ["e1"], ["h1"], ["p1"]),
    )
    graph = SimpleNamespace(replay=lambda: calls.append("replay"))
    wrapper.cudagraphs[wrapper_module.AscendNPUGraphKey((4, 4), False, 0)] = (
        wrapper_module.AscendNPUGraphMetaData(
            aclgraph=graph,
            ubatch_metadata=[],
            outputs="output",
            mla_graph_params=graph_params,
        )
    )
    original_metadata = [{"layer0": "m0"}, {"layer0": "m1"}]
    original_kwargs = {"afd_metadata": object()}
    context = SimpleNamespace(
        ubatch_slices=_two_slices(4, 4),
        cudagraph_runtime_mode=wrapper_module.CUDAGraphMode.FULL,
        batch_descriptor=_batch_descriptor(),
        attn_metadata=original_metadata,
        additional_kwargs=original_kwargs,
        is_draft_model=False,
        max_tokens_across_pcp=None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_forward_context",
        lambda: context,
    )
    monkeypatch.setattr(
        wrapper_module.torch.npu,
        "current_stream",
        lambda: SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
        ),
    )

    def update_params(active_context, num_tokens):
        calls.append("update")
        assert num_tokens == 4
        assert list(active_context.attn_metadata) == [
            ("layer0", 0),
            ("layer0", 1),
        ]
        merged = active_context.additional_kwargs["afd_mla_graph_params"]
        assert merged.events[4] == ["e0", "e1"]
        assert merged.handles[4] == ["h0", "h1"]
        assert merged.attn_params[4] == ["p0", "p1"]

    wrapper.full_graph_params_updater = update_params
    position_tensor = object()

    results = [
        wrapper(
            input_ids=object(),
            positions=position_tensor,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        for _ in range(2)
    ]

    assert results == ["output", "output"]
    assert calls == ["synchronize", "replay", "update"] * 2
    assert context.attn_metadata is original_metadata
    assert context.additional_kwargs is original_kwargs
    assert context.dbo_enabled is True


def test_non_mla_graph_replay_keeps_stream_fence(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=False,
    )
    calls = []
    wrapper.cudagraphs[wrapper_module.AscendNPUGraphKey((4, 4), False, 0)] = (
        wrapper_module.AscendNPUGraphMetaData(
            aclgraph=SimpleNamespace(
                replay=lambda: calls.append("replay"),
            ),
            ubatch_metadata=[],
            outputs="output",
        )
    )
    context = SimpleNamespace(
        ubatch_slices=_two_slices(4, 4),
        cudagraph_runtime_mode=wrapper_module.CUDAGraphMode.FULL,
        batch_descriptor=_batch_descriptor(),
        attn_metadata=None,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_forward_context",
        lambda: context,
    )
    monkeypatch.setattr(
        wrapper_module.torch.npu,
        "current_stream",
        lambda: SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
        ),
    )

    result = wrapper(
        input_ids=object(),
        positions=object(),
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "output"
    assert calls == ["synchronize", "replay"]
    assert context.dbo_enabled is True


@pytest.mark.parametrize(
    (
        "mla_enabled",
        "ubatch_slices",
        "runtime_mode_name",
        "max_tokens_across_pcp",
        "expected",
    ),
    [
        (True, _two_slices(4, 4), "FULL", None, True),
        (True, _two_slices(4, 4), "FULL", 0, True),
        (False, _two_slices(4, 4), "FULL", None, False),
        (True, None, "FULL", None, False),
        (True, _two_slices(4, 4), "PIECEWISE", None, False),
    ],
)
def test_wrapper_owns_only_supported_mla_full_graph_updates(
    monkeypatch,
    mla_enabled,
    ubatch_slices,
    runtime_mode_name,
    max_tokens_across_pcp,
    expected,
):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=mla_enabled,
    )
    context = SimpleNamespace(
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=getattr(
            wrapper_module.CUDAGraphMode,
            runtime_mode_name,
        ),
        max_tokens_across_pcp=max_tokens_across_pcp,
    )

    assert wrapper.owns_full_graph_update(context) is expected


def test_mla_full_graph_rejects_pcp_before_graph_routing(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    context = SimpleNamespace(
        ubatch_slices=_two_slices(4, 4),
        cudagraph_runtime_mode=wrapper_module.CUDAGraphMode.FULL,
        batch_descriptor=_batch_descriptor(),
        attn_metadata=[{"layer0": "m0"}, {"layer0": "m1"}],
        max_tokens_across_pcp=8,
    )
    monkeypatch.setattr(
        wrapper_module,
        "get_forward_context",
        lambda: context,
    )

    with pytest.raises(
        RuntimeError,
        match="does not support PCP execution",
    ):
        wrapper(
            input_ids=object(),
            positions=object(),
            intermediate_tensors=None,
            inputs_embeds=None,
        )

    assert wrapper.cudagraphs == {}


def test_mla_graph_replay_rejects_missing_capture_registry(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    wrapper.full_graph_params_updater = lambda *_args: None
    replay_calls = []
    graph_metadata = wrapper_module.AscendNPUGraphMetaData(
        aclgraph=SimpleNamespace(
            replay=lambda: replay_calls.append("replay"),
        ),
        ubatch_metadata=[],
        mla_graph_params=None,
    )
    context = SimpleNamespace(
        attn_metadata=[{"layer0": "m0"}, {"layer0": "m1"}],
        additional_kwargs={},
    )

    with pytest.raises(RuntimeError, match="no capture registry"):
        wrapper._replay_mla_graph(graph_metadata, context, 4)

    assert replay_calls == []


def test_mla_graph_replay_rejects_missing_updater_before_replay(monkeypatch):
    wrapper_module = _load_ubatch_wrapper_module(monkeypatch)
    wrapper = _new_wrapper_for_unit_test(
        wrapper_module,
        mla_full_graph_enabled=True,
    )
    wrapper.full_graph_params_updater = None
    workspace = object()
    replay_calls: list[str] = []
    graph_metadata = wrapper_module.AscendNPUGraphMetaData(
        aclgraph=SimpleNamespace(replay=lambda: replay_calls.append("replay")),
        ubatch_metadata=[],
        mla_graph_params=(
            _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
            _graph_params(4, workspace, ["e1"], ["h1"], ["p1"]),
        ),
    )
    context = SimpleNamespace(
        attn_metadata=[{"layer0": "m0"}, {"layer0": "m1"}],
        additional_kwargs={},
    )
    with pytest.raises(RuntimeError, match="no parameter updater"):
        wrapper._replay_mla_graph(graph_metadata, context, 4)

    assert replay_calls == []
