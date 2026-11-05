# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU row/storage contracts with native imports replaced at module boundaries.

Tensor operations and AFD metadata materialization are real. The DSA builder
double models bd42eea0's padded/actual row contract, not its NPU operators.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import load_host_module

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


class _GraphMode(Enum):
    NONE = 0
    FULL = 1


class _DSAMetadata(SimpleNamespace):
    pass


class _RopeProxy:
    def __init__(self, data, is_cos=True):
        self._data = data
        self.idx = 0 if is_cos else 1

    def __getitem__(self, _layer):
        return self._data["rope"]["default"][self.idx]


def _rope(positions, use_cache=False):
    values = positions.float().reshape(-1, 1, 1, 1).clone()
    data = {"rope": {"default": (values, values + 1000)}}
    return _RopeProxy(data), _RopeProxy(data, is_cos=False)


class _DSAGroup:
    def __init__(self, ratio):
        self.ratio = ratio
        self.compressor_buffers = tuple(torch.empty(16) for _ in range(3))

    def build(self, values, full_graph):
        # bd42eea0 dsa_v1.py:1014 and :1565-1572. FULL uses physical rows;
        # eager uses actual tokens, including the min() bound for tiny batches.
        tokens = (
            values["num_input_tokens"] if full_graph else values["num_actual_tokens"]
        )
        rows = min(tokens, tokens // self.ratio + values["num_reqs"])
        cos, sin = _rope(values["positions"][:tokens])
        table = values["block_tables"][0]
        table[values["num_actual_reqs"] :].zero_()
        for buffer in self.compressor_buffers:
            buffer[:rows].copy_(values["positions"][:rows])
        return _DSAMetadata(
            req_metadata=SimpleNamespace(
                num_compressed_tokens=rows,
                compressor_metadata=tuple(
                    buffer[:rows] for buffer in self.compressor_buffers
                ),
                block_table=table[: values["num_reqs"]],
                cos=cos,
                sin=sin,
            )
        )


def _build_metadata(*, full_graph_mode=False, for_cudagraph_capture=False, **values):
    group = values["attn_groups"][0][0]
    # bd42eea0 attn_utils.py:588 also enables the padded contract at capture.
    metadata = group.build(values, full_graph_mode or for_cudagraph_capture)
    return {"layer": metadata}


class _NativeUBatchRunner:
    model_state: SimpleNamespace
    stage_attn_groups: list[list[list[_DSAGroup]]]
    kv_cache_config: object

    def prepare(self, batch, tables, slots, cg_mode, for_capture):
        return self.model_state.prepare_attn(
            batch,
            cg_mode,
            tables,
            slots,
            self.stage_attn_groups[0],
            self.kv_cache_config,
            for_capture=for_capture,
        )


@pytest.fixture
def runtime(monkeypatch):
    def stub(name, **attributes):
        parent, _, field = name.rpartition(".")
        if parent and parent not in sys.modules:
            stub(parent)
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        if parent:
            monkeypatch.setattr(sys.modules[parent], field, module, raising=False)
        return module

    # Unused runtime entry points stay inert; no device or graph is simulated.
    for name, exports in {
        "vllm.forward_context": (
            "ForwardContext get_forward_context "
            "override_forward_context set_forward_context"
        ),
        "vllm.v1.kv_cache_interface": "KVCacheConfig",
        "vllm.v1.worker.gpu.cudagraph_utils": (
            "BatchExecutionDescriptor ModelCudaGraphManager"
        ),
        "vllm.v1.worker.gpu.input_batch": "InputBatch",
        "vllm.v1.worker.gpu.model_states.interface": "ModelState",
        "vllm.v1.worker.ubatch_utils": "UBatchSlices",
        "vllm.v1.worker.utils": "AttentionGroup",
        "vllm_ascend.compilation.acl_graph": (
            "GraphParams get_graph_params update_full_graph_params"
        ),
        "vllm_ascend.worker.v2.aclgraph_utils": (
            "ModelAclGraphManager _get_graph_update_backend"
        ),
        "afd_plugin.v1.worker.npu.mla_graph": (
            "merge_mla_graph_params new_mla_graph_params override_mla_graph_params"
        ),
    }.items():
        stub(name, **dict.fromkeys(exports.split(), object))
    stub(
        "vllm.config",
        CUDAGraphMode=_GraphMode,
        VllmConfig=object,
        set_current_vllm_config=nullcontext,
    )
    stub("vllm.platforms", current_platform=object())
    stub("vllm.v1.worker.ubatching")
    stub("vllm.v1.worker.dp_utils", skip_dp_coordination=nullcontext)
    stub("vllm.v1.worker.workspace", use_workspace_lane=nullcontext)
    stub(
        "vllm.v1.worker.gpu.ubatch_utils",
        UBatchRunner=_NativeUBatchRunner,
        UBatchState=object,
    )
    stub("vllm_ascend.worker.v2.input_batch", AscendInputBatch=object)
    stub(
        "vllm_ascend.worker.v2.attn_utils",
        build_attn_metadata=_build_metadata,
        build_attn_state=lambda *args: "eager",
    )
    stub(
        "vllm_ascend.ascend_forward_context",
        get_mrv2_in_profile_run=lambda: False,
        override_mrv2_in_profile_run=nullcontext,
    )
    stub("afd_plugin.compat.npu.runtime_config", uses_async_cam_dbo=lambda _: False)
    stub("afd_plugin.compat.patches.npu.mla_graph", AFD_MLA_GRAPH_PARAMS_KEY="mla")
    stub(
        "afd_plugin.compat.patches.npu.mrv2_ubatching",
        **dict.fromkeys(
            (
                "use_async_cam_child_dp_metadata",
                "use_npu_input_batch_slicing",
                "use_npu_ubatch_contexts",
                "use_npu_ubatch_output_merge",
            ),
            nullcontext,
        ),
    )
    stub(
        "vllm.v1.attention.backend",
        AttentionMetadata=object,
        AttentionMetadataBuilder=object,
    )
    stub(
        "vllm_ascend.attention.dsa_v1",
        AscendDSAMetadata=_DSAMetadata,
        AscendDSAMetadataBuilder=object,
    )
    stub(
        "vllm_ascend.attention.sfa_v1",
        AscendSFAMetadata=type("SFA", (), {}),
        AscendSFAMetadataBuilder=object,
    )
    stub("vllm_ascend.attention.mla_v1", AscendMLAMetadata=type("MLA", (), {}))
    stub("vllm_ascend.attention.utils", AscendCommonAttentionMetadata=object)
    stub(
        "vllm_ascend.ops.rope_dsv4", RopeDataProxy=_RopeProxy, get_cos_and_sin_dsa=_rope
    )
    load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_attention_metadata"
    )
    module = load_host_module(monkeypatch, "afd_plugin.v1.worker.npu.ubatch_runner")
    return module


@pytest.fixture
def runner(runtime):
    value = object.__new__(runtime.NPUUBatchRunner)
    value.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )
    value.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=1, decode_context_parallel_size=1
    )
    value.model_state = SimpleNamespace(max_model_len=128)
    value.kv_cache_config = object()
    value.stage_attn_groups = [[[_DSAGroup(4)]], [[_DSAGroup(4)]]]
    value.ubatch_seq_lens = [torch.zeros(4), torch.zeros(4)]
    value._uses_mla_graph_params = False
    value._stage_block_tables = [None, None]
    value._stage_rope_buffers = [{}, {}]
    return value


def _batch(actual, padded, *, offset=0):
    return SimpleNamespace(
        num_reqs=1 if actual else 0,
        num_reqs_after_padding=min(2, padded),
        num_tokens=actual,
        num_tokens_after_padding=padded,
        query_start_loc=torch.tensor([0, actual, actual]),
        query_start_loc_np=np.array([0, actual, actual]),
        num_scheduled_tokens=np.array([actual]),
        seq_lens=torch.tensor([actual, 0]),
        seq_lens_np=np.array([actual, 0]),
        seq_lens_cpu_upper_bound=torch.tensor([actual, 0]),
        positions=torch.arange(padded) + offset,
        attn_state="decode",
        is_prefilling_np=np.array([False, False]),
    )


def _prepare(runner, batch, mode, *, stage=0, capture=False, table=None):
    if table is None:
        table = torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    return runner._prepare_attn(
        batch,
        mode,
        (table,),
        torch.arange(batch.num_tokens_after_padding)[None, :],
        runner.stage_attn_groups[stage],
        runner.kv_cache_config,
        for_capture=capture,
        ubatch_idx=stage,
    )["layer"].req_metadata


@pytest.mark.parametrize("capture", [False, True], ids=["replay", "capture"])
@pytest.mark.parametrize(
    ("actual", "padded", "ratio", "rows"),
    [
        (1, 1, 4, 1),
        (1, 2, 4, 2),
        (3, 5, 4, 3),
        (3, 7, 128, 2),
    ],
)
def test_full_metadata_covers_small_and_nondivisible_padded_rows(
    runner, capture, actual, padded, ratio, rows
):
    runner.stage_attn_groups[0][0][0] = _DSAGroup(ratio)
    batch = _batch(actual, padded)
    metadata = _prepare(runner, batch, _GraphMode.FULL, capture=capture)

    assert metadata.num_compressed_tokens == rows
    assert metadata.compressor_metadata[0].shape == (rows,)
    torch.testing.assert_close(metadata.cos["layer"].flatten(), batch.positions.float())
    torch.testing.assert_close(
        metadata.sin["layer"].flatten(), batch.positions.float() + 1000
    )


@pytest.mark.parametrize(
    ("actual", "padded", "rows"), [(1, 2, 1), (3, 5, 1), (5, 7, 2)]
)
def test_eager_metadata_keeps_actual_rows(runner, actual, padded, rows):
    batch = _batch(actual, padded)
    metadata = _prepare(runner, batch, _GraphMode.NONE)

    assert metadata.num_compressed_tokens == rows
    assert metadata.compressor_metadata[0].shape == (rows,)
    torch.testing.assert_close(
        metadata.cos["layer"].flatten(), batch.positions[:actual].float()
    )


def test_full_stage_storage_stays_isolated_and_replay_updates_captured_views(runner):
    source = torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    captured = [
        _prepare(
            runner,
            _batch(3, 5, offset=stage * 20),
            _GraphMode.FULL,
            stage=stage,
            capture=True,
            table=source,
        )
        for stage in range(2)
    ]
    pointers = [metadata.cos["layer"].data_ptr() for metadata in captured]
    assert pointers[0] != pointers[1]
    assert captured[0].block_table.data_ptr() != captured[1].block_table.data_ptr()
    assert (
        captured[0].compressor_metadata[0].data_ptr()
        != captured[1].compressor_metadata[0].data_ptr()
    )
    second_before = captured[1].cos["layer"].clone()
    live_table = torch.tensor([[41, 42], [43, 44]], dtype=torch.int32)

    replay = _prepare(
        runner, _batch(4, 5, offset=10), _GraphMode.FULL, table=live_table
    )

    assert replay.cos["layer"].data_ptr() == pointers[0]
    assert replay.block_table.data_ptr() == captured[0].block_table.data_ptr()
    torch.testing.assert_close(
        captured[0].cos["layer"].flatten(), torch.arange(10, 15).float()
    )
    torch.testing.assert_close(captured[1].cos["layer"], second_before)
    torch.testing.assert_close(
        captured[0].block_table, torch.tensor([[41, 42], [0, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        source, torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        live_table, torch.tensor([[41, 42], [43, 44]], dtype=torch.int32)
    )


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("existing_override", [False, True])
def test_prepare_restores_native_metadata_factory(
    runner, monkeypatch, fail, existing_override
):
    def original(*_args, **_kwargs):
        return None

    if existing_override:
        runner.model_state.prepare_attn = original
    if fail:

        def failed_prepare(*args, **kwargs):
            raise RuntimeError("native preparation failed")

        monkeypatch.setattr(_NativeUBatchRunner, "prepare", failed_prepare)
    batch = _batch(3, 5)
    table = torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    slots = torch.arange(5)[None, :]

    if fail:
        with pytest.raises(RuntimeError, match="native preparation failed"):
            runner.prepare(batch, (table,), slots, _GraphMode.FULL)
    else:
        result = runner.prepare(batch, (table,), slots, _GraphMode.FULL)
        assert result["layer"].req_metadata.num_compressed_tokens == 3
    if existing_override:
        assert runner.model_state.prepare_attn is original
    else:
        assert "prepare_attn" not in vars(runner.model_state)
