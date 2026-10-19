# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real graph reads of stage-owned inputs, independent of checkpoint attention.

The builders below expose the upstream mutable storage contract. NPU kernels
read the pointers captured from the actual adapter before and after live input
updates; full MLA/DSA operators and A/F transport belong to the model E2Es.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

import vllm_ascend.ops  # noqa: E402, F401
from vllm.config import CUDAGraphMode  # noqa: E402

# Match the real worker's import order before Ascend attn_utils resolves the
# native speculator module through its parent package.
from vllm.v1.worker.gpu import model_runner  # noqa: E402, F401
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata  # noqa: E402
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata  # noqa: E402
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata  # noqa: E402
from vllm_ascend.ops import rope_dsv4  # noqa: E402
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy  # noqa: E402

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_attention_metadata,
)
from afd_plugin.v1.worker.npu import ubatch_runner  # noqa: E402


@pytest.mark.parametrize("backend", ["mla", "sfa", "dsa"])
def test_full_graph_reads_updated_block_ids_and_stage_rope(monkeypatch, backend):
    if not torch.npu.is_available():
        pytest.skip("requires an allocated NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    runner = object.__new__(ubatch_runner.NPUUBatchRunner)
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8)
    )
    runner.parallel_config = object()
    runner.model_state = SimpleNamespace(max_model_len=128)
    runner.stage_attn_groups = [0, 1]
    runner._uses_mla_graph_params = backend == "mla"
    runner._stage_block_tables = [None, None]
    runner._stage_rope_buffers = [{}, {}]
    runner.ubatch_seq_lens = [torch.empty(4), torch.empty(4)]
    shared_cos = torch.empty(2, 1, device=device)
    shared_sin = torch.empty_like(shared_cos)
    built_tables = []
    if backend == "dsa":
        state = rope_dsv4.RopeGlobalState()
        state.layer_info["layer"] = ("stage-rope", ["default"])
        monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", state)

    def dsa_rope(cos, sin):
        data = {"stage-rope": {"default": (cos, sin)}}
        return RopeDataProxy(data), RopeDataProxy(data, is_cos=False)

    def build(**kwargs):
        table = kwargs["block_tables"][0]
        built_tables.append(table)
        if backend != "mla":
            table[kwargs["num_actual_reqs"] :].zero_()
        shared_cos.copy_(kwargs["positions"].float().view(2, 1) * 3)
        shared_sin.copy_(kwargs["positions"].float().view(2, 1) * 5)
        if backend == "mla":
            metadata = object.__new__(AscendMLAMetadata)
            metadata.prefill = None
            metadata.decode = SimpleNamespace(cos=shared_cos, sin=shared_sin)
        elif backend == "sfa":
            metadata = object.__new__(AscendSFAMetadata)
            metadata.cos, metadata.sin = shared_cos, shared_sin
        else:
            metadata = object.__new__(AscendDSAMetadata)
            cos, sin = dsa_rope(shared_cos, shared_sin)
            metadata.req_metadata = SimpleNamespace(cos=cos, sin=sin)
        return {"layer": metadata}

    monkeypatch.setattr(ubatch_runner, "build_attn_metadata", build)
    monkeypatch.setattr(
        deepseek_attention_metadata,
        "get_cos_and_sin_dsa",
        lambda positions, use_cache: dsa_rope(
            positions.float().view(2, 1) * 3,
            positions.float().view(2, 1) * 5,
        ),
    )
    positions = [
        torch.tensor([1, 2], device=device),
        torch.tensor([6, 7], device=device),
    ]
    tables = [
        torch.tensor([[3, 4], [9, 10]], dtype=torch.int32, device=device),
        torch.tensor([[17, 18], [25, 26]], dtype=torch.int32, device=device),
    ]

    def prepare(stage, capturing):
        batch = SimpleNamespace(
            num_tokens=1,
            num_tokens_after_padding=2,
            num_reqs=1,
            num_reqs_after_padding=2,
            query_start_loc=torch.tensor([0, 1, 1], device=device),
            query_start_loc_np=np.array([0, 1, 1], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            seq_lens=torch.tensor([1, 0], device=device),
            seq_lens_np=np.array([1, 0], dtype=np.int32),
            seq_lens_cpu_upper_bound=torch.tensor([1, 0]),
            positions=positions[stage],
            attn_state=object(),
            is_prefilling_np=np.array([False, False]),
        )
        return runner._prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            (tables[stage],),
            torch.full((1, 2), -1, device=device),
            [],
            None,
            for_capture=capturing,
            ubatch_idx=stage,
        )["layer"]

    def rope(metadata):
        if backend == "mla":
            return metadata.decode.cos, metadata.decode.sin
        if backend == "sfa":
            return metadata.cos, metadata.sin
        assert isinstance(metadata.req_metadata.cos, RopeDataProxy)
        assert isinstance(metadata.req_metadata.sin, RopeDataProxy)
        return metadata.req_metadata.cos["layer"], metadata.req_metadata.sin["layer"]

    captured = [prepare(stage, True) for stage in range(2)]
    captured_tables = list(built_tables)
    addresses = [
        (table.data_ptr(), *(value.data_ptr() for value in rope(metadata)))
        for table, metadata in zip(captured_tables, captured, strict=True)
    ]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [
            table[:, :1].float() + sum(rope(metadata))
            for table, metadata in zip(captured_tables, captured, strict=True)
        ]

    for stage in range(2):
        tables[stage].add_(100)
        positions[stage].add_(7)
    live = [prepare(stage, False) for stage in range(2)]
    assert [
        (table.data_ptr(), *(value.data_ptr() for value in rope(metadata)))
        for table, metadata in zip(built_tables[2:], live, strict=True)
    ] == addresses
    assert addresses[0][1] != addresses[1][1]
    graph.replay()
    for stage, output in enumerate(outputs):
        expected_table = tables[stage].clone()
        if backend != "mla":
            expected_table[1:].zero_()
            assert tables[stage][1, 0].item() != 0
        torch.testing.assert_close(
            output.cpu(),
            (expected_table[:, :1].float() + positions[stage].view(2, 1) * 8).cpu(),
        )
    graph.reset()
