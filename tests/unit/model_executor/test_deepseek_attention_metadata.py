# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

from vllm_ascend.attention.dsa_v1 import (  # noqa: E402
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata  # noqa: E402
from vllm_ascend.attention.sfa_v1 import (  # noqa: E402
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_attention_metadata,
)


def test_materializes_sfa_mla_and_dsa_rope_metadata(monkeypatch):
    sfa_cos_source = torch.arange(8, dtype=torch.float32)
    sfa_sin_source = sfa_cos_source + 10
    sfa_metadata = object.__new__(AscendSFAMetadata)
    sfa_metadata.cos = sfa_cos_source[:4]
    sfa_metadata.sin = sfa_sin_source[:4]
    deepseek_attention_metadata.materialize_deepseek_attention_metadata(
        sfa_metadata,
        torch.arange(4),
        4,
    )

    mla_cos_source = torch.arange(8, dtype=torch.float32)
    mla_sin_source = mla_cos_source + 20
    mla_metadata = object.__new__(AscendMLAMetadata)
    mla_metadata.prefill = SimpleNamespace(
        cos=mla_cos_source[:5],
        sin=mla_sin_source[:5],
    )
    mla_metadata.decode = SimpleNamespace(
        cos=mla_cos_source[5:],
        sin=mla_sin_source[5:],
    )
    deepseek_attention_metadata.materialize_deepseek_attention_metadata(
        mla_metadata,
        torch.arange(8),
        8,
    )

    sfa_cos_source.fill_(-1)
    sfa_sin_source.fill_(-1)
    mla_cos_source.fill_(-1)
    mla_sin_source.fill_(-1)
    assert sfa_metadata.cos.tolist() == [0, 1, 2, 3]
    assert sfa_metadata.sin.tolist() == [10, 11, 12, 13]
    assert mla_metadata.prefill.cos.tolist() == [0, 1, 2, 3, 4]
    assert mla_metadata.prefill.sin.tolist() == [20, 21, 22, 23, 24]
    assert mla_metadata.decode.cos.tolist() == [5, 6, 7]
    assert mla_metadata.decode.sin.tolist() == [25, 26, 27]

    dsa_calls = []

    def get_dsa_rope(positions, use_cache=False):
        positions = positions.clone()
        dsa_calls.append((positions, use_cache))
        return positions + 100, positions + 200

    monkeypatch.setattr(
        deepseek_attention_metadata,
        "get_cos_and_sin_dsa",
        get_dsa_rope,
    )
    dsa_metadata = object.__new__(AscendDSAMetadata)
    dsa_metadata.req_metadata = SimpleNamespace(cos=object(), sin=object())
    deepseek_attention_metadata.materialize_deepseek_attention_metadata(
        dsa_metadata,
        torch.arange(6),
        4,
    )

    assert [(positions.tolist(), use_cache) for positions, use_cache in dsa_calls] == [
        ([0, 1, 2, 3], False),
    ]
    assert dsa_metadata.req_metadata.cos.tolist() == [100, 101, 102, 103]
    assert dsa_metadata.req_metadata.sin.tolist() == [200, 201, 202, 203]


def test_isolates_mutable_sfa_and_dsa_builder_inputs():
    group_len = torch.tensor([1, 2], dtype=torch.int32)
    group_key_idx = torch.tensor([3, 4], dtype=torch.int32)
    group_key_cache_idx = torch.tensor([5, 6], dtype=torch.int32)
    sfa_common_metadata = SimpleNamespace(
        group_len=group_len,
        group_key_idx=group_key_idx,
        group_key_cache_idx=group_key_cache_idx,
    )
    deepseek_attention_metadata.isolate_deepseek_attention_builder_inputs(
        object.__new__(AscendSFAMetadataBuilder),
        sfa_common_metadata,
    )
    sfa_common_metadata.group_len.fill_(9)
    sfa_common_metadata.group_key_idx.fill_(9)
    sfa_common_metadata.group_key_cache_idx.fill_(9)
    assert group_len.tolist() == [1, 2]
    assert group_key_idx.tolist() == [3, 4]
    assert group_key_cache_idx.tolist() == [5, 6]

    block_table = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    dsa_common_metadata = SimpleNamespace(block_table_tensor=block_table)
    deepseek_attention_metadata.isolate_deepseek_attention_builder_inputs(
        object.__new__(AscendDSAMetadataBuilder),
        dsa_common_metadata,
    )
    dsa_common_metadata.block_table_tensor.fill_(0)
    assert block_table.tolist() == [[0, 1, 2], [3, 4, 5]]
