# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

from torch import Tensor  # noqa: E402
from vllm_ascend.attention.dsa_v1 import (  # noqa: E402
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata  # noqa: E402
from vllm_ascend.attention.sfa_v1 import (  # noqa: E402
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.ops import rope_dsv4  # noqa: E402
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy  # noqa: E402

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
        data = {"test": {"default": (positions + 100, positions + 200)}}
        return RopeDataProxy(data), RopeDataProxy(data, is_cos=False)

    state = rope_dsv4.RopeGlobalState()
    state.layer_info["layer"] = ("test", ["default"])
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", state)
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
    assert dsa_metadata.req_metadata.cos["layer"].tolist() == [100, 101, 102, 103]
    assert dsa_metadata.req_metadata.sin["layer"].tolist() == [200, 201, 202, 203]


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


@pytest.fixture
def dsa_rope_source(monkeypatch):
    # Use the real pinned getter and proxy, including multi-group selection.
    state = rope_dsv4.RopeGlobalState()
    for key, width, offset in (("config-a", 2, 100), ("config-b", 3, 300)):
        cos = torch.arange(32 * width, dtype=torch.float32).reshape(32, 1, 1, width)
        state.full_rope_cache[key] = (cos + offset, cos + offset + 1000)
        state.registry_summary[key] = {"default", "compressed"}
    state.layer_info = {
        "layer-a": ("config-a", ["default"]),
        "layer-b": ("config-b", ["default"]),
        "layer-groups": ("config-a", ["default", "compressed"]),
    }
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", state)
    calls = []
    returned = []

    def get_rope(positions, use_cache=False):
        calls.append((positions.clone(), use_cache))
        result = rope_dsv4.get_cos_and_sin_dsa(
            {"default": positions, "compressed": positions + 1},
            use_cache=use_cache,
        )
        returned.append(result)
        return result

    monkeypatch.setattr(deepseek_attention_metadata, "get_cos_and_sin_dsa", get_rope)

    def materialize(positions, num_tokens, buffers=None):
        metadata = object.__new__(AscendDSAMetadata)
        metadata.req_metadata = SimpleNamespace(cos=None, sin=None)
        # Both layers share one request metadata object: one getter call is enough.
        deepseek_attention_metadata.materialize_deepseek_attention_metadata_by_layer(
            {"layer-a": metadata, "layer-b": metadata},
            positions,
            num_tokens,
            rope_buffers=buffers,
            rope_capacity=8,
        )
        return metadata.req_metadata

    return SimpleNamespace(
        state=state, calls=calls, returned=returned, materialize=materialize
    )


def test_dsa_eager_keeps_real_rope_proxies(dsa_rope_source):
    source = dsa_rope_source
    positions = torch.tensor([1.0, 4.0, 6.0, 100.0])
    first = source.materialize(positions, 3)
    assert first.cos is source.returned[0][0]
    assert first.sin is source.returned[0][1]
    assert isinstance(first.cos, RopeDataProxy)
    assert isinstance(first.sin, RopeDataProxy)
    expected = source.state.full_rope_cache["config-a"][0][[1, 4, 6]]
    second = source.materialize(torch.tensor([9, 10]), 2)
    assert second.cos is source.returned[1][0]
    assert first.cos["layer-a"].data_ptr() != second.cos["layer-a"].data_ptr()
    torch.testing.assert_close(first.cos["layer-a"], expected)
    assert source.calls[0][0].dtype == torch.long
    assert [(p.tolist(), cached) for p, cached in source.calls] == [
        ([1, 4, 6], False),
        ([9, 10], False),
    ]


@pytest.mark.parametrize("num_tokens", [1, 4])
def test_dsa_full_preserves_proxy_selection_slicing_and_padding(
    dsa_rope_source, num_tokens
):
    source = dsa_rope_source
    positions = torch.tensor([1, 4, 6, 9, 100])
    buffers: dict[str, dict[str, Tensor]] = {}
    request = source.materialize(positions, num_tokens, buffers)
    assert isinstance(request.cos, RopeDataProxy)
    assert isinstance(request.sin, RopeDataProxy)
    assert request.cos.idx == 0 and request.sin.idx == 1
    assert request.cos._data is request.sin._data
    assert set(buffers) == {"layer-a"}
    assert len(buffers["layer-a"]) == 8
    assert all(value.shape[0] == 8 for value in buffers["layer-a"].values())
    assert len(source.calls) == 1
    assert source.calls[0][0].tolist() == positions[:num_tokens].tolist()
    for key, groups in request.cos._data.items():
        for group, values in groups.items():
            indices = positions[:num_tokens] + (group == "compressed")
            for index, value in enumerate(values):
                torch.testing.assert_close(
                    value, source.state.full_rope_cache[key][index][indices]
                )
                original = source.returned[0][0]._data[key][group][index]
                assert value.data_ptr() != original.data_ptr()
    torch.testing.assert_close(
        request.sin["layer-b"],
        source.state.full_rope_cache["config-b"][1][positions[:num_tokens]],
    )
    groups = request.cos["layer-groups"]
    assert set(groups) == {"default", "compressed"}
    torch.testing.assert_close(groups["default"], request.cos["layer-a"])
    torch.testing.assert_close(
        request.sin[:1]["layer-groups"]["compressed"],
        request.sin["layer-groups"]["compressed"][:1],
    )
    padded = request.sin.pad_to(num_tokens + 2)["layer-b"]
    torch.testing.assert_close(padded[:num_tokens], request.sin["layer-b"])
    assert torch.count_nonzero(padded[num_tokens:]).item() == 0


def test_dsa_full_updates_persistent_leaf_storage_across_stages(dsa_rope_source):
    source = dsa_rope_source
    buffers: list[dict[str, dict[str, Tensor]]] = [{}, {}]
    captured = [
        source.materialize(torch.tensor([1, 3]) + stage * 4, 2, buffers[stage])
        for stage in range(2)
    ]

    def addresses(proxy):
        return [
            value.data_ptr()
            for groups in proxy._data.values()
            for values in groups.values()
            for value in values
        ]

    pointers = [addresses(request.cos) for request in captured]
    assert set(pointers[0]).isdisjoint(pointers[1])
    for values in ([8, 10], [11, 13, 15, 17], [19, 21, 23]):
        stage_one_before = captured[1].sin["layer-b"].clone()
        for stage in range(2):
            positions = torch.tensor(values) + stage
            live = source.materialize(positions, len(values), buffers[stage])
            assert addresses(live.cos) == pointers[stage]
            assert addresses(captured[stage].cos) == pointers[stage]
            torch.testing.assert_close(
                captured[stage].sin["layer-b"],
                source.state.full_rope_cache["config-b"][1][positions[:2]],
            )
            if stage == 0:
                torch.testing.assert_close(captured[1].sin["layer-b"], stage_one_before)
    assert len(source.calls) == 8
