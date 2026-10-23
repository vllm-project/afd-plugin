# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.connectors.gpu import fastafd  # noqa: E402

CONNECTOR = "FastAFDAFDConnector"


def _qwen_text_config(**overrides):
    values = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


_FP8_BLOCK = {"quant_method": "fp8", "weight_block_size": [128, 128]}


def _vllm_config(
    *,
    enable_dbo=False,
    tensor_parallel_size=1,
    dtype=torch.bfloat16,
    quantization=_FP8_BLOCK,
    **text_overrides,
):
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_config=SimpleNamespace(quantization_config=quantization),
            hf_text_config=_qwen_text_config(**text_overrides),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            enable_dbo=enable_dbo,
            use_ubatching=False,
            tensor_parallel_size=tensor_parallel_size,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
    )


def _afd_config(*, role="attention", num_attention_ranks=2, num_ffn_ranks=2):
    return afd_config_from_mapping(
        {
            "connector": CONNECTOR,
            "role": role,
            "num_attention_ranks": num_attention_ranks,
            "num_ffn_ranks": num_ffn_ranks,
            "compute_gate_on_attention": True,
        },
    )


def _connector(*, role="attention", role_rank=0, **vllm_overrides):
    return fastafd.FastAFDAFDConnector(
        rank=role_rank,
        local_rank=0,
        vllm_config=_vllm_config(**vllm_overrides),
        afd_config=_afd_config(role=role),
        role_rank=role_rank,
    )


def _payload(*stage_tokens):
    return AFDControlPayload(
        dp_metadata_list={
            stage: AFDDPMetadata(num_tokens_across_dp_cpu=tokens)
            for stage, tokens in enumerate(stage_tokens)
        },
        is_graph_capturing=False,
        is_warmup=False,
    )


def test_fastafd_connector_is_registered():
    cls = AFDConnectorFactory.get_connector_class(CONNECTOR)

    assert cls is fastafd.FastAFDAFDConnector


def test_fastafd_requires_attention_side_routing():
    with pytest.raises(ValueError, match="compute_gate_on_attention=true"):
        afd_config_from_mapping({"connector": CONNECTOR})


def test_fastafd_rejects_more_union_ranks_than_the_kernel_addresses():
    with pytest.raises(ValueError, match="at most 72"):
        afd_config_from_mapping(
            {
                "connector": CONNECTOR,
                "compute_gate_on_attention": True,
                "num_attention_ranks": 64,
                "num_ffn_ranks": 16,
            },
        )


def test_fastafd_rejects_connector_extra_config():
    assert fastafd.FastAFDAFDConnector.parse_extra_config({}) is not None
    with pytest.raises(ValueError, match="does not support connector_extra_config"):
        fastafd.FastAFDAFDConnector.parse_extra_config({"core_num": 8})


def test_m2n_shape_reads_qwen_text_config():
    shape = fastafd.m2n_shape_from_config(_vllm_config(), _afd_config())

    assert shape == fastafd.M2NShape(
        num_attention_ranks=2,
        num_ffn_ranks=2,
        num_layers=40,
        num_experts=256,
        top_k=8,
        hidden_size=2048,
        intermediate_size=512,
        max_tokens_per_rank=1024,
    )
    assert shape.num_experts_per_ffn_rank == 128


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_type": "deepseek_v3"}, "supports model types"),
        ({"hidden_size": 2304}, "multiples of 512"),
        ({"moe_intermediate_size": 768}, "multiples of 512"),
        ({"num_experts": 255}, "divide the 255 routed experts"),
        ({"mlp_only_layers": [3]}, "MoE block in every layer"),
        ({"decoder_sparse_step": 2}, "MoE block in every layer"),
        ({"dtype": torch.float16}, "bfloat16 activations"),
        ({"quantization": None}, "FP8 checkpoint"),
        (
            {"quantization": {"quant_method": "fp8", "weight_block_size": None}},
            "FP8 checkpoint",
        ),
        ({"quantization": {"quant_method": "awq"}}, "FP8 checkpoint"),
    ],
)
def test_m2n_shape_rejects_unsupported_models(overrides, message):
    with pytest.raises(ValueError, match=message):
        fastafd.m2n_shape_from_config(_vllm_config(**overrides), _afd_config())


def _fake_all_reduce(other_rank_fields):
    """All-reduce double that folds in one other rank's values."""

    def all_reduce(bounds, op, group):
        other = torch.tensor(list(other_rank_fields.values()), dtype=torch.int64)
        bounds.copy_(torch.maximum(bounds, torch.cat([other, -other])))

    return all_reduce


def test_matching_ranks_pass_the_agreement_check(monkeypatch):
    fields = {"num_lanes": 1, "num_layers": 40, "max_tokens_per_rank": 1024}
    monkeypatch.setattr(fastafd.dist, "all_reduce", _fake_all_reduce(fields))

    fastafd.check_same_across_ranks(fields, None, torch.device("cpu"))


@pytest.mark.parametrize("field", ["num_lanes", "num_layers", "max_tokens_per_rank"])
def test_mismatched_ranks_fail_before_any_buffer_exists(monkeypatch, field):
    fields = {"num_lanes": 1, "num_layers": 40, "max_tokens_per_rank": 1024}
    other = {**fields, field: fields[field] + 1}
    monkeypatch.setattr(fastafd.dist, "all_reduce", _fake_all_reduce(other))

    with pytest.raises(RuntimeError, match=f"'{field}'"):
        fastafd.check_same_across_ranks(fields, None, torch.device("cpu"))


def test_close_releases_buffers_and_the_union_group_once(monkeypatch):
    destroyed: list[object] = []
    monkeypatch.setattr(fastafd.dist, "destroy_process_group", destroyed.append)
    connector = _connector()
    group = object()
    connector.group = group
    connector.lanes = [object()]

    connector.close()
    connector.close()

    assert destroyed == [group]
    assert connector.lanes == []
    assert not connector.is_initialized


def test_close_before_initialization_is_a_no_op(monkeypatch):
    monkeypatch.setattr(
        fastafd.dist,
        "destroy_process_group",
        lambda group: pytest.fail("no group to destroy"),
    )

    _connector().close()


@pytest.mark.parametrize(
    ("role", "role_rank", "union_rank"),
    [("attention", 1, 1), ("ffn", 0, 2), ("ffn", 1, 3)],
)
def test_union_group_places_attention_ranks_first(role, role_rank, union_rank):
    connector = _connector(role=role, role_rank=role_rank)

    assert connector.union_rank == union_rank
    assert not connector.is_initialized


@pytest.mark.parametrize(("enable_dbo", "num_lanes"), [(False, 1), (True, 2)])
def test_dbo_selects_one_lane_per_microbatch(enable_dbo, num_lanes):
    assert _connector(enable_dbo=enable_dbo).num_lanes == num_lanes


def test_fastafd_rejects_tensor_parallel_roles():
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _connector(tensor_parallel_size=2)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("send_attn_output", (None, None)),
        ("recv_ffn_output", (None,)),
        ("recv_attn_output", ()),
        ("send_ffn_output", (None, None)),
    ],
)
def test_per_item_transfers_are_not_part_of_the_protocol(method, args):
    connector = _connector()

    with pytest.raises(NotImplementedError):
        getattr(connector, method)(*args)


def test_control_plane_rejects_more_stages_than_lanes():
    control_plane = _connector().control_plane

    control_plane.update_state_from_dp_metadata(_payload([3, 4]))
    with pytest.raises(RuntimeError, match="1 lanes, got a 2-stage step"):
        control_plane.update_state_from_dp_metadata(_payload([3, 4], [2, 2]))


@pytest.mark.parametrize(("role_rank", "sends"), [(0, True), (1, False)])
def test_only_union_rank_zero_sends_the_step_plan(monkeypatch, role_rank, sends):
    calls = []
    monkeypatch.setattr(
        fastafd,
        "send_control_payload",
        lambda payload, **kwargs: calls.append((payload, kwargs)),
    )
    connector = _connector(role_rank=role_rank)
    connector.group = object()
    payload = _payload([3, 4])

    connector.control_plane.send_dp_metadata_list(payload)

    if sends:
        assert len(calls) == 1
        assert calls[0][0] is payload
        assert calls[0][1]["dst"] == [2, 3]
        assert calls[0][1]["group"] is connector.group
    else:
        assert calls == []


def test_ffn_ranks_receive_the_step_plan_from_union_rank_zero(monkeypatch):
    payload = _payload([3, 4])
    calls = []

    def recv_control_payload(**kwargs):
        calls.append(kwargs)
        return payload

    monkeypatch.setattr(fastafd, "recv_control_payload", recv_control_payload)
    connector = _connector(role="ffn", role_rank=1)
    connector.group = object()

    assert connector.control_plane.recv_dp_metadata_list() is payload
    assert calls[0]["src"] == 0
    assert calls[0]["group"] is connector.group
