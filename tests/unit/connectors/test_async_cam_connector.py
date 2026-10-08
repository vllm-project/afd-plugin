# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig, afd_config_from_mapping

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import torch  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDConnectorFactory,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu import async_cam as async_cam_module  # noqa: E402
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFD_ASYNC_CAM_GROUP_NAME,
    CAM_COMM_ID,
    AFDAsyncExtraInfo,
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
    build_async_topology,
)


class _FakeTensorLike:
    def __init__(self, name, *, shape=None, device="npu:0"):
        self.name = name
        self.shape = shape
        self.device = device

    def __getitem__(self, item):
        start = "" if item.start is None else item.start
        stop = "" if item.stop is None else item.stop
        return f"{self.name}[{start}:{stop}]"


class _FakeTensor:
    def __init__(self, shape, *, dtype="bf16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def to(self, *, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self

    def new_zeros(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def new_empty(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)


class _FakeCamOps:
    def __init__(self):
        self.calls = []

    def afd_async_dispatch_send(self, *args):
        self.calls.append(("dispatch_send", args))
        return args[0]

    def afd_async_dispatch_recv(self, *args):
        self.calls.append(("dispatch_recv", args))
        batch_size = args[3]
        hidden_size = args[4]
        expert_per_rank = args[8]
        tp_size = args[11]
        return (
            _FakeTensor((batch_size, hidden_size)),
            _FakeTensor((batch_size,), dtype="fp32"),
            _FakeTensor((5 + tp_size * (1 + expert_per_rank),), dtype="int64"),
            _FakeTensor((expert_per_rank,), dtype="int64"),
        )

    def afd_async_combine_send(self, *args):
        self.calls.append(("combine_send", args))
        return args[0]

    def afd_async_combine_recv(self, *args):
        self.calls.append(("combine_recv", args))
        batch_size = args[5]
        hidden_size = args[6]
        return _FakeTensor((batch_size, hidden_size))


class _FakeTorch:
    def __init__(self):
        self.bfloat16 = "bf16"
        self.float16 = "fp16"
        self.float32 = "fp32"
        self.int32 = "int32"
        self.int64 = "int64"
        self.ops = SimpleNamespace(afd_ascend=_FakeCamOps())

    def device(self, name):
        return name

    def empty(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)

    def zeros(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


def _vllm_config(*, tp_size: int = 1, pcp_size: int = 1, extra_config=None):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector_extra_config": extra_config
                if extra_config is not None
                else {"attn_ranks_per_dp": 4},
            },
        },
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=8,
            ),
        ),
    )


def _afd_config(*, role: str):
    return AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )


def _dp2_afd_config(*, role: str):
    return AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=8,
        num_ffn_ranks=8,
    )


def _topk_payload(batch_size: int, topk: int = 2):
    return {
        "topk_ids": _FakeTensor((batch_size, topk), dtype="int32"),
        "topk_weights": _FakeTensor((batch_size, topk), dtype="fp32"),
    }


def test_config_accepts_async_connector_name():
    config = afd_config_from_mapping(
        {
            "role": "attention",
            "connector": "CAMAsyncAFDConnector",
        },
    )

    assert config.connector == "CAMAsyncAFDConnector"


def test_async_extra_info_rejects_unknown_and_invalid_fields():
    with pytest.raises(ValueError, match="unknown AFD async connector_extra_config"):
        AFDAsyncExtraInfo.from_mapping({"core_num": 8})
    with pytest.raises(ValueError, match="attn_ranks_per_dp must be positive"):
        AFDAsyncExtraInfo.from_mapping({"attn_ranks_per_dp": 0})


def test_async_extra_info_parses_async_moe_fields():
    extra_info = AFDAsyncExtraInfo.from_mapping(
        {
            "async_moe_ubatching": "true",
            "async_moe_num_ubatches": "2",
            "async_moe_split": "Request",
        },
    )

    assert extra_info.async_moe_ubatching is True
    assert extra_info.async_moe_num_ubatches == 2
    assert extra_info.async_moe_split == "request"


def test_async_connector_factory_creates_import_safe_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(extra_config={"attn_ranks_per_dp": 2}),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, CAMAsyncAFDConnector)
    assert not connector.is_initialized
    assert connector.control_plane is None
    # tp_size is derived from the connector_extra_config attn_ranks_per_dp,
    # which the factory reads through the same path as direct construction.
    assert connector.tp_size == 2


def test_async_connector_uses_attn_ranks_per_dp_for_cam_tp_size():
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            tp_size=4,
            pcp_size=2,
            extra_config={"attn_ranks_per_dp": "4"},
        ),
        _afd_config(role="attention"),
        0,
    )

    assert connector.tp_size == 4


@pytest.mark.parametrize("value", [True, "bad"])
def test_async_connector_rejects_invalid_attn_ranks_per_dp(value):
    with pytest.raises(TypeError, match="attn_ranks_per_dp"):
        CAMAsyncAFDConnector(
            0,
            0,
            _vllm_config(extra_config={"attn_ranks_per_dp": value}),
            _afd_config(role="attention"),
            0,
        )


def test_async_connector_rejects_nonpositive_attn_ranks_per_dp():
    with pytest.raises(ValueError, match="attn_ranks_per_dp"):
        CAMAsyncAFDConnector(
            0,
            0,
            _vllm_config(extra_config={"attn_ranks_per_dp": 0}),
            _afd_config(role="attention"),
            0,
        )


def test_async_topology_uses_cam_attention_first_rank_layout():
    attn = build_async_topology(_afd_config(role="attention"), 3)
    ffn = build_async_topology(
        _afd_config(role="ffn"),
        1,
        num_routed_experts=8,
    )

    assert attn.world_rank == 3
    assert ffn.world_rank == 5
    assert ffn.world_size == 6
    assert ffn.expert_per_rank == 4


@pytest.mark.parametrize(
    ("role", "role_rank", "expected_world_rank"),
    [
        ("attention", 3, 3),
        ("attention", 4, 4),
        ("ffn", 3, 11),
        ("ffn", 4, 12),
    ],
)
def test_async_topology_uses_one_world_for_all_dp_replicas(
    role,
    role_rank,
    expected_world_rank,
):
    topology = build_async_topology(
        _dp2_afd_config(role=role),
        role_rank,
        num_routed_experts=32,
    )

    assert topology.world_rank == expected_world_rank
    assert topology.attn_size == 8
    assert topology.ffn_size == 8
    assert topology.world_size == 16
    assert topology.expert_per_rank == 4


def test_async_extra_info_rejects_removed_shared_ffn_pool_option():
    with pytest.raises(ValueError, match="unknown AFD async connector_extra_config"):
        AFDAsyncExtraInfo.from_mapping({"shared_ffn_pool": "true"})


def test_async_connector_init_creates_attention_first_hccl_group(monkeypatch):
    calls = []
    pg_options = object()
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    monkeypatch.setattr(
        async_cam_module,
        "ensure_cam_async_ops_available",
        lambda: None,
    )

    def fake_init_afd_process_group(**kwargs):
        calls.append(kwargs)
        backend = SimpleNamespace(
            get_hccl_comm_name=lambda rank: f"hccl:{kwargs['group_name']}:{rank}",
        )
        return SimpleNamespace(
            group_name=kwargs["group_name"],
            _get_backend=lambda device: backend,
        )

    monkeypatch.setattr(
        async_cam_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    monkeypatch.setattr(
        async_cam_module,
        "create_hccl_process_group_options",
        lambda hccl_buffer_size_mb: pg_options if hccl_buffer_size_mb == 6144 else None,
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(extra_config={"hccl_buffer_size": 6144}),
        _afd_config(role="ffn"),
        1,
    )

    connector.init_afd_connector()

    assert calls == [
        {
            "backend": "hccl",
            "world_size": 6,
            "rank": 5,
            "group_name": AFD_ASYNC_CAM_GROUP_NAME,
            "timeout": calls[0]["timeout"],
            "init_method": "tcp://127.0.0.1:1239",
            "pg_options": pg_options,
        },
    ]
    assert connector.cam_pg is not None
    assert connector.group_name == f"hccl:{AFD_ASYNC_CAM_GROUP_NAME}:5"
    assert connector.comm_args.shape == (1,)
    assert connector.comm_args.dtype == fake_torch.float16
    assert connector._placeholder.shape == (1,)


def test_async_connector_init_uses_one_hccl_group_for_all_dp(monkeypatch):
    calls = []
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    monkeypatch.setattr(
        async_cam_module,
        "ensure_cam_async_ops_available",
        lambda: None,
    )

    def fake_init_afd_process_group(**kwargs):
        calls.append(kwargs)
        backend = SimpleNamespace(
            get_hccl_comm_name=lambda rank: f"hccl:{kwargs['group_name']}:{rank}",
        )
        return SimpleNamespace(_get_backend=lambda device: backend)

    monkeypatch.setattr(
        async_cam_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(extra_config={"attn_ranks_per_dp": 4}),
        _dp2_afd_config(role="ffn"),
        4,
    )

    connector.init_afd_connector()

    assert calls[0]["world_size"] == 16
    assert calls[0]["rank"] == 12
    assert calls[0]["group_name"] == AFD_ASYNC_CAM_GROUP_NAME
    assert calls[0]["init_method"] == "tcp://127.0.0.1:1239"
    assert connector.group_name == "hccl:afd_async_cam:12"


def test_async_connector_disables_dp_metadata_control_plane():
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )

    assert connector.control_plane is None


def test_async_connector_calls_cam_shaped_ops(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            pcp_size=3,
            extra_config={"attn_ranks_per_dp": 4},
        ),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    connector._placeholder = _FakeTensor((8, 16))
    hidden_states = _FakeTensor((3, 16))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=2,
        stage_idx=0,
        seq_len=3,
    )
    context = AFDTransferContext(metadata=metadata)

    output = connector.send_attn_output(
        hidden_states,
        context,
        **_topk_payload(3),
    )
    combined = connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=0,
    )

    assert output is None
    assert combined.shape == (3, 16)
    assert fake_torch.ops.afd_ascend.calls[0][0] == "dispatch_send"
    assert fake_torch.ops.afd_ascend.calls[1][0] == "combine_recv"
    assert fake_torch.ops.afd_ascend.calls[0][1][3] == CAM_COMM_ID
    assert fake_torch.ops.afd_ascend.calls[1][1][4] == CAM_COMM_ID
    assert fake_torch.ops.afd_ascend.calls[0][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.afd_ascend.calls[1][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.afd_ascend.calls[0][1][14] == 4
    assert isinstance(context.states, AFDAsyncTransferState)
    assert isinstance(context.states, AFDTransferState)


def test_async_ffn_side_dispatch_recv_and_combine_send(monkeypatch):
    logs = []
    monkeypatch.setattr(
        async_cam_module,
        "_log_cam_op_values",
        lambda op, label, **values: logs.append((op, label, values)),
    )
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            pcp_size=2,
            extra_config={"attn_ranks_per_dp": 2},
        ),
        _afd_config(role="ffn"),
        0,
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    connector._placeholder = _FakeTensor((8, 16))

    recv_output = connector.recv_attn_output(batch_size=4, layer_idx=1)
    connector.send_ffn_output(recv_output.hidden_states, recv_output.context)

    states = recv_output.context.states
    assert recv_output.hidden_states.shape == (connector.max_num_batched_tokens, 16)
    assert states.dynamic_scales.shape == (connector.max_num_batched_tokens,)
    assert states.group_list.shape == (connector.topology.expert_per_rank,)
    assert fake_torch.ops.afd_ascend.calls[0][0] == "dispatch_recv"
    assert fake_torch.ops.afd_ascend.calls[1][0] == "combine_send"
    assert fake_torch.ops.afd_ascend.calls[0][1][11] == 2
    assert fake_torch.ops.afd_ascend.calls[1][1][12] == 2
    assert fake_torch.ops.afd_ascend.calls[1][1][2] is states.token_nums_rankid_layeridx

    assert [(op, label) for op, label, _ in logs] == [
        ("async_dispatch_recv", "inputs"),
        ("async_dispatch_recv", "outputs"),
        ("async_combine_send", "inputs"),
    ]
    assert logs[1][2] == {
        "hidden_states": recv_output.hidden_states,
        "dynamic_scales": states.dynamic_scales,
        "batch_info": states.token_nums_rankid_layeridx,
        "expert_token_nums": states.group_list,
    }
    for index in (0, 2):
        assert logs[index][2]["max_seq_len"] == connector.max_num_batched_tokens


def test_async_combine_send_requires_dispatch_recv_token_metadata(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )
    connector._initialized = True
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=0,
        seq_lens=[4],
    )
    # token_nums_rankid_layeridx is left unset (None) so combine send must fail.
    states = AFDAsyncTransferState(
        batch_size=4,
        hidden_size=16,
        topk=2,
        layer_idx=1,
        group_list=_FakeTensor((4,), dtype="int64"),
    )
    context = AFDTransferContext(metadata=metadata, states=states)

    with pytest.raises(RuntimeError, match="TokenNums_Rankid_Layeridx"):
        connector.send_ffn_output(_FakeTensor((4, 16)), context)


def test_async_ffn_work_item_uses_cam_layer_and_token_metadata(monkeypatch):
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )
    connector.ffn_size = 2

    def fake_recv_attn_output(*, stage_idx, layer_idx, batch_size, ubatch_idx):
        assert ubatch_idx == 0
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, batch_size)],
        )
        states = AFDAsyncTransferState(
            batch_size=max(1, batch_size),
            hidden_size=connector.hidden_size,
            topk=connector.topk,
            layer_idx=layer_idx,
            token_nums_rankid_layeridx=torch.tensor(
                [7, 0, 11, 0, 1], dtype=torch.int64
            ),
            group_list=torch.tensor([2, 3], dtype=torch.int64),
            dynamic_scales=_FakeTensorLike("scales"),
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(
        stage_idx=0,
        max_num_tokens=16,
    )

    states = work_item.context.states
    assert work_item.layer_idx == 11
    assert work_item.stage_idx == 0
    assert work_item.total_num_tokens == 7
    assert work_item.num_tokens == 5
    assert work_item.hidden_states == "hidden[:5]"
    assert work_item.context.metadata.layer_idx == 11
    assert work_item.context.metadata.seq_lens == [5]
    assert states.dynamic_scales == "scales[:5]"


def test_async_ffn_work_item_uses_expert_counts_for_routed_tokens(monkeypatch):
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )

    import torch

    # The routed token count comes from summing the per-expert group_list.
    # The counts below sum to 6.
    group_list = torch.tensor(
        [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        dtype=torch.int64,
    )

    def fake_recv_attn_output(*, stage_idx, layer_idx, batch_size, ubatch_idx):
        assert ubatch_idx == 0
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, batch_size)],
        )
        states = AFDAsyncTransferState(
            batch_size=max(1, batch_size),
            hidden_size=connector.hidden_size,
            topk=connector.topk,
            layer_idx=layer_idx,
            token_nums_rankid_layeridx=torch.tensor(
                [6, 0, 23, 0, 15], dtype=torch.int64
            ),
            group_list=group_list,
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(
        stage_idx=0,
        max_num_tokens=16,
    )

    assert work_item.layer_idx == 23
    assert work_item.total_num_tokens == 6
    assert work_item.num_tokens == 6
    assert work_item.hidden_states == "hidden[:6]"
    assert work_item.context.metadata.seq_lens == [6]


def test_async_send_ffn_work_item_output_notifies_empty_rank(
    monkeypatch,
):
    fake_torch = _FakeTorch()
    # Keep real CPU control tensors; only the NPU placeholder allocation is fake.
    monkeypatch.setattr(async_cam_module.torch, "zeros", fake_torch.zeros)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )
    sent_outputs = []

    def fake_send_ffn_output(ffn_output, metadata, **kwargs):
        sent_outputs.append((ffn_output, metadata, kwargs))

    monkeypatch.setattr(connector, "send_ffn_output", fake_send_ffn_output)

    def fake_recv_attn_output(*, stage_idx, layer_idx, batch_size, ubatch_idx):
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, batch_size)],
        )
        states = AFDAsyncTransferState(
            batch_size=max(1, batch_size),
            hidden_size=connector.hidden_size,
            topk=connector.topk,
            layer_idx=layer_idx,
            token_nums_rankid_layeridx=torch.tensor([0, 0, 7, 0, 7], dtype=torch.int64),
            group_list=torch.tensor([0] * 8, dtype=torch.int64),
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden", shape=(5, 16)),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(
        stage_idx=0,
        max_num_tokens=16,
    )
    sent_output = connector.send_ffn_work_item_output(
        work_item,
        AFDF2ATransferPayload(
            routed_output="computed-routed",
        ),
    )

    # Empty ranks still send a floating placeholder with untouched zero counts.
    assert work_item.num_tokens == 0
    assert work_item.hidden_states == "hidden[:0]"
    assert work_item.context.metadata.seq_lens == [0]
    assert isinstance(sent_output, AFDF2ATransferPayload)
    assert sent_output.shared_output is None
    assert sent_output.routed_output.shape == (1, 16)
    assert sent_output.routed_output.dtype == torch.bfloat16
    assert sent_outputs == [
        (
            sent_output.routed_output,
            work_item.context,
            {"ubatch_idx": 0},
        ),
    ]


def test_async_select_experts_uses_native_router_factory(monkeypatch):
    router_module = pytest.importorskip(
        "vllm_ascend.ops.fused_moe.router.router_factory"
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    router_logits = torch.tensor(
        [
            [1.0, 1.0 + 1e-7, 0.0, -1.0, 0.0, -1.0, -2.0, -3.0],
            [-1.0, -2.0, -3.0, -4.0, 1.0, 1.0 - 1e-7, 0.0, -1.0],
        ],
    )
    correction_bias = torch.tensor(
        [-1e-7, 1e-7, 0.0, 0.0, 1e-7, -1e-7, 0.0, 0.0],
    )
    captured = {}

    class NativeRouterStub:
        def select_experts(
            self,
            hidden_states,
            logits,
            *,
            topk_indices_dtype,
        ):
            captured["hidden_states"] = hidden_states
            captured["router_logits"] = logits
            captured["topk_indices_dtype"] = topk_indices_dtype
            return (
                torch.tensor([[1.5, 0.5], [0.25, 1.75]]),
                torch.tensor([[0, 4], [3, 7]], dtype=topk_indices_dtype),
            )

    def create_router(**kwargs):
        captured["factory_kwargs"] = kwargs
        return NativeRouterStub()

    monkeypatch.setattr(
        router_module,
        "create_ascend_fused_moe_router",
        create_router,
    )
    router_options = dict(
        router_logits=router_logits,
        top_k=2,
        use_grouped_topk=True,
        renormalize=True,
        scoring_func="softmax",
        num_expert_group=4,
        topk_group=2,
        routed_scaling_factor=2.0,
        e_score_correction_bias=correction_bias,
        num_logical_experts=8,
        num_shared_experts=1,
        num_experts=9,
    )
    weights, ids = connector.select_experts(
        hidden_states=torch.zeros(2, 16),
        mix_placement=False,
        **router_options,
    )

    factory_kwargs = captured["factory_kwargs"]
    assert factory_kwargs.pop("e_score_correction_bias") is correction_bias
    assert factory_kwargs == {
        "top_k": 2,
        "global_num_experts": 8,
        "use_grouped_topk": True,
        "renormalize": True,
        "scoring_func": "softmax",
        "num_expert_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.0,
    }
    assert captured["router_logits"] is router_logits
    assert captured["topk_indices_dtype"] is torch.int32
    assert ids.shape == weights.shape == (2, 2)
    assert torch.equal(ids, torch.tensor([[0, 4], [3, 7]], dtype=torch.int32))
    torch.testing.assert_close(weights.sum(dim=-1), torch.tensor([2.0, 2.0]))

    cached_router = NativeRouterStub()
    monkeypatch.setattr(
        router_module,
        "create_ascend_fused_moe_router",
        lambda **kwargs: pytest.fail("cached router must be reused"),
    )
    connector.select_experts(
        hidden_states=torch.zeros(2, 16),
        mix_placement=False,
        router=cached_router,
        **router_options,
    )
    assert captured["router_logits"] is router_logits


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("renormalize,hidden_size", [(False, 16), (True, 4)])
def test_async_select_experts_native_fallback(with_bias, renormalize, hidden_size):
    # sigmoid without renormalization chooses the fallback at factory time;
    # small hidden states choose it inside the fused router at execution time.
    scores = torch.tensor([[0.90, 0.08, 0.04, 0.02, 0.60, 0.59, 0.30, 0.01]])
    bias = (
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0]) if with_bias else None
    )
    weights, ids = async_cam_module.select_cam_experts(
        hidden_states=torch.zeros(1, hidden_size),
        router_logits=torch.logit(scores),
        top_k=2,
        use_grouped_topk=True,
        renormalize=renormalize,
        scoring_func="sigmoid",
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias,
        mix_placement=False,
        num_logical_experts=8,
        num_shared_experts=1,
        num_experts=9,
    )
    # Native fallback uses each group's maximum: group 0 wins (0.90 > 0.70).
    order = ids.argsort(dim=-1)
    assert torch.equal(ids.gather(1, order), torch.tensor([[0, 1]], dtype=torch.int32))
    expected = torch.tensor([[0.90, 0.08]])
    if renormalize:
        expected /= 0.98
    torch.testing.assert_close(weights.gather(1, order), expected * 2.0)


@pytest.mark.npu
@pytest.mark.parametrize(
    "scoring_func,renormalize,with_bias",
    [
        ("sigmoid", True, True),
        ("sigmoid", True, False),
        ("softmax", True, False),
        ("softmax", False, False),
    ],
)
def test_async_select_experts_fused_group_scores(scoring_func, renormalize, with_bias):
    if not torch.npu.is_available():
        pytest.skip("Ascend device required for real fused routing")
    # Match worker initialization: choose the device before loading CANN ops.
    torch.npu.set_device(0)
    assert enable_custom_op()
    scores = torch.tensor([[0.90, 0.08, 0.04, 0.02, 0.60, 0.59, 0.30, 0.01]])
    logits = torch.logit(scores) if scoring_func == "sigmoid" else scores.log()
    bias = (
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0]) if with_bias else None
    )
    weights, ids = async_cam_module.select_cam_experts(
        hidden_states=torch.zeros(1, 16, device="npu"),
        router_logits=logits.npu(),
        top_k=2,
        use_grouped_topk=True,
        renormalize=renormalize,
        scoring_func=scoring_func,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias.npu() if bias is not None else None,
        mix_placement=False,
        num_logical_experts=8,
        num_shared_experts=1,
        num_experts=9,
    )
    # Top-2 sums choose group 1; the former hard-coded max fallback picks
    # group 0. Bias changes selection to expert 6 but never its raw weight.
    expected_ids = [4, 6] if with_bias else [4, 5]
    expected = torch.tensor([[0.60, 0.30 if with_bias else 0.59]])
    if renormalize:
        expected /= 0.90 if with_bias else 1.19
    else:
        expected /= 2.54  # Sum of the eight scores, before softmax top-k.
    ids, weights = ids.cpu(), weights.cpu()
    order = ids.argsort(dim=-1)
    assert ids.dtype == torch.int32
    assert torch.equal(
        ids.gather(1, order), torch.tensor([expected_ids], dtype=torch.int32)
    )
    torch.testing.assert_close(weights.gather(1, order), expected * 2.0)


@pytest.mark.parametrize("router", [None, SimpleNamespace()])
def test_async_select_experts_rejects_mix_placement(router):
    with pytest.raises(RuntimeError, match="routed-only.*mix_placement"):
        async_cam_module.select_cam_experts(
            hidden_states=torch.zeros(1, 16),
            router_logits=torch.zeros(1, 8),
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            scoring_func="softmax",
            num_expert_group=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            mix_placement=True,
            num_logical_experts=8,
            num_shared_experts=1,
            num_experts=9,
            router=router,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_async_receive_anchor_uses_model_dtype(monkeypatch, dtype):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    monkeypatch.setattr(
        async_cam_module, "ensure_cam_async_ops_available", lambda: None
    )
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "source-test")
    monkeypatch.setattr(
        async_cam_module,
        "init_afd_process_group",
        lambda **kwargs: SimpleNamespace(_get_backend=lambda device: backend),
    )
    monkeypatch.setattr(
        async_cam_module, "create_hccl_process_group_options", lambda _: None
    )
    config = _vllm_config()
    config.model_config.dtype = dtype
    connector = CAMAsyncAFDConnector(0, 0, config, _afd_config(role="ffn"), 0)
    connector.init_afd_connector()
    assert connector._placeholder.dtype == dtype


def test_async_rejects_nondivisible_expert_placement():
    with pytest.raises(ValueError, match="divisible"):
        build_async_topology(_afd_config(role="ffn"), 0, num_routed_experts=7)


def test_async_rejects_fused_shared_expert_ids():
    from afd_plugin.compat.npu.feature_validation import (
        _fail_if_unsupported_npu_afd_async_features,
    )

    config = _vllm_config()
    config.additional_config["mix_placement"] = True
    with pytest.raises(RuntimeError, match="routed-only.*mix_placement"):
        _fail_if_unsupported_npu_afd_async_features(
            config,
            _afd_config(role="attention"),
            AFDAsyncExtraInfo(),
        )


def test_dispatch_failure_does_not_leave_pending_routing(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=2,
            stage_idx=0,
            seq_len=3,
        )
    )

    def fail_dispatch(*_args):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(
        fake_torch.ops.afd_ascend, "afd_async_dispatch_send", fail_dispatch
    )
    with pytest.raises(RuntimeError, match="dispatch failed"):
        connector.send_attn_output(_FakeTensor((3, 16)), context, **_topk_payload(3))
    assert connector._pending_attention_payloads == {}


def test_close_releases_every_stage_routing(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    for stage in (0, 1):
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=2,
                stage_idx=stage,
                seq_len=3,
            )
        )
        connector.send_attn_output(_FakeTensor((3, 16)), context, **_topk_payload(3))
    assert set(connector._pending_attention_payloads) == {0, 1}
    connector.close()
    assert connector._pending_attention_payloads == {}
