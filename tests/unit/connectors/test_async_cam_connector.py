from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig, afd_config_from_mapping

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import torch  # noqa: E402

from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDConnectorExtension,
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


class _FakeScalar:
    def __init__(self, value):
        self._value = int(value)

    def item(self):
        return self._value


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

    def new_zeros(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def new_empty(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)


class _FakeCamOps:
    def __init__(self):
        self.calls = []

    def async_dispatch_send(self, *args):
        self.calls.append(("dispatch_send", args))
        return args[0]

    def async_dispatch_recv(self, *args):
        self.calls.append(("dispatch_recv", args))
        batch_size = args[3]
        hidden_size = args[4]
        expert_per_rank = args[8]
        tp_size = args[11]
        return (
            _FakeTensor((batch_size, hidden_size)),
            _FakeTensor((max(1, batch_size // 2), hidden_size)),
            _FakeTensor((batch_size,), dtype="fp32"),
            _FakeTensor((max(1, batch_size // 2),), dtype="fp32"),
            _FakeTensor((5 + tp_size * (2 + expert_per_rank),), dtype="int64"),
            _FakeTensor((expert_per_rank,), dtype="int64"),
            _FakeTensor((1,), dtype="int64"),
        )

    def async_combine_send(self, *args):
        self.calls.append(("combine_send", args))
        return args[0]

    def async_combine_recv(self, *args):
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
        self.ops = SimpleNamespace(umdk_cam_op_lib=_FakeCamOps())

    def device(self, name):
        return name

    def empty(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)

    def zeros(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


def _vllm_config(*, tp_size: int = 1, pcp_size: int = 1, extra_config=None):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": extra_config or {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
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
            extra_config={"attn_ranks_per_dp": "3"},
        ),
        _afd_config(role="attention"),
        0,
    )

    assert connector.tp_size == 3


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


def test_async_connector_init_creates_attention_first_hccl_group(monkeypatch):
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
        return SimpleNamespace(
            group_name=kwargs["group_name"],
            _get_backend=lambda device: backend,
        )

    monkeypatch.setattr(
        async_cam_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        1,
    )

    connector.init_afd_connector()

    assert calls == [
        {
            "backend": "hccl",
            "init_method": "tcp://127.0.0.1:1239",
            "world_size": 6,
            "rank": 5,
            "group_name": AFD_ASYNC_CAM_GROUP_NAME,
            "timeout": calls[0]["timeout"],
        },
    ]
    assert connector.cam_pg is not None
    assert connector.group_name == f"hccl:{AFD_ASYNC_CAM_GROUP_NAME}:5"
    assert connector.comm_args.shape == (1,)
    assert connector.comm_args.dtype == fake_torch.float16
    assert connector._placeholder.shape == (1,)


def test_async_connector_disables_dp_metadata_control_plane():
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
        0,
    )

    assert connector.control_plane is None
    assert isinstance(connector.extension, AFDConnectorExtension)


def test_async_connector_calls_cam_shaped_ops(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            pcp_size=3,
            extra_config={"attn_ranks_per_dp": 3},
        ),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    connector._placeholder = _FakeTensor((8, 16))
    extension_calls = []
    connector.extension.send_attn_extention = (
        lambda bound_connector, bound_context, **kwargs: extension_calls.append(
            (bound_connector, bound_context),
        )
    )
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
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][0] == "dispatch_send"
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][0] == "combine_recv"
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][3] == CAM_COMM_ID
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][4] == CAM_COMM_ID
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][14] == 3
    assert isinstance(context.states, AFDAsyncTransferState)
    assert isinstance(context.states, AFDTransferState)
    assert extension_calls == [(connector, context)]


def test_async_ffn_side_dispatch_recv_and_combine_send(monkeypatch):
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
    connector.extension.recv_attn_extention = lambda bound_connector, ubatch_idx: (
        "async-extension"
    )

    recv_output = connector.recv_attn_output(batch_size=4, layer_idx=1)
    connector.send_ffn_output(recv_output.hidden_states, recv_output.context)

    states = recv_output.context.states
    assert recv_output.hidden_states.shape == (4, 16)
    assert recv_output.context.metadata.extension == "async-extension"
    assert states.dynamic_scales.shape == (4,)
    assert states.expand_x_shared.shape == (2, 16)
    assert states.dynamic_scales_shared.shape == (2,)
    assert states.group_list.shape == (4,)
    assert states.expert_token_nums_shared.shape == (1,)
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][0] == "dispatch_recv"
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][0] == "combine_send"
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][11] == 2
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][13] == 2
    assert (
        fake_torch.ops.umdk_cam_op_lib.calls[1][1][3]
        is states.token_nums_rankid_layeridx
    )


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
            token_nums_rankid_layeridx=[
                _FakeScalar(7),
                _FakeScalar(0),
                _FakeScalar(11),
            ],
            expert_token_nums_shared=[_FakeScalar(2)],
            group_list=torch.tensor([2, 3], dtype=torch.int64),
            dynamic_scales=_FakeTensorLike("scales"),
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)

    states = work_item.context.states
    assert work_item.layer_idx == 11
    assert work_item.stage_idx == 0
    assert work_item.total_num_tokens == 7
    assert work_item.shared_num_tokens == 2
    assert work_item.num_tokens == 5
    assert work_item.hidden_states == "hidden[:5]"
    assert work_item.context.metadata.layer_idx == 11
    assert work_item.context.metadata.seq_lens == [5]
    assert states.dynamic_scales == "scales[:5]"
    assert states.expand_x_shared == "shared-hidden[:2]"
    assert states.dynamic_scales_shared == "shared-scales[:2]"


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
            token_nums_rankid_layeridx=[
                _FakeScalar(6),
                _FakeScalar(0),
                _FakeScalar(23),
            ],
            expert_token_nums_shared=[_FakeScalar(0)],
            group_list=group_list,
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)

    states = work_item.context.states
    assert work_item.layer_idx == 23
    assert work_item.total_num_tokens == 6
    assert work_item.shared_num_tokens == 0
    assert work_item.num_tokens == 6
    assert work_item.hidden_states == "hidden[:6]"
    assert work_item.context.metadata.seq_lens == [6]
    assert states.expand_x_shared == "shared-hidden[:0]"
    assert states.dynamic_scales_shared == "shared-scales[:0]"


def test_async_send_ffn_work_item_output_preserves_all_shared_passthrough(
    monkeypatch,
):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
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
            token_nums_rankid_layeridx=[
                _FakeScalar(5),
                _FakeScalar(0),
                _FakeScalar(7),
            ],
            expert_token_nums_shared=[_FakeScalar(5)],
            group_list=torch.zeros(8, dtype=torch.int64),
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
        )
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden", shape=(5, 16)),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)
    sent_output = connector.send_ffn_work_item_output(
        work_item,
        AFDF2ATransferPayload(
            routed_output="computed-routed",
            shared_output="computed-shared",
        ),
    )

    # This FFN rank received no routed tokens, so the connector applies the
    # empty-rank NPU workaround: it replaces the routed output with a single
    # fake bf16 token and resizes the metadata to one token, while still
    # passing the computed shared output through untouched.
    assert work_item.num_tokens == 0
    assert work_item.hidden_states == "hidden[:0]"
    assert work_item.context.metadata.seq_lens == [1]
    assert isinstance(sent_output, AFDF2ATransferPayload)
    assert sent_output.shared_output == "computed-shared"
    assert sent_output.routed_output.shape == (1, 16)
    assert sent_output.routed_output.dtype == fake_torch.bfloat16
    assert sent_outputs == [
        (
            sent_output.routed_output,
            work_item.context,
            {"ubatch_idx": 0, "expand_x_shared": "computed-shared"},
        ),
    ]


def test_async_select_experts_uses_v026_num_experts_contract(monkeypatch):
    calls = []
    fake_package = ModuleType("vllm_ascend")
    fake_ops = ModuleType("vllm_ascend.ops")
    fake_fused_moe = ModuleType("vllm_ascend.ops.fused_moe")
    fake_selector = ModuleType("vllm_ascend.ops.fused_moe.experts_selector")

    def select_experts(*, num_experts=-1, **kwargs):
        calls.append((num_experts, kwargs))
        return "weights", "ids"

    fake_selector.select_experts = select_experts
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fake_fused_moe)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        fake_selector,
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )

    result = connector.select_experts(router_logits="logits", num_experts=8)

    assert result == ("weights", "ids")
    assert calls == [(8, {"router_logits": "logits"})]
