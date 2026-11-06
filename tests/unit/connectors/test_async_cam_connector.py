# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig

pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import torch  # noqa: E402

from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu import async_cam as async_cam_module  # noqa: E402
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    CAM_COMM_ID,
    AFDAsyncExtraInfo,
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
    build_async_topology,
)
from tests.unit.fakes.async_cam_startup import MemoryStore  # noqa: E402


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

    afd_mega_dispatch_send = afd_async_dispatch_send
    afd_mega_combine_recv = afd_async_combine_recv


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


def test_async_extra_info_rejects_unknown_and_invalid_fields():
    with pytest.raises(ValueError, match="unknown AFD async connector_extra_config"):
        AFDAsyncExtraInfo.from_mapping({"core_num": 8})
    with pytest.raises(ValueError, match="attn_ranks_per_dp must be positive"):
        AFDAsyncExtraInfo.from_mapping({"attn_ranks_per_dp": 0})


def test_mega_config_round_trips_protocol_and_normalized_quantization():
    info = AFDAsyncExtraInfo.from_mapping(
        {"ffn_mega_moe": "true", "dynamicQuant": "1", "ffn_two_stream": "false"}
    )
    assert info.ffn_mega_moe and info.dynamic_quant == 1
    assert AFDAsyncExtraInfo.from_mapping(info.to_mapping()) == info


@pytest.mark.parametrize("capacity", [1, 16])
def test_mega_warmup_launches_once_per_source(monkeypatch, capacity):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", str(capacity / 262144))
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            extra_config={
                "ffn_mega_moe": True,
                "dynamicQuant": 1,
                "attn_ranks_per_dp": 4,
            }
        ),
        _afd_config(role="ffn"),
        0,
    )
    assert connector.graph_startup_chunks() == 1


@pytest.mark.parametrize(
    "raw",
    [
        {"ffn_mega_moe": True},
        {"ffn_mega_moe": True, "dynamicQuant": 1, "ffn_two_stream": True},
    ],
)
def test_mega_requires_quantized_single_stream(raw):
    with pytest.raises(ValueError, match="ffn_mega_moe"):
        AFDAsyncExtraInfo.from_mapping(raw)


@pytest.mark.parametrize(
    "field",
    [
        "async_moe_ubatching",
        "async_moe_num_ubatches",
        "async_moe_split",
    ],
)
def test_async_extra_info_rejects_removed_fields(field):
    with pytest.raises(ValueError, match="unknown AFD async connector_extra_config"):
        AFDAsyncExtraInfo.from_mapping({field: True})


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


def test_async_explicit_receive_consumes_matching_pending_dispatch(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0, 0, _vllm_config(), _afd_config(role="attention"), 0
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    hidden_states = _FakeTensor((3, 16))
    contexts = []
    payloads = []
    for layer_idx in (2, 3):
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx, stage_idx=0, seq_len=3
            )
        )
        payload = _topk_payload(3)
        connector.send_attn_output(hidden_states, context, **payload)
        contexts.append(context)
        payloads.append(payload)

    with pytest.raises(RuntimeError, match="does not match the next pending"):
        connector.recv_ffn_output(hidden_states, ubatch_idx=0, context=contexts[1])
    assert len(fake_torch.ops.afd_ascend.calls) == 2

    connector.recv_ffn_output(
        hidden_states,
        ubatch_idx=0,
        context=contexts[0],
        **payloads[1],
    )
    assert fake_torch.ops.afd_ascend.calls[-1][1][1] is payloads[0]["topk_ids"]

    connector.recv_ffn_output(hidden_states, ubatch_idx=0)
    assert fake_torch.ops.afd_ascend.calls[-1][1][1] is payloads[1]["topk_ids"]


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


def test_async_select_experts_rejects_mix_placement():
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
            router=None,
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
    connector.graph_startup.mode = "eager"
    connector.init_afd_connector()
    connector.recv_attn_output(batch_size=1, layer_idx=0)
    assert fake_torch.ops.afd_ascend.calls[0][1][0].dtype == dtype


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


@pytest.mark.parametrize("mega", [False, True])
def test_dispatch_failure_preserves_protocol_window_ownership(monkeypatch, mega):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            extra_config={
                "attn_ranks_per_dp": 4,
                "ffn_mega_moe": mega,
                "dynamicQuant": 1,
            }
        ),
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
        fake_torch.ops.afd_ascend,
        "afd_mega_dispatch_send" if mega else "afd_async_dispatch_send",
        fail_dispatch,
    )
    with pytest.raises(RuntimeError, match="dispatch failed"):
        connector.send_attn_output(_FakeTensor((3, 16)), context, **_topk_payload(3))
    if mega:
        with pytest.raises(RuntimeError, match="pending"):
            connector.close()
    else:
        connector.close()
        assert not connector.is_initialized


@pytest.fixture
def pull_attention(monkeypatch):
    # CPU status words and host transfer recording, not NPU validation.
    fake_torch = _FakeTorch()
    fake_torch.int64 = torch.int64
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(
            extra_config={
                "ffn_mega_moe": True,
                "attn_ranks_per_dp": 4,
                "dynamicQuant": 1,
            }
        ),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    state = torch.zeros(128, dtype=torch.int64)
    connector.comm_args = state.view(torch.float16)
    store = MemoryStore()
    connector.graph_startup._store = store
    connector.graph_startup._prefix = "ffn_graph/test/"
    for rank in (4, 5):
        store.set(f"ffn_graph/test/pull_stopped/{rank}", "ready")
    hidden = _FakeTensor((3, 16))
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=2, stage_idx=0, seq_len=3
        )
    )
    return SimpleNamespace(
        connector=connector,
        state=state,
        hidden=hidden,
        context=context,
        torch=fake_torch,
    )


@pytest.mark.parametrize("phase", ["launch", "status"])
def test_pull_combine_failure_retains_window_until_completion(
    pull_attention, monkeypatch, phase
):
    h = pull_attention
    connector = h.connector
    connector.send_attn_output(h.hidden, h.context, **_topk_payload(3))
    combine = h.torch.ops.afd_ascend.afd_mega_combine_recv

    def fail(*_args):
        raise RuntimeError("combine failed")

    if phase == "launch":
        monkeypatch.setattr(h.torch.ops.afd_ascend, "afd_mega_combine_recv", fail)
    else:
        h.state[async_cam_module.PULL_FAILURE_INDEX] = -5
    with pytest.raises(RuntimeError, match="combine failed|status -5"):
        connector.recv_ffn_output(h.hidden, context=h.context)
    with pytest.raises(RuntimeError, match="pending"):
        connector.close()
    monkeypatch.setattr(h.torch.ops.afd_ascend, "afd_mega_combine_recv", combine)
    h.state[async_cam_module.PULL_FAILURE_INDEX] = 0
    connector.recv_ffn_output(h.hidden, context=h.context)
    connector.close()
    assert not connector.is_initialized and connector.comm_args is None


@pytest.mark.parametrize("nested", [False, True])
def test_attention_step_reports_status_after_queued_combine(pull_attention, nested):
    h = pull_attention
    connector = h.connector
    with pytest.raises(RuntimeError, match="status -5"), connector.attention_step():
        connector.send_attn_output(h.hidden, h.context, **_topk_payload(3))
        h.state[async_cam_module.PULL_FAILURE_INDEX] = -5
        if nested:
            with connector.attention_step():
                output = connector.recv_ffn_output(h.hidden, context=h.context)
        else:
            output = connector.recv_ffn_output(h.hidden, context=h.context)
        assert output.shape == h.hidden.shape
    # An exception at the step boundary must restore immediate status checks.
    connector.send_attn_output(h.hidden, h.context, **_topk_payload(3))
    with pytest.raises(RuntimeError, match="status -5"):
        connector.recv_ffn_output(h.hidden, context=h.context)
    h.state[async_cam_module.PULL_FAILURE_INDEX] = 0
    connector.recv_ffn_output(h.hidden, context=h.context)
    connector.close()
    assert not connector.is_initialized


def test_attention_step_restores_status_checks_after_model_error(pull_attention):
    h = pull_attention
    with pytest.raises(ValueError, match="model failed"), h.connector.attention_step():
        raise ValueError("model failed")
    h.connector.send_attn_output(h.hidden, h.context, **_topk_payload(3))
    h.state[async_cam_module.PULL_FAILURE_INDEX] = -5
    with pytest.raises(RuntimeError, match="status -5"):
        h.connector.recv_ffn_output(h.hidden, context=h.context)


def test_close_releases_every_stage_routing(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0, 0, _vllm_config(), _afd_config(role="attention"), 0
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    references: list[weakref.ReferenceType] = []
    for stage in (0, 1):
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=2, stage_idx=stage, seq_len=3
            )
        )
        payload = _topk_payload(3)
        references.extend(weakref.ref(tensor) for tensor in payload.values())
        connector.send_attn_output(_FakeTensor((3, 16)), context, **payload)
    del payload, context
    fake_torch.ops.afd_ascend.calls.clear()
    assert all(reference() is not None for reference in references)
    connector.close()
    assert all(reference() is None for reference in references)


def test_async_routing_preserves_supplied_router_results():
    expected_weights = torch.tensor([[1.5, 0.5]])
    expected_ids = torch.tensor([[2, 6]], dtype=torch.int32)

    def select_experts(hidden_states, router_logits, *, topk_indices_dtype):
        return expected_weights, expected_ids.to(topk_indices_dtype)

    weights, ids = async_cam_module.select_cam_experts(
        hidden_states=torch.zeros(1, 4),
        router_logits=torch.zeros(1, 8),
        top_k=2,
        use_grouped_topk=True,
        renormalize=True,
        scoring_func="softmax",
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=2.0,
        e_score_correction_bias=None,
        mix_placement=False,
        num_logical_experts=8,
        num_shared_experts=1,
        num_experts=9,
        router=SimpleNamespace(select_experts=select_experts),
    )

    assert ids.dtype == torch.int32
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights)
