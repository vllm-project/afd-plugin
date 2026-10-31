from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu import p2p_hccl as hccl_module  # noqa: E402
from afd_plugin.connectors.npu.p2p_hccl import (  # noqa: E402
    HCCLAttentionGraphStreamPlan,
    P2pHcclAFDConnector,
)


def _vllm_config(
    *,
    num_ubatches: int = 1,
    dsv4: bool = True,
    max_num_batched_tokens: int = 16,
    mtp: bool = False,
    dspark: bool = False,
    mtp_draft_enforce_eager: bool = True,
    num_speculative_tokens: int = 1,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    assert not (mtp and dspark)
    draft_hf_config = SimpleNamespace(dspark_block_size=4) if dspark else None
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"] if dsv4 else [],
                hidden_size=4,
                hc_mult=4,
                num_hidden_layers=3,
                vocab_size=32,
            ),
        ),
        speculative_config=(
            SimpleNamespace(
                method="mtp",
                enforce_eager=mtp_draft_enforce_eager,
                num_speculative_tokens=num_speculative_tokens,
                draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
            )
            if mtp or dspark
            else None
        ),
    )


def _afd_config(*, role: str, attention: int = 1, ffn: int = 1):
    return AFDConfig(
        connector="P2pHcclAFDConnector",
        role=role,
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _connector(
    *,
    role: str,
    role_rank: int = 0,
    attention: int = 1,
    ffn: int = 1,
    num_ubatches: int = 1,
    max_num_batched_tokens: int = 16,
    mtp: bool = False,
    mtp_draft_enforce_eager: bool = True,
    num_speculative_tokens: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(
            num_ubatches=num_ubatches,
            max_num_batched_tokens=max_num_batched_tokens,
            mtp=mtp,
            mtp_draft_enforce_eager=mtp_draft_enforce_eager,
            num_speculative_tokens=num_speculative_tokens,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
        ),
        _afd_config(role=role, attention=attention, ffn=ffn),
        role_rank,
    )
    connector._initialized = True
    connector.data_pg_list = [object() for _ in range(num_ubatches)]
    connector.ids_pg_list = [object() for _ in range(num_ubatches)]
    connector.input_ids_buffers = [
        torch.empty(16, dtype=torch.int32) for _ in range(num_ubatches)
    ]
    connector.mtp_header_buffers = [
        torch.empty(4 + ffn, dtype=torch.int32) for _ in range(num_ubatches)
    ]
    if ffn > attention:
        connector.mtp_header_buffers_by_peer = [
            {
                peer_rank: torch.empty(4 + ffn, dtype=torch.int32)
                for peer_rank in connector.mapping.ffn_peer_ranks
            }
            for _ in range(num_ubatches)
        ]
    return connector


def _attention_context(*, layer_idx: int, stage_idx: int, num_tokens: int):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_len=num_tokens,
        ),
    )


def _mtp_attention_context(*, num_tokens: int):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=num_tokens,
            phase="mtp",
            speculative_step=0,
        ),
    )


def test_p2p_hccl_factory_registration():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, P2pHcclAFDConnector)
    assert connector.requires_input_ids is True
    assert connector.topology.role_rank == 0
    assert connector.is_initialized is False


def test_p2p_hccl_keeps_dspark_draft_on_attention():
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(dspark=True, num_speculative_tokens=4),
        _afd_config(role="attention"),
        0,
    )

    assert connector.requires_mtp is False
    assert connector.num_speculative_tokens == 0
    assert connector.mtp_draft_graph_enabled is False


@pytest.mark.parametrize(
    ("role", "role_rank", "expected_subgroup", "expected_peers"),
    [
        ("attention", 0, 0, (0, 2, 3)),
        ("attention", 3, 1, (1, 4, 5)),
        ("ffn", 0, 0, (0, 2, 3)),
        ("ffn", 1, 1, (1, 4, 5)),
    ],
)
def test_p2p_hccl_accepts_integer_multiple_topology(
    role,
    role_rank,
    expected_subgroup,
    expected_peers,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role=role, attention=4, ffn=2),
        role_rank,
    )

    assert connector.ratio == 2
    assert connector.mapping.subgroup_index == expected_subgroup
    assert connector.mapping.subgroup_ranks == expected_peers


@pytest.mark.parametrize(
    ("attention", "ffn"),
    [(3, 2), (0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_p2p_hccl_rejects_invalid_unequal_topology(attention, ffn):
    with pytest.raises(ValueError, match="P2P AFD connectors require"):
        P2pHcclAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", attention=attention, ffn=ffn),
            0,
        )


def test_p2p_hccl_attention_sends_ids_before_hidden(monkeypatch):
    connector = _connector(role="attention")
    events = []
    forward_context = SimpleNamespace(afd_input_ids_pretransferred=False)
    monkeypatch.setattr(hccl_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: events.append(("yield", tensor.clone())),
    )

    def send(tensor, *, dst, group):
        kind = "ids" if tensor.dtype == torch.int32 else "hidden"
        events.append((kind, dst, group, tensor.clone()))

    monkeypatch.setattr(hccl_module.dist, "send", send)
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=0, stage_idx=0, num_tokens=3),
        input_ids=torch.tensor([-1, 2, 31], dtype=torch.int64),
    )

    assert [event[0] for event in events] == ["ids", "yield", "hidden"]
    assert events[0][1:3] == (0, connector.ids_pg_list[0])
    assert events[2][1:3] == (0, connector.data_pg_list[0])
    assert events[0][3].dtype == torch.int32


def test_p2p_hccl_mtp_sends_fixed_header_before_moe_input(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2, mtp=True)
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: events.append(
            (tensor.clone(), dst, group),
        ),
    )
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _mtp_attention_context(num_tokens=3),
        num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
    )

    assert len(events) == 2
    header, dst, group = events[0]
    assert (dst, group) == (0, connector.ids_pg_list[0])
    assert header.dtype == torch.int32
    assert header[1:].tolist() == [0, 3, 1, 3]
    assert events[1][1:] == (0, connector.data_pg_list[0])
    assert events[1][0].shape == (3, 4)


def test_p2p_hccl_mtp_header_carries_each_configured_speculative_step(monkeypatch):
    connector = _connector(
        role="attention",
        mtp=True,
        num_speculative_tokens=3,
    )
    headers = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: headers.append(tensor.clone()),
    )

    for speculative_step in range(3):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=speculative_step,
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
            stage_idx=0,
        )

    assert [int(header[1]) for header in headers] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="speculative step must be in"):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=3,
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
            stage_idx=0,
        )


def test_p2p_hccl_mtp_fans_out_header_and_projected_counts(monkeypatch):
    connector = _connector(
        role="attention",
        attention=1,
        ffn=2,
        mtp=True,
        num_speculative_tokens=3,
    )
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=5,
        speculative_step=2,
        num_tokens_across_dp=torch.tensor([5], dtype=torch.int32),
        stage_idx=0,
    )

    assert [(dst, header.tolist()) for header, dst, _group in sent] == [
        (0, [0x4D545031, 2, 3, 2, 3, 2]),
        (1, [0x4D545031, 2, 2, 2, 3, 2]),
    ]
    assert all(group is connector.ids_pg_list[0] for _, _, group in sent)


def test_p2p_hccl_mtp_expands_dp_counts_for_equal_tp2(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=4,
        mtp=True,
        tensor_parallel_size=2,
        data_parallel_size=2,
    )
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=5,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([3, 5], dtype=torch.int32),
        stage_idx=0,
    )

    assert len(sent) == 1
    header, dst, group = sent[0]
    assert (dst, group) == (3, connector.ids_pg_list[0])
    assert header.tolist() == [0x4D545031, 0, 5, 4, 3, 3, 5, 5]


def test_p2p_hccl_control_plane_rejects_mismatched_tp_size():
    connector = _connector(
        role="ffn",
        attention=2,
        ffn=2,
        tensor_parallel_size=2,
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([3])},
        is_graph_capturing=False,
        is_warmup=False,
        tensor_parallel_size=1,
    )

    with pytest.raises(RuntimeError, match="matching Attention/FFN"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_p2p_hccl_ffn_aggregates_unequal_mtp_headers_and_splits_output(
    monkeypatch,
):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
        mtp=True,
    )
    connector.mtp_hidden_recv_buffers[0] = torch.empty((9, 4), dtype=torch.bfloat16)
    headers = {
        4: torch.tensor([0x4D545031, 0, 4, 2, 5, 9], dtype=torch.int32),
        5: torch.tensor([0x4D545031, 0, 5, 2, 5, 9], dtype=torch.int32),
    }
    events = []

    def recv(tensor, *, src, group):
        events.append((src, group, tuple(tensor.shape)))
        if tensor.dtype == torch.int32:
            tensor.copy_(headers[src])
        else:
            tensor.fill_(src)

    sent = []
    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    header = connector.recv_mtp_header(stage_idx=0)
    payload = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        phase="mtp",
        speculative_step=header.speculative_step,
        num_tokens=header.num_tokens,
    )
    connector.send_ffn_output(payload.hidden_states + 1, payload.context)

    assert header.num_tokens == 9
    assert header.speculative_step == 0
    assert header.num_tokens_across_dp.tolist() == [5, 9]
    assert events == [
        (4, connector.ids_pg_list[0], (6,)),
        (5, connector.ids_pg_list[0], (6,)),
        (4, connector.data_pg_list[0], (4, 4)),
        (5, connector.data_pg_list[0], (5, 4)),
    ]
    assert payload.context.metadata.seq_lens == [4, 5]
    assert payload.hidden_states[:, 0].tolist() == [4] * 4 + [5] * 5
    assert [(dst, tensor.shape[0]) for tensor, dst, _group in sent] == [
        (4, 4),
        (5, 5),
    ]
    assert connector.mtp_stage_layouts == {}


def test_p2p_hccl_attention_stream_pipeline_orders_sync_send_recv(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    active_stream = [None]
    compute_stream = object()
    send_stream = object()
    recv_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    connector.a2f_send_stream = send_stream
    connector.f2a_recv_stream = recv_stream
    connector.attention_pipeline_events = {
        (1, 0): hccl_module.HCCLAttentionPipelineEvents(
            compute_done=FakeEvent("compute"),
            send_done=FakeEvent("send"),
            recv_done=FakeEvent("recv"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(dbo_enabled=True, num_ubatches=2),
    )
    monkeypatch.setattr(hccl_module.torch.npu, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: calls.append(
            ("dist.send", active_stream[0], dst, group)
        ),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, *, src, group: calls.append(
            ("dist.recv", active_stream[0], src, group, tensor)
        ),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: calls.append(("yield", tensor)),
    )
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    output = connector.recv_ffn_output(hidden, ubatch_idx=0)

    assert output is not hidden
    assert connector.pending_attention_transfers == {}
    assert calls == [
        ("compute", "record", compute_stream),
        ("compute", "wait", send_stream),
        ("dist.send", send_stream, 0, connector.data_pg_list[0]),
        ("send", "record", send_stream),
        ("send", "wait", recv_stream),
        ("dist.recv", recv_stream, 0, connector.data_pg_list[0], output),
        ("recv", "record", recv_stream),
        ("yield", output),
        ("recv", "wait", compute_stream),
    ]


@pytest.mark.parametrize(
    ("three_stream_enabled", "expected_side_transport"),
    [("1", True), ("0", False)],
)
def test_p2p_hccl_initializes_configured_attention_graph_stream_plan(
    monkeypatch,
    three_stream_enabled,
    expected_side_transport,
):
    monkeypatch.setenv(
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        three_stream_enabled,
    )
    connector = _connector(role="attention", num_ubatches=2)
    send_stream = object()
    recv_stream = object()
    compute_stream = object()
    streams = iter((send_stream, recv_stream, compute_stream))
    monkeypatch.setattr(
        hccl_module.torch.npu, "Stream", lambda **_kwargs: next(streams)
    )
    monkeypatch.setattr(hccl_module.torch.npu, "Event", object)

    connector._initialize_attention_stream_pipeline()

    plan = connector.attention_graph_stream_plan
    assert plan is not None
    assert plan.compute_stream is compute_stream
    assert plan.send_stream is (send_stream if expected_side_transport else None)
    assert plan.recv_stream is (recv_stream if expected_side_transport else None)


def test_p2p_hccl_graph_compute_pipeline_can_be_disabled_for_comparison(
    monkeypatch,
):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "0")
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    connector.attention_graph_compute_stream = object()
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=connector.attention_graph_compute_stream,
    )
    connector.attention_graph_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )

    assert connector.graph_u2_compute_overlap_enabled is False
    assert connector.attention_graph_compute_pipeline_active() is False


def test_p2p_hccl_eager_u2_stream_overlap_can_be_disabled_for_comparison(
    monkeypatch,
):
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "0")
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=False,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )

    assert connector.stream_overlap_enabled is True
    assert connector.eager_u2_stream_overlap_enabled is False
    assert connector._attention_stream_pipeline_active() is False


def test_p2p_hccl_stage_diagnostics_default_off(monkeypatch):
    monkeypatch.delenv("AFD_HCCL_STAGE_DIAGNOSTICS", raising=False)

    connector = _connector(role="attention", num_ubatches=2)

    assert connector.stage_diagnostics_enabled is False


def test_p2p_hccl_stage_diagnostics_can_be_enabled_for_u1(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_STAGE_DIAGNOSTICS", "1")

    connector = _connector(role="attention", num_ubatches=1)

    assert connector.stage_diagnostics_enabled is True
    assert connector.num_stages == 1


def test_p2p_hccl_ffn_compute_sync_diagnostics_default_off(monkeypatch):
    monkeypatch.delenv(
        "AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS",
        raising=False,
    )

    connector = _connector(role="ffn", num_ubatches=1)

    assert connector.ffn_compute_sync_diagnostics_enabled is False


def test_p2p_hccl_ffn_compute_sync_diagnostics_can_be_enabled(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS", "1")

    connector = _connector(role="ffn", num_ubatches=1)

    assert connector.ffn_compute_sync_diagnostics_enabled is True


@pytest.mark.parametrize(
    "name",
    [
        "AFD_HCCL_EAGER_U2_STREAM_OVERLAP",
        "AFD_HCCL_STAGE_DIAGNOSTICS",
        "AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS",
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
    ],
)
def test_p2p_hccl_rejects_invalid_graph_physical_pipeline_values(
    monkeypatch,
    name,
):
    monkeypatch.setenv(name, "invalid")

    with pytest.raises(RuntimeError, match=rf"{name} must be 0 or 1"):
        _connector(role="attention", num_ubatches=2)


def test_p2p_hccl_graph_physical_pipeline_defaults_are_enabled(monkeypatch):
    for name in (
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
    ):
        monkeypatch.delenv(name, raising=False)
    connector = _connector(role="attention", num_ubatches=2)

    assert connector.graph_u2_attention_three_stream_enabled is True
    assert connector.graph_u2_ffn_recv_stream_enabled is True
    assert connector.graph_u2_ffn_cross_layer_enabled is True


@pytest.mark.parametrize(
    ("is_compiling", "is_graph_capturing", "is_warmup", "uses_graph_ops"),
    [
        (False, False, False, False),
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_p2p_hccl_graph_transport_activation(
    monkeypatch,
    is_compiling,
    is_graph_capturing,
    is_warmup,
    uses_graph_ops,
):
    connector = _connector(role="attention")
    connector.is_graph_capturing = is_graph_capturing
    connector.is_warmup = is_warmup
    monkeypatch.setattr(
        hccl_module.torch.compiler,
        "is_compiling",
        lambda: is_compiling,
    )

    assert connector._graph_transport_active() is uses_graph_ops


def test_p2p_hccl_capture_uses_graph_send_recv(monkeypatch):
    connector = _connector(role="ffn")
    connector.is_graph_capturing = True
    graph_calls = []
    side_stream = object()
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        connector,
        "_record_stream",
        lambda tensor, stream: graph_calls.append(("record", tensor, stream)),
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_send",
        lambda tensor, *, dst, group: graph_calls.append(("send", tensor, dst, group)),
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_recv",
        lambda tensor, *, src, group: graph_calls.append(("recv", tensor, src, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.send"),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.recv"),
    )
    tensor = torch.ones((2, 4), dtype=torch.bfloat16)

    connector._send_tensor(
        tensor,
        dst=1,
        group=connector.data_pg_list[0],
        stream=side_stream,
    )
    connector._recv_tensor(
        tensor,
        src=1,
        group=connector.data_pg_list[0],
        stream=side_stream,
    )

    assert graph_calls == [
        ("record", tensor, side_stream),
        ("send", tensor, 1, connector.data_pg_list[0]),
        ("record", tensor, side_stream),
        ("recv", tensor, 1, connector.data_pg_list[0]),
    ]


def test_p2p_hccl_attention_uses_mapped_ffn_rank(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=2,
    )
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: events.append(("send", dst, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda _tensor, *, src, group: events.append(("recv", src, group)),
    )
    monkeypatch.setattr(hccl_module, "maybe_apply_dbo_yield", lambda *_a, **_k: None)

    connector.send_attn_output(
        torch.ones((2, 4), dtype=torch.bfloat16),
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    connector.recv_ffn_output(torch.empty((2, 4), dtype=torch.bfloat16))

    assert events == [
        ("send", 1, connector.data_pg_list[0]),
        ("recv", 1, connector.data_pg_list[0]),
    ]


def test_p2p_hccl_attention_fans_out_ids_hidden_and_gathers_output(monkeypatch):
    connector = _connector(role="attention", attention=1, ffn=2)
    sent_ids = []
    sent_hidden = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent_ids.append(
            (tensor.clone(), dst, group),
        ),
    )
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent_hidden.append(
            (tensor.clone(), dst, group),
        ),
    )
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    connector.send_input_ids(torch.arange(5), ubatch_idx=0)
    connector._send_attention_tensor(hidden, group=connector.data_pg_list[0])

    assert [(dst, tensor.tolist()) for tensor, dst, _group in sent_ids] == [
        (0, [0, 1, 2]),
        (1, [3, 4]),
    ]
    assert [(dst, tuple(tensor.shape)) for tensor, dst, _group in sent_hidden] == [
        (0, (3, 4)),
        (1, (2, 4)),
    ]
    assert torch.equal(sent_hidden[0][0], hidden[:3])
    assert torch.equal(sent_hidden[1][0], hidden[3:])

    def recv(tensor, *, src, group):
        assert group is connector.data_pg_list[0]
        tensor.fill_(src + 1)

    monkeypatch.setattr(connector, "_recv_tensor", recv)
    output = torch.empty_like(hidden)
    connector._recv_attention_tensor(output, group=connector.data_pg_list[0])

    assert output[:, 0].tolist() == [1, 1, 1, 2, 2]


def test_balanced_split_sizes_supports_fullgraph_dynamic_shapes():
    def split_tensor(tensor):
        first, second = hccl_module._balanced_split_sizes(int(tensor.shape[0]), 2)
        return tensor[:first], tensor[first : first + second]

    compiled = torch.compile(
        split_tensor,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )

    for num_tokens, expected_sizes in ((1, (1, 0)), (2, (1, 1)), (5, (3, 2))):
        shards = compiled(torch.ones((num_tokens, 4)))
        assert tuple(shard.shape[0] for shard in shards) == expected_sizes


def test_p2p_hccl_attention_fanout_pads_and_discards_dummy_token(monkeypatch):
    connector = _connector(role="attention", attention=1, ffn=2)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )
    hidden = torch.full((1, 4), 7, dtype=torch.float32)

    connector._send_attention_tensor(hidden, group=connector.data_pg_list[0])

    assert [tuple(tensor.shape) for tensor, _dst, _group in sent] == [(1, 4), (1, 4)]
    assert torch.equal(sent[0][0], hidden)
    assert torch.equal(sent[1][0], torch.zeros_like(hidden))

    def recv(tensor, *, src, group):
        tensor.fill_(src + 11)

    monkeypatch.setattr(connector, "_recv_tensor", recv)
    output = torch.empty_like(hidden)
    connector._recv_attention_tensor(output, group=connector.data_pg_list[0])

    assert output.tolist() == [[11, 11, 11, 11]]


@pytest.mark.parametrize("num_peers", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 5, 8, 9])
@pytest.mark.parametrize("direction", ["send", "recv"])
def test_p2p_hccl_fanout_reuses_graph_for_short_and_odd_batches(
    monkeypatch,
    num_peers,
    num_tokens,
    direction,
):
    connector = _connector(role="attention", attention=1, ffn=num_peers)
    shards = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: shards.append(tensor.clone()),
    )

    def recv(tensor, *, src, group):
        tensor.fill_(src + 11)
        shards.append(tensor.clone())

    monkeypatch.setattr(connector, "_recv_tensor", recv)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append((graph, example_inputs))
        return graph.forward

    def transfer(tensor):
        if direction == "send":
            connector._send_attention_tensor(tensor, group=connector.data_pg_list[0])
        else:
            connector._recv_attention_tensor(tensor, group=connector.data_pg_list[0])
        return tuple(shards)

    warmup = torch.ones((8, 4))
    torch._dynamo.mark_dynamic(warmup, 0)
    torch._dynamo.reset_code(transfer.__code__)
    torch.compile(transfer, backend=backend, fullgraph=True)(warmup)
    assert len(graphs) == 1
    graph, example_inputs = graphs[0]
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    # vLLM reuses the range graph without Dynamo's shape guards. Calling the
    # saved graph directly must handle sizes absent from the warmup batch.
    graph_inputs = []
    for value in example_inputs:
        if isinstance(value, torch.SymInt):
            # Dynamo may also lift connector.ratio as a symbolic argument.
            graph_inputs.append(num_tokens if int(value) == 8 else int(value))
        else:
            graph_inputs.append(hidden)
    result = graph(*graph_inputs)
    base, remainder = divmod(max(num_tokens, num_peers), num_peers)
    sizes = [base + (peer < remainder) for peer in range(num_peers)]
    assert [shard.shape[0] for shard in result] == sizes, graph.code
    if direction == "send":
        expected = torch.cat((hidden, torch.zeros((max(num_peers - num_tokens, 0), 4))))
        torch.testing.assert_close(torch.cat(result), expected)
    else:
        expected = torch.cat(
            [
                torch.full((size, 4), peer + 11, dtype=torch.float32)
                for peer, size in enumerate(sizes)
            ]
        )[:num_tokens]
        torch.testing.assert_close(hidden, expected)


def test_p2p_hccl_control_plane_uses_one_sender_per_subgroup(monkeypatch):
    payload = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    for role_rank in range(4):
        connector = _connector(
            role="attention",
            role_rank=role_rank,
            attention=4,
            ffn=2,
        )
        connector.p2p_pg = object()
        connector.control_plane.send_dp_metadata_list(payload)

    assert [(dst, device.type) for _value, dst, _group, device in sent] == [
        (0, "cpu"),
        (1, "cpu"),
    ]


def test_p2p_hccl_mtp_phase_marker_uses_control_sender(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=2,
        attention=4,
        ffn=2,
    )
    connector.p2p_pg = object()
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    connector.control_plane.send_mtp_phase_ready(graph_replay=True)

    assert len(sent) == 1
    payload, dst, group, device = sent[0]
    assert payload.mtp_phase_ready is True
    assert payload.mtp_phase_graph_replay is True
    assert payload.dp_metadata_list == {}
    assert dst == 1
    assert group is connector.p2p_pg
    assert device == torch.device("cpu")


def test_p2p_hccl_mtp_phase_receive_stashes_next_target(monkeypatch):
    connector = _connector(role="ffn")
    connector.p2p_pg = object()
    next_target = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
        mtp_phase_control_enabled=True,
    )
    monkeypatch.setattr(
        connector.control_plane,
        "_recv_payload",
        lambda: next_target,
    )

    assert connector.control_plane.recv_mtp_phase_ready() is False
    assert connector.control_plane.recv_dp_metadata_list() is next_target
    assert connector.mtp_phase_control_enabled is True


def test_p2p_hccl_mtp_phase_receive_consumes_marker(monkeypatch):
    connector = _connector(role="ffn")
    marker = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
        mtp_phase_ready=True,
        mtp_phase_graph_replay=True,
    )
    monkeypatch.setattr(
        connector.control_plane,
        "_recv_payload",
        lambda: marker,
    )

    assert connector.control_plane.recv_mtp_phase_ready() is True
    assert connector.mtp_phase_graph_replay is True
    assert connector.control_plane._pending_payload is None


def test_p2p_hccl_close_destroys_all_groups(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    connector.p2p_pg = object()
    groups = [
        connector.p2p_pg,
        *connector.ids_pg_list,
        *connector.data_pg_list,
    ]
    destroyed = []
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    connector.close()

    assert destroyed == groups
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.input_ids_buffers == []
    assert connector.hidden_recv_buffers == {}
    assert connector.stage_layouts == {}
    assert connector.mtp_stage_layouts == {}
    assert connector.is_initialized is False


def test_p2p_hccl_partial_init_failure_destroys_created_groups(monkeypatch):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    data_group = object()
    calls = iter((data_group, RuntimeError("IDs group failed")))

    def init_group(**_kwargs):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    destroyed = []
    monkeypatch.setattr(hccl_module, "init_afd_process_group", init_group)
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    with pytest.raises(RuntimeError, match="IDs group failed"):
        connector.init_afd_connector()

    assert destroyed == [data_group]
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.is_initialized is False


def test_p2p_hccl_graph_init_registers_hccl_graph_ops_before_groups(monkeypatch):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(enforce_eager=False),
        _afd_config(role="attention"),
        0,
    )
    calls = []
    monkeypatch.setattr(
        hccl_module,
        "_ensure_graph_hccl_ops_registered",
        lambda: calls.append("register"),
    )
    monkeypatch.setattr(
        hccl_module,
        "init_afd_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after register")),
    )

    with pytest.raises(RuntimeError, match="stop after register"):
        connector.init_afd_connector()

    assert calls == ["register"]


@pytest.mark.parametrize("input_ids", [[-2], [32]])
def test_p2p_hccl_rejects_invalid_input_ids(monkeypatch, input_ids):
    connector = _connector(role="attention")
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=False),
    )

    with pytest.raises(ValueError, match="-1 padding"):
        connector.send_attn_output(
            torch.ones((1, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=0, stage_idx=0, num_tokens=1),
            input_ids=torch.tensor(input_ids),
        )


def test_p2p_hccl_rejects_non_integer_input_ids_before_send():
    connector = _connector(role="attention")

    with pytest.raises(TypeError, match="int32 or int64"):
        connector._validate_input_ids(torch.ones(1, dtype=torch.float32), 1)


def test_p2p_hccl_device_input_ids_skip_host_value_readback():
    connector = _connector(role="attention")
    device_ids = torch.empty(1, dtype=torch.int64, device="meta")

    connector._validate_input_ids(device_ids, 1)


def test_p2p_hccl_module_has_no_camp2p_custom_op_reference():
    source = __import__("inspect").getsource(hccl_module)

    assert "torch.ops.vllm.afd_camp2p" not in source
    assert "torch.ops.afd_ascend" not in source


def test_p2p_hccl_graph_lowering_omits_dynamic_shape_guard(monkeypatch):
    calls = []

    class FakeOp:
        def default(self, *args):
            calls.append(args)
            return args[0]

    monkeypatch.setattr(
        hccl_module.torch,
        "ops",
        SimpleNamespace(
            npu_define=SimpleNamespace(_send=FakeOp(), _recv=FakeOp()),
        ),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "get_process_group_ranks",
        lambda group: [0, 1] if group == "data-group" else pytest.fail(),
    )
    monkeypatch.setattr(
        hccl_module.c10d,
        "_get_group_tag",
        lambda group: "afd-data" if group == "data-group" else pytest.fail(),
    )
    tensor = torch.zeros((2, 4), dtype=torch.bfloat16)

    hccl_module._graph_hccl_send(tensor, dst=1, group="data-group")
    hccl_module._graph_hccl_recv(tensor, src=0, group="data-group")

    assert calls[0][1:] == (1, [0, 1], "afd-data", 0, None, None)
    assert calls[1][1:] == (0, [0, 1], "afd-data", 0, None, None)
