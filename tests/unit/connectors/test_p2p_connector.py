# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import importlib
import sys
import types
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig, afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.base import AFDControlPlaneClosedError  # noqa: E402


@pytest.fixture
def lifecycle_connector():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
    )
    return module, connector


def test_p2p_close_cancels_receive_and_releases_owned_resources_idempotently(
    monkeypatch, lifecycle_connector
):
    module, connector = lifecycle_connector
    owned = SimpleNamespace(active=True)
    connector.p2p_pg = owned
    communicators = [SimpleNamespace(active=True) for _ in range(2)]
    for communicator in communicators:
        communicator.destroy = lambda owner=communicator: setattr(
            owner, "active", False
        )
    connector.a2e_pynccl, connector.e2a_pynccl = communicators

    def abort(group):
        if not group.active:
            raise ValueError("group already unregistered")
        group.active = False

    monkeypatch.setattr(module, "_abort_process_group", abort)

    connector.control_plane.close()
    connector.close()
    connector.close()

    assert not owned.active
    assert all(not communicator.active for communicator in communicators)
    assert not connector.is_initialized
    with pytest.raises(AFDControlPlaneClosedError):
        connector.control_plane.recv_dp_metadata_list()


def test_p2p_control_plane_receive_requires_initialized_group(lifecycle_connector):
    _, connector = lifecycle_connector

    with pytest.raises(RuntimeError, match="process group is not initialized"):
        connector.control_plane.recv_dp_metadata_list()


def _fake_vllm_config(
    *,
    data_parallel_size=1,
    data_parallel_rank=0,
    enforce_eager=True,
    tensor_parallel_size=1,
):
    text_config = SimpleNamespace(hidden_size=16, num_hidden_layers=2)
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
        ),
    )


def test_p2p_connector_can_be_constructed_without_runtime_initialization():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )

    assert connector.is_initialized is False
    assert connector.world_rank == 1
    assert connector.dst_list == [0]


def test_p2p_connector_uses_nested_text_config_for_multimodal_model():
    text_config = SimpleNamespace(hidden_size=24, num_hidden_layers=40)
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        SimpleNamespace(
            additional_config={},
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                enforce_eager=True,
                hf_config=SimpleNamespace(model_type="qwen3_5_moe"),
                hf_text_config=text_config,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                prefill_context_parallel_size=1,
                tensor_parallel_size=1,
            ),
        ),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
    )

    assert connector.num_hidden_layers == (40,)
    assert connector.hidden_size == 24


def test_p2p_connector_uses_factory_resolved_role_rank():
    connector = AFDConnectorFactory.create_connector(
        3,
        3,
        _fake_vllm_config(
            data_parallel_size=4,
            data_parallel_rank=3,
        ),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=4,
            num_ffn_ranks=4,
        ),
    )

    assert connector.mapping.role_rank == 3
    assert connector.world_rank == 7
    assert connector.p2p_rank == 7


def test_p2p_tp2_maps_shared_dp_payload_one_to_one(
    monkeypatch,
):  # long-test: paired A/F control and data routing over the same TP2 topology
    p2p_module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    parallel_state = importlib.import_module("vllm.distributed.parallel_state")
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([7])},
        is_graph_capturing=False,
        is_warmup=False,
    )

    attention_connectors: list[AFDConnectorBase] = []
    ffn_connectors: list[AFDConnectorBase] = []
    for role, connectors in (
        ("attention", attention_connectors),
        ("ffn", ffn_connectors),
    ):
        for role_rank in range(2):
            monkeypatch.setattr(
                parallel_state,
                "get_tensor_model_parallel_rank",
                lambda role_rank=role_rank: role_rank,
            )
            connector = AFDConnectorFactory.create_connector(
                role_rank,
                0,
                _fake_vllm_config(tensor_parallel_size=2),
                AFDConfig(
                    role=role,
                    connector="P2pNcclAFDConnector",
                    num_attention_ranks=2,
                    num_ffn_ranks=2,
                ),
            )
            connector.control_plane.update_state_from_dp_metadata(payload)
            connectors.append(connector)

    assert [
        (
            connector.mapping.role_rank,
            connector.mapping.subgroup_ranks,
            tuple(connector.dst_list),
        )
        for connector in attention_connectors
    ] == [
        (0, (0, 2), (0,)),
        (1, (1, 3), (1,)),
    ]
    assert [
        (connector.mapping.role_rank, connector.mapping.subgroup_ranks)
        for connector in ffn_connectors
    ] == [
        (0, (0, 2)),
        (1, (1, 3)),
    ]

    sent_destinations = []
    monkeypatch.setattr(
        p2p_module,
        "send_control_payload",
        lambda _payload, **kwargs: sent_destinations.append(tuple(kwargs["dst"])),
    )
    for connector in attention_connectors:
        connector.p2p_pg = object()
        connector.control_plane.send_dp_metadata_list(payload)

    received_sources = []

    def receive_payload(**kwargs):
        received_sources.append(kwargs["src"])
        return payload

    monkeypatch.setattr(
        p2p_module,
        "recv_control_payload",
        receive_payload,
    )
    for connector in ffn_connectors:
        connector.p2p_pg = object()
        connector.control_plane.recv_dp_metadata_list()

    assert sent_destinations == [(0,), (1,)]
    assert received_sources == [2, 3]

    attention_routes = []
    attention_context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=7,
        ),
    )
    attention_hidden_states = torch.zeros((7, 16), dtype=torch.bfloat16)
    for connector in attention_connectors:

        def record_attention_send(
            hidden_states,
            dst,
            process_group,
            comm_id,
            *,
            connector=connector,
        ):
            assert dst == 0
            attention_routes.append(
                (connector.world_rank, connector.mapping.subgroup_ranks[dst]),
            )

        monkeypatch.setattr(connector, "_send_hidden_states", record_attention_send)
        connector.send_attn_output(attention_hidden_states, attention_context)

    assert attention_routes == [(2, 0), (3, 1)]

    ffn_routes = []
    ffn_context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[7],
        ),
    )
    ffn_output = torch.zeros((7, 16), dtype=torch.bfloat16)
    for connector in ffn_connectors:

        def record_ffn_send(
            tensors,
            dsts,
            process_group,
            comm_id,
            *,
            connector=connector,
        ):
            assert dsts == [1]
            ffn_routes.append(
                (connector.world_rank, connector.mapping.subgroup_ranks[dsts[0]]),
            )

        monkeypatch.setattr(connector, "_send_tensors", record_ffn_send)
        connector.send_ffn_output(ffn_output, ffn_context)

    assert ffn_routes == [(0, 2), (1, 3)]


@pytest.fixture
def receive_peer_payload(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    # Only allocation/transport are replaced; real control parsing and fan-in run.
    monkeypatch.setattr(
        module,
        "_recv_buffer",
        lambda metadata, reference: torch.empty(
            metadata.size, dtype=metadata.dtype, device="cpu"
        ),
    )

    def deliver(buffers, sources, comm_id):
        for tensor, source in zip(buffers, sources, strict=True):
            tensor.fill_(source)

    monkeypatch.setattr(torch.ops.vllm, "afd_p2p_recv_group", deliver, raising=False)

    def receive(connector):
        connector.a2e_group = SimpleNamespace(world_size=connector.group_size)
        connector.a2e_comm_id = 17
        return connector.recv_attn_output()

    return receive


@pytest.mark.parametrize(
    (
        "attention_size",
        "ffn_size",
        "ffn_rank",
        "token_counts",
        "expected_peer_tokens",
    ),
    [
        (2, 1, 0, [3, 5], [3, 5]),
        (4, 2, 0, [3, 5, 7, 11], [3, 5]),
        (4, 2, 1, [3, 5, 7, 0], [7, 1]),
        (6, 3, 2, [2, 3, 5, 7, 11, 13], [11, 13]),
        # A % F != 0: subgroup 0 holds A0 and A1, subgroup 1 holds A2 alone.
        (3, 2, 0, [3, 5, 7], [3, 5]),
        (3, 2, 1, [3, 5, 7], [7]),
    ],
)
def test_p2p_ffn_metadata_tracks_each_attention_peer_in_xayf(
    receive_peer_payload,
    attention_size,
    ffn_size,
    ffn_rank,
    token_counts,
    expected_peer_tokens,
):
    connector = AFDConnectorFactory.create_connector(
        ffn_rank,
        0,
        _fake_vllm_config(
            data_parallel_size=ffn_size,
            data_parallel_rank=ffn_rank,
        ),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
        ),
    )

    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata(token_counts)},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )

    payload = receive_peer_payload(connector)
    assert payload.context.metadata.seq_lens == expected_peer_tokens
    assert payload.hidden_states.shape == (sum(expected_peer_tokens), 16)
    assert payload.hidden_states[:, 0].tolist() == [
        source
        for source, count in enumerate(expected_peer_tokens, 1)
        for _ in range(count)
    ]


def test_p2p_tensor_metadata_clamps_idle_attention_rank_to_dummy_token(
    receive_peer_payload,
):
    ffn_connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([0, 4])},
        is_graph_capturing=False,
        is_warmup=False,
    )

    ffn_connector.control_plane.update_state_from_dp_metadata(payload)

    received = receive_peer_payload(ffn_connector)
    assert received.context.metadata.seq_lens == [1, 4]
    assert received.hidden_states.shape == (5, 16)

    attention_connector = AFDConnectorFactory.create_connector(
        1,
        1,
        _fake_vllm_config(data_parallel_size=2, data_parallel_rank=0),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    attention_connector.control_plane.update_state_from_dp_metadata(payload)

    assert attention_connector.tensor_metadata_list[0].size == torch.Size([1, 16])


@pytest.fixture
def host_hidden_allocations(monkeypatch):
    """Observe Python buffer ownership, without claiming CUDA validation."""
    allocations = []
    empty = torch.empty

    def allocate(*args, **kwargs):
        kwargs["device"] = "cpu"
        tensor = empty(*args, **kwargs)
        if tensor.ndim == 2:
            allocations.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(
        torch.ops.vllm, "afd_p2p_recv_group", lambda *args: None, raising=False
    )
    return allocations


def test_eager_steps_release_hidden_buffers_with_graphs_enabled(
    lifecycle_connector, host_hidden_allocations
):
    _, connector = lifecycle_connector
    connector.vllm_config.model_config.enforce_eager = False
    connector.a2e_group = SimpleNamespace(world_size=connector.group_size)
    connector.a2e_comm_id = 17

    for num_tokens in (3, 7, 5):
        connector.control_plane.update_state_from_dp_metadata(
            AFDControlPayload(
                dp_metadata_list={0: AFDDPMetadata([num_tokens])},
                is_graph_capturing=False,
                is_warmup=False,
            ),
        )
        payload = connector.recv_attn_output()
        assert payload.hidden_states.shape == (num_tokens, 16)
        del payload
        assert host_hidden_allocations
        assert all(reference() is None for reference in host_hidden_allocations)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": 1,
                "num_ffn_ranks": 2,
            },
            "num_attention_ranks >= num_ffn_ranks",
        ),
    ],
)
def test_p2p_topology_validation_errors_are_clear(raw, message):
    with pytest.raises(ValueError, match=message):
        afd_config_from_mapping(raw)


@pytest.fixture
def registered_p2p_ops(monkeypatch):
    """Run the op registration against a capturing registrar.

    Returns the P2P module and the ``direct_register_custom_op`` kwargs in
    registration order, so tests can call the real op implementations.
    """
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    calls = []

    # Empty ops namespace: the registration helper skips ops that already
    # exist on torch.ops.vllm, so the fake must report none registered.
    torch_module = SimpleNamespace(
        Tensor=object,
        ops=SimpleNamespace(vllm=SimpleNamespace()),
        cuda=SimpleNamespace(current_stream=lambda device: f"stream:{device}"),
    )

    vllm_module = types.ModuleType("vllm")
    utils_module = types.ModuleType("vllm.utils")
    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        torch_utils_module,
        "direct_register_custom_op",
        direct_register_custom_op,
        raising=False,
    )
    monkeypatch.setattr(utils_module, "torch_utils", torch_utils_module, raising=False)
    monkeypatch.setattr(vllm_module, "utils", utils_module, raising=False)

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_module)
    monkeypatch.setattr(module, "torch", torch_module)
    monkeypatch.setattr(module, "direct_register_custom_op", direct_register_custom_op)
    monkeypatch.setattr(module, "_AFD_CUSTOM_OPS_REGISTERED", False)

    module._register_p2p_custom_ops()
    return module, calls


def test_p2p_custom_ops_register_send_recv_with_fake_impls(registered_p2p_ops):
    _, calls = registered_p2p_ops

    assert [(call["op_name"], call["mutates_args"]) for call in calls] == [
        ("afd_p2p_send", ["tensor"]),
        ("afd_p2p_recv", ["out"]),
        ("afd_p2p_send_group", ["tensors"]),
        ("afd_p2p_recv_group", ["outs"]),
    ]
    assert all(callable(call["fake_impl"]) for call in calls)


class _RecordingCommunicator:
    """Stands in for PyNcclCommunicator and records the NCCL call sequence."""

    def __init__(self):
        self.events = []

    def group_start(self):
        self.events.append("group_start")

    def group_end(self):
        self.events.append("group_end")

    def send(self, tensor, dst, stream):
        self.events.append(("send", tensor, dst, stream))

    def recv(self, tensor, src, stream):
        self.events.append(("recv", tensor, src, stream))


# The grouped ops are the FFN rank's whole transport: every transfer, for one
# peer or several, must sit inside a single group_start()/group_end() pair, in
# the order given.
@pytest.mark.parametrize(
    ("op_name", "transfer"),
    [("afd_p2p_send_group", "send"), ("afd_p2p_recv_group", "recv")],
)
@pytest.mark.parametrize("peers", [[1], [1, 2], [1, 1, 2, 2]])
def test_p2p_group_ops_issue_transfers_inside_one_nccl_group(
    registered_p2p_ops,
    monkeypatch,
    op_name,
    transfer,
    peers,
):
    module, calls = registered_p2p_ops
    op_func = next(call["op_func"] for call in calls if call["op_name"] == op_name)
    communicator = _RecordingCommunicator()
    monkeypatch.setitem(module._AFD_COMMUNICATORS, 7, communicator)
    tensors = [SimpleNamespace(device="cuda:0") for _ in peers]

    op_func(tensors, peers, 7)

    assert communicator.events == [
        "group_start",
        *[
            (transfer, tensor, peer, "stream:cuda:0")
            for tensor, peer in zip(tensors, peers, strict=True)
        ],
        "group_end",
    ]


def test_p2p_recv_preserves_dynamic_ref_tensor_first_dim(
    monkeypatch,
):  # long-test: real/fake custom-op shape behavior across dynamic token counts
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    connector.e2a_comm_id = 23

    calls = []

    torch_module = SimpleNamespace(
        ops=SimpleNamespace(
            vllm=SimpleNamespace(
                afd_p2p_recv=lambda tensor, src, comm_id: calls.append(
                    (tensor, src, comm_id),
                ),
            ),
        ),
        empty=lambda *_args, **_kwargs: pytest.fail(
            "recv should reuse the dynamic ref tensor",
        ),
    )
    monkeypatch.setattr(module, "torch", torch_module)

    ref_tensor = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(7, 16),
        dtype="bf16",
    )
    tensor_metadata = SimpleNamespace(
        device="cuda:0",
        dtype="bf16",
        size=(64, 16),
    )

    output = connector._recv_hidden_states(
        0,
        SimpleNamespace(world_size=2, rank=1),
        connector.e2a_comm_id,
        tensor_metadata,
        ref_tensor=ref_tensor,
    )

    assert output is ref_tensor
    assert calls == [(ref_tensor, 0, 23)]


def _ffn_connector_with_peers(monkeypatch, recorded, peer_tokens):
    """FFN rank whose Attention peers send ``peer_tokens``; records wire calls."""
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=len(peer_tokens),
            num_ffn_ranks=1,
        ),
    )
    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata(peer_tokens)},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )
    connector.a2e_group = connector.e2a_group = SimpleNamespace(
        world_size=len(peer_tokens) + 1,
    )
    connector.a2e_comm_id, connector.e2a_comm_id = 5, 6
    real_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda size, **kwargs: real_empty(size, **{**kwargs, "device": "meta"}),
    )

    def record(op_name):
        def op(tensors, peers, comm_id):
            recorded.append(
                (
                    op_name,
                    [(tuple(tensor.shape), tensor.dtype) for tensor in tensors],
                    list(peers),
                    comm_id,
                ),
            )

        return op

    for op_name in ("afd_p2p_send_group", "afd_p2p_recv_group"):
        monkeypatch.setattr(torch.ops.vllm, op_name, record(op_name), raising=False)
    for op_name in ("afd_p2p_send", "afd_p2p_recv"):
        monkeypatch.setattr(
            torch.ops.vllm,
            op_name,
            lambda *args, op_name=op_name: pytest.fail(f"{op_name} bypassed the group"),
            raising=False,
        )
    return connector


# One NCCL group carries every peer's tensors, for one peer as well as several;
# within a peer the receive order matches its wire order (hidden states, then
# router logits).
@pytest.mark.parametrize(
    ("peer_tokens", "routing_spec", "expected_buffers", "expected_srcs"),
    [
        ([3], None, [((3, 16), torch.bfloat16)], [1]),
        (
            [3, 5],
            None,
            [((3, 16), torch.bfloat16), ((5, 16), torch.bfloat16)],
            [1, 2],
        ),
        (
            [3, 5],
            "router_logits",
            [
                ((3, 16), torch.bfloat16),
                ((3, 4), torch.float32),
                ((5, 16), torch.bfloat16),
                ((5, 4), torch.float32),
            ],
            [1, 1, 2, 2],
        ),
    ],
)
def test_p2p_ffn_receives_all_attention_peers_in_one_group(
    monkeypatch,
    peer_tokens,
    routing_spec,
    expected_buffers,
    expected_srcs,
):
    recorded: list[tuple] = []
    connector = _ffn_connector_with_peers(monkeypatch, recorded, peer_tokens)
    recv_kwargs = (
        {"routing_spec": AFDExpertRoutingSpec(4, torch.float32)} if routing_spec else {}
    )

    payload = connector.recv_attn_output(ubatch_idx=0, **recv_kwargs)

    assert recorded == [("afd_p2p_recv_group", expected_buffers, expected_srcs, 5)]
    assert tuple(payload.hidden_states.shape) == (sum(peer_tokens), 16)
    assert payload.context.metadata.seq_lens == peer_tokens
    if routing_spec:
        assert tuple(payload.router_logits.shape) == (sum(peer_tokens), 4)


@pytest.mark.parametrize(
    ("peer_tokens", "expected_slices", "expected_dsts"),
    [
        ([8], [((8, 16), torch.bfloat16)], [1]),
        ([3, 5], [((3, 16), torch.bfloat16), ((5, 16), torch.bfloat16)], [1, 2]),
    ],
)
def test_p2p_ffn_sends_all_attention_peers_in_one_group(
    monkeypatch,
    peer_tokens,
    expected_slices,
    expected_dsts,
):
    recorded: list[tuple] = []
    connector = _ffn_connector_with_peers(monkeypatch, recorded, peer_tokens)
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=peer_tokens,
        ),
    )

    ffn_output = torch.zeros((8, 16), dtype=torch.bfloat16, device="meta")
    connector.send_ffn_output(ffn_output, context)

    assert recorded == [("afd_p2p_send_group", expected_slices, expected_dsts, 6)]


def test_p2p_recv_single_rank_requires_ref_tensor():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    connector.e2a_comm_id = 23
    tensor_metadata = SimpleNamespace(
        device="cuda:0",
        dtype="bf16",
        size=(64, 16),
    )

    with pytest.raises(RuntimeError, match="requires a reference tensor"):
        connector._recv_hidden_states(
            0,
            SimpleNamespace(world_size=1, rank=0),
            connector.e2a_comm_id,
            tensor_metadata,
        )


@pytest.mark.parametrize(
    ("role", "role_rank", "expected_subgroup"),
    [("ffn", 0, 0), ("ffn", 1, 1), ("attention", 0, 0), ("attention", 3, 1)],
)
def test_p2p_subgroup_rendezvous_reuses_the_afd_world_store(
    monkeypatch, role, role_rank, expected_subgroup
):
    """Subgroups share the AFD world's store under a per-subgroup prefix.

    Creating a store of their own would bind ``afd.host``, which only the rank
    that lives on that host can do, so FFN ranks on other nodes could not come
    up. Keys must stay separated per subgroup or the two subgroups overwrite
    each other's ncclUniqueId.
    """
    from torch.distributed import HashStore

    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    root_store = HashStore()
    afd_pg = SimpleNamespace(get_group_store=lambda: root_store)

    monkeypatch.setattr(module, "init_afd_process_group", lambda **kwargs: afd_pg)
    monkeypatch.setattr(module, "_get_default_group", lambda: None)
    monkeypatch.setattr(
        module, "DefaultProcessGroupSwitcher", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(module, "PyNcclCommunicator", lambda **kwargs: object())
    monkeypatch.setattr(module, "_register_comm", lambda communicator: 0)
    monkeypatch.setattr(module, "_register_p2p_custom_ops", lambda: None)

    connector = AFDConnectorFactory.create_connector(
        role_rank,
        0,
        _fake_vllm_config(
            data_parallel_size=4 if role == "attention" else 2,
            data_parallel_rank=role_rank,
        ),
        AFDConfig(
            role=role,
            connector="P2pNcclAFDConnector",
            num_attention_ranks=4,
            num_ffn_ranks=2,
            host="10.0.0.1",
            port=6269,
        ),
    )
    connector.init_afd_connector()

    assert connector.mapping.subgroup_index == expected_subgroup

    # A write through the subgroup store lands under that subgroup's prefix
    # only, so the sibling subgroup never sees the key.
    connector.a2e_group.store.set("probe", b"value")
    assert root_store.get(f"afd_subgroup_{expected_subgroup}/probe") == b"value"
    assert root_store.check([f"afd_subgroup_{1 - expected_subgroup}/probe"]) is False


def test_factory_rejects_unknown_connector():
    with pytest.raises(ValueError, match="unsupported AFD connector type"):
        AFDConnectorFactory.get_connector_class("UnknownConnector")


class _LazilyExportedConnector(AFDConnectorBase):
    """Stands in for a connector served only through module ``__getattr__``."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - never built
        raise NotImplementedError


@pytest.fixture
def isolated_connector_registry(monkeypatch):
    """Keep a temporary registration out of the registry other tests share."""

    monkeypatch.setattr(
        AFDConnectorFactory,
        "_registry",
        dict(AFDConnectorFactory._registry),
    )


def test_factory_resolves_a_connector_served_only_by_package_getattr(
    monkeypatch, isolated_connector_registry
):
    """The factory must read lazily exported names out of ``__getattr__``."""

    module = types.ModuleType("afd_lazily_exported_connector")
    class_name = "_LazilyExportedConnector"

    def _resolve(name: str):
        if name == class_name:
            return _LazilyExportedConnector
        raise AttributeError(name)

    # A module-level __getattr__ (PEP 562) is what the CAMP2P package uses.
    module.__getattr__ = _resolve  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    AFDConnectorFactory.register_connector(
        "LazilyExportedConnector",
        module.__name__,
        class_name,
        replace=True,
    )

    assert class_name not in vars(module), "the class must not be in the module dict"
    resolved = AFDConnectorFactory.get_connector_class("LazilyExportedConnector")
    assert resolved is _LazilyExportedConnector
