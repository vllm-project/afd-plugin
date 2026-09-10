from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig, afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.distributed import build_rank_mapping  # noqa: E402


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


def _tolist(value):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def test_p2p_connector_is_registered():
    sys.modules.pop("afd_plugin.connectors.gpu.p2p", None)

    cls = AFDConnectorFactory.get_connector_class("P2pNcclAFDConnector")

    assert cls.__name__ == "P2pNcclAFDConnector"


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


def test_p2p_connector_preserves_deepseek_attention_side_gate():
    vllm_config = _fake_vllm_config()
    vllm_config.model_config.hf_config.model_type = "deepseek_v2"

    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        vllm_config,
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
            compute_gate_on_attention=True,
        ),
    )

    assert connector.is_initialized is False


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


@pytest.mark.parametrize(
    ("attention_size", "ffn_size", "role", "role_rank", "subgroup_ranks", "dsts"),
    [
        (2, 2, "attention", 1, (1, 3), (1,)),
        (2, 1, "attention", 0, (0, 1, 2), (0,)),
        (4, 2, "attention", 2, (1, 4, 5), ()),
        (4, 2, "ffn", 1, (1, 4, 5), ()),
    ],
)
def test_p2p_topology_supports_equal_and_integer_multiple_attention_counts(
    attention_size,
    ffn_size,
    role,
    role_rank,
    subgroup_ranks,
    dsts,
):
    mapping = build_rank_mapping(
        AFDConfig(
            role=role,
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
        ),
        role_rank,
    )

    assert mapping.ratio == attention_size // ffn_size
    assert mapping.subgroup_ranks == subgroup_ranks
    assert mapping.dp_metadata_destinations == dsts


def test_p2p_tp2_maps_shared_dp_payload_one_to_one(monkeypatch):
    p2p_module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    parallel_state = importlib.import_module("vllm.distributed.parallel_state")
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([7])},
        is_graph_capturing=False,
        is_warmup=False,
    )

    attention_connectors = []
    ffn_connectors = []
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
    monkeypatch.setattr(
        p2p_module,
        "recv_control_payload",
        lambda **kwargs: received_sources.append(kwargs["src"]) or payload,
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
            hidden_states,
            dst,
            process_group,
            comm_id,
            *,
            connector=connector,
        ):
            assert dst == 1
            ffn_routes.append(
                (connector.world_rank, connector.mapping.subgroup_ranks[dst]),
            )

        monkeypatch.setattr(connector, "_send_hidden_states", record_ffn_send)
        connector.send_ffn_output(ffn_output, ffn_context)

    assert ffn_routes == [(0, 2), (1, 3)]


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
    ],
)
def test_p2p_ffn_metadata_tracks_each_attention_peer_in_xayf(
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

    for src_rank, expected_tokens in enumerate(expected_peer_tokens, start=1):
        assert connector._recv_attn_tensor_metadata_list[
            (0, src_rank)
        ].size == torch.Size([expected_tokens, 16])

    assert connector.tensor_metadata_list[0].size == torch.Size(
        [sum(expected_peer_tokens), 16],
    )


def test_p2p_tensor_metadata_clamps_idle_attention_rank_to_dummy_token():
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

    assert ffn_connector._recv_attn_tensor_metadata_list[(0, 1)].size == torch.Size(
        [1, 16],
    )
    assert ffn_connector._recv_attn_tensor_metadata_list[(0, 2)].size == torch.Size(
        [4, 16],
    )
    assert ffn_connector.tensor_metadata_list[0].size == torch.Size([5, 16])

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
        (
            {
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": 3,
                "num_ffn_ranks": 2,
            },
            "multiple of num_ffn_ranks",
        ),
    ],
)
def test_p2p_topology_validation_errors_are_clear(raw, message):
    with pytest.raises(ValueError, match=message):
        afd_config_from_mapping(raw)


def test_p2p_module_exports_connector_class():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    assert module.P2pNcclAFDConnector.__module__ == "afd_plugin.connectors.gpu.p2p"


def test_p2p_dp_metadata_serialization_uses_json_payload():
    module = importlib.import_module("afd_plugin.connectors.metadata")
    metadata = AFDDPMetadata(num_tokens_across_dp_cpu=[3, 5])

    payload = module.encode_control_payload(
        AFDControlPayload(
            dp_metadata_list={7: metadata},
            is_graph_capturing=True,
            is_warmup=False,
            is_graph_replaying=True,
            is_profile=True,
        ),
    )
    decoded_payload = module.decode_control_payload(payload)
    decoded = decoded_payload.dp_metadata_list

    assert payload.startswith(b"{")
    assert isinstance(decoded[7], AFDDPMetadata)
    assert _tolist(decoded[7].num_tokens_across_dp_cpu) == [3, 5]
    assert int(decoded[7].max_tokens_across_dp_cpu) == 5
    with decoded[7].sp_local_sizes(sequence_parallel_size=1):
        assert decoded[7].get_chunk_sizes_across_dp_rank() == [3, 5]
    assert _tolist(decoded[7].cu_tokens_across_sp(1)) == [3, 8]
    assert decoded_payload.is_graph_capturing is True
    assert decoded_payload.is_warmup is False
    assert decoded_payload.is_graph_replaying is True
    assert decoded_payload.is_profile is True


def test_graph_state_adds_input_ids_buffer_when_hidden_buffer_exists(monkeypatch):
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(enforce_eager=False),
        AFDConfig(
            role="ffn",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
    )
    hidden_key = (0, 1, (2, 16))
    existing_hidden = object()
    connector._recv_attn_buffers[hidden_key] = existing_hidden
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())

    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata([2])},
            is_graph_capturing=True,
            is_warmup=False,
        ),
    )

    assert connector._recv_attn_buffers[hidden_key] is existing_hidden
    assert (0, 1, (2,)) in connector._recv_attn_input_ids_buffers


def test_p2p_custom_ops_register_send_recv_with_fake_impls(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    calls = []

    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object
    # Empty ops namespace: the registration helper skips ops that already
    # exist on torch.ops.vllm, so the fake must report none registered.
    torch_module.ops = SimpleNamespace(vllm=SimpleNamespace())

    vllm_module = types.ModuleType("vllm")
    utils_module = types.ModuleType("vllm.utils")
    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(**kwargs):
        calls.append(kwargs)

    torch_utils_module.direct_register_custom_op = direct_register_custom_op
    utils_module.torch_utils = torch_utils_module
    vllm_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_module)
    monkeypatch.setattr(module, "torch", torch_module)
    monkeypatch.setattr(module, "direct_register_custom_op", direct_register_custom_op)
    monkeypatch.setattr(module, "_AFD_CUSTOM_OPS_REGISTERED", False)

    module._register_p2p_custom_ops()

    assert [call["op_name"] for call in calls] == [
        "afd_p2p_send",
        "afd_p2p_recv",
    ]
    assert calls[0]["mutates_args"] == ["tensor"]
    assert calls[1]["mutates_args"] == ["out"]
    assert callable(calls[0]["fake_impl"])
    assert callable(calls[1]["fake_impl"])


def test_p2p_hidden_state_send_uses_registered_custom_op(monkeypatch):
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
    connector.a2e_comm_id = 17

    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_send=lambda tensor, dst, comm_id: (
                calls.append((tensor, dst, comm_id)) or None
            ),
        ),
    )
    monkeypatch.setattr(module, "torch", torch_module)

    hidden_states = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(4, 16),
        dtype="bf16",
    )
    output = connector._send_hidden_states(
        hidden_states,
        1,
        SimpleNamespace(world_size=2, rank=0),
        connector.a2e_comm_id,
    )

    assert calls == [(hidden_states, 1, 17)]
    assert output is None


def test_p2p_recv_preserves_dynamic_ref_tensor_first_dim(monkeypatch):
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
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_recv=lambda tensor, src, comm_id: (
                calls.append((tensor, src, comm_id)) or None
            ),
        ),
    )
    torch_module.empty = lambda *_args, **_kwargs: pytest.fail(
        "recv should reuse the dynamic ref tensor",
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
