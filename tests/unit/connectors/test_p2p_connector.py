from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig, afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransportSpec,
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.distributed import build_rank_mapping  # noqa: E402


def _fake_vllm_config(
    *,
    data_parallel_size=1,
    data_parallel_rank=0,
    enforce_eager=True,
):
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
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
            transport_spec=AFDA2FTransportSpec(),
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
    assert decoded_payload.transport_spec == AFDA2FTransportSpec()


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
            transport_spec=AFDA2FTransportSpec(),
        ),
    )

    assert connector._recv_attn_buffers[hidden_key] is existing_hidden
    assert (0, 1, (2,), torch.int64) in connector._recv_attn_input_ids_buffers


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
