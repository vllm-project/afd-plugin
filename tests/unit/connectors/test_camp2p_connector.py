# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu import camp2p as camp2p_module
from afd_plugin.connectors.npu.camp2p import (
    CAMP2pAFDConnector,
    CAMP2PExtraInfo,
    CAMP2PTransferState,
    build_camp2p_topology,
)


class _FakeDPMetadata:
    def __init__(self, values):
        import torch

        # The connector reads token counts with .flatten().tolist(), so this
        # must be a tensor like the real DP metadata, not a plain list.
        self.num_tokens_across_dp_cpu = torch.tensor(values, dtype=torch.int32)


def _vllm_config(
    *,
    num_ubatches: int = 1,
    n_shared_experts: int = 0,
    extra_config=None,
):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": extra_config or {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=64),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=n_shared_experts,
                vocab_size=256,
            ),
        ),
    )


def _afd_config(*, role: str):
    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )


def test_camp2p_factory_creates_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(extra_config={"core_num": 12}),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, CAMP2pAFDConnector)
    assert not connector.is_initialized
    assert connector.max_num_reqs == 8
    assert connector.extra_info.core_num == 12


def test_camp2p_topology_matches_original_rank_layout():
    attn0 = build_camp2p_topology(_afd_config(role="attention"), 0)
    attn1 = build_camp2p_topology(_afd_config(role="attention"), 1)
    attn2 = build_camp2p_topology(_afd_config(role="attention"), 2)
    ffn1 = build_camp2p_topology(_afd_config(role="ffn"), 1)

    assert (attn0.world_rank, attn0.p2p_rank, attn0.dp_metadata_destinations) == (
        2,
        2,
        (0,),
    )
    assert (attn1.world_rank, attn1.p2p_rank, attn1.dp_metadata_destinations) == (
        3,
        3,
        (1,),
    )
    assert not attn2.participates_in_p2p_group
    assert (ffn1.world_rank, ffn1.p2p_rank) == (1, 1)


def _init_ffn_connector(rank, vllm_config):
    connector = CAMP2pAFDConnector(
        rank,
        rank,
        vllm_config,
        _afd_config(role="ffn"),
        rank,
    )
    connector._initialized = True
    connector.hccl_comm_name = "hccl0"
    connector.hccl_comm_name2 = "hccl1"
    connector.hccl_comm_name3 = ""
    connector.hccl_comm_name1 = "moe"
    return connector


def _publish_dp_metadata(connector, dp_metadata_list):
    """Publish DP metadata the way the control plane does.

    The connector derives its A2E tile from an integer snapshot that the control
    plane takes when it publishes a payload, so knowing the counts means going
    through the same path.
    """

    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )


def test_camp2p_recv_attn_output_uses_the_padded_a2e_tile_layout(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: ("hidden", None, None, "atten-batch", "active-mask"),
        raising=False,
    )
    # A2E pairs FFN rank r with Attention ranks r, r + ffn_size, ... and reads one
    # equal tile per peer, so each FFN rank sizes the transfer from the largest
    # count of its own peer group: F0 owns {A0, A2} and F1 owns {A1, A3}.
    dp_metadata_list = {0: _FakeDPMetadata([2, 2, 5, 7])}
    rank0 = _init_ffn_connector(0, _vllm_config())
    rank1 = _init_ffn_connector(1, _vllm_config())
    _publish_dp_metadata(rank0, dp_metadata_list)
    _publish_dp_metadata(rank1, dp_metadata_list)

    context0 = rank0.recv_attn_output(ubatch_idx=0, layer_idx=3).context
    context1 = rank1.recv_attn_output(ubatch_idx=0, layer_idx=3).context

    assert context0.metadata.seq_lens == [10]
    assert context1.metadata.seq_lens == [14]
    assert isinstance(context0.states, CAMP2PTransferState)
    assert isinstance(context0.states, AFDTransferState)
    assert context0.states.batch_size == 10
    assert context1.states.batch_size == 14
    assert context0.states.h == 16
    assert context0.states.k == 2


def test_camp2p_sizes_a_missing_metadata_step_like_the_attention_rank(monkeypatch):
    """Both roles have to derive the same tile when no counts arrived.

    The FFN rank cannot learn the sender's row count from the transfer, so a step
    whose metadata is unusable has to fall back to the run-level token count both
    roles share. Falling back to a larger tile on the receiving side alone is what
    makes A2E read past the rows the Attention rank wrote.
    """

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: ("hidden", None, None, "atten-batch", "active-mask"),
        raising=False,
    )
    connector = _init_ffn_connector(0, _vllm_config())

    context = connector.recv_attn_output(ubatch_idx=0, layer_idx=0).context

    # Stage 0 has no counts: two tiles of ``max_num_batched_tokens`` rows, which is
    # exactly the tile the Attention ranks pad their payloads up to.
    assert connector.max_num_tokens == 64
    assert context.states.batch_size == 128


def test_camp2p_attention_tile_equals_the_ffn_ranks_per_peer_rows(monkeypatch):
    """Both roles have to derive the same tile from the same metadata.

    A2E reads one equal tile per Attention peer and divides the FFN rank's total
    by its peer ratio, so the rows the Attention rank writes and the rows the FFN
    rank reads per peer have to be one number. The failure this pins is
    asymmetric sizing: a rank that sends fewer rows than its peer makes the
    receive read that peer's scales and then its activations as token ids.
    """

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: ("hidden", None, None, "atten-batch", "active-mask"),
        raising=False,
    )
    vllm_config = _vllm_config()
    attention = CAMP2pAFDConnector(
        2,
        0,
        vllm_config,
        _afd_config(role="attention"),
        0,
    )
    attention._initialized = True
    ffn = _init_ffn_connector(0, vllm_config)
    dp_metadata_list = {0: _FakeDPMetadata([24, 24, 24, 24])}
    _publish_dp_metadata(attention, dp_metadata_list)
    _publish_dp_metadata(ffn, dp_metadata_list)

    tile = attention._padding_rows_for_step(0)
    context = ffn.recv_attn_output(ubatch_idx=0, layer_idx=0).context

    # F0 owns two tiles of A0's and A2's 24 rows each.
    assert tile == 24
    assert context.states.batch_size == 48
    assert context.states.batch_size // 2 == tile


def test_camp2p_recv_attn_output_drives_the_operator_ids_mode(monkeypatch):
    """Requesting ids must reach the operator, not stay a connector-local flag.

    The ``a2e`` operator only writes its ids slot in the ids mode, and the
    sending rank selects that mode independently. A receiving rank that reads the
    slot in the other mode would install uninitialised device memory as token
    ids, which a token-keyed router turns into an out-of-range table read. The
    ids that arrive are model-specific tensors, so they travel on the payload
    rather than in the backend transfer state.
    """

    torch = pytest.importorskip("torch")
    calls: list[tuple] = []

    def fake_a2e(*args):
        calls.append(args)
        tokens, topk = args[3], args[5]
        ids = torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk)
        return ("hidden", ids, None, "atten-batch", "active-mask")

    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", fake_a2e, raising=False)
    connector = _init_ffn_connector(0, _vllm_config())
    _publish_dp_metadata(connector, {0: _FakeDPMetadata([2, 2, 5, 5])})

    with_ids = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_input_ids=True,
    )
    without_ids = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        recv_input_ids=False,
    )

    assert calls[0][-1] == 1
    # F0 owns A0 and A2, which hold 2 and 5 tokens, so the padded tile is 5.
    assert with_ids.input_ids.tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert calls[1][-1] == 0
    assert without_ids.input_ids is None


def test_camp2p_extra_info_rejects_unknown_mix_placement():
    with pytest.raises(ValueError, match="unknown CAMP2P connector_extra_config"):
        CAMP2PExtraInfo.from_mapping({"mix_placement": True})


def test_camp2p_extra_info_validates_values():
    with pytest.raises(ValueError, match="core_num must be positive"):
        CAMP2PExtraInfo.from_mapping({"core_num": 0})
    with pytest.raises(TypeError, match="core_num must be an integer"):
        CAMP2PExtraInfo.from_mapping({"core_num": 8.5})


def test_camp2p_extra_info_coerces_integer_bool_values():
    assert (
        CAMP2PExtraInfo.from_mapping(
            {"compute_gate_on_attention": 1},
        ).compute_gate_on_attention
        is True
    )
    assert (
        CAMP2PExtraInfo.from_mapping(
            {"compute_gate_on_attention": 0},
        ).compute_gate_on_attention
        is False
    )


def test_camp2p_connector_uses_role_specific_core_num(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: ("hidden", None, None, "atten-batch", "active-mask"),
        raising=False,
    )
    connector = _init_ffn_connector(
        0,
        _vllm_config(
            n_shared_experts=3,
            extra_config={
                "core_num": 8,
                "ffn_core_num": 13,
            },
        ),
    )
    _publish_dp_metadata(connector, {0: _FakeDPMetadata([2, 2, 5, 5])})

    states = connector.recv_attn_output(ubatch_idx=0, layer_idx=3).context.states

    assert states.k == 2
    # F0 owns A0 and A2, whose largest count is 5, and A2E reads two such tiles.
    assert states.batch_size == 10
    # The ffn_core_num override applies because this is an FFN-role connector.
    assert states.aiv_num == 13


def test_camp2p_init_scopes_fresh_options_to_each_hccl_group(monkeypatch):
    calls = []
    option_calls = []

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(camp2p_module, "ensure_cam_p2p_ops_available", lambda: None)
    monkeypatch.setattr(camp2p_module, "_register_camp2p_custom_ops", lambda: None)

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
        camp2p_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )

    def fake_create_hccl_process_group_options(hccl_buffer_size_mb):
        option_calls.append(hccl_buffer_size_mb)
        return object()

    monkeypatch.setattr(
        camp2p_module,
        "create_hccl_process_group_options",
        fake_create_hccl_process_group_options,
    )
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(
            num_ubatches=2,
            extra_config={"hccl_buffer_size": 2048},
        ),
        _afd_config(role="ffn"),
        0,
    )

    connector.init_afd_connector()

    assert [call["group_name"] for call in calls] == ["afd", "afd1", "afd_moe", "p2p"]
    assert option_calls == [2048, 2048, 2048]
    hccl_options = [call["pg_options"] for call in calls[:3]]
    assert len({id(options) for options in hccl_options}) == 3
    assert "pg_options" not in calls[3]
    assert connector.hccl_comm_name_list == ["hccl:afd:0", "hccl:afd1:0"]
    assert connector.hccl_comm_name == "hccl:afd:0"
    assert connector.hccl_comm_name2 == "hccl:afd1:0"
    assert (
        camp2p_module._get_group_ep(
            0,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd:0"
    )
    assert (
        camp2p_module._get_group_ep(
            1,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd1:0"
    )


def test_camp2p_send_attn_custom_op_receives_all_hccl_names(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=2),
        _afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.hccl_comm_name = "hccl0"
    connector.hccl_comm_name2 = "hccl1"
    connector.hccl_comm_name3 = ""
    hidden_states = torch.empty((3, 16))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=1,
        seq_len=3,
    )
    context = AFDTransferContext(metadata=metadata)

    # The connector stows the CAMP2P transfer state and ubatch index on the
    # forward context; capture that instead of a dedicated helper.
    forward_context = SimpleNamespace()
    monkeypatch.setattr(camp2p_module, "get_forward_context", lambda: forward_context)

    def fake_send_attn_output(*args):
        captured["args"] = args
        return args[0]

    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        fake_send_attn_output,
        raising=False,
    )

    output = connector.send_attn_output(hidden_states, context)

    assert output is None
    assert forward_context.ubatch_idx == 1
    assert captured["args"][1:4] == ("hccl0", "hccl1", "")
    assert captured["args"][4] == 3
    assert forward_context.cam_afdtransfer_state.batch_size == 3


def test_camp2p_init_fails_cleanly_without_ascend_runtime(monkeypatch):
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )

    def _raise_missing_ops():
        raise RuntimeError(
            "CAMP2P Ascend custom ops are not available. Build the package with "
            "Ascend ops enabled in a torch-npu/CANN environment.",
        )

    # Force the "ascend runtime missing" path so the test is deterministic on
    # real NPU hosts too: otherwise init proceeds into init_afd_process_group
    # and blocks forever on the HCCL rendezvous waiting for absent peers.
    monkeypatch.setattr(
        camp2p_module,
        "ensure_cam_p2p_ops_available",
        _raise_missing_ops,
    )

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops|torch-npu"):
        connector.init_afd_connector()
