from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu import camp2p as camp2p_module  # noqa: E402
from afd_plugin.connectors.npu.camp2p import (  # noqa: E402
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
    dsv4: bool = False,
    max_num_batched_tokens: int = 16,
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
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"] if dsv4 else [],
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=n_shared_experts,
                vocab_size=32,
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


def _dsv4_afd_config(*, role: str):
    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role=role,
        num_attention_ranks=1,
        num_ffn_ranks=1,
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


def _force_cpu_tensor_allocations(monkeypatch):
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs.pop("device")
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(camp2p_module.torch, "tensor", cpu_tensor)


def test_camp2p_recv_attn_output_uses_original_contiguous_af_grouping(monkeypatch):
    torch = pytest.importorskip("torch")
    _force_cpu_tensor_allocations(monkeypatch)
    monkeypatch.setattr(
        torch.ops.afd_ascend,
        "a2e",
        lambda *args: ("hidden", None, None, "atten-batch", "active-mask"),
        raising=False,
    )
    dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}
    rank0 = _init_ffn_connector(0, _vllm_config())
    rank1 = _init_ffn_connector(1, _vllm_config())
    rank0.dp_metadata_list = dp_metadata_list
    rank1.dp_metadata_list = dp_metadata_list

    context0 = rank0.recv_attn_output(ubatch_idx=0, layer_idx=3).context
    context1 = rank1.recv_attn_output(ubatch_idx=0, layer_idx=3).context

    assert context0.metadata.seq_lens == [5]
    assert context1.metadata.seq_lens == [12]
    assert isinstance(context0.states, CAMP2PTransferState)
    assert isinstance(context0.states, AFDTransferState)
    assert context0.states.batch_size == 5
    assert context0.states.h == 16
    assert context0.states.k == 2


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
    _force_cpu_tensor_allocations(monkeypatch)
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
    connector.dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    states = connector.recv_attn_output(ubatch_idx=0, layer_idx=3).context.states

    assert states.k == 2
    assert states.batch_size == 5
    # The ffn_core_num override applies because this is an FFN-role connector.
    assert states.aiv_num == 13


def test_camp2p_init_creates_one_hccl_group_per_ubatch(monkeypatch):
    calls = []

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
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=2),
        _afd_config(role="attention"),
        0,
    )

    connector.init_afd_connector()

    assert [call["group_name"] for call in calls[:2]] == ["afd", "afd1"]
    assert connector.hccl_comm_name_list == ["hccl:afd:2", "hccl:afd1:2"]
    assert connector.hccl_comm_name == "hccl:afd:2"
    assert connector.hccl_comm_name2 == "hccl:afd1:2"
    assert (
        camp2p_module._get_group_ep(
            0,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd:2"
    )
    assert (
        camp2p_module._get_group_ep(
            1,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd1:2"
    )


def test_dsv4_camp2p_init_creates_ids_group_and_buffer_per_stage(monkeypatch):
    calls = []
    real_empty = torch.empty

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

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(
        camp2p_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    monkeypatch.setattr(camp2p_module.torch, "empty", cpu_empty)
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=2, dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )

    connector.init_afd_connector()

    assert [call["group_name"] for call in calls] == [
        "afd",
        "afd_ids",
        "afd1",
        "afd_ids1",
        "p2p",
    ]
    assert [group.group_name for group in connector.ids_pg_list] == [
        "afd_ids",
        "afd_ids1",
    ]
    assert [tuple(buffer.shape) for buffer in connector.input_ids_buffers] == [
        (16,),
        (16,),
    ]
    assert all(buffer.dtype == torch.int32 for buffer in connector.input_ids_buffers)


def test_dsv4_camp2p_sends_ids_before_hidden(monkeypatch):
    events = []
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.ids_pg_list = [object()]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    connector.hccl_comm_name = "hidden"
    connector.hccl_comm_name2 = "hidden"
    forward_context = SimpleNamespace()
    monkeypatch.setattr(camp2p_module, "get_forward_context", lambda: forward_context)

    def send_ids(tensor, *, dst, group):
        events.append(("ids", tensor.clone(), dst, group))

    def send_hidden(*args):
        events.append(("hidden", args[0].clone()))
        return args[0]

    monkeypatch.setattr(camp2p_module.dist, "send", send_ids)
    monkeypatch.setattr(
        camp2p_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: events.append(("yield",)) or tensor,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        send_hidden,
        raising=False,
    )
    hidden_states = torch.ones(3, 16)
    input_ids = torch.tensor([-1, 0, 31], dtype=torch.int64)
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=3,
        )
    )

    connector.send_attn_output(hidden_states, context, input_ids=input_ids)

    assert [event[0] for event in events] == ["ids", "yield", "hidden"]
    assert events[0][1].dtype == torch.int32
    assert events[0][1].tolist() == [-1, 0, 31]
    assert events[0][2:] == (0, connector.ids_pg_list[0])


def test_dsv4_camp2p_skips_duplicate_ids_after_runner_pretransfer(monkeypatch):
    events = []
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.ids_pg_list = [object()]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    connector.hccl_comm_name = "hidden"
    connector.hccl_comm_name2 = "hidden"
    monkeypatch.setattr(
        camp2p_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    monkeypatch.setattr(
        camp2p_module.dist,
        "send",
        lambda *_args, **_kwargs: events.append("ids"),
    )
    monkeypatch.setattr(
        camp2p_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: events.append("yield") or tensor,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda hidden_states, *_args: events.append("hidden") or hidden_states,
        raising=False,
    )
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=2,
        )
    )

    connector.send_attn_output(
        torch.ones(2, 16),
        context,
        input_ids=torch.tensor([1, 2]),
    )

    assert events == ["hidden"]


@pytest.mark.parametrize("input_ids", [[-2], [32]])
def test_dsv4_camp2p_rejects_invalid_input_ids(monkeypatch, input_ids):
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.ids_pg_list = [object()]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    monkeypatch.setattr(
        camp2p_module,
        "get_forward_context",
        lambda: SimpleNamespace(),
    )
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=1,
        )
    )

    with pytest.raises(ValueError, match="-1 padding"):
        connector.send_attn_output(
            torch.ones(1, 16),
            context,
            input_ids=torch.tensor(input_ids),
        )


def test_dsv4_camp2p_skips_value_guard_while_graph_compiling(monkeypatch):
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )
    connector._initialized = True
    connector.ids_pg_list = [object()]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    connector.hccl_comm_name = "hidden"
    connector.hccl_comm_name2 = "hidden"
    monkeypatch.setattr(
        camp2p_module,
        "get_forward_context",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    sent_ids = []
    monkeypatch.setattr(
        camp2p_module.dist,
        "send",
        lambda tensor, **_kwargs: sent_ids.append(tensor.clone()),
    )
    monkeypatch.setattr(
        camp2p_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        lambda hidden_states, *_args: hidden_states,
        raising=False,
    )
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=1,
        )
    )

    connector.send_attn_output(
        torch.ones(1, 16),
        context,
        input_ids=torch.tensor([32]),
    )

    assert sent_ids[0].tolist() == [32]


def test_dsv4_camp2p_receives_ids_before_hidden(monkeypatch):
    events = []
    _force_cpu_tensor_allocations(monkeypatch)
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="ffn"),
        0,
    )
    connector._initialized = True
    connector.ids_pg_list = [object()]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    connector.hccl_comm_name = "hidden"
    connector.hccl_comm_name2 = "hidden"
    connector.hccl_comm_name1 = "moe"
    connector.dp_metadata_list = {0: _FakeDPMetadata([3])}

    def recv_ids(tensor, *, src, group):
        events.append(("ids", src, group))
        tensor.copy_(torch.tensor([-1, 0, 31], dtype=torch.int32))

    def recv_hidden(*args):
        events.append(("hidden",))
        return (torch.ones(3, 16), None, None, torch.tensor(3), torch.ones(3))

    monkeypatch.setattr(camp2p_module.dist, "recv", recv_ids)
    monkeypatch.setattr(torch.ops.afd_ascend, "a2e", recv_hidden, raising=False)

    layer0 = connector.recv_attn_output(ubatch_idx=0, layer_idx=0)
    layer1 = connector.recv_attn_output(ubatch_idx=0, layer_idx=1)

    assert [event[0] for event in events] == [
        "ids",
        "hidden",
        "hidden",
    ]
    assert events[0][1:] == (1, connector.ids_pg_list[0])
    assert layer0.input_ids.tolist() == [-1, 0, 31]
    assert layer1.input_ids is None


def test_dsv4_camp2p_close_releases_ids_groups_and_buffers(monkeypatch):
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(dsv4=True),
        _dsv4_afd_config(role="attention"),
        0,
    )
    hidden_group = object()
    ids_group = object()
    connector._initialized = True
    connector.afd_pg_list = [hidden_group]
    connector.afd_pg = hidden_group
    connector.ids_pg_list = [ids_group]
    connector.input_ids_buffers = [torch.empty(16, dtype=torch.int32)]
    destroyed = []
    monkeypatch.setattr(
        camp2p_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    connector.close()

    assert destroyed == [ids_group, hidden_group]
    assert connector.ids_pg_list == []
    assert connector.input_ids_buffers == []
    assert connector.is_initialized is False


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
