# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector


class Execution:
    def __init__(self, tracer, exec_ref):
        self.tracer = tracer
        self.exec_ref = exec_ref


class Tracer:
    def __init__(self, *, active=True):
        self.active = active
        self.spans = []
        self.deferred = []
        self.released = []
        self.transitions = []

    def execution(self):
        return Execution(self, "pipeline") if self.active else None

    def new_work(self):
        return self.execution()

    def retain_execution(self, execution):
        return self.active

    def release_execution(self, execution):
        self.released.append(execution)

    def transition(self, execution, phase):
        self.transitions.append(phase)

    def record_metadata(self, execution, control):
        self.deferred.append((execution, control))


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.delenv("AFD_CAM_OP_IO_LOG", raising=False)
    operators = SimpleNamespace(
        afd_async_dispatch_send=Mock(),
        afd_async_dispatch_recv=Mock(),
        afd_async_combine_send=Mock(),
        afd_async_combine_recv=Mock(return_value=torch.empty((2, 16))),
    )
    monkeypatch.setattr(torch.ops, "afd_ascend", operators)
    return operators


def connector(monkeypatch, *, role, enabled=True):
    monkeypatch.setenv("AFD_CAM_PROFILE_ENABLE", "1" if enabled else "0")
    config = SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {"attn_ranks_per_dp": 2}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=16, num_experts_per_tok=2, n_routed_experts=8
            ),
        ),
    )
    afd_config = AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )
    result = CAMAsyncAFDConnector(0, 0, config, afd_config, 0)
    result._initialized = True
    result.comm_args = torch.empty((1,))
    result._placeholder = torch.empty((1,))
    return result


def profile_session(tracer):
    return SimpleNamespace(
        tracer=tracer,
        capture=SimpleNamespace(admits=lambda: tracer.active, mode="LIGHTWEIGHT"),
        advance_profiler=Mock(),
        device_waiter=SimpleNamespace(
            read_ffn_control=lambda control: control.tolist(),
            begin_receive=Mock(),
            wait_for_stream=Mock(),
            end_receive=Mock(),
            begin=object(),
            end=object(),
            stream=object(),
            host_begin_ns=1,
            host_end_ns=2,
        ),
    )


def dispatch(connector, *, stage=0, annotation=""):
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=3, stage_idx=stage, seq_len=2
        ),
        profile_annotation=annotation,
    )
    connector.send_attn_output(
        torch.empty((2, 16)),
        context,
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        topk_weights=torch.ones((2, 2)),
    )
    return context


def test_attention_dispatch_uses_its_captured_annotation(monkeypatch, ops):
    attn = connector(monkeypatch, role="attention")
    tracer = Tracer()
    transition = Mock()
    monkeypatch.setattr(tracer, "transition", transition)
    attn.profile_session = profile_session(tracer)
    first = dispatch(attn, stage=0, annotation="0|7|2|9")
    second = dispatch(attn, stage=1, annotation="1|8|3|5")
    assert [call.args for call in transition.call_args_list] == [
        (first.profile, "COMMUNICATION", "0|7|2|9"),
        (second.profile, "COMMUNICATION", "1|8|3|5"),
    ]


def recv_outputs(connector):
    prefix_size = 5 + connector.tp_size * (1 + connector.expert_per_rank)
    batch_info = torch.arange(prefix_size, dtype=torch.int64)
    return (
        torch.empty((16, 16)),
        torch.ones((16,)),
        batch_info,
        torch.ones((connector.expert_per_rank,), dtype=torch.int64),
    )


def test_local_call_identity_skips_idle_and_never_changes_native_comm_id(
    monkeypatch, ops
):
    attn = connector(monkeypatch, role="attention")
    without_session = dispatch(attn)
    first_tracer = Tracer()
    attn.profile_session = profile_session(first_tracer)
    first_capture = dispatch(attn, stage=1)
    first_tracer.active = False
    idle = dispatch(attn)
    second_tracer = Tracer()
    attn.profile_session = profile_session(second_tracer)
    second_capture = dispatch(attn)

    contexts = [without_session, first_capture, idle, second_capture]
    assert all(not hasattr(context, "dispatch_seq") for context in contexts)
    assert [call.args[3] for call in ops.afd_async_dispatch_send.call_args_list] == (
        [attn.comm_id] * 4
    )
    assert without_session.profile is None
    assert idle.profile is None
    assert first_capture.profile.tracer is first_tracer
    assert second_capture.profile.tracer is second_tracer

    attn.recv_ffn_output(torch.empty((2, 16)), ubatch_idx=1, context=first_capture)

    assert first_tracer.transitions == ["COMMUNICATION", "WAIT"]
    assert second_tracer.transitions == ["COMMUNICATION"]
    assert ops.afd_async_combine_recv.call_args.args[4] == 0


def test_recv_reuses_control_copy_and_preserves_full_combine_metadata(monkeypatch, ops):
    ffn = connector(monkeypatch, role="ffn")
    tracer = Tracer()
    ffn.profile_session = profile_session(tracer)
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs
    batch_info = outputs[2]
    original = batch_info.clone()

    received = ffn.recv_attn_output(batch_size=2)
    states = received.context.states
    execution = received.context.profile
    assert execution is not None
    assert not tracer.deferred
    assert states.host_control == [0, 2, 4]
    assert states.token_nums_rankid_layeridx is batch_info

    ffn.send_ffn_output(received.hidden_states, received.context)

    assert ops.afd_async_combine_send.call_args.args[2] is batch_info
    torch.testing.assert_close(batch_info, original)
    assert tracer.released == [execution]
    assert tracer.transitions == ["WAIT", "COMMUNICATION", "WAIT"]


@pytest.mark.parametrize("mode", ["FULL", "LIGHTWEIGHT"])
def test_graph_capture_skips_eager_host_waits_and_metadata_reads(
    monkeypatch, ops, mode
):
    ffn = connector(monkeypatch, role="ffn")
    ffn.graph_startup.mode = "full"
    tracer = Tracer()
    ffn.profile_session = profile_session(tracer)
    ffn.profile_session.capture.mode = mode
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs
    received = ffn.recv_attn_output()
    ffn.send_ffn_output(received.hidden_states, received.context)
    assert received.context.profile is None
    assert received.context.states.host_control is None
    assert received.context.states.token_nums_rankid_layeridx is outputs[2]
    ffn.profile_session.device_waiter.begin_receive.assert_not_called()
    ffn.profile_session.device_waiter.wait_for_stream.assert_not_called()
    ffn.profile_session.advance_profiler.assert_not_called()
    assert not tracer.transitions


@pytest.mark.parametrize("active_at_completion", [False, True])
def test_receive_admits_only_after_device_completion(
    monkeypatch, ops, active_at_completion
):
    ffn = connector(monkeypatch, role="ffn")
    tracer = Tracer(active=not active_at_completion)
    ffn.profile_session = profile_session(tracer)
    actual = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = actual

    def complete(control):
        assert not tracer.deferred and not tracer.released
        tracer.active = active_at_completion
        return control.tolist()

    ffn.profile_session.device_waiter.read_ffn_control = complete
    received = ffn.recv_attn_output(batch_size=2)

    assert received.context.states.token_nums_rankid_layeridx is actual[2]
    assert ops.afd_async_dispatch_recv.call_count == 1
    assert bool(received.context.profile) == active_at_completion
    assert not tracer.deferred
    assert tracer.transitions == ([] if active_at_completion else ["WAIT"])


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "idle"])
def test_no_capture_skips_metadata_views_and_keeps_full_metadata(
    monkeypatch, ops, enabled
):
    ffn = connector(monkeypatch, role="ffn", enabled=enabled)
    tracer = Tracer(active=False)
    if enabled:
        ffn.profile_session = profile_session(tracer)
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs
    batch_info = outputs[2]
    getitem = torch.Tensor.__getitem__

    def checked_getitem(tensor, item):
        assert tensor is not batch_info or item in (0, 1, 2), (
            "inactive profiling must not slice source identities"
        )
        return getitem(tensor, item)

    monkeypatch.setattr(torch.Tensor, "__getitem__", checked_getitem)
    received = ffn.recv_attn_output(batch_size=2)
    ffn.send_ffn_output(received.hidden_states, received.context)

    assert received.context.profile is None
    assert not tracer.deferred
    assert ops.afd_async_dispatch_recv.call_args.args[2] == 0
    assert ops.afd_async_combine_send.call_args.args[2] is batch_info
    assert ops.afd_async_combine_send.call_args.args[3] == 0

    if not enabled:
        attn = connector(monkeypatch, role="attention", enabled=False)
        context = dispatch(attn)
        assert not hasattr(context, "dispatch_seq")
        assert ops.afd_async_dispatch_send.call_args.args[3] == 0


@pytest.mark.parametrize("mode", [None, "LIGHTWEIGHT", "FULL"])
def test_layered_receive_skips_host_reads_but_keeps_completion_fences(
    monkeypatch, ops, mode
):
    ffn = connector(monkeypatch, role="ffn")
    ffn.layered_gmm_enabled = True
    ffn.profile_session = profile_session(Tracer(active=mode == "LIGHTWEIGHT"))
    if mode is None:
        ffn.profile_session.capture = None
    else:
        ffn.profile_session.capture.mode = mode
    waiter = ffn.profile_session.device_waiter
    waiter.read_ffn_control = Mock(side_effect=lambda control: control.tolist())
    outputs = recv_outputs(ffn)
    ops.afd_async_dispatch_recv.return_value = outputs

    def reject_control(*args, **kwargs):
        raise AssertionError("unused device control tensor constructed")

    if mode != "FULL":
        monkeypatch.setattr(torch, "stack", reject_control)
    received = ffn.recv_attn_output()
    assert received.context.states.token_nums_rankid_layeridx is outputs[2]
    waiter.begin_receive.assert_called_once()
    if mode == "FULL":
        assert received.context.states.host_control == [0, 2, 4]
        waiter.read_ffn_control.assert_called_once()
        waiter.wait_for_stream.assert_not_called()
    else:
        assert received.context.states.host_control is None
        waiter.read_ffn_control.assert_not_called()
        waiter.wait_for_stream.assert_called_once()
        ffn.profile_session.advance_profiler.assert_called_once()
