# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async FFN host protocol checks; device graphs require GPU validation."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("afd_plugin.connectors.gpu.async_nvshmem.connector")

from afd_plugin.v1.worker import async_ffn_model_runner as runtime  # noqa: E402
from tests.unit.fakes.async_ffn import GPUAsyncHostCase  # noqa: E402
from tests.unit.fakes.async_gpu import recording_async_connector  # noqa: E402


@pytest.fixture
def host_case(monkeypatch):
    case = GPUAsyncHostCase(runtime, torch)
    monkeypatch.setattr(
        runtime, "_ffn_forward_context", lambda config: nullcontext(case.context)
    )
    return case


@pytest.mark.parametrize("routed_rows, shared_rows", [(2, 1), (0, 1), (2, 0), (0, 0)])
def test_each_arrival_replies_with_local_expert_outputs_and_its_original_context(
    host_case, routed_rows, shared_rows
):
    payload = host_case.add(routed_rows=routed_rows, shared_rows=shared_rows)

    assert (
        host_case.runner.execute_connector_driven_step(cancel_requested=lambda: False)
        is True
    )

    routed, context, shared = host_case.replies[0]
    assert torch.equal(routed, payload.hidden_states * 2)
    assert shared is payload.context.states.expand_x_shared
    assert context is payload.context
    assert (context.metadata.layer_idx, context.metadata.stage_idx) == (3, 1)
    assert host_case.context.dp_metadata is None


def test_idle_poll_keeps_receiver_available_without_sending_a_reply(host_case):
    assert (
        host_case.runner.execute_connector_driven_step(cancel_requested=lambda: False)
        is True
    )
    assert host_case.replies == []
    assert host_case.runner.session.keep_serving(shutdown_requested=False)


def test_local_stop_is_delegated_to_the_remote_producer_session(host_case):
    host_case.runner.session.keep_serving = lambda local_stop: True
    payload = host_case.add()

    assert (
        host_case.runner.execute_connector_driven_step(cancel_requested=lambda: True)
        is True
    )
    assert host_case.replies[0][1] is payload.context


def test_terminal_producer_shutdown_stops_the_receive_loop(host_case):
    host_case.arrivals.append(runtime.ConnectorShutdown("all producers closed"))
    assert (
        host_case.runner.execute_connector_driven_step(cancel_requested=lambda: False)
        is False
    )
    assert host_case.replies == []


def test_missing_transport_state_fails_before_a_reply(host_case):
    payload = host_case.add()
    payload.context.states = None
    with pytest.raises(RuntimeError, match="GpuAsyncTransferState"):
        host_case.runner.execute_connector_driven_step(cancel_requested=lambda: False)
    assert host_case.replies == []


@pytest.mark.parametrize("failure_site", ["compute", "send"])
def test_peer_observes_normal_ffn_step_failure(host_case, monkeypatch, failure_site):
    runner = host_case.runner
    peer = recording_async_connector(
        role="ffn", role_rank=1, store=runner.connector.window.runtime.store
    )
    failure = TimeoutError(f"{failure_site} failed")

    def fail(*args, **kwargs):
        raise failure

    if failure_site == "compute":
        monkeypatch.setattr(runner.model, "compute_ffn_output", fail)
    else:
        monkeypatch.setattr(runner.connector, "send_ffn_output", fail)
    host_case.add()
    with pytest.raises(TimeoutError) as exc:
        runner.execute_connector_driven_step(cancel_requested=lambda: False)
    assert exc.value is failure
    with pytest.raises(RuntimeError, match=f"peer failed: rank 2: {failure}"):
        peer.recv_attn_output(timeout_ms=1)
