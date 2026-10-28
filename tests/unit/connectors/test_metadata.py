# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU token-count contracts for the shared AFD DP metadata payload."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.base import AFDControlPlaneClosedError  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDControlPayload,
    AFDDPMetadata,
    encode_control_payload,
    recv_control_payload,
)


@pytest.fixture
def receive_control_payload(monkeypatch):
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([3])},
        is_graph_capturing=False,
        is_warmup=False,
    )
    encoded = encode_control_payload(payload)

    def receive(tensor, **kwargs):
        if tensor.dtype == torch.long:
            tensor.fill_(len(encoded))
        else:
            tensor.copy_(torch.tensor(list(encoded), dtype=torch.uint8))
        return 0

    monkeypatch.setattr(torch.distributed, "recv", receive)

    def run(check_closed=None):
        return recv_control_payload(
            src=0, group=None, device=torch.device("cpu"), check_closed=check_closed
        )

    return run, payload


@pytest.mark.parametrize(
    "sp_size, context_options, expected",
    [
        pytest.param(2, {}, [2, 2, 3, 3], id="legacy-one-argument"),
        pytest.param(
            2, {"pcp_size": 1, "use_ep": True}, [2, 2, 3, 3], id="gpu-pcp1-ep"
        ),
        pytest.param(
            2, {"pcp_size": 2, "use_ep": False}, [3, 3, 5, 5], id="pcp-without-ep"
        ),
        pytest.param(
            2,
            {"pcp_size": 2, "use_ep": True},
            [2, 2, 2, 2, 3, 3, 3, 3],
            id="pcp-with-ep",
        ),
    ],
)
def test_sp_local_sizes_preserves_native_token_layout(
    sp_size, context_options, expected
):
    metadata = AFDDPMetadata(torch.tensor([3, 5]))

    with metadata.sp_local_sizes(sp_size, **context_options) as local_sizes:
        assert local_sizes == expected
        assert metadata.get_chunk_sizes_across_dp_rank() == expected

    assert metadata.get_chunk_sizes_across_dp_rank() is None
    assert metadata.num_tokens_across_dp_cpu.tolist() == [3, 5]


def test_sp_local_sizes_clears_chunk_sizes_after_forward_failure():
    metadata = AFDDPMetadata(torch.tensor([3, 5]))

    with (
        pytest.raises(RuntimeError, match="forward failed"),
        metadata.sp_local_sizes(2, pcp_size=1, use_ep=True),
    ):
        raise RuntimeError("forward failed")

    assert metadata.get_chunk_sizes_across_dp_rank() is None


def test_control_payload_receive_keeps_default_codec_contract(receive_control_payload):
    receive, expected = receive_control_payload

    actual = receive()

    assert actual.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist() == [3]
    assert actual.is_warmup == expected.is_warmup


def test_control_payload_close_prevents_allocation_from_cancelled_size(monkeypatch):
    def receive_cancelled_size(tensor, **kwargs):
        tensor.fill_(-1)
        return 0

    def check_closed():
        raise AFDControlPlaneClosedError("closed after scalar wait")

    monkeypatch.setattr(torch.distributed, "recv", receive_cancelled_size)

    with pytest.raises(AFDControlPlaneClosedError, match="scalar wait"):
        recv_control_payload(
            src=0, group=None, device=torch.device("cpu"), check_closed=check_closed
        )


def test_control_payload_checks_close_after_payload_copy(
    monkeypatch, receive_control_payload
):
    receive, _ = receive_control_payload
    copied = False
    copy_to_cpu = torch.Tensor.cpu

    def close_after_copy(tensor):
        nonlocal copied
        result = copy_to_cpu(tensor)
        copied = True
        return result

    def check_closed():
        if copied:
            raise AFDControlPlaneClosedError("closed after payload copy")

    monkeypatch.setattr(torch.Tensor, "cpu", close_after_copy)

    with pytest.raises(AFDControlPlaneClosedError, match="payload copy"):
        receive(check_closed)


@pytest.mark.parametrize("locally_closed", [False, True])
def test_control_payload_unregister_race_preserves_original_cause(
    monkeypatch, locally_closed
):
    original = ValueError(
        "Group p2p is not registered, please create group with "
        "torch.distributed.new_group API"
    )

    def receive_unregistered(tensor, **kwargs):
        raise original

    def check_closed():
        if locally_closed:
            raise AFDControlPlaneClosedError("local close")

    monkeypatch.setattr(torch.distributed, "recv", receive_unregistered)
    expected = AFDControlPlaneClosedError if locally_closed else ValueError

    with pytest.raises(expected) as caught:
        recv_control_payload(
            src=0, group=None, device=torch.device("cpu"), check_closed=check_closed
        )

    if locally_closed:
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original


@pytest.mark.parametrize("error", [ValueError("bad transport"), RuntimeError("NCCL")])
def test_control_payload_keeps_unknown_transport_errors_on_close(monkeypatch, error):
    def receive_failed(tensor, **kwargs):
        raise error

    def check_closed():
        raise AFDControlPlaneClosedError("local close")

    monkeypatch.setattr(torch.distributed, "recv", receive_failed)

    with pytest.raises(type(error)) as caught:
        recv_control_payload(
            src=0, group=None, device=torch.device("cpu"), check_closed=check_closed
        )

    assert caught.value is error
