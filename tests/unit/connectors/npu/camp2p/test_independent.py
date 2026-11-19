# SPDX-License-Identifier: Apache-2.0
"""Independent CAMP metadata scheduling; no device behavior is simulated."""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import CAMP2P_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.connectors.base import AFDControlPlaneClosedError  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDControlPayload,
    AFDDPMetadata,
)
from afd_plugin.connectors.npu.camp2p.connector import CAMP2pAFDConnector  # noqa: E402
from afd_plugin.connectors.npu.camp2p.independent import (  # noqa: E402
    IndependentCAMP2pControlPlane,
    IndependentMetadataMailbox,
)
from tests.unit.fakes.vllm_config import (  # noqa: E402
    camp2p_config,
    npu_parallel_config,
)


def payload(tokens):
    return AFDControlPayload({0: AFDDPMetadata([tokens])}, False, False)


@pytest.fixture
def mailboxes():
    # HashStore exercises real torch Store key and payload semantics on CPU.
    stores = {0: torch.distributed.HashStore(), 1: torch.distributed.HashStore()}
    return (
        IndependentMetadataMailbox(stores),
        IndependentMetadataMailbox({0: stores[0]}),
        IndependentMetadataMailbox({1: stores[1]}),
    )


@pytest.mark.parametrize("peer,tokens", [(0, 63), (1, 1)])
def test_one_attention_progresses_without_other_attention(mailboxes, peer, tokens):
    receiver, prefill, decode = mailboxes
    (prefill, decode)[peer].send(peer, payload(tokens))
    source, received = receiver.recv()
    assert (source, received.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist()) == (
        peer,
        [tokens],
    )


def test_ready_clients_are_served_in_round_robin_without_padding(mailboxes):
    receiver, prefill, decode = mailboxes
    prefill.send(0, payload(63))
    prefill.send(0, payload(95))
    decode.send(1, payload(1))
    results = [receiver.recv() for _ in range(3)]
    assert [
        (peer, p.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist())
        for peer, p in results
    ] == [(0, [63]), (1, [1]), (0, [95])]


def test_close_cancels_an_idle_receive_without_destroying_store(mailboxes):
    receiver, prefill, _ = mailboxes
    entered = threading.Event()

    def receive():
        entered.set()
        return receiver.recv()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receive)
        assert entered.wait(timeout=5)
        receiver.close()
        with pytest.raises(AFDControlPlaneClosedError):
            future.result(timeout=5)
    # Closing the receiver owns cancellation, not deletion of peer resources.
    prefill.send(0, payload(1))


@pytest.mark.parametrize("counts", [[2, 2], [0], [-1]])
def test_mailbox_rejects_non_single_attention_counts(mailboxes, counts):
    _, prefill, _ = mailboxes
    with pytest.raises(ValueError, match="one positive"):
        prefill.send(0, AFDControlPayload({0: AFDDPMetadata(counts)}, False, False))


def independent_configs(role="attention", attention_id=0, **parallel):
    extra = {"independent_attention": True}
    if attention_id is not None:
        extra["attention_id"] = attention_id
    config = camp2p_config(extra_config=extra)
    config.parallel_config = npu_parallel_config(**parallel)
    config.model_config.enforce_eager = True
    config.speculative_config = None
    afd = AFDConfig(
        connector=CAMP2P_CONNECTOR,
        role=role,
        num_attention_ranks=2,
        connector_extra_config=extra,
    )
    return config, afd


@pytest.mark.parametrize(
    "role,peer,world_rank", [("attention", 0, 1), ("attention", 1, 1), ("ffn", None, 0)]
)
def test_independent_services_use_pair_topology(role, peer, world_rank):
    config, afd = independent_configs(role, peer)
    connector = CAMP2pAFDConnector(0, 0, config, afd, role_rank=0)
    assert (connector.world_rank, connector.attn_size, connector.ffn_size) == (
        world_rank,
        1,
        1,
    )
    assert isinstance(connector.control_plane, IndependentCAMP2pControlPlane)


@pytest.mark.parametrize(
    "role,peer", [("attention", None), ("attention", -1), ("attention", 2), ("ffn", 0)]
)
def test_invalid_peer_identity_is_rejected_before_communication(role, peer):
    config, afd = independent_configs(role, peer)
    with pytest.raises(ValueError, match="attention_id"):
        CAMP2pAFDConnector(0, 0, config, afd, role_rank=0)


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_local_parallel_groups_are_rejected_before_communication(field):
    config, afd = independent_configs(**{field: 2})
    with pytest.raises(ValueError, match=field):
        CAMP2pAFDConnector(0, 0, config, afd, role_rank=0)


@pytest.mark.parametrize("mode", ["graph", "dbo", "ubatching", "speculation"])
def test_unsupported_execution_is_rejected_before_communication(mode):
    config, afd = independent_configs()
    if mode == "graph":
        config.model_config.enforce_eager = False
    elif mode == "dbo":
        config.parallel_config.enable_dbo = True
    elif mode == "ubatching":
        config.parallel_config.use_ubatching = True
    else:
        config.speculative_config = object()
    with pytest.raises(ValueError, match="independent Attention"):
        CAMP2pAFDConnector(0, 0, config, afd, role_rank=0)
