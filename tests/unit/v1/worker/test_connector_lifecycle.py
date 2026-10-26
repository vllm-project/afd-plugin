# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Workers create a routed-experts connector before vLLM's distributed init."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import AFDConnectorFactory  # noqa: E402
from afd_plugin.v1.worker.connector_lifecycle import (  # noqa: E402
    check_worker_connector,
    create_worker_connector,
)
from tests.unit.fakes.connector import FakeRoutedConnector  # noqa: E402
from tests.unit.fakes.platform import use_cuda_like_platform  # noqa: E402
from tests.unit.fakes.vllm_config import gpu_worker_vllm_config  # noqa: E402

ROUTED = "FakeRoutedConnector"


@pytest.fixture(autouse=True)
def _routed_connector_and_cuda_like_platform(monkeypatch):
    monkeypatch.setitem(
        AFDConnectorFactory._registry, ROUTED, lambda: FakeRoutedConnector
    )
    use_cuda_like_platform(monkeypatch)


def test_other_connectors_are_left_to_their_runner():
    """A connector that is not an ``AFDRoutedExpertsConnector`` is not created
    before vLLM's distributed init.

    Why: P2P and the NPU connectors keep their runner-owned bring-up.
    """
    connector = create_worker_connector(
        gpu_worker_vllm_config(),
        AFDConfig(connector="P2pNcclAFDConnector", role="ffn"),
        rank=0,
        local_rank=0,
    )

    assert connector is None


def test_routed_connector_is_created_on_vllms_device_and_prepared():
    """The connector gets the worker's ranks, the device vLLM will select and
    has joined the AFD world.

    Why: its NCCL environment must precede vLLM's first communicator, and its
    device must be the one vLLM gives the worker.
    """
    config = gpu_worker_vllm_config(data_parallel_rank_local=1)

    connector = create_worker_connector(
        config, AFDConfig(connector=ROUTED, role="ffn"), rank=3, local_rank=0
    )

    assert isinstance(connector, FakeRoutedConnector)
    assert (connector.rank, connector.local_rank, connector.role_rank) == (3, 1, 0)
    assert connector.events == [("prepare",)]


def test_assigned_device_ids_select_the_connector_device():
    """With ``--device-ids 4,5`` worker 0's connector runs on cuda:4.

    Why: vLLM maps device indices through the assigned ids, so they must be
    registered before the connector's device is resolved.
    """
    config = gpu_worker_vllm_config(assigned_physical_gpu_ids=[4, 5])

    connector = create_worker_connector(
        config, AFDConfig(connector=ROUTED, role="ffn"), rank=0, local_rank=0
    )

    assert connector is not None
    assert connector.local_rank == 4


def test_tp_part_of_the_role_rank_comes_from_the_worker_rank(monkeypatch):
    """With TP > 1 the role rank's TP part comes from the worker rank.

    Why: vLLM's TP group does not exist before its distributed init; reading
    it there would fail. TP is vLLM's innermost rank dimension, so the worker
    rank modulo the TP size is the TP rank.
    """
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", None)
    config = gpu_worker_vllm_config(
        data_parallel_size=2, data_parallel_rank=1, tensor_parallel_size=2
    )

    connector = create_worker_connector(
        config,
        AFDConfig(connector=ROUTED, role="attention", num_attention_ranks=4),
        rank=1,
        local_rank=1,
    )

    assert connector is not None
    # DP rank 1, TP rank 1 of 2.
    assert connector.role_rank == 3


def test_connector_is_closed_when_it_fails_to_join(monkeypatch):
    """A failed ``prepare_before_distributed_init`` closes the connector.

    Why: the worker never receives the connector, so nobody else closes it.
    """
    created: list[FakeRoutedConnector] = []
    original_init = FakeRoutedConnector.__init__

    def recording_init(self, *args):
        original_init(self, *args)
        created.append(self)

    monkeypatch.setattr(FakeRoutedConnector, "__init__", recording_init)
    monkeypatch.setattr(
        FakeRoutedConnector, "prepare_error", RuntimeError("join failed")
    )

    with pytest.raises(RuntimeError, match="join failed"):
        create_worker_connector(
            gpu_worker_vllm_config(),
            AFDConfig(connector=ROUTED, role="ffn"),
            rank=0,
            local_rank=0,
        )

    assert created[0].events == [("prepare",), ("close",)]


def _connector(*, local_rank=0, role_rank=0):
    return SimpleNamespace(
        local_rank=local_rank,
        role_rank=role_rank,
        afd_config=AFDConfig(role="attention", num_attention_ranks=2),
    )


def _tp_rank(monkeypatch, tp_rank):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )


def test_check_accepts_vllms_device_and_role_rank(monkeypatch):
    _tp_rank(monkeypatch, 1)

    check_worker_connector(
        _connector(local_rank=1, role_rank=1),
        gpu_worker_vllm_config(tensor_parallel_size=2),
        torch.device("cuda", 1),
    )


def test_check_rejects_another_device(monkeypatch):
    """A connector on another GPU than the worker's fails with both devices.

    Why: a vLLM upgrade that changes device selection must fail at startup
    instead of moving traffic onto another rank's GPU.
    """
    _tp_rank(monkeypatch, 0)

    with pytest.raises(RuntimeError, match="uses cuda:0 but vLLM placed"):
        check_worker_connector(
            _connector(local_rank=0),
            gpu_worker_vllm_config(tensor_parallel_size=2),
            torch.device("cuda", 4),
        )


def test_check_rejects_a_different_role_rank(monkeypatch):
    """A connector whose early role rank differs from vLLM's fails startup.

    Why: the TP part of the role rank is derived from the worker rank before
    vLLM's TP group exists; a vLLM change to its rank order must fail here
    instead of routing another rank's tokens.
    """
    _tp_rank(monkeypatch, 0)

    with pytest.raises(RuntimeError, match="has role rank 1 but vLLM"):
        check_worker_connector(
            _connector(role_rank=1),
            gpu_worker_vllm_config(tensor_parallel_size=2),
            torch.device("cuda", 0),
        )
