# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.engine import EngineCoreOutputs  # noqa: E402
from vllm.v1.engine.core_client import DPLBAsyncMPClient  # noqa: E402

pytestmark = pytest.mark.vllm_runtime


def _make_native_dplb_client(
    request_counts: tuple[tuple[int, float], ...],
) -> DPLBAsyncMPClient:
    """Build the smallest client state needed by vLLM's native DPLB method.

    vLLM 0.28.0 tracks per-engine counts as ``[waiting, running,
    kv_cache_usage]``.
    """

    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.core_engines = [
        index.to_bytes(2, "little") for index in range(len(request_counts))
    ]
    client.lb_engines = [list(counts) for counts in request_counts]
    client.eng_start_index = 0
    return client


def _request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data_parallel_rank=None,
        pooling_params=None,
    )


def test_native_dplb_uses_reported_request_counts():
    client = _make_native_dplb_client(((2, 0, 0.0), (0, 1, 0.0)))

    chosen_engine = client.get_core_engine_for_request(_request("loaded-route"))

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines == [[2, 0, 0.0], [1, 1, 0.0]]


def test_native_dplb_distributes_tied_burst_with_optimistic_counts():
    client = _make_native_dplb_client(((0, 0, 0.0), (0, 0, 0.0)))

    chosen_engines = [
        client.get_core_engine_for_request(_request(f"burst-{index}"))
        for index in range(4)
    ]

    assert chosen_engines == [
        client.core_engines[0],
        client.core_engines[1],
        client.core_engines[0],
        client.core_engines[1],
    ]
    assert client.lb_engines == [[2, 0, 0.0], [2, 0, 0.0]]


def test_native_dplb_releases_completed_request():
    client = _make_native_dplb_client(((0, 0, 0.0), (0, 0, 0.0)))
    request = _request("finished")
    client.get_core_engine_for_request(request)

    outputs = EngineCoreOutputs(finished_requests={request.request_id})
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, outputs))

    assert request.request_id not in client.reqs_in_flight
