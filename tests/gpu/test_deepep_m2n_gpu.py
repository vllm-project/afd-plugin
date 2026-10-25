# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU tests for ``DeepEPM2NAFDConnector``: N->M transport and expert compute.

Each test spawns one process per AFD rank on one node (FFN ranks first). FFN
ranks check the received rows exactly (per-expert counts, then every row and
top-k weight matched bit-exactly) and the MoE FFN output against an fp32
reference and vLLM ``fused_experts``; Attention ranks check the combined
output against a dense MoE reference. The same checks run in eager mode, with
two lanes (DBO order), under ``torch.compile`` and after CUDA-graph replay.

Each case is one step: one control payload, then several MoE layers under the
plan it sets, as the runtime does.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from tests.gpu.deepep_m2n_harness import (  # noqa: E402
    CLOSE_FAILED,
    NO_RESULT,
    Case,
    RankResult,
    available_gpus,
    compile_scenario,
    correctness_scenario,
    dbo_pipeline_scenario,
    graph_scenario,
    hot_cases,
    lane_cases,
    mixed_mode_cases,
    peer_exit_scenario,
    run_ranks,
    standard_cases,
)
from tests.unit.fakes.vllm_config import M2NTopology  # noqa: E402

pytestmark = pytest.mark.gpu

CAPACITY = 256
BF16 = "bf16"
FP8 = "fp8_block128_fp32"
PAYLOADS = [BF16, FP8]
GRAPH_BUCKETS = [1, 8, 64, CAPACITY]
GRAPH_REPLAYS = 3
# Layers per captured graph: several dispatch/combine pairs per graph, as in
# a decode graph that holds a whole step.
GRAPH_LAYERS = 3
# DeepEP timeouts for the peer-exit test, and the wall-clock budget for the
# whole run (spawn, init, the failing layer, teardown).
PEER_EXIT_TIMEOUT_S = 10
PEER_EXIT_BUDGET_S = 300


def _skip_without_gpus(topology: M2NTopology) -> None:
    if available_gpus() < topology.world_size:
        pytest.skip(f"needs {topology.world_size} GPUs")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepEP M2N needs SM90+")


def _run(
    topology: M2NTopology,
    extra_config: dict,
    tmp_path,
    scenario: Callable[..., dict[str, Any]] = correctness_scenario,
    scenario_args: dict[str, Any] | None = None,
) -> None:
    _skip_without_gpus(topology)
    if scenario_args is None:
        scenario_args = {
            "cases": _cases(standard_cases(topology.attn_dp_size, CAPACITY))
        }
    results = run_ranks(
        topology,
        scenario,
        scenario_args,
        {"capacity_tokens_per_rank": CAPACITY, **extra_config},
        tmp_path,
    )
    failures = [
        f"rank {r.world_rank} ({r.role}):\n{r.error}" for r in results if not r.ok
    ]
    assert not failures, "\n".join(failures)
    for result in results:
        print(f"rank {result.world_rank} {result.role}: {result.metrics}")


def _cases(cases: list[Case]) -> list[dict[str, Any]]:
    return [asdict(case) for case in cases]


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    ("ffn_size", "attn_size"),
    [(1, 1), (2, 2), (1, 3)],
    ids=["1F1A", "2F2A", "1F3A"],
)
def test_transport_and_deepgemm_experts(ffn_size, attn_size, payload_format, tmp_path):
    """Standard cases (even, uneven, decode, empty, mixed; with and without
    host sync) through dispatch, DeepGEMM experts and combine.

    Why: the core claim, for several rank ratios and both payload formats.
    """
    _run(M2NTopology(ffn_size, attn_size), {"payload_format": payload_format}, tmp_path)


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "topology",
    [M2NTopology(2, 2, attn_tp_size=2), M2NTopology(2, 4, attn_tp_size=2)],
    ids=["2F-1DPx2TP", "2F-2DPx2TP"],
)
def test_attention_tp_peers_dispatch_disjoint_slices(
    topology, payload_format, tmp_path
):
    """With Attention TP, each TP peer dispatches its padded slice of its DP
    rank's tokens (padding rows with expert id -1): the FFN receives every
    token's routes exactly once and each slice combines to the dense MoE
    reference. Covers single-lane and two-lane steps, including decode
    batches smaller than the TP size and empty DP ranks.

    Why: TP peers hold the same tokens; dispatching all of them from every
    peer would run each token's experts TP times, and a lost or doubled slice
    would silently corrupt the output.
    """
    cases = standard_cases(topology.attn_dp_size, CAPACITY) + lane_cases(
        topology.attn_dp_size, CAPACITY
    )
    _run(
        topology,
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario_args={"cases": _cases(cases)},
    )


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_transport_with_fused_experts_backend(payload_format, tmp_path):
    """Standard cases on 2F2A with vLLM's Triton ``fused_experts`` backend.

    Why: the reference backend must keep working to cross-check DeepGEMM.
    """
    _run(
        M2NTopology(2, 2),
        {"expert_backend": "fused_experts", "payload_format": payload_format},
        tmp_path,
    )


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_transport_with_masked_protocol(payload_format, tmp_path):
    """Standard cases on 2F2A with ``role_aware: false``: afdEP's Masked DeepEP
    protocol instead of its AFD role path.

    Why: the masked protocol is the comparison baseline for the role path and
    must stay correct on its own.
    """
    _run(
        M2NTopology(2, 2),
        {"role_aware": False, "payload_format": payload_format},
        tmp_path,
    )


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_hot_ffn_rank_routing(payload_format, tmp_path):
    """Standard cases with every top-k choice on FFN rank 0's experts: rank 0
    receives every token ``top_k`` times, rank 1 receives nothing.

    Why: real routing is skewed; this is the worst-case receive size the
    buffers and the no-sync path must handle.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": payload_format},
        tmp_path,
        scenario_args={"cases": _cases(hot_cases(2, CAPACITY))},
    )


def test_fp8_without_host_sync_for_small_buckets(tmp_path):
    """FP8 standard cases with ``cpu_sync_min_tokens`` above capacity, so no
    eager step reads receive counts on the host.

    Why: the no-sync path uses worst-case receive shapes and must stay correct.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": FP8, "cpu_sync_min_tokens": CAPACITY + 1},
        tmp_path,
    )


def test_graph_and_eager_plans_mixed_across_ranks(tmp_path):
    """Attention plans a step as graph (no host sync) while FFN plans it eager
    (host sync), and the reverse, for even, decode and mixed token counts.

    Why: an FFN runner without a graph for a step that Attention replays runs
    it eagerly; the two sides then disagree on the host sync.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": FP8},
        tmp_path,
        scenario_args={"cases": _cases(mixed_mode_cases(2))},
    )


@pytest.mark.parametrize(
    ("ffn_size", "attn_size", "payload_format"),
    [(2, 2, BF16), (2, 2, FP8), (1, 3, FP8)],
    ids=["2F2A-bf16", "2F2A-fp8", "1F3A-fp8"],
)
def test_two_lanes_dbo_order(ffn_size, attn_size, payload_format, tmp_path):
    """Two lanes with different token counts per lane (including an empty
    lane); Attention dispatches both lanes, then combines both.

    Why: DBO keeps two micro-batches in flight; they must not corrupt each other.
    """
    topology = M2NTopology(ffn_size, attn_size)
    _run(
        topology,
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario_args={"cases": _cases(lane_cases(attn_size, CAPACITY))},
    )


# GPU sleep long enough (~5 ms on H100) to dwarf a combine.
DBO_SLEEP_CYCLES = 10_000_000


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_dbo_pipeline_eager_lanes_do_not_wait_for_each_other(payload_format, tmp_path):
    """Four MoE layers in vLLM's DBO order with the FFN's deferred sends; every
    E0 on the FFN is stretched by a GPU sleep, and C1(L-1) must still finish
    before the stretched E0(L).

    Why: a lane's combine that waited for the other lane's experts serialized
    Attention and FFN compute, so DBO overlapped nothing.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario=dbo_pipeline_scenario,
        scenario_args={
            "num_layers": 4,
            "tokens": 64,
            "steps": 2,
            "sleep_cycles": DBO_SLEEP_CYCLES,
        },
    )


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_dbo_pipeline_cuda_graph_capture_and_replay(payload_format, tmp_path):
    """The same DBO order captured in one graph per side and replayed with fresh
    inputs; every layer checked after each replay.

    Why: the per-lane waits become graph edges; a lane that never joined the
    capture or a missing edge would corrupt rows only in replays.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario=dbo_pipeline_scenario,
        scenario_args={"num_layers": 4, "tokens": 64, "graph": True, "replays": 3},
    )


@pytest.mark.parametrize("payload_format", PAYLOADS, ids=["bf16", "fp8"])
def test_dbo_graph_capture_reuses_sent_rows(payload_format, tmp_path):
    """DBO steps of 2 and 8 MoE layers captured with an FFN that drops each
    layer's rows once sent (as the routed runner does), every experts launch
    stretched; Attention's outputs are checked after each replay, and the
    memory the FFN's capture reserves must not grow with the layer count.

    Why: rows kept alive across streams during capture were never reused
    within it, so every layer's rows stayed allocated and Qwen3-235B
    graph+DBO (94 MoE layers) ran out of FFN memory at startup. Reusing them
    early must still not let a lane overwrite rows the experts read.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": payload_format, "num_lanes": 2},
        tmp_path,
        scenario=dbo_pipeline_scenario,
        scenario_args={
            "num_layers": 2,
            "tokens": 64,
            "graph": True,
            "replays": 2,
            "growth_layers": (2, 8),
            "stretch_cycles": DBO_SLEEP_CYCLES // 10,
        },
    )


@pytest.mark.parametrize(
    ("num_lanes", "payload_format", "hot"),
    [(1, BF16, False), (1, FP8, False), (2, FP8, False), (1, FP8, True)],
    ids=["1lane-bf16", "1lane-fp8", "2lanes-fp8", "1lane-fp8-hot"],
)
def test_cuda_graph_capture_and_replay(num_lanes, payload_format, hot, tmp_path):
    """Capture one graph per bucket (1, 8, 64, 256), each holding several MoE
    layers, then replay them 3 times interleaved with fresh inputs; check all
    layers after every replay. ``hot`` routes everything to FFN rank 0.

    Why: decode runs from replays of whole-step graphs; stale pointers, values
    baked in at capture, or buffer reuse across layers would show up only here.
    """
    _run(
        M2NTopology(2, 2),
        {"payload_format": payload_format, "num_lanes": num_lanes},
        tmp_path,
        scenario=graph_scenario,
        scenario_args={
            "buckets": GRAPH_BUCKETS,
            "replays": GRAPH_REPLAYS,
            "num_layers": GRAPH_LAYERS,
            "hot_ffn_rank": hot,
        },
    )


def test_torch_compile_attention_side(tmp_path):
    """Attention's send and receive inside ``torch.compile(fullgraph=True)``
    for 64, 1 and 17/22 tokens per rank.

    Why: vLLM compiles the Attention model; the custom ops must not break it.
    """
    topology = M2NTopology(2, 2)
    cases = [
        Case("even64", (64,) * topology.attn_size),
        Case("decode1", (1,) * topology.attn_size),
        Case("mixed", (17, 22)),
    ]
    _run(
        topology,
        {"payload_format": FP8},
        tmp_path,
        scenario=compile_scenario,
        scenario_args={"cases": _cases(cases)},
    )


def test_failed_init_reports_and_closes(tmp_path):
    """FFN rank 0 asks for FP8 while the others use BF16: every rank's init
    must fail on the handshake, and ``close()`` must then succeed.

    Why: a misconfigured deployment must fail cleanly on all ranks, and
    teardown after a partial init must not raise or hang.
    """
    topology = M2NTopology(2, 2)
    _skip_without_gpus(topology)
    results = run_ranks(
        topology,
        correctness_scenario,
        {"cases": []},
        {"capacity_tokens_per_rank": CAPACITY},
        tmp_path,
        rank_extra_config={0: {"payload_format": FP8}},
    )

    for result in results:
        assert not result.ok, f"rank {result.world_rank} initialized"
        assert "disagree on payload_format" in result.error, result.error
        assert CLOSE_FAILED not in result.error, result.error


def test_peer_exit_fails_within_timeout(tmp_path):
    """FFN rank 1 exits abruptly after the step's control payload; the other
    ranks then run the layer with DeepEP timeouts of 10 s. Every survivor must
    stop (error or abort) well within the budget instead of waiting forever.

    Why: a crashed peer must not hang the rest of the deployment.
    """
    topology = M2NTopology(2, 2)
    _skip_without_gpus(topology)
    start = time.monotonic()
    results = run_ranks(
        topology,
        peer_exit_scenario,
        {"tokens": 64, "exit_world_rank": 1},
        {
            "capacity_tokens_per_rank": CAPACITY,
            "control_backend": "gloo",
            "cpu_timeout_s": PEER_EXIT_TIMEOUT_S,
            "gpu_timeout_s": PEER_EXIT_TIMEOUT_S,
        },
        tmp_path,
        timeout_s=PEER_EXIT_BUDGET_S,
    )
    elapsed = time.monotonic() - start

    print(f"peer exit: {elapsed:.0f} s")
    for result in results:
        print(f"rank {result.world_rank}: ok={result.ok} {result.error[-300:]}")
    assert results[1].error.startswith(NO_RESULT), "the exiting rank reported"
    survivors: list[RankResult] = [r for r in results if r.world_rank != 1]
    assert all(not r.ok for r in survivors), "a layer completed without its peer"
    assert all("hung=True" not in r.error for r in survivors), "a survivor hung"
    assert elapsed < PEER_EXIT_BUDGET_S
