# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for ``DeepEPM2NAFDConnector``: config, layout, plans, handshake."""

from __future__ import annotations

import contextlib
import itertools
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from tests.unit.fakes.afdep import FakeElasticBuffer, install_fake_afdep  # noqa: E402

# The connector module imports afdEP's transport when it loads.
install_fake_afdep()

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.gpu.deepep_m2n import (  # noqa: E402
    connector as connector_module,
)
from afd_plugin.connectors.gpu.deepep_m2n.connector import (  # noqa: E402
    DeepEPM2NConfig,
    bucket_ladder,
    expert_weight_format,
    local_handshake_info,
    local_model_info,
    no_sync_bucket_limit,
    plan_stage,
    precompile_buckets,
    resolve_capacity_tokens_per_rank,
    resolve_num_lanes,
    tokens_per_attention_rank,
    validate_handshake,
    validate_model_infos,
    worst_case_receive_rows,
)
from afd_plugin.connectors.gpu.deepep_m2n.experts import (  # noqa: E402
    expert_row_ranges,
)
from afd_plugin.connectors.gpu.deepep_m2n.fp8_quant import (  # noqa: E402
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from afd_plugin.connectors.gpu.deepep_m2n.layout import DeepEPM2NLayout  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n.network import (  # noqa: E402
    NetworkConfig,
    RankTopology,
)
from afd_plugin.connectors.routed_experts import (  # noqa: E402
    FFNLocalExperts,
    LocalExpertWeights,
)
from tests.unit.fakes.vllm_config import (  # noqa: E402
    M2N_HIDDEN_SIZE,
    M2NTopology,
    m2n_vllm_config,
)

CONNECTOR = "DeepEPM2NAFDConnector"
HOPPER = (9, 0)


def _dp(*tokens: int) -> AFDDPMetadata:
    return AFDDPMetadata(num_tokens_across_dp_cpu=torch.tensor(tokens))


def _connector(
    role: str, role_rank: int, topology: M2NTopology, extra: dict | None = None
):
    vllm_config = m2n_vllm_config(topology, role, role_rank, extra or {}, port=12345)
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])
    return AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


def _ffn_deployment(layout: DeepEPM2NLayout, weight_format: str = "bf16") -> dict:
    """A valid FFN deployment: linear EP over the FFN ranks, idle all2all."""
    return {
        "expert_parallel": True,
        "ep_size": layout.ffn_size,
        "tensor_parallel_size": 1,
        "expert_placement_strategy": "linear",
        "eplb": False,
        "all2all_backend": "allgather_reducescatter",
        "weight_format": weight_format,
    }


def _handshake(
    layout: DeepEPM2NLayout,
    overrides: dict | None = None,
    hosts: str | None = None,
    attn_tp_size: int = 1,
) -> list[dict]:
    """Handshake infos; rank r runs on host hosts[r] (one NVLink domain per host).

    Attention ranks report their TP size, a capacity of 256 and a bucket
    ladder (8, 16, 64); FFN ranks report none, as without an explicit capacity.
    """
    hosts = hosts or "a" * layout.union_size
    infos = []
    for world_rank in range(layout.union_size):
        is_ffn = world_rank < layout.ffn_size
        info = local_handshake_info(
            role="ffn" if is_ffn else "attention",
            role_rank=world_rank if is_ffn else world_rank - layout.ffn_size,
            world_rank=world_rank,
            config=DeepEPM2NConfig(),
            hidden_size=M2N_HIDDEN_SIZE,
            dtype=torch.bfloat16,
            num_experts=layout.num_experts,
            top_k=8,
            num_lanes=1,
            capacity_tokens_per_rank=None if is_ffn else 256,
            bucket_ladder=None if is_ffn else (8, 16, 64),
            attn_tp_size=None if is_ffn else attn_tp_size,
            compute_gate_on_attention=True,
            expert_alignment=128 if is_ffn else None,
            ffn_deployment=_ffn_deployment(layout) if is_ffn else None,
            sm=HOPPER,
            topology=RankTopology(
                world_rank=world_rank,
                host=hosts[world_rank],
                nvlink_domain=f"host:{hosts[world_rank]}",
                local_index=hosts[: world_rank + 1].count(hosts[world_rank]) - 1,
            ),
        )
        infos.append(info)
    for (world_rank, key), value in (overrides or {}).items():
        infos[world_rank][key] = value
    return infos


def test_connector_is_registered():
    """The factory resolves ``DeepEPM2NAFDConnector`` to this module's class.

    Why: deployments select the connector by name in the AFD config.
    """
    cls = AFDConnectorFactory.get_connector_class(CONNECTOR)

    assert cls.__name__ == CONNECTOR
    assert cls.__module__ == "afd_plugin.connectors.gpu.deepep_m2n.connector"


def test_config_defaults_and_round_trip():
    """An empty config gives DeepGEMM experts, a BF16 payload and afdEP's role
    path, and the config survives to_mapping/from_mapping unchanged.

    Why: defaults are what deployments get, and the handshake compares
    serialized configs.
    """
    config = DeepEPM2NConfig.from_mapping(None)
    masked = DeepEPM2NConfig.from_mapping({"role_aware": False})

    assert config.expert_backend == "deepgemm"
    assert config.payload_format == "bf16"
    assert config.role_aware is True
    assert masked.role_aware is False
    assert DeepEPM2NConfig.from_mapping(config.to_mapping()) == config
    assert DeepEPM2NConfig.from_mapping(masked.to_mapping()) == masked


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"bogus": 1}, "unknown DeepEPM2NAFDConnector options"),
        ({"expert_backend": "cutlass"}, "expert_backend"),
        ({"payload_format": "fp4"}, "payload_format"),
        # Removed in Phase 4b (now under ``network``): rejected, not ignored.
        ({"disable_gin": "auto"}, "unknown DeepEPM2NAFDConnector options.*disable_gin"),
        ({"num_lanes": 3}, "num_lanes"),
        ({"capacity_tokens_per_rank": 0}, "capacity_tokens_per_rank"),
        ({"bucket_granularity": 0}, "bucket_granularity"),
        ({"cpu_sync_min_tokens": -1}, "cpu_sync_min_tokens"),
        ({"control_backend": "mpi"}, "control_backend"),
        ({"network": {"preset": "token-ring"}}, "network.preset"),
        ({"num_qps": -1}, "num_qps"),
        ({"expert_num_sms": -2}, "expert_num_sms"),
        ({"expert_num_sms": 131}, "expert_num_sms"),
        ({"cpu_timeout_s": 0}, "cpu_timeout_s"),
        ({"role_aware": "true"}, "role_aware must be a boolean"),
        ({"role_aware": 1}, "role_aware must be a boolean"),
    ],
)
def test_config_rejects_invalid_options(raw, message):
    """Each bad or stale option (unknown or removed key, bad enum, zero or
    negative size, bad network preset, bad timeout, non-boolean protocol
    switch) raises a ValueError naming the field.

    Why: typos and removed keys must fail at startup, not be ignored.
    """
    with pytest.raises(ValueError, match=message):
        DeepEPM2NConfig.from_mapping(raw)


def test_config_accepts_fp8_payload_and_sync_threshold():
    """The FP8 payload format and ``cpu_sync_min_tokens`` parse as given.

    Why: the Phase 2 options must reach the connector intact.
    """
    config = DeepEPM2NConfig.from_mapping(
        {"payload_format": "fp8_block128_fp32", "cpu_sync_min_tokens": 512}
    )

    assert config.payload_format == "fp8_block128_fp32"
    assert config.cpu_sync_min_tokens == 512


def test_expert_num_sms_sizes_the_deepgemm_grid(monkeypatch):
    """``expert_num_sms`` defaults to all SMs and, when set, sizes DeepGEMM's
    persistent grid once, at expert construction; above the device's SM count
    it is rejected.

    Why: under DBO a grid that needs every SM straggles behind the other
    lane's waiting transport kernel.
    """
    from afd_plugin.connectors.gpu.deepep_m2n import experts as experts_module

    calls: list[int] = []
    monkeypatch.setattr(
        experts_module.deepgemm, "set_contiguous_m_alignment", lambda a: None
    )
    monkeypatch.setattr(experts_module.deepgemm, "set_num_sms", calls.append)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )

    assert DeepEPM2NConfig.from_mapping({}).expert_num_sms == 0
    experts_module.DeepEPM2NExperts("deepgemm", 128)
    assert calls == []
    experts_module.DeepEPM2NExperts("deepgemm", 128, 130)
    assert calls == [130]
    with pytest.raises(ValueError, match="exceeds the device"):
        experts_module.DeepEPM2NExperts("deepgemm", 128, 134)


def test_plan_stage_skips_host_sync_below_threshold():
    """With a threshold of 256, an eager 255-token step skips the host sync
    and a 256-token step keeps it.

    Why: pins the boundary of the small-batch no-sync shortcut.
    """

    def plan(tokens: int):
        return plan_stage(
            _dp(tokens),
            attn_size=1,
            attn_tp_size=1,
            capacity_tokens_per_rank=1024,
            bucket_granularity=1,
            graph_mode=False,
            bucket_ladder=(255,),
            cpu_sync_min_tokens=256,
        )

    assert not plan(255).do_cpu_sync
    assert plan(256).do_cpu_sync


def test_fp8_block128_round_trip():
    """Block-128 FP8 quantization gives e4m3 values and one scale per
    128x128 block, and dequantizes back within 5% relative error.

    Why: test weights and the reference math rely on this helper.
    """
    w = torch.randn(2, 256, 384)

    w_q, scale = quantize_fp8_block128(w)

    assert w_q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 2, 3)
    error = (dequantize_fp8_block128(w_q, scale) - w).norm() / w.norm()
    assert error < 5e-2


def test_expert_weights_require_matching_fp8_scales():
    """FP8 weights with scales are FP8, BF16 weights without scales are not,
    and FP8 weights without scales are rejected.

    Why: a half-configured FP8 layer must fail loudly, not compute garbage.
    """
    w13, w2 = torch.zeros(1, 256, 128), torch.zeros(1, 128, 128)
    w13_q, w13_s = quantize_fp8_block128(w13)
    w2_q, w2_s = quantize_fp8_block128(w2)

    assert LocalExpertWeights(w13_q, w2_q, w13_s, w2_s).is_fp8
    assert not LocalExpertWeights(w13, w2).is_fp8
    with pytest.raises(ValueError, match="block scales"):
        LocalExpertWeights(w13_q, w2_q)


def test_layout_is_ffn_first_with_identity_expert_ids():
    """2F3A with 128 experts: 64 experts per FFN rank, 5x64 DeepEP expert
    slots, FFN ranks first in world order, FFN 1 owning experts 64-127.

    Why: real expert ids must equal DeepEP ids in the plugin's rank order.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=3, num_experts=128)

    assert layout.experts_per_rank == 64
    assert layout.deepep_num_experts == 5 * 64
    assert layout.world_rank("ffn", 1) == 1
    assert layout.world_rank("attention", 0) == 2
    assert layout.expert_range(1) == range(64, 128)


def test_layout_rejects_indivisible_experts():
    """128 experts over 3 FFN ranks is rejected.

    Why: uneven expert splits are unsupported and must be refused.
    """
    with pytest.raises(ValueError, match="divisible"):
        DeepEPM2NLayout(ffn_size=3, attn_size=1, num_experts=128)


def test_tokens_per_attention_rank_splits_each_dp_rank_over_its_tp_peers():
    """Without TP each Attention rank dispatches its DP rank's count; with TP
    each TP peer dispatches a ceil(count / TP) slice; a count list that does
    not cover the Attention ranks raises.

    Why: TP peers hold the same tokens; dispatching them all from every peer
    would run each token's experts TP times.
    """
    assert tokens_per_attention_rank(_dp(3, 5), attn_size=2, attn_tp_size=1) == (
        3,
        5,
    )
    assert tokens_per_attention_rank(_dp(3, 5), attn_size=4, attn_tp_size=2) == (
        2,
        2,
        3,
        3,
    )
    assert tokens_per_attention_rank(_dp(1, 0), attn_size=8, attn_tp_size=4) == (
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    )
    with pytest.raises(ValueError, match="token counts"):
        tokens_per_attention_rank(_dp(3, 5, 7), attn_size=2, attn_tp_size=1)
    # Two counts with TP 1 do not cover four ranks: no TP is guessed.
    with pytest.raises(ValueError, match="token counts"):
        tokens_per_attention_rank(_dp(3, 5), attn_size=4, attn_tp_size=1)


def test_plan_stage_sizes_the_bucket_for_the_tp_slice():
    """A 17-token DP rank with Attention TP 2 dispatches 9-token slices.

    Why: the bucket and capacity check apply to what one rank dispatches.
    """
    plan = plan_stage(
        _dp(17),
        attn_size=2,
        attn_tp_size=2,
        capacity_tokens_per_rank=9,
        bucket_granularity=1,
        graph_mode=False,
        bucket_ladder=(9,),
    )

    assert plan.tokens_per_attn_rank == (9, 9)
    assert plan.bucket == 9


def test_plan_stage_rounds_graph_buckets_up_the_ladder_without_host_sync():
    """With ladder (8, 16, 32, 64), graph-mode steps of (17, 4), (16, 16) and
    (0, 0) tokens get buckets 32, 16 and 8 and never a host sync.

    Why: graph steps sit on a rung (vLLM pads them to a capture size) and must
    keep it; anything else rounds up to the next rung, never to 0, and graph
    mode must never block on the host.
    """

    def plan(*tokens: int):
        return plan_stage(
            _dp(*tokens),
            attn_size=2,
            attn_tp_size=1,
            capacity_tokens_per_rank=64,
            bucket_granularity=1,
            graph_mode=True,
            bucket_ladder=(8, 16, 32, 64),
        )

    assert (plan(17, 4).bucket, plan(17, 4).do_cpu_sync) == (32, False)
    assert plan(17, 4).tokens_per_attn_rank == (17, 4)
    assert plan(16, 16).bucket == 16
    assert plan(0, 0).bucket == 8


def test_plan_stage_gives_buckets_above_the_no_sync_limit_the_capacity():
    """With ladder (8, 16) and capacity 64, eager steps of 17, 40 and 64
    tokens all get bucket 64, while a 5-token step rounds up to rung 8.

    Why: DeepEP compiles its kernels per bucket; a bucket per eager token
    count ran nvcc on the request path (seconds per new prefill size).
    """

    def plan(tokens: int):
        return plan_stage(
            _dp(tokens),
            attn_size=1,
            attn_tp_size=1,
            capacity_tokens_per_rank=64,
            bucket_granularity=1,
            graph_mode=False,
            bucket_ladder=(8, 16),
        )

    assert [plan(tokens).bucket for tokens in (17, 40, 64)] == [64] * 3
    assert (plan(5).bucket, plan(5).do_cpu_sync) == (8, True)


@pytest.mark.parametrize("tokens", [8, 40])
def test_plan_stage_bucket_does_not_depend_on_the_rank_s_mode(tokens):
    """A step planned as graph on one rank and eager on another gets the same
    bucket, both on the ladder (8 tokens) and above it (40).

    Why: an FFN eager fallback can plan a step Attention replays in another
    mode, and DeepEP needs the same bucket on every rank; a capacity bucket
    chosen by the sync decision broke test_graph_and_eager_plans_mixed_across_ranks.
    """
    buckets = {
        plan_stage(
            _dp(tokens),
            attn_size=1,
            attn_tp_size=1,
            capacity_tokens_per_rank=64,
            bucket_granularity=1,
            graph_mode=graph_mode,
            bucket_ladder=(8, 16),
            cpu_sync_min_tokens=sync_min,
        ).bucket
        for graph_mode in (False, True)
        for sync_min in (0, 10)
    }

    assert len(buckets) == 1


def test_plan_stage_keeps_a_ladder_bucket_for_eager_steps_without_host_sync():
    """With a threshold of 256 and ladder (32, 128, 255), an eager 20-token
    step skips the sync with rung 32, while a 300-token step syncs with the
    capacity bucket.

    Why: without the sync DeepEP allocates the worst case for the bucket, so
    a capacity bucket there would allocate for the capacity on every step.
    """

    def plan(tokens: int):
        return plan_stage(
            _dp(tokens),
            attn_size=1,
            attn_tp_size=1,
            capacity_tokens_per_rank=1024,
            bucket_granularity=1,
            graph_mode=False,
            bucket_ladder=(32, 128, 255),
            cpu_sync_min_tokens=256,
        )

    assert (plan(20).bucket, plan(20).do_cpu_sync) == (32, False)
    assert (plan(300).bucket, plan(300).do_cpu_sync) == (1024, True)


@pytest.mark.parametrize(
    ("sizes", "tp", "granularity", "sync_min", "capacity", "expected"),
    [
        ((1, 2, 4, 8, 16, 24, 32), 1, 1, 0, 1024, (1, 2, 4, 8, 16, 24, 32)),
        ((1, 2, 4, 8, 16, 24, 32), 2, 1, 0, 1024, (1, 2, 4, 8, 12, 16)),
        ((1, 2, 4, 8, 16, 24, 32), 1, 8, 0, 1024, (8, 16, 24, 32)),
        ((1, 2, 4, 8, 16, 24, 32), 1, 1, 100, 1024, (1, 2, 4, 8, 16, 24, 32, 99)),
        ((1, 2, 4, 8, 16, 24, 32), 1, 1, 0, 20, (1, 2, 4, 8, 16, 20)),
        ((), 1, 1, 0, 1024, ()),
    ],
    ids=["plain", "tp2", "granularity8", "sync-threshold", "capped", "no-graphs"],
)
def test_bucket_ladder(sizes, tp, granularity, sync_min, capacity, expected):
    """The ladder holds each capture size's TP slice, rounded to the
    granularity, plus the no-sync limit as its top rung, capped at capacity;
    without graphs or a sync threshold it is empty.

    Why: graph steps must land on a rung, the top rung sizes the combine zero
    buffer, and an empty ladder sends every eager step to the capacity.
    """
    ladder = bucket_ladder(
        capacity_tokens_per_rank=capacity,
        cudagraph_capture_sizes=sizes,
        attn_tp_size=tp,
        bucket_granularity=granularity,
        cpu_sync_min_tokens=sync_min,
    )

    assert ladder == expected


@pytest.mark.parametrize(
    ("ladder", "sync_min", "expected"),
    [
        ((8, 16, 32), 0, ((8, True), (16, True), (32, True), (64, True))),
        (
            (8, 16, 32),
            20,
            ((32, True), (64, True), (8, False), (16, False), (32, False)),
        ),
        ((), 0, ((64, True),)),
    ],
    ids=["all-synced", "sync-threshold", "no-ladder"],
)
def test_precompile_buckets(ladder, sync_min, expected):
    """Synced eager steps can land on rungs at or above the sync threshold
    and on the capacity (64); steps below the threshold skip the sync on any
    rung up to the first one covering the threshold.

    Why: every pair plan_stage can give an eager step must be compiled at
    startup, or nvcc runs on the request path again.
    """
    assert precompile_buckets(ladder, 64, sync_min) == expected


class _RecordingLane:
    """A DeepEP buffer stand-in that records dispatch and combine calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def dispatch(self, x, x_scale, topk_ids, topk_weights, **kwargs):
        self.calls.append(
            ("dispatch", kwargs["bucket"], kwargs["do_cpu_sync"], x.shape[0])
        )
        return SimpleNamespace(handle=SimpleNamespace(recv_shape=(0, M2N_HIDDEN_SIZE)))

    def combine(self, expert_output, handle):
        self.calls.append(("combine", tuple(expert_output.shape)))


def test_precompile_runs_the_same_empty_steps_on_attention_and_ffn(monkeypatch):
    """Attention and FFN each run one empty dispatch and combine per lane for
    every precompile bucket, in the same order.

    Why: dispatch and combine are collectives; ranks that warm up different
    buckets would hang at startup.
    """
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    topology = M2NTopology(ffn_size=2, attn_size=2)
    calls = {}
    for role in ("attention", "ffn"):
        connector = _connector(role, 0, topology, {"capacity_tokens_per_rank": 64})
        connector.device = torch.device("cpu")
        connector.capacity_tokens_per_rank = 64
        connector.bucket_ladder = (8, 16, 64)
        connector.expert_alignment = DEEPGEMM_ALIGNMENT
        connector.lanes = [_RecordingLane(), _RecordingLane()]
        if role == "attention":
            connector._allocate_combine_zeros()
        connector._precompile_dispatch_kernels()
        calls[role] = [lane.calls for lane in connector.lanes]

    expected = [
        call
        for bucket in (8, 16, 64)
        for call in (("dispatch", bucket, True, 0), ("combine", (0, M2N_HIDDEN_SIZE)))
    ]
    assert calls["attention"] == calls["ffn"] == [expected, expected]


def test_plan_stage_rejects_bucket_above_capacity():
    """65 tokens against a capacity of 64 raises.

    Why: an overflow must be an error, not memory corruption in DeepEP.
    """
    with pytest.raises(ValueError, match="exceeds capacity"):
        plan_stage(
            _dp(65),
            attn_size=1,
            attn_tp_size=1,
            capacity_tokens_per_rank=64,
            bucket_granularity=1,
            graph_mode=False,
            bucket_ladder=(64,),
        )


def test_handshake_accepts_consistent_ranks():
    """Four consistent handshake records pass, giving expert alignment 128
    and no GIN (one host).

    Why: the normal path must pass and yield the right settings.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    result = validate_handshake(_handshake(layout), layout, NetworkConfig())

    assert result.expert_alignment == 128
    assert not result.network_plan.use_gin
    assert (result.attn_tp_size, result.capacity_tokens_per_rank) == (1, 256)


def test_handshake_gives_ffn_the_attention_tp_size_capacity_and_ladder():
    """FFN ranks report no TP size, capacity or bucket ladder; the result
    carries Attention's.

    Why: the FFN runs without TP or Attention's CUDA graphs, so only
    Attention knows its TP size, the capacity of a TP slice and its graph
    buckets; both roles must size buffers and pick buckets alike.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=4, num_experts=128)

    result = validate_handshake(
        _handshake(layout, attn_tp_size=2), layout, NetworkConfig()
    )

    assert (
        result.attn_tp_size,
        result.capacity_tokens_per_rank,
        result.bucket_ladder,
    ) == (2, 256, (8, 16, 64))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({(3, "attn_tp_size"): 2}, "disagree on their TP size"),
        ({(2, "attn_tp_size"): 3, (3, "attn_tp_size"): 3}, "does not divide"),
        ({(3, "capacity_tokens_per_rank"): 128}, "capacity_tokens_per_rank"),
        ({(0, "capacity_tokens_per_rank"): 128}, "capacity_tokens_per_rank"),
        ({(3, "bucket_ladder"): [8, 32]}, "bucket_ladder"),
        ({(0, "cpu_sync_min_tokens"): 128}, "disagree on cpu_sync_min_tokens"),
        ({(0, "bucket_granularity"): 16}, "disagree on bucket_granularity"),
    ],
)
def test_handshake_rejects_attention_tp_and_capacity_mismatches(overrides, message):
    """Attention ranks that disagree on TP, a TP size that does not divide
    the Attention ranks, or a capacity some rank disagrees with (an FFN rank
    reports one only when configured explicitly) fail the handshake.

    Why: ranks that plan different slices or buffer sizes would hang or
    overflow DeepEP buffers.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    with pytest.raises(RuntimeError, match=message):
        validate_handshake(_handshake(layout, overrides), layout, NetworkConfig())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({(3, "hidden_size"): 4096}, "disagree on hidden_size"),
        ({(0, "payload_format"): "fp8"}, "disagree on payload_format"),
        ({(1, "expert_alignment"): 64}, "expert_alignment"),
        ({(0, "role"): "attention"}, "expected 'ffn'"),
        ({(2, "sm"): [8, 0]}, "SM90 or newer"),
    ],
)
def test_handshake_rejects_mismatches(overrides, message):
    """One rank's record differs (hidden size, payload format, expert
    alignment, role order, or a pre-Hopper GPU): the handshake raises.

    Why: disagreeing ranks would corrupt data or hang; fail before buffers.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    with pytest.raises(RuntimeError, match=message):
        validate_handshake(_handshake(layout, overrides), layout, NetworkConfig())


def test_handshake_plans_gin_across_nvlink_domains():
    """Ranks on hosts ``aabb`` give a plan of 2 domains of 2 ranks with GIN.

    Why: the handshake must hand the network plan the right domains.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    result = validate_handshake(
        _handshake(layout, hosts="aabb"), layout, NetworkConfig()
    )

    assert result.network_plan.use_gin
    assert (result.network_plan.num_domains, result.network_plan.ranks_per_domain) == (
        2,
        2,
    )


@pytest.mark.parametrize(
    ("hosts", "message"),
    [("aaab", "different rank counts"), ("abab", "not contiguous")],
)
def test_handshake_rejects_unsupported_domain_layouts(hosts, message):
    """Hosts ``aaab`` (unequal counts) and ``abab`` (not contiguous) raise.

    Why: DeepEP's hybrid mode needs equal, contiguous ranks per domain.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)

    with pytest.raises(RuntimeError, match=message):
        validate_handshake(_handshake(layout, hosts=hosts), layout, NetworkConfig())


def test_handshake_rejects_mismatched_network_config():
    """One rank with a different ``network`` config fails the handshake.

    Why: every rank must derive the same NIC and GIN plan.
    """
    layout = DeepEPM2NLayout(ffn_size=1, attn_size=1, num_experts=128)
    infos = _handshake(layout)
    infos[1]["network"] = "something else"

    with pytest.raises(RuntimeError, match="disagree on network"):
        validate_handshake(infos, layout, NetworkConfig())


def test_gloo_control_backend_stages_payloads_on_cpu():
    """With ``control_backend: gloo`` control payloads are staged on CPU.

    Why: gloo cannot send CUDA tensors.
    """
    connector = _connector(
        "attention",
        0,
        M2NTopology(ffn_size=1, attn_size=1),
        {"control_backend": "gloo"},
    )

    assert connector.control_device == torch.device("cpu")


def test_config_parses_network_section_and_tunables():
    """The ``network`` section (preset plus NIC), ``num_qps`` and timeouts
    parse and survive a round trip.

    Why: the Phase 4b options must reach the connector intact.
    """
    config = DeepEPM2NConfig.from_mapping(
        {
            "network": {"preset": "roce-rail-isolated", "nics": ["mlx5_0"]},
            "num_qps": 33,
            "gpu_timeout_s": 60,
        }
    )

    assert config.network.nic_policy == "shared"
    assert config.network.nics == ("mlx5_0",)
    assert (config.num_qps, config.gpu_timeout_s) == (33, 60)
    assert DeepEPM2NConfig.from_mapping(config.to_mapping()) == config


def test_expert_row_ranges_follow_alignment():
    """Per-expert row ranges from the prefix sum: contiguous with alignment 1,
    each group starting on a multiple of 128 with alignment 128.

    Why: every receive check reads DeepEP's padded per-expert layout.
    """
    psum = torch.tensor([3, 128 + 5, 256])

    assert expert_row_ranges(psum, 1) == [(0, 3), (3, 133), (133, 256)]
    assert expert_row_ranges(psum, 128) == [(0, 3), (128, 133), (256, 256)]


def test_connector_constructs_without_initialization():
    """Constructing Attention and FFN connectors computes world ranks without
    initializing. Attention knows its TP size and capacity; the FFN learns
    both in the handshake unless the capacity is explicit.

    Why: construction must not communicate, and rank math must be right.
    """
    topology = M2NTopology(ffn_size=2, attn_size=3)
    attention = _connector("attention", 1, topology, {"capacity_tokens_per_rank": 64})
    default_attention = _connector("attention", 0, topology)
    ffn = _connector("ffn", 1, topology)
    explicit_ffn = _connector("ffn", 0, topology, {"capacity_tokens_per_rank": 64})

    assert attention.is_initialized is False
    assert attention.world_rank == 3
    assert (attention.attn_tp_size, attention.capacity_tokens_per_rank) == (1, 64)
    # scheduler max_num_batched_tokens
    assert default_attention.capacity_tokens_per_rank == 8192
    assert ffn.world_rank == 1
    assert (ffn.attn_tp_size, ffn.capacity_tokens_per_rank) == (None, None)
    assert explicit_ffn.capacity_tokens_per_rank == 64


def test_control_plane_requires_the_handshake_on_ffn():
    """An FFN connector cannot plan a step before the handshake.

    Why: it does not know the Attention TP size until then.
    """
    connector = _connector("ffn", 0, M2NTopology(ffn_size=1, attn_size=1))
    payload = AFDControlPayload(
        dp_metadata_list={0: _dp(1)},
        is_graph_capturing=False,
        is_warmup=False,
    )

    with pytest.raises(RuntimeError, match="before bootstrap"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_connector_rejects_non_bf16_model():
    """An FP16 model is rejected at construction.

    Why: only BF16 hidden states are supported.
    """
    topology = M2NTopology(ffn_size=1, attn_size=1)
    vllm_config = m2n_vllm_config(topology, "ffn", 0, {}, port=12345)
    vllm_config.model_config.dtype = torch.float16
    afd_config = afd_config_from_mapping(vllm_config.additional_config["afd"])

    with pytest.raises(ValueError, match="bfloat16"):
        AFDConnectorFactory.create_connector(0, 0, vllm_config, afd_config)


def test_control_plane_derives_one_plan_per_stage():
    """An eager control payload with counts (5, 9) leaves stage 0 with a
    host sync and bucket 16, the next rung of the ladder Attention derives
    from its capture sizes (1, 2, 4, 8, 16, ... capped at capacity 64).

    Why: the runtime path from control payload to plan must work.
    """
    connector = _connector(
        "attention",
        0,
        M2NTopology(ffn_size=1, attn_size=2),
        {"capacity_tokens_per_rank": 64},
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: _dp(5, 9)},
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert connector.stage_plans[0].bucket == 16
    assert connector.stage_plans[0].do_cpu_sync


def test_control_plane_rejects_more_stages_than_lanes():
    """Two stages on a one-lane connector raise.

    Why: each in-flight micro-batch needs its own buffer.
    """
    connector = _connector("ffn", 0, M2NTopology(ffn_size=1, attn_size=1))
    payload = AFDControlPayload(
        dp_metadata_list={0: _dp(1), 1: _dp(1)},
        is_graph_capturing=False,
        is_warmup=False,
    )

    with pytest.raises(ValueError, match="num_lanes"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_close_without_init_is_safe():
    """``close()`` on a never-initialized connector succeeds, twice.

    Why: runtimes call close() on every exit path, including before init.
    """
    connector = _connector("ffn", 0, M2NTopology(ffn_size=1, attn_size=1))

    connector.close()
    connector.close()

    assert connector.is_initialized is False


def test_data_path_requires_initialization():
    """``send_attn_output`` before init raises "not initialized".

    Why: misuse must give a clear Python error, not a native crash.
    """
    connector = _connector("attention", 0, M2NTopology(ffn_size=1, attn_size=1))
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0, stage_idx=0, seq_len=1
        )
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        connector.send_attn_output(
            torch.zeros(1, M2N_HIDDEN_SIZE),
            context,
            topk_ids=torch.zeros(1, 8, dtype=torch.int64),
            topk_weights=torch.zeros(1, 8),
        )


# ---------------------------------------------------------------------------
# Bring-up: defaults, FFN deployment in the handshake, post-load checks
# ---------------------------------------------------------------------------

SMALL_HIDDEN = 256
SMALL_INTERMEDIATE = 128
FP8_BLOCK = 128
SMALL_LAYOUT = DeepEPM2NLayout(ffn_size=2, attn_size=1, num_experts=8)
MOE_LAYERS = [1, 3]


def _ffn_vllm_config():
    return m2n_vllm_config(M2NTopology(ffn_size=1, attn_size=1), "ffn", 0, {}, port=1)


@pytest.mark.parametrize(
    ("capture_size", "tp_size", "explicit", "expected"),
    [
        (512, 1, None, 8192),
        (16384, 1, None, 16384),
        (16384, 1, 64, 64),
        (512, 2, None, 4096),
        (512, 3, None, 2731),
        (512, 2, 64, 64),
    ],
)
def test_capacity_covers_prefill_budget_and_largest_graph(
    capture_size, tp_size, explicit, expected
):
    """Default capacity is the larger of max_num_batched_tokens (8192) and
    the largest CUDA-graph capture size, divided (rounded up) by the
    Attention TP size; an explicit value wins.

    Why: a decode graph larger than the prefill budget must still fit, and
    with TP each rank dispatches only its slice.
    """
    vllm_config = _ffn_vllm_config()
    vllm_config.compilation_config.max_cudagraph_capture_size = capture_size
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    config = DeepEPM2NConfig(capacity_tokens_per_rank=explicit)

    assert resolve_capacity_tokens_per_rank(config, vllm_config) == expected


@pytest.mark.parametrize(
    ("use_ubatching", "num_ubatches", "explicit", "expected"),
    [(False, 1, None, 1), (True, 2, None, 2), (True, 2, 1, 1)],
)
def test_num_lanes_follows_vllm_micro_batches(
    use_ubatching, num_ubatches, explicit, expected
):
    """One lane without micro-batching, two under DBO; explicit wins.

    Why: a DBO deployment needs a buffer per in-flight micro-batch.
    """
    vllm_config = _ffn_vllm_config()
    vllm_config.parallel_config.use_ubatching = use_ubatching
    vllm_config.parallel_config.num_ubatches = num_ubatches

    lanes = resolve_num_lanes(DeepEPM2NConfig(num_lanes=explicit), vllm_config)

    assert lanes == expected


def test_num_lanes_rejects_more_micro_batches_than_lanes():
    vllm_config = _ffn_vllm_config()
    vllm_config.parallel_config.use_ubatching = True
    vllm_config.parallel_config.num_ubatches = 3

    with pytest.raises(ValueError, match="at most 2 lanes"):
        resolve_num_lanes(DeepEPM2NConfig(), vllm_config)


def test_expert_weight_format_names_what_vllm_loads():
    """BF16 checkpoints, block-128 FP8 checkpoints, and anything else.

    Why: the handshake compares this with the payload format before loading.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    block_fp8 = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[FP8_BLOCK, FP8_BLOCK]
    )
    per_tensor_fp8 = Fp8Config(is_checkpoint_fp8_serialized=True)

    assert expert_weight_format(None) == "bf16"
    assert expert_weight_format(block_fp8) == "fp8_block128"
    assert expert_weight_format(per_tensor_fp8) == "unsupported (fp8)"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("expert_parallel", False),
        ("ep_size", 4),
        ("tensor_parallel_size", 2),
        ("expert_placement_strategy", "round_robin"),
        ("eplb", True),
        ("all2all_backend", "deepep_low_latency"),
        ("weight_format", "fp8_block128"),
    ],
)
def test_handshake_rejects_ffn_deployments_that_misplace_experts(key, value):
    """An FFN rank whose vLLM deployment would hold other experts, run
    vLLM's own all-to-all, or load weights in another format fails the
    handshake on every rank.

    Why: rows would reach the wrong experts with no error later.
    """
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)
    infos = _handshake(layout)
    infos[1]["ffn_deployment"] = {**infos[1]["ffn_deployment"], key: value}

    with pytest.raises(
        RuntimeError, match=f"FFN world rank 1 is deployed with .*{key}"
    ):
        validate_handshake(infos, layout, NetworkConfig())


def test_handshake_accepts_fp8_block_weights_for_fp8_payload():
    layout = DeepEPM2NLayout(ffn_size=2, attn_size=2, num_experts=128)
    infos = _handshake(
        layout,
        {(rank, "payload_format"): "fp8_block128_fp32" for rank in range(4)},
    )
    for rank in range(layout.ffn_size):
        infos[rank]["ffn_deployment"] = _ffn_deployment(layout, "fp8_block128")

    validate_handshake(infos, layout, NetworkConfig())


def test_handshake_requires_gate_on_attention():
    """Both roles must route on Attention; a mix, or false everywhere, fails.

    Why: DeepEP M2N dispatches by top-k ids computed on Attention.
    """
    layout = DeepEPM2NLayout(ffn_size=1, attn_size=1, num_experts=128)
    mixed = _handshake(layout, {(0, "compute_gate_on_attention"): False})
    neither = _handshake(
        layout,
        {(rank, "compute_gate_on_attention"): False for rank in range(2)},
    )

    with pytest.raises(RuntimeError, match="disagree on compute_gate_on_attention"):
        validate_handshake(mixed, layout, NetworkConfig())
    with pytest.raises(RuntimeError, match="compute_gate_on_attention=true"):
        validate_handshake(neither, layout, NetworkConfig())


def _local_experts(
    expert_ids: range, *, fp8: bool = False, w2_scale_blocks: int | None = None
) -> FFNLocalExperts:
    num_experts = len(expert_ids)
    w13_shape = (num_experts, 2 * SMALL_INTERMEDIATE, SMALL_HIDDEN)
    w2_shape = (num_experts, SMALL_HIDDEN, SMALL_INTERMEDIATE)
    if fp8:
        weights = LocalExpertWeights(
            w13=torch.zeros(w13_shape, dtype=torch.float8_e4m3fn),
            w2=torch.zeros(w2_shape, dtype=torch.float8_e4m3fn),
            w13_scale=torch.ones(
                num_experts,
                2 * SMALL_INTERMEDIATE // FP8_BLOCK,
                SMALL_HIDDEN // FP8_BLOCK,
            ),
            w2_scale=torch.ones(
                num_experts,
                SMALL_HIDDEN // FP8_BLOCK,
                w2_scale_blocks or SMALL_INTERMEDIATE // FP8_BLOCK,
            ),
        )
    else:
        weights = LocalExpertWeights(
            w13=torch.zeros(w13_shape, dtype=torch.bfloat16),
            w2=torch.zeros(w2_shape, dtype=torch.bfloat16),
        )
    return FFNLocalExperts(expert_ids=tuple(expert_ids), weights=weights)


def _model_infos(
    *,
    ffn_ranges: tuple[range, range] = (range(0, 4), range(4, 8)),
    attention_layers: list[int] = MOE_LAYERS,
    fp8: bool = False,
    w2_scale_blocks: int | None = None,
) -> list:
    infos = [
        local_model_info(
            MOE_LAYERS,
            {
                layer: _local_experts(experts, fp8=fp8, w2_scale_blocks=w2_scale_blocks)
                for layer in MOE_LAYERS
            },
        )
        for experts in ffn_ranges
    ]
    infos.append(local_model_info(attention_layers, None))
    return infos


def test_model_check_accepts_matching_layers_and_expert_placement():
    """Two FFN ranks hold experts [0, 4) and [4, 8) of layers 1 and 3, and
    Attention dispatches the same layers: the check passes, BF16 and FP8.

    Why: this is the deployment DeepEP M2N routes for.
    """
    validate_model_infos(
        _model_infos(), SMALL_LAYOUT, hidden_size=SMALL_HIDDEN, payload_format="bf16"
    )
    validate_model_infos(
        _model_infos(fp8=True),
        SMALL_LAYOUT,
        hidden_size=SMALL_HIDDEN,
        payload_format="fp8_block128_fp32",
    )


def test_model_check_skips_when_no_rank_reports():
    """Ranks that report nothing (the connector harness) skip the check."""
    validate_model_infos(
        [None, None, None],
        SMALL_LAYOUT,
        hidden_size=SMALL_HIDDEN,
        payload_format="bf16",
    )


@pytest.mark.parametrize(
    ("infos_kwargs", "payload_format", "message"),
    [
        ({"attention_layers": [1]}, "bf16", "disagree on MoE layers"),
        (
            {"ffn_ranges": (range(0, 4), range(0, 4))},
            "bf16",
            r"routes experts \[4, 8\) to it",
        ),
        ({}, "fp8_block128_fp32", "do not match payload_format"),
        ({"fp8": True, "w2_scale_blocks": 2}, "fp8_block128_fp32", "w2_scale"),
    ],
    ids=["layers", "placement", "format", "scale-shape"],
)
def test_model_check_rejects_what_would_compute_wrong_rows(
    infos_kwargs, payload_format, message
):
    """Ranks running different MoE layers, an FFN rank holding experts DeepEP
    does not route to it, weights in another format, or scales of the wrong
    shape fail on every rank after loading.

    Why: each of these hangs the collectives or silently corrupts outputs.
    """
    with pytest.raises(RuntimeError, match=message):
        validate_model_infos(
            _model_infos(**infos_kwargs),
            SMALL_LAYOUT,
            hidden_size=SMALL_HIDDEN,
            payload_format=payload_format,
        )


def test_model_check_requires_ffn_experts_once_anyone_reports():
    infos = _model_infos()
    infos[1] = None

    with pytest.raises(RuntimeError, match="FFN world rank 1 reported no local"):
        validate_model_infos(
            infos, SMALL_LAYOUT, hidden_size=SMALL_HIDDEN, payload_format="bf16"
        )


def test_attention_init_rejects_local_experts_before_communicating():
    """Passing experts on an Attention rank raises before any rendezvous.

    Why: a wiring bug must not leave the peers waiting in a collective.
    """
    connector = _connector("attention", 0, M2NTopology(ffn_size=1, attn_size=1))

    with pytest.raises(ValueError, match="only FFN ranks hold local experts"):
        connector.init_afd_connector(local_experts={})

    assert connector.bootstrap_group is None


@pytest.mark.parametrize(("role", "role_rank"), [("attention", 2), ("ffn", 1)])
@pytest.mark.parametrize("role_aware", [True, False])
def test_lane_transport_is_afdep_with_the_protocol_and_ffn_rank_count(
    monkeypatch, role, role_rank, role_aware
):
    """Each lane's transport is afdEP's ``ElasticBuffer``, built with the
    selected protocol and the FFN rank count; the DeepEP M2N arguments pass
    through unchanged.

    Why: afdEP's role path needs the FFN-first rank split, and every rank must
    request the same protocol.
    """
    monkeypatch.setattr(connector_module, "ElasticBuffer", FakeElasticBuffer)
    connector = _connector(
        role,
        role_rank,
        M2NTopology(ffn_size=2, attn_size=3),
        {"role_aware": role_aware},
    )

    buffer = connector._new_buffer(capacity_tokens_per_rank=64, top_k=4)

    assert isinstance(buffer, FakeElasticBuffer)
    assert buffer.kwargs == {
        "capacity_tokens_per_rank": 64,
        "top_k": 4,
        "role_aware": role_aware,
        "ffn_ranks": 2,
    }


def test_attention_ops_take_the_stage_from_the_live_dbo_micro_batch(monkeypatch):
    """Under DBO the dispatch/combine op bodies use the calling micro-batch's
    id, not the ``stage_idx`` argument; outside DBO they use the argument.

    Why: torch.compile froze ``stage_idx=0`` into the graph both DBO
    micro-batch threads share, so micro-batch 1 dispatched "stage 0" again
    and raised, and every rank hung at the DeepEP barrier.
    """
    from afd_plugin.connectors.gpu.deepep_m2n import connector as connector_module

    class _RecordingConnector:
        def __init__(self):
            self.calls = []

        def dispatch_attention(self, hidden_states, topk_ids, topk_weights, stage):
            self.calls.append(("dispatch", stage))

        def combine_attention(self, out, stage):
            self.calls.append(("combine", stage))

    recorder = _RecordingConnector()
    connector_id = -1
    monkeypatch.setitem(connector_module._CONNECTORS, connector_id, recorder)
    tensor = torch.zeros(1)
    traced_stage_idx = 0

    monkeypatch.setattr(connector_module, "dbo_enabled", lambda: True)
    monkeypatch.setattr(connector_module, "dbo_current_ubatch_id", lambda: 1)
    connector_module._dispatch_op_impl(
        tensor, tensor, tensor, connector_id, traced_stage_idx
    )
    connector_module._combine_op_impl(tensor, connector_id, traced_stage_idx)

    monkeypatch.setattr(connector_module, "dbo_enabled", lambda: False)
    connector_module._dispatch_op_impl(tensor, tensor, tensor, connector_id, 1)

    assert recorder.calls == [("dispatch", 1), ("combine", 1), ("dispatch", 1)]


# ---------------------------------------------------------------------------
# Attention-side combine zeros: sized for the largest receive
# ---------------------------------------------------------------------------

DEEPGEMM_ALIGNMENT = 128


@pytest.mark.parametrize(
    ("num_ranks", "bucket", "top_k", "experts_per_rank", "alignment", "expected"),
    [
        # Qwen3-30B 2A2F at the default capacity, and at a 512 bucket.
        (4, 8192, 8, 64, DEEPGEMM_ALIGNMENT, 270_336),
        (4, 512, 8, 64, DEEPGEMM_ALIGNMENT, 24_576),
        # Qwen3-235B 4A2F at a 512 bucket.
        (6, 512, 8, 64, DEEPGEMM_ALIGNMENT, 32_768),
        # No alignment (fused_experts backend): exactly ranks x bucket x top-k.
        (4, 10, 8, 64, 1, 320),
        # A token sends at most one route per local expert.
        (4, 10, 8, 4, 1, 160),
    ],
)
def test_worst_case_receive_rows_mirrors_deepep(
    num_ranks, bucket, top_k, experts_per_rank, alignment, expected
):
    """Rows match DeepEP's worst-case receive without the host sync.

    Why: the Attention-side zero buffer is sized with this; too few rows
    would fail combine, and DeepEP's formula is the contract.
    """
    rows = worst_case_receive_rows(
        num_ranks=num_ranks,
        bucket=bucket,
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        expert_alignment=alignment,
    )

    assert rows == expected


@pytest.mark.parametrize(
    ("capture", "tp", "granularity", "sync_min", "capacity", "expected"),
    [
        (512, 1, 1, 0, 8192, 512),  # graph buckets only
        (512, 2, 1, 0, 4096, 256),  # each TP rank dispatches its slice
        (100, 1, 16, 0, 8192, 112),  # graph bucket rounded to granularity
        (512, 1, 1, 1000, 8192, 999),  # eager buckets below the sync threshold
        (512, 1, 1, 0, 256, 256),  # never above capacity
        (0, 1, 1, 0, 8192, 0),  # eager-only and always syncing
    ],
)
def test_no_sync_bucket_limit(capture, tp, granularity, sync_min, capacity, expected):
    """The largest bucket that skips the host sync: the largest graph bucket or
    the largest eager bucket below cpu_sync_min_tokens, capped by capacity.

    Why: only those dispatches give an Attention rank a worst-case receive;
    sizing the zero buffer for capacity instead would waste ~1 GiB per GPU.
    """
    limit = no_sync_bucket_limit(
        capacity_tokens_per_rank=capacity,
        max_cudagraph_capture_size=capture,
        attn_tp_size=tp,
        bucket_granularity=granularity,
        cpu_sync_min_tokens=sync_min,
    )

    assert limit == expected


def test_attention_combine_zeros_cover_the_largest_no_sync_receive():
    """Attention allocates one zero buffer for its largest no-sync receive and
    hands combine a contiguous view of its first rows.

    Why: a cache keyed by receive shape grew without bound when eager buckets
    skipped the sync; a fixed buffer bounds memory and keeps one address for
    captured graphs.
    """
    connector = _connector(
        "attention",
        0,
        M2NTopology(ffn_size=2, attn_size=2),
        {"capacity_tokens_per_rank": 16},
    )
    connector.device = torch.device("cpu")
    connector.expert_alignment = DEEPGEMM_ALIGNMENT

    connector._allocate_combine_zeros()

    # Bucket min(16, 512 capture) = 16: 4 ranks x 16 x top-k 8 + 127 x 64.
    expected_rows = worst_case_receive_rows(
        num_ranks=4,
        bucket=16,
        top_k=8,
        experts_per_rank=64,
        expert_alignment=DEEPGEMM_ALIGNMENT,
    )
    zeros = connector._combine_zeros
    assert zeros is not None
    assert zeros.shape == (expected_rows, M2N_HIDDEN_SIZE)
    view = connector._combine_zero_rows((5, M2N_HIDDEN_SIZE))
    assert view.shape == (5, M2N_HIDDEN_SIZE) and view.is_contiguous()
    assert view.data_ptr() == zeros.data_ptr()
    assert connector._combine_zero_rows((0, M2N_HIDDEN_SIZE)).shape == (
        0,
        M2N_HIDDEN_SIZE,
    )
    with pytest.raises(RuntimeError, match="no_sync_bucket_limit"):
        connector._combine_zero_rows((expected_rows + 1, M2N_HIDDEN_SIZE))


class _LoggedEvent:
    """CPU stand-in for ``torch.cuda.Event``: logs where it was recorded."""

    _ids = itertools.count()

    def __init__(self, log: list) -> None:
        self._log = log
        self.name = f"ev{next(self._ids)}"

    def record(self, stream) -> None:
        self._log.append(("record", self.name, stream.name))


class _LoggedStream:
    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self._log = log

    def wait_event(self, event) -> None:
        self._log.append(("wait_event", self.name, event.name))

    def wait_stream(self, other) -> None:
        self._log.append(("wait_stream", self.name, other.name))


class _FFNLane:
    def dispatch(self, x, x_scale, topk_ids, topk_weights, **kwargs):
        rows = torch.zeros((0, M2N_HIDDEN_SIZE))
        return SimpleNamespace(
            recv_x=rows,
            recv_scale=None,
            recv_topk_weights=torch.zeros(0),
            recv_count_psum=torch.zeros(0, dtype=torch.int32),
            recv_counts_host=[],
            handle=SimpleNamespace(recv_shape=(0, M2N_HIDDEN_SIZE)),
        )

    def combine(self, expert_output, handle):
        return expert_output


def _logged_ffn_connector(monkeypatch, num_lanes: int):
    """FFN connector whose streams and events log every record and wait, and
    every allocation (with the stream it is made under) and keep-alive."""
    from afd_plugin.connectors.gpu.deepep_m2n import connector as m2n

    log: list = []
    main = _LoggedStream("main", log)
    active: list[str] = []

    @contextlib.contextmanager
    def stream_context(stream):
        active.append(stream.name)
        try:
            yield
        finally:
            active.pop()

    real_empty = torch.empty

    def logged_empty(*args, **kwargs):
        log.append(("alloc", active[-1] if active else "main"))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: _LoggedEvent(log))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch, "empty", logged_empty)
    monkeypatch.setattr(
        m2n,
        "_keep_alive_on",
        lambda stream, *tensors: log.append(("keep_alive", stream.name)),
    )

    connector = _connector("ffn", 0, M2NTopology(ffn_size=2, attn_size=2))
    connector.device = torch.device("cpu")
    connector._initialized = True
    connector.lanes = [_FFNLane() for _ in range(num_lanes)]
    connector.lane_streams = (
        [_LoggedStream(f"lane{i}", log) for i in range(num_lanes)]
        if num_lanes > 1
        else []
    )
    connector.stage_plans = {
        lane: SimpleNamespace(bucket=8, do_cpu_sync=False) for lane in range(num_lanes)
    }

    def experts(x, scale, weights, psum, local, out=None):
        log.append(("experts", "main", None))
        return out

    connector.experts = experts
    return connector, log


def _ffn_dbo_step(connector, num_layers: int, num_stages: int) -> None:
    """The routed FFN runner's order: deferred sends, then join lanes."""
    pending: dict[int, tuple] = {}
    for _layer in range(num_layers):
        for stage in range(num_stages):
            if stage in pending:
                connector.send_ffn_output(*pending.pop(stage))
            payload = connector.recv_attn_output(stage)
            out = connector.run_experts(payload, None)
            pending[stage] = (out, payload.context)
    for stage in range(num_stages):
        connector.send_ffn_output(*pending.pop(stage))
    connector.synchronize_lanes()


def test_ffn_lane_combine_waits_only_for_its_own_experts(monkeypatch):
    """Under DBO each lane's combine waits on the event recorded right after
    that lane's experts, and no lane waits on the whole caller stream.

    Why: waiting on the whole stream also waited for the other micro-batch's
    experts, which serialized Attention and FFN compute (no DBO overlap).
    """
    connector, log = _logged_ffn_connector(monkeypatch, num_lanes=2)

    _ffn_dbo_step(connector, num_layers=3, num_stages=2)

    assert not [e for e in log if e[0] == "wait_stream"]
    # The event each experts launch is followed by on the caller stream.
    experts_events = []
    for i, entry in enumerate(log):
        if entry[0] == "experts":
            record = log[i + 1]
            assert record[0] == "record" and record[2] == "main"
            experts_events.append(record[1])
    # Experts run E0(L0) E1(L0) E0(L1) E1(L1) ...; combine C_s(L) waits E_s(L).
    lane_waits = {
        lane: [e[2] for e in log if e[0] == "wait_event" and e[1] == lane]
        for lane in ("lane0", "lane1")
    }
    step_fork = lane_waits["lane0"][0]
    assert lane_waits["lane1"][0] == step_fork
    assert lane_waits["lane0"][1:] == experts_events[0::2]
    assert lane_waits["lane1"][1:] == experts_events[1::2]
    # The step fork is recorded before any experts of the step.
    fork_at = log.index(("record", step_fork, "main"))
    assert fork_at < next(i for i, e in enumerate(log) if e[0] == "experts")


def test_ffn_lanes_join_each_step_once(monkeypatch):
    """Each lane joins the caller stream once per step; synchronize_lanes
    starts a new step (a new fork event).

    Why: the per-step join orders a lane after the previous step and, while
    capturing a CUDA graph, joins the lane to the capture.
    """
    connector, log = _logged_ffn_connector(monkeypatch, num_lanes=2)

    _ffn_dbo_step(connector, num_layers=1, num_stages=2)
    first = len(log)
    _ffn_dbo_step(connector, num_layers=1, num_stages=2)

    def forks(entries):
        return [
            e
            for e in entries
            if e[0] == "wait_event"
            and e[1].startswith("lane")
            and e[2] not in experts_ids
        ]

    experts_ids = {log[i + 1][1] for i, e in enumerate(log) if e[0] == "experts"}
    first_forks, second_forks = forks(log[:first]), forks(log[first:])
    assert [e[1] for e in first_forks] == ["lane0", "lane1"]
    assert [e[1] for e in second_forks] == ["lane0", "lane1"]
    assert first_forks[0][2] != second_forks[0][2]


def test_ffn_rows_and_expert_output_stay_in_the_lane_pool(monkeypatch):
    """Under DBO every FFN allocation of a stage (its dispatch inputs and the
    expert output) is made on that stage's lane, and nothing is kept alive on
    the caller stream.

    Why: rows allocated on a lane and recorded on the caller stream (or an
    output allocated on the caller stream and recorded on the lane) cannot be
    reused within one CUDA-graph capture; every MoE layer's rows stayed
    allocated, and Qwen3-235B graph+DBO ran out of FFN memory at startup.
    """
    connector, log = _logged_ffn_connector(monkeypatch, num_lanes=2)

    _ffn_dbo_step(connector, num_layers=3, num_stages=2)

    assert {e[1] for e in log if e[0] == "keep_alive"} <= {"lane0", "lane1"}
    assert {e[1] for e in log if e[0] == "alloc"} == {"lane0", "lane1"}
    # The expert output is the last allocation before each experts launch.
    output_lanes = []
    for i, entry in enumerate(log):
        if entry[0] == "experts":
            output_lanes.append(next(e for e in reversed(log[:i]) if e[0] == "alloc"))
    assert [e[1] for e in output_lanes] == ["lane0", "lane1"] * 3


def test_single_lane_ffn_has_no_cross_stream_waits(monkeypatch):
    """One lane runs on the caller stream: no events, no waits.

    Why: the single-lane path must keep its zero fork/join cost.
    """
    connector, log = _logged_ffn_connector(monkeypatch, num_lanes=1)

    _ffn_dbo_step(connector, num_layers=2, num_stages=1)

    assert not [e for e in log if e[0] == "wait_stream"]
    assert not [e for e in log if e[0] == "wait_event" and e[1] != "main"]
    # One ready event per receive and one done event per send; no experts event.
    assert len([e for e in log if e[0] == "record"]) == 2 + 2


def test_keep_alive_records_stream_uses_while_capturing(monkeypatch):
    """Cross-stream uses are recorded also during CUDA graph capture.

    Why: skipping them let the allocator reuse a freed graph-pool block on the
    freeing stream while a DeepEP lane still read it on replay (a micro-batch's
    hidden states under its dispatch); Qwen3-235B graph+DBO died with an
    illegal instruction on the first concurrent requests.
    """
    from afd_plugin.connectors.gpu.deepep_m2n import connector as m2n

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    recorded = []

    class _Tensor:
        def record_stream(self, stream):
            recorded.append(stream)

    m2n._keep_alive_on("lane0", _Tensor(), None, _Tensor())

    assert recorded == ["lane0", "lane0"]
