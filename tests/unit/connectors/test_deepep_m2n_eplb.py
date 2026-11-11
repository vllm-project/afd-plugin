# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for the DeepEP M2N connector's EPLB coordination.

The distributed boundary exchange is exercised with a recording stand-in for
``torch.distributed``: what is asserted is the protocol (which tensors are
reduced, which are adopted), not the transport.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n import eplb as eplb_module  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n.eplb import (  # noqa: E402
    DeepEPM2NAttentionEplb,
    DeepEPM2NFFNEplb,
    real_rows_in_tp_slice,
)
from tests.unit.fakes.connector import FakeRoutedConnector  # noqa: E402

MOE_LAYERS = (1, 3)
NUM_LOGICAL = 12
NUM_PHYSICAL = 12
WINDOW_SIZE = 4
STEP_INTERVAL = 100
INITIAL_REARRANGE_STEP = 8


def _schedule(**overrides) -> dict[str, int]:
    schedule = {
        "num_moe_layers": len(MOE_LAYERS),
        "num_logical_experts": NUM_LOGICAL,
        "num_physical_experts": NUM_PHYSICAL,
        "map_slots": 4,
        "window_size": WINDOW_SIZE,
        "step_interval": STEP_INTERVAL,
        "initial_rearrangement_step": INITIAL_REARRANGE_STEP,
        "log_interval": 0,
    }
    schedule.update(overrides)
    return schedule


def _attention_connector(**overrides):
    attrs = {
        "device": torch.device("cpu"),
        "attn_tp_size": 1,
        "attention_tp_rank": 0,
        "world_rank": 2,
        "bootstrap_group": object(),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _recorder(*, log_interval: int = 0, **connector_overrides):
    initial_map = torch.zeros((len(MOE_LAYERS), NUM_LOGICAL, 4), dtype=torch.int64)
    initial_count = torch.ones((len(MOE_LAYERS), NUM_LOGICAL), dtype=torch.int64)
    return DeepEPM2NAttentionEplb(
        _attention_connector(**connector_overrides),
        _schedule(log_interval=log_interval),
        MOE_LAYERS,
        initial_map,
        initial_count,
    )


def _payload(*, is_warmup=False, is_graph_capturing=False, num_real_tokens=5):
    return AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(
                num_tokens_across_dp_cpu=torch.tensor([num_real_tokens]),
            ),
        },
        is_graph_capturing=is_graph_capturing,
        is_warmup=is_warmup,
        num_real_tokens=num_real_tokens,
    )


class _FakeDist:
    """Records the EPLB exchange collectives; the map broadcast fills the
    receiving tensor so adoption is observable."""

    def __init__(self, *, map_value: int = 7) -> None:
        self.reduces: list[tuple[Any, int]] = []
        self.broadcasts: list[Any] = []
        self._map_value = map_value

    def reduce(self, tensor, dst, group):
        self.reduces.append((tensor, dst))

    def broadcast(self, tensor, src, group):
        tensor.fill_(self._map_value)
        self.broadcasts.append(tensor)


def test_real_rows_in_tp_slice_splits_the_dp_rank_tokens():
    """Each TP rank's slice holds its share of the DP rank's real tokens;
    padding rows (beyond the real count) are excluded."""
    assert real_rows_in_tp_slice(10, tp_size=1, tp_rank=0) == 10
    assert real_rows_in_tp_slice(10, tp_size=2, tp_rank=0) == 5
    assert real_rows_in_tp_slice(10, tp_size=2, tp_rank=1) == 5
    assert real_rows_in_tp_slice(9, tp_size=2, tp_rank=0) == 5
    assert real_rows_in_tp_slice(9, tp_size=2, tp_rank=1) == 4
    assert real_rows_in_tp_slice(1, tp_size=4, tp_rank=2) == 0
    assert real_rows_in_tp_slice(0, tp_size=2, tp_rank=0) == 0


def test_recorder_rotates_the_window_and_drops_dummy_load():
    """Real steps move the recorded load into the window and zero the pass;
    dummy steps drop their load; the upcoming transaction's flags gate
    recording and set its real token count."""
    recorder = _recorder()
    pass_sum = lambda: int(recorder.expert_load_pass.sum())  # noqa: E731
    recorder.expert_load_pass.fill_(3)

    recorder.on_control_step(_payload())
    # The first transaction has no predecessor: only its own state is set.
    assert recorder.record_enabled.item() is True
    assert recorder.num_unpadded_tokens[0].item() == 5
    assert pass_sum() == 3 * NUM_PHYSICAL * len(MOE_LAYERS)

    recorder.on_control_step(_payload(num_real_tokens=7))
    # The real transaction rotated its load into the window and reset the pass.
    assert int(recorder.expert_load_window[0].sum()) == 3 * NUM_PHYSICAL * len(
        MOE_LAYERS
    )
    assert pass_sum() == 0
    assert recorder.num_unpadded_tokens[0].item() == 7

    # A warmup (dummy) transaction's load is dropped, not recorded.
    recorder.on_control_step(_payload(is_warmup=True, num_real_tokens=2))
    assert int(recorder.expert_load_window[1].sum()) == 0
    recorder.expert_load_pass.fill_(9)
    recorder.on_control_step(_payload())
    assert pass_sum() == 0
    assert int(recorder.expert_load_window[2].sum()) == 0
    # Dummy transactions do not record.
    recorder.on_control_step(_payload(is_graph_capturing=True))
    assert recorder.record_enabled.item() is False


def test_recorder_steps_tp_rank_real_counts():
    """With Attention TP the recorder counts only this rank's real rows."""
    recorder = _recorder(attn_tp_size=2, attention_tp_rank=1)

    recorder.on_control_step(_payload(num_real_tokens=9))

    assert recorder.num_unpadded_tokens[0].item() == 4


def test_recorder_exchanges_window_and_adopts_maps_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    """At a rearrangement boundary the recorder contributes its window to the
    reduce aimed at FFN world rank 0 and adopts the broadcast maps."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    initial_map = torch.zeros((len(MOE_LAYERS), NUM_LOGICAL, 4), dtype=torch.int64)
    initial_count = torch.ones((len(MOE_LAYERS), NUM_LOGICAL), dtype=torch.int64)
    recorder = DeepEPM2NAttentionEplb(
        _attention_connector(),
        _schedule(step_interval=10),
        MOE_LAYERS,
        initial_map,
        initial_count,
    )
    recorder.expert_load_pass.fill_(2)

    # initial_rearrangement_step=8, interval=10: the third step reaches it.
    recorder.on_control_step(_payload())
    recorder.on_control_step(_payload())
    recorder.on_control_step(_payload())

    assert len(fake_dist.reduces) == 1
    window, dst = fake_dist.reduces[0]
    assert dst == 0
    # The window carries the load recorded before the boundary: the first
    # real transaction's pass (filled with 2), the second recorded nothing.
    assert int(window.sum()) == 2 * NUM_PHYSICAL * len(MOE_LAYERS)
    # Both maps were adopted from the broadcast (filled with 7).
    assert int(recorder.logical_to_physical_map.sum()) == 7 * len(MOE_LAYERS) * (
        NUM_LOGICAL * 4
    )
    assert int(recorder.logical_replica_count.sum()) == 7 * NUM_LOGICAL * len(
        MOE_LAYERS
    )
    # Slot identities changed with the map, so the next window starts empty.
    assert int(recorder.expert_load_window.sum()) == 0
    assert int(recorder.expert_load_window[0].sum()) == 0


def test_recorder_reduces_the_logical_window_at_each_log_interval(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every log interval the recorder joins a reduce carrying its load window
    aggregated into logical experts -- the only load EPLB's policy sees -- and
    it changes no EPLB state."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    recorder = _recorder(log_interval=2)
    per_logical_expert = NUM_LOGICAL * len(MOE_LAYERS)

    recorder.on_control_step(_payload())
    for step_load in (1, 2, 3, 4):
        recorder.expert_load_pass.fill_(step_load)
        recorder.on_control_step(_payload())

    # Counters 10 and 12 are log steps. All the recorder's slots hold logical
    # 0 (its map is all zeros), so each expert's load is slot 0's window
    # total: 1+2 tokens at the first log step, then 1+2+3+4.
    assert [dst for _load, dst in fake_dist.reduces] == [0, 0]
    assert [load.shape for load, _dst in fake_dist.reduces] == [
        (len(MOE_LAYERS), NUM_LOGICAL),
    ] * 2
    assert [int(load.sum()) for load, _dst in fake_dist.reduces] == [
        3 * per_logical_expert,
        10 * per_logical_expert,
    ]
    assert fake_dist.broadcasts == []


def test_recorder_keeps_rearranging_without_the_loss_exchange(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without log_balancedness the recorder reduces nothing until the
    rearrangement boundary, which still runs the map exchange."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    recorder = _recorder()

    for _ in range(STEP_INTERVAL - INITIAL_REARRANGE_STEP + 1):
        recorder.on_control_step(_payload())

    assert len(fake_dist.reduces) == 1
    assert len(fake_dist.broadcasts) == 2


def _logical_view(physical_to_logical_map: torch.Tensor):
    """The logical-to-physical map and replica counts a placement implies."""
    num_layers = physical_to_logical_map.shape[0]
    logical_to_physical = torch.zeros((num_layers, NUM_LOGICAL, 4), dtype=torch.int64)
    replica_count = torch.zeros((num_layers, NUM_LOGICAL), dtype=torch.int64)
    for layer in range(num_layers):
        for slot, logical in enumerate(physical_to_logical_map[layer].tolist()):
            if logical < 0:
                continue
            used = int(replica_count[layer, logical])
            logical_to_physical[layer, logical, used] = slot
            replica_count[layer, logical] = used + 1
    return logical_to_physical, replica_count


def _ffn_state():
    model = SimpleNamespace(
        num_moe_layers=len(MOE_LAYERS),
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
    )
    # Two layers of a 12-slot world over 2 FFN ranks: layer 0 keeps the
    # identity placement with logical 8 replicated in a redundant slot,
    # layer 1 swaps logical 5 and 6 across the ranks.
    layer0 = [0, 1, 2, 3, 4, 8, 6, 7, 8, 9, 10, 11]
    layer1 = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11]
    physical_to_logical = torch.tensor([layer0, layer1])
    logical_to_physical, replica_count = _logical_view(physical_to_logical)
    model_state = SimpleNamespace(
        model=model,
        expert_load_window=torch.zeros((WINDOW_SIZE, len(MOE_LAYERS), NUM_PHYSICAL)),
        physical_to_logical_map=physical_to_logical,
        logical_to_physical_map=logical_to_physical,
        logical_replica_count=replica_count,
    )
    rearrangements: list[None] = []

    def rearrange():
        rearrangements.append(None)

    state = SimpleNamespace(
        model_states={"model-hash": model_state},
        expert_rearrangement_step_interval=STEP_INTERVAL,
        expert_rearrangement_step=INITIAL_REARRANGE_STEP,
        rearrange=rearrange,
    )
    return state, model_state, rearrangements


def _ffn_connector(
    world_rank: int, *, log_placement: bool = False, log_interval: int = 0
):
    return SimpleNamespace(
        bootstrap_group=object(),
        world_rank=world_rank,
        ffn_size=2,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(
                    log_balancedness=log_placement,
                    log_balancedness_interval=log_interval,
                ),
            ),
        ),
    )


def test_ffn_driver_rearranges_only_at_the_boundary(monkeypatch: pytest.MonkeyPatch):
    """The FFN driver counts every step and runs the exchange (reduce,
    EplbState.rearrange, map broadcasts) only when the counter reaches the
    interval; FFN ranks reduce zeros and only rank 0 adopts the window."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    state, model_state, rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = 10
    connector = _ffn_connector(world_rank=1)
    driver = DeepEPM2NFFNEplb(connector, state)

    for _ in range(10 - INITIAL_REARRANGE_STEP - 1):
        driver.step()

    assert rearrangements == []
    assert fake_dist.reduces == []

    driver.step()

    assert len(rearrangements) == 1
    # FFN ranks contribute an empty window; rank 0 is the reduce destination.
    window, dst = fake_dist.reduces[0]
    assert dst == 0
    assert int(window.sum()) == 0
    # Rank != 0 keeps its (empty) state window; maps went to every rank.
    assert int(model_state.expert_load_window.sum()) == 0
    assert len(fake_dist.broadcasts) == 2


def test_ffn_driver_rank_zero_adopts_the_reduced_window(
    monkeypatch: pytest.MonkeyPatch,
):
    """FFN world rank 0 copies the reduced Attention window into its
    EplbState before the rearrangement reads it."""

    def reduce(tensor, dst, group):
        tensor.fill_(5)

    monkeypatch.setattr(eplb_module.dist, "reduce", reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", _FakeDist().broadcast)
    state, model_state, rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = 10
    connector = _ffn_connector(world_rank=0)
    driver = DeepEPM2NFFNEplb(connector, state)

    for _ in range(10 - INITIAL_REARRANGE_STEP):
        driver.step()

    assert len(rearrangements) == 1
    assert (
        int(model_state.expert_load_window.sum())
        == 5 * WINDOW_SIZE * len(MOE_LAYERS) * NUM_PHYSICAL
    )


def test_routed_connector_base_seams_default_to_no_op():
    """A routed connector without dynamic expert placement maps ids
    unchanged and steps nothing."""
    connector = FakeRoutedConnector()
    ids = torch.tensor([[1, 2]])

    assert torch.equal(connector.apply_eplb_routing(ids, 0), ids)
    connector.step_ffn_eplb()
    assert connector.events == []


def test_ffn_driver_logs_chosen_experts_when_balancedness_logging_is_on(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """With log_balancedness the FFN world rank 0 logs the committed
    placement and the recorded load that chose it at each boundary."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    state, model_state, rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = 10
    driver = DeepEPM2NFFNEplb(_ffn_connector(0, log_placement=True), state)

    with caplog.at_level(logging.INFO, logger=eplb_module.logger.name):
        for _ in range(10 - INITIAL_REARRANGE_STEP):
            driver.step()

    assert len(rearrangements) == 1
    message = caplog.messages[0]
    assert "EPLB chose experts" in message
    assert "rank 0=0 tokens, rank 1=0 tokens" in message
    assert "layer 0: rank 0=[0, 1, 2, 3, 4, 8]" in message


def test_ffn_driver_stays_silent_without_balancedness_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Without log_balancedness no placement is logged, on any rank."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    for world_rank in (0, 1):
        state, _model_state, _rearrangements = _ffn_state()
        state.expert_rearrangement_step_interval = 10
        driver = DeepEPM2NFFNEplb(_ffn_connector(world_rank), state)

        with caplog.at_level(logging.INFO, logger=eplb_module.logger.name):
            for _ in range(10 - INITIAL_REARRANGE_STEP):
                driver.step()

    assert caplog.messages == []


def _rank_heavy_reduce(tensor, dst, group):
    """Attention ranks contribute a window whose load sits on the logical
    experts FFN rank 0's slots hold (the fake map's first four)."""
    tensor.zero_()
    tensor[:, :4] = 10


def test_ffn_driver_logs_the_deployed_map_loss_at_each_log_interval(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Every log interval FFN world rank 0 scores the deployed map on the
    Attention-reduced window with EPLB's own load model."""
    monkeypatch.setattr(eplb_module.dist, "reduce", _rank_heavy_reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", _FakeDist().broadcast)
    state, _model_state, rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = STEP_INTERVAL
    state.expert_rearrangement_step = 0
    driver = DeepEPM2NFFNEplb(
        _ffn_connector(0, log_placement=True, log_interval=2), state
    )

    with caplog.at_level(logging.INFO, logger=eplb_module.logger.name):
        for _ in range(5):
            driver.step()

    assert rearrangements == []
    assert len(caplog.messages) == 2
    message = caplog.messages[0]
    assert "AFD FFN EPLB step 2:" in message
    # Rank 0's slots hold 40 of the 80 tokens in every layer: half balanced.
    assert "loss=0.5000 over 2 steps (avg_tokens=40.00, max_tokens=80)" in message
    assert "steps until the next rearrangement: 98" in message
    assert "AFD FFN EPLB step 4:" in caplog.messages[1]


def test_ffn_driver_skips_loss_logging_without_recorded_load(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """A log step that recorded no load (warmup, graph capture) logs nothing,
    though every rank still joined the reduce."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    state, _model_state, _rearrangements = _ffn_state()
    state.expert_rearrangement_step = 0
    driver = DeepEPM2NFFNEplb(
        _ffn_connector(0, log_placement=True, log_interval=2), state
    )

    with caplog.at_level(logging.INFO, logger=eplb_module.logger.name):
        for _ in range(2):
            driver.step()

    assert len(fake_dist.reduces) == 1
    assert caplog.messages == []


def test_ffn_driver_logs_the_loss_of_the_map_it_replaced(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """At a boundary the FFN world rank 0 logs EPLB's own objective for the
    deployed map and for the proposal on the same window, so the gain is the
    improvement the rearrangement made rather than workload drift."""

    # Both layers recorded rank 0 holding 80 of the 100 tokens: logical 0
    # and 1 are its hot experts, logical 6 and 7 are rank 1's cold ones.
    def reduce(tensor, dst, group):
        tensor.zero_()
        tensor[:, :, 0] = 50
        tensor[:, :, 1] = 30
        tensor[:, :, 6] = 10
        tensor[:, :, 7] = 10

    monkeypatch.setattr(eplb_module.dist, "reduce", reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", _FakeDist().broadcast)
    state, model_state, _rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = 10
    # Both layers serve the same placement while the window is recorded.
    deployed = [0, 1, 2, 3, 4, 8, 6, 7, 8, 9, 10, 11]
    model_state.physical_to_logical_map = torch.tensor([deployed, deployed])
    (
        model_state.logical_to_physical_map,
        model_state.logical_replica_count,
    ) = _logical_view(model_state.physical_to_logical_map)
    # The proposal moves the cold expert 7 onto rank 0 and the hot expert 1
    # onto rank 1: 60/40 instead of 80/20.
    balanced = [0, 7, 2, 3, 4, 8, 1, 6, 8, 9, 10, 11]

    def rearrange():
        model_state.physical_to_logical_map = torch.tensor([balanced, balanced])

    state.rearrange = rearrange
    driver = DeepEPM2NFFNEplb(_ffn_connector(0, log_placement=True), state)

    with caplog.at_level(logging.INFO, logger=eplb_module.logger.name):
        for _ in range(10 - INITIAL_REARRANGE_STEP):
            driver.step()

    assert (
        "AFD FFN EPLB loss: before=0.6250, after=0.8333, gain=+33.33%"
        in (caplog.messages[-1])
    )
