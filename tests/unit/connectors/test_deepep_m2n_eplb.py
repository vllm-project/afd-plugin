# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only tests for the DeepEP M2N connector's EPLB coordination.

The distributed boundary exchange is exercised with a recording stand-in for
``torch.distributed``: what is asserted is the protocol (which tensors are
reduced, which are adopted), not the transport.
"""

from __future__ import annotations

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
NUM_LOGICAL = 8
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


def _recorder(**connector_overrides):
    initial_map = torch.zeros((len(MOE_LAYERS), NUM_LOGICAL, 4), dtype=torch.int64)
    initial_count = torch.ones((len(MOE_LAYERS), NUM_LOGICAL), dtype=torch.int64)
    return DeepEPM2NAttentionEplb(
        _attention_connector(**connector_overrides),
        _schedule(),
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


def _ffn_state():
    model = SimpleNamespace(
        num_moe_layers=len(MOE_LAYERS),
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
    )
    model_state = SimpleNamespace(
        model=model,
        expert_load_window=torch.zeros((WINDOW_SIZE, len(MOE_LAYERS), NUM_PHYSICAL)),
        logical_to_physical_map=torch.zeros((len(MOE_LAYERS), NUM_LOGICAL, 4)),
        logical_replica_count=torch.zeros((len(MOE_LAYERS), NUM_LOGICAL)),
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


def test_ffn_driver_rearranges_only_at_the_boundary(monkeypatch: pytest.MonkeyPatch):
    """The FFN driver counts every step and runs the exchange (reduce,
    EplbState.rearrange, map broadcasts) only when the counter reaches the
    interval; FFN ranks reduce zeros and only rank 0 adopts the window."""
    fake_dist = _FakeDist()
    monkeypatch.setattr(eplb_module.dist, "reduce", fake_dist.reduce)
    monkeypatch.setattr(eplb_module.dist, "broadcast", fake_dist.broadcast)
    state, model_state, rearrangements = _ffn_state()
    state.expert_rearrangement_step_interval = 10
    connector = SimpleNamespace(bootstrap_group=object(), world_rank=1)
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
    connector = SimpleNamespace(bootstrap_group=object(), world_rank=0)
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
