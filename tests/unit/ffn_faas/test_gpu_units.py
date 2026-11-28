# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Take First Ready serves one MoE layer of whichever unit the pick names."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

CHOICE_STOP, CHOICE_TIMEOUT = -1, -3


@dataclass(frozen=True)
class FakeAFDConfig:
    port: int


class FakeConnector:
    def __init__(self, rank, local_rank, vllm_config, afd_config, role_rank) -> None:
        self.rank, self.local_rank, self.role_rank = rank, local_rank, role_rank
        self.afd_config, self.device = afd_config, "cuda:0"
        self.calls: list[object] = []
        self.control_plane = SimpleNamespace(
            update_state_from_dp_metadata=lambda payload: self.calls.append(payload),
            close=lambda: self.calls.append("control closed"),
        )

    def init_afd_connector(self, **kwargs) -> None:
        self.calls.append("init")

    def _lanes(self, lane):
        return f"buffer-{self.afd_config.port}"

    def recv_attn_output(self, ubatch_idx):
        return SimpleNamespace(context="context")

    def run_experts(self, rows, weights):
        return weights

    def send_ffn_output(self, output, context) -> None:
        self.calls.append(output)

    def synchronize_lanes(self) -> None:
        pass

    def close(self) -> None:
        self.calls.append("closed")


class FakeService:
    def __init__(self, rows, graphs, *, stream, order) -> None:
        self.rows, self.order, self.stopped = rows, order, False
        self.picks: list[int] = []

    def scan(self) -> int:
        return self.picks.pop(0)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def units_module(monkeypatch):
    service = ModuleType("afdep.service")
    service.__dict__.update(
        CHOICE_TIMEOUT=CHOICE_TIMEOUT,
        FfnService=FakeService,
        client_row=lambda buffer: buffer,
        order_ring=lambda buffer: ("ring", buffer),
    )
    monkeypatch.setitem(sys.modules, "afdep.service", service)
    name = "afd_plugin.ffn_faas.platforms.gpu.units"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "DeepEPM2NAFDConnector", FakeConnector)
    # No GPU here: the streams and the DeepGEMM warm-up are GPU-suite matters.
    monkeypatch.setattr(module.torch.cuda, "Stream", lambda device: "stream")
    monkeypatch.setattr(module.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(module.AttentionUnits, "_warm_experts", lambda self: None)
    yield module
    sys.modules.pop(name, None)


def test_each_pick_runs_the_next_layer_of_that_unit(units_module):
    runner = SimpleNamespace(
        connector=FakeConnector(0, 0, None, FakeAFDConfig(port=7000), 0),
        vllm_config=None,
        moe_layer_indices=(3, 5),
        local_experts={
            3: SimpleNamespace(weights="layer 3"),
            5: SimpleNamespace(weights="layer 5"),
        },
    )
    cancelled: list[bool] = []
    units = units_module.AttentionUnits(runner, 2, lambda: bool(cancelled))
    first, second = units.connectors

    assert second.afd_config.port == 7001 and second.calls == ["init"]
    assert units.service.rows == ["buffer-7000", "buffer-7001"]
    assert units.service.order == ("ring", "buffer-7000")

    units.payloads[0].put("unit 0 step")
    units.payloads[1].put("unit 1 step 1")
    units.payloads[1].put("unit 1 step 2")
    # Unit 1 finishes a whole step while unit 0 is between its two layers.
    units.service.picks = [0, 1, CHOICE_TIMEOUT, 1, 1, 0, CHOICE_STOP]
    assert [units.step() for _ in range(5)] == [True] * 5
    assert first.calls == ["unit 0 step", "layer 3", "layer 5"]
    assert second.calls == [
        "init",
        "unit 1 step 1",
        "layer 3",
        "layer 5",
        "unit 1 step 2",
        "layer 3",
    ]

    with pytest.raises(RuntimeError, match="without a cancel"):
        units.step()
    cancelled.append(True)
    units.service.picks = [CHOICE_STOP]
    assert units.step() is False

    units.cancel()
    units.close()
    assert units.service.stopped
    assert first.calls[-1] == "control closed"
    assert second.calls[-2:] == ["control closed", "closed"]


def test_a_control_receiver_error_fails_the_step(units_module):
    runner = SimpleNamespace(
        connector=FakeConnector(0, 0, None, FakeAFDConfig(port=7000), 0),
        vllm_config=None,
        moe_layer_indices=(3,),
        local_experts={},
    )
    units = units_module.AttentionUnits(runner, 2, lambda: False)
    units.errors.append(ValueError("bad payload"))
    units.service.picks = [CHOICE_STOP]

    with pytest.raises(ValueError, match="bad payload"):
        units.step()
