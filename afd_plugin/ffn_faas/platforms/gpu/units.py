# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Take First Ready: one F rank serves several Attention units in arrival order.

Each Attention unit is its own ``vllm serve`` deployment with its own AFD world
on ``afd.port + unit``; this rank holds one ``DeepEPM2NAFDConnector`` per unit.
afdEP-comm's ``FfnService`` picks the next unit whose dispatch has arrived
(round robin when several have), and this rank runs one MoE layer of that unit.
A unit with a long step therefore does not hold a unit with a short one.

Limits: eager, one lane, BF16 experts, and every control payload runs all MoE
layers. An idle rank spins the pick kernel on its GPU and blocks one CPU thread
per unit in the control receive.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import torch
from afdep.service import CHOICE_TIMEOUT, FfnService, client_row, order_ring

from afd_plugin.connectors.base import AFDControlPlaneClosedError
from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector
from afd_plugin.connectors.metadata import AFDControlPayload
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
    GPUFFNRoutedExpertsModelRunner,
)

# The only lane of an eager step without DBO.
LANE = 0
PAYLOAD_POLL_S = 1.0
THREAD_JOIN_S = 5.0
# Rows of the expert warm-up; DeepGEMM compiles its kernels for N and K only.
WARM_ROWS = 1024


class AttentionUnits:
    """Owner of the per-unit connectors, control receivers and the pick."""

    def __init__(
        self,
        runner: GPUFFNRoutedExpertsModelRunner,
        units: int,
        cancel_requested: Callable[[], bool],
    ) -> None:
        """Collective with every unit: connect units 1.. after unit 0.

        ``runner.connector`` (unit 0) is already initialized. The pick's
        arrival counters start here, before any control receiver runs, so no
        unit can have dispatched yet: Attention dispatches only after every F
        rank received its control payload.
        """
        # The backend checked the connector type before building this.
        first = cast(DeepEPM2NAFDConnector, runner.connector)
        self.runner = runner
        self.cancel_requested = cancel_requested
        self.connectors = [first]
        for unit in range(1, units):
            connector = DeepEPM2NAFDConnector(
                first.rank,
                first.local_rank,
                runner.vllm_config,
                replace(first.afd_config, port=first.afd_config.port + unit),
                first.role_rank,
            )
            self.connectors.append(connector)
            connector.init_afd_connector(
                moe_layer_indices=runner.moe_layer_indices,
                local_experts=runner.local_experts,
            )
        buffers = [connector._lanes(LANE) for connector in self.connectors]
        # Not the default stream: CUDA orders it with every other stream, so
        # a step would wait for the control receives that are still pending.
        self.stream = torch.cuda.Stream(first.device)
        self.service = FfnService(
            [client_row(buffer) for buffer in buffers],
            # scan() never replays a graph; the experts run eagerly in step().
            [None] * units,  # type: ignore[list-item]
            # Behind the experts: a pick spinning beside them slows their GEMMs.
            stream=self.stream,
            # F rank 0 picks; the other F ranks serve its order.
            order=order_ring(buffers[0]),
        )
        self.payloads: list[queue.Queue[AFDControlPayload]] = [
            queue.Queue() for _ in range(units)
        ]
        self.next_layer = [0] * units
        self.errors: list[BaseException] = []
        self.threads: list[threading.Thread] = []
        self._warm_experts()

    def _warm_experts(self) -> None:
        """Load the expert GEMM kernels before any control receive is pending.

        DeepGEMM's first call of a kernel synchronizes the whole device, which
        never returns while another unit's control receive waits for a payload.
        """
        first = self.connectors[0]
        weights = self.runner.local_experts[self.runner.moe_layer_indices[0]].weights
        if weights.is_fp8:
            raise NotImplementedError("several Attention units need BF16 experts")
        assert first.experts is not None  # set on every FFN rank at connector init
        with torch.cuda.stream(self.stream):
            first.experts(
                torch.zeros(
                    (WARM_ROWS, weights.w13.shape[2]),
                    dtype=torch.bfloat16,
                    device=first.device,
                ),
                None,
                torch.zeros(WARM_ROWS, device=first.device),
                # Prefix sum of rows per expert: every row belongs to expert 0.
                torch.full(
                    (weights.num_experts,),
                    WARM_ROWS,
                    dtype=torch.int32,
                    device=first.device,
                ),
                weights,
            )
        self.stream.synchronize()

    def start(self) -> None:
        """Start one control receiver per unit (a receive blocks in NCCL)."""
        for unit in range(len(self.connectors)):
            thread = threading.Thread(
                target=self._receive, args=(unit,), name=f"afd-unit-{unit}", daemon=True
            )
            self.threads.append(thread)
            thread.start()

    def _receive(self, unit: int) -> None:
        connector = self.connectors[unit]
        torch.cuda.set_device(connector.device)
        try:
            # A pending NCCL receive occupies the caller's current stream, so
            # it gets its own, apart from the experts and the pick.
            with torch.cuda.stream(torch.cuda.Stream(connector.device)):
                while True:
                    self.payloads[unit].put(
                        connector.control_plane.recv_dp_metadata_list()
                    )
        except AFDControlPlaneClosedError as error:
            if not self.cancel_requested():
                self.errors.append(error)
        except BaseException as error:  # reported by step() on the receiver thread
            self.errors.append(error)
        self.service.stop()

    def _check(self) -> None:
        if self.errors:
            raise self.errors[0]

    def step(self) -> bool:
        """Run one MoE layer of the first ready unit; ``False`` after a cancel."""
        with torch.cuda.stream(self.stream):
            return self._step()

    def _step(self) -> bool:
        while (unit := self.service.scan()) == CHOICE_TIMEOUT:
            self._check()
        self._check()
        if unit < 0:
            if not self.cancel_requested():
                raise RuntimeError(f"FFN pick ended without a cancel ({unit})")
            return False
        connector, layer = self.connectors[unit], self.next_layer[unit]
        if layer == 0:
            # The unit dispatched, so its payload was sent; it may still be
            # in decoding on the control receiver.
            while True:
                try:
                    payload = self.payloads[unit].get(timeout=PAYLOAD_POLL_S)
                    break
                except queue.Empty:
                    self._check()
                    if self.cancel_requested():
                        return False
            connector.control_plane.update_state_from_dp_metadata(payload)
        layers = self.runner.moe_layer_indices
        rows = connector.recv_attn_output(ubatch_idx=LANE)
        weights = self.runner.local_experts[layers[layer]].weights
        output = connector.run_experts(rows, weights)
        connector.send_ffn_output(output, rows.context)
        connector.synchronize_lanes()
        self.next_layer[unit] = (layer + 1) % len(layers)
        return True

    def cancel(self) -> None:
        """End the pick and every control receive (any thread)."""
        self.service.stop()
        for connector in self.connectors:
            connector.control_plane.close()

    def close(self) -> None:
        """Close units 1..; the runner closes unit 0."""
        for thread in self.threads:
            thread.join(THREAD_JOIN_S)
        for connector in self.connectors[1:]:
            connector.close()
