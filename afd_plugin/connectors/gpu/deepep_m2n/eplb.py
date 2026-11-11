# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""EPLB support for the DeepEP M2N connector.

With DeepEP M2N, routing (gate and top-k) runs on Attention, so the two EPLB
duties split across the AFD roles:

- **Attention** maps logical top-k ids to physical expert ids before dispatch
  (``eplb_map_to_physical_and_record``) and records the expert load of its own
  rows. :class:`DeepEPM2NAttentionEplb` owns that recorder: per-layer load
  buffers, a sliding load window, and the current logical-to-physical maps.
- **FFN** holds the experts, so it owns vLLM's ``EplbState`` (registered by
  ``GPUFFNModelRunner.load_model``) and runs the rearrangement.
  :class:`DeepEPM2NFFNEplb` drives it: it never calls ``EplbState.step``
  (nothing on the FFN records load), it steps its own rearrangement counter
  and calls ``EplbState.rearrange`` directly at each boundary.

Both sides count every connector step (real, warmup and graph-capture alike,
matching the FFN runner's ``eplb_step`` cadence) so the boundary is detected
in lockstep. All exchanges run on the connector's CPU bootstrap group (gloo)
with CPU-staged tensors: rearrangements are rare, so a host sync there is
cheap and keeps DeepEP's NCCL communicators out of it.

Boundary exchange (every rank enters, in this order):

1. ``reduce``: every Attention rank contributes its recorded load window; FFN
   ranks contribute zeros. FFN world rank 0 receives the global window and
   copies it into its ``EplbState``; the other FFN ranks' windows stay zero,
   so the all-reduce inside ``EplbState.rearrange`` (over the FFN-only EP
   group) distributes exactly one copy of the global window to every FFN rank.
2. FFN only: ``EplbState.rearrange`` computes the new placement, moves the
   expert weights and commits the new maps in place.
3. ``broadcast``: FFN world rank 0 sends the committed
   ``logical_to_physical_map`` and ``logical_replica_count``; Attention copies
   them into its stable map tensors, whose addresses CUDA graphs captured.

Attention blocks on the exchange before it routes the next step's tokens, and
the FFN runs it after its step's last combine, so Attention can never dispatch
rows placed by an older map than the FFN's weights.

Loss logging (``log_balancedness``, one line every
``log_balancedness_interval`` steps) reuses that exchange shape without
changing anything: Attention ranks reduce their load window aggregated into
logical experts, and the FFN's world rank 0 scores the deployed map on it with
EPLB's own load model (an expert's window load split evenly over its replicas).
The result is the quantity ``DefaultEplbPolicy`` optimises, so the series moves
only when a rearrangement changes the placement. At each boundary the same
window scores the replaced map and the committed one, which is what tells
whether the rearrangement helped. It is opt-in because it adds a host sync on
every participating rank; the map exchange above is unaffected and still runs
only at a rearrangement boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.distributed.eplb.eplb_state import EplbModelState, EplbState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)

from afd_plugin.connectors.metadata import AFDControlPayload
from afd_plugin.distributed.eplb import (
    describe_eplb_placement,
    describe_eplb_rank_load,
    eplb_map_loss,
    logical_load_from_window,
)

if TYPE_CHECKING:
    from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector

logger = init_logger(__name__)

# Keys of the schedule FFN world rank 0 broadcasts to every rank at connector
# init. Everything the Attention recorder needs to mirror the FFN's
# rearrangement cadence and allocate identically shaped tensors.
_EPLB_SCHEDULE_FIELDS = (
    "num_moe_layers",
    "num_logical_experts",
    "num_physical_experts",
    "map_slots",
    "window_size",
    "step_interval",
    "initial_rearrangement_step",
    "log_interval",
)


def recv_initial_eplb_state(
    connector: DeepEPM2NAFDConnector,
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    """Collective: Attention side of ``broadcast_initial_state``.

    Returns the FFN's EPLB schedule and the initial
    ``(logical_to_physical_map, logical_replica_count)`` CPU tensors.
    """
    holder: list[dict[str, int] | None] = [None]
    dist.broadcast_object_list(holder, src=0, group=connector.bootstrap_group)
    schedule = holder[0]
    assert schedule is not None
    missing = [key for key in _EPLB_SCHEDULE_FIELDS if key not in schedule]
    if missing:
        raise RuntimeError(f"FFN EPLB schedule is missing {missing}: {schedule}")
    map_cpu = torch.empty(
        (
            int(schedule["num_moe_layers"]),
            int(schedule["num_logical_experts"]),
            int(schedule["map_slots"]),
        ),
        dtype=torch.int64,
    )
    dist.broadcast(map_cpu, src=0, group=connector.bootstrap_group)
    count_cpu = torch.empty(
        (int(schedule["num_moe_layers"]), int(schedule["num_logical_experts"])),
        dtype=torch.int64,
    )
    dist.broadcast(count_cpu, src=0, group=connector.bootstrap_group)
    return schedule, map_cpu, count_cpu


def real_rows_in_tp_slice(
    num_real_tokens: int,
    *,
    tp_size: int,
    tp_rank: int,
) -> int:
    """Real rows one Attention TP rank's dispatch slice holds.

    The DP rank's ``num_real_tokens`` rows are padded to a multiple of
    ``tp_size`` and chunked; a chunk's real rows are the ones before
    ``num_real_tokens``. Eager steps are exact. A graph-replayed batch is
    padded to its capture size, so with TP > 1 the chunk boundary is computed
    from the real count: the total recorded load stays exact (per-expert
    attribution may shift by the padding rows), and TP 1 is exact in every
    mode. Padding rows route to no expert and are never recorded.
    """
    chunk = -(-num_real_tokens // tp_size)
    return min(max(num_real_tokens - tp_rank * chunk, 0), chunk)


def _single_model_state(eplb_state: EplbState) -> EplbModelState:
    """The one ``EplbModelState`` an AFD FFN registers (speculative decoding
    and draft models are rejected by AFD validation, and a state without any
    model was never registered)."""
    if not eplb_state.model_states:
        raise RuntimeError(
            "DeepEP M2N EPLB needs the FFN model registered with EPLB; "
            "the EplbState holds no model"
        )
    if len(eplb_state.model_states) > 1:
        raise RuntimeError(
            "DeepEP M2N EPLB supports exactly one registered model, got "
            f"{len(eplb_state.model_states)}"
        )
    return next(iter(eplb_state.model_states.values()))


class DeepEPM2NAttentionEplb:
    """Attention-side EPLB recorder for one DeepEP M2N deployment.

    Owns stable-address GPU tensors that the routing path (inside CUDA graphs)
    reads and writes: the per-layer expert load buffer, the sliding load
    window, the logical-to-physical maps adopted from the FFN, and the
    per-step ``record_enabled`` / ``num_unpadded_tokens`` scalars. The
    connector's control plane calls :meth:`on_control_step` once per
    transaction, before the transaction's forward.
    """

    def __init__(
        self,
        connector: DeepEPM2NAFDConnector,
        schedule: dict[str, int],
        moe_layer_indices: tuple[int, ...],
        initial_logical_map: torch.Tensor,
        initial_replica_count: torch.Tensor,
    ) -> None:
        self._connector = connector
        self._device = connector.device
        self._step_interval = int(schedule["step_interval"])
        self._log_interval = int(schedule["log_interval"])
        window_size = int(schedule["window_size"])
        num_layers = int(schedule["num_moe_layers"])
        num_logical = int(schedule["num_logical_experts"])
        num_physical = int(schedule["num_physical_experts"])
        self._map_slots = int(schedule["map_slots"])
        if num_layers != len(moe_layer_indices):
            raise RuntimeError(
                f"FFN reports {num_layers} MoE layers, this Attention model "
                f"runs {len(moe_layer_indices)}"
            )
        self._layer_rows = {
            layer_idx: row for row, layer_idx in enumerate(moe_layer_indices)
        }
        # Read and written by the routing kernel; stable addresses, updated
        # in place so captured CUDA graphs follow the current placement.
        self.expert_load_pass = torch.zeros(
            (num_layers, num_physical), dtype=torch.int32, device=self._device
        )
        self.expert_load_window = torch.zeros(
            (window_size, num_layers, num_physical),
            dtype=torch.int32,
            device=self._device,
        )
        self.logical_to_physical_map = torch.zeros(
            (num_layers, num_logical, self._map_slots),
            dtype=torch.int64,
            device=self._device,
        )
        self.logical_replica_count = torch.zeros(
            (num_layers, num_logical), dtype=torch.int64, device=self._device
        )
        self.record_enabled = torch.ones((), dtype=torch.bool, device=self._device)
        # DBO is rejected with EPLB, so one ubatch is enough.
        self.num_unpadded_tokens = [
            torch.zeros((), dtype=torch.int32, device=self._device)
        ]
        self.logical_to_physical_map.copy_(initial_logical_map.to(self._device))
        self.logical_replica_count.copy_(initial_replica_count.to(self._device))
        self._window_step = 0
        self._rearrange_step = int(schedule["initial_rearrangement_step"])
        self._previous_step_is_dummy: bool | None = None

    # ==============================
    # Control-plane stepping
    # ==============================

    def on_control_step(self, payload: AFDControlPayload) -> None:
        """Step the recorder for the transaction that just finished, then
        prepare the recording state for the transaction about to run.

        The FFN steps EPLB after its forward, while the Attention control
        plane publishes before its forward, so the finished transaction's
        flags are the previous payload's. The very first transaction has no
        predecessor and only prepares its own recording state.
        """
        if self._previous_step_is_dummy is not None:
            self._step_finished_transaction(self._previous_step_is_dummy)
        self._prepare_current_transaction(payload)
        self._previous_step_is_dummy = payload.is_warmup or payload.is_graph_capturing

    def _step_finished_transaction(self, is_dummy: bool) -> None:
        """Mirror the FFN's per-step EPLB transitions.

        Dummy steps (warmup, graph capture) never enter the window; every
        step advances the rearrangement counter, exactly like vLLM's
        ``EplbState.step``, so both roles detect the boundary together.
        """
        if not is_dummy:
            # Record every real step: the window then always holds the last
            # ``window_size`` real steps, the same set vLLM's
            # should_record schedule would have recorded before a rearrange.
            self.expert_load_window[self._window_step].copy_(self.expert_load_pass)
            window_size = self.expert_load_window.shape[0]
            self._window_step = (self._window_step + 1) % window_size
        self._rearrange_step += 1
        if self._log_interval and self._rearrange_step % self._log_interval == 0:
            self._exchange_window_load()
        # The pass holds this step's load until here: dummy steps recorded
        # nothing, so dropping it is a no-op for them.
        self.expert_load_pass.zero_()
        if self._rearrange_step >= self._step_interval:
            self._rearrange_step = 0
            self._exchange_at_boundary()

    def _exchange_window_load(self) -> None:
        """Join the loss-logging exchange.

        Contributes this rank's load window aggregated into logical experts,
        which is all EPLB's policy looks at, so the FFN's world rank 0 can
        score the deployed map on the same traffic that drives the next
        rearrangement. Shipped every ``log_interval`` steps instead of at
        every boundary, which keeps the figures a window average rather than
        one step's noise. The tensor is CPU-staged because the connector's
        bootstrap group is gloo.
        """
        load_cpu = logical_load_from_window(
            self.expert_load_window,
            self.logical_to_physical_map,
            self.logical_replica_count,
        ).cpu()
        dist.reduce(load_cpu, dst=0, group=self._connector.bootstrap_group)

    def _prepare_current_transaction(self, payload: AFDControlPayload) -> None:
        """Gate recording and fill the real token count for the upcoming
        forward from the payload this rank is about to execute."""
        is_dummy = payload.is_warmup or payload.is_graph_capturing
        self.record_enabled.fill_(not is_dummy)
        self.num_unpadded_tokens[0].fill_(
            real_rows_in_tp_slice(
                payload.num_real_tokens,
                tp_size=self._connector.attn_tp_size or 1,
                tp_rank=self._connector.attention_tp_rank,
            )
        )

    def _exchange_at_boundary(self) -> None:
        """Join the boundary exchange: contribute the load window, adopt the
        FFN's new maps. Blocks until the FFN finished its rearrangement."""
        group = self._connector.bootstrap_group
        # Copied, not just moved: the window is cleared below once the new map
        # is in place, and on a CPU device ``.cpu()`` would alias it.
        window_cpu = torch.empty(
            self.expert_load_window.shape, dtype=self.expert_load_window.dtype
        )
        window_cpu.copy_(self.expert_load_window)
        dist.reduce(window_cpu, dst=0, group=group)
        map_cpu = torch.empty(self.logical_to_physical_map.shape, dtype=torch.int64)
        dist.broadcast(map_cpu, src=0, group=group)
        count_cpu = torch.empty(self.logical_replica_count.shape, dtype=torch.int64)
        dist.broadcast(count_cpu, src=0, group=group)
        self.logical_to_physical_map.copy_(map_cpu.to(self._device))
        self.logical_replica_count.copy_(count_cpu.to(self._device))
        # Slot identities changed with the placement: load recorded under the
        # old map cannot be read through the new one, and EPLB does not need
        # it either (a window is fully refreshed before the next boundary
        # whenever window_size <= step_interval). Start the next window clean
        # so every logged figure describes traffic routed by one map.
        self.expert_load_window.zero_()
        self._window_step = 0

    # ==============================
    # Routing path
    # ==============================

    def apply_routing(self, topk_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Map logical top-k ids to physical ids and record the expert load."""
        row = self._layer_rows[layer_idx]
        return eplb_map_to_physical_and_record(
            topk_ids=topk_ids,
            expert_load_view=self.expert_load_pass[row],
            logical_to_physical_map=self.logical_to_physical_map[row],
            logical_replica_count=self.logical_replica_count[row],
            record_enabled=self.record_enabled,
            num_unpadded_tokens=self.num_unpadded_tokens[0],
        )


class DeepEPM2NFFNEplb:
    """FFN-side EPLB rearrangement driver for one DeepEP M2N deployment.

    Steps a rearrangement counter (seeded from the ``EplbState`` so the first
    rearrangement keeps vLLM's initial three-quarter progress) and, at each
    boundary, runs the exchange described in the module docstring and calls
    ``EplbState.rearrange`` directly: on the routed-experts path nothing on
    the FFN records load, so ``EplbState.step``'s window bookkeeping has no
    input and its rearrangement trigger is replaced by this counter. With
    ``log_balancedness`` the same counter also paces the loss log, which
    changes no state.
    """

    def __init__(
        self,
        connector: DeepEPM2NAFDConnector,
        eplb_state: EplbState,
    ) -> None:
        self._connector = connector
        self._state = eplb_state
        self._model_state = _single_model_state(eplb_state)
        self._step_interval = eplb_state.expert_rearrangement_step_interval
        self._rearrange_step = eplb_state.expert_rearrangement_step
        # vLLM's log_balancedness flag gates placement logging: on this path
        # no EplbState.step runs, so the flag cannot log from there. It also
        # enables the loss log: every ``log_balancedness_interval`` steps the
        # deployed map is scored on the Attention ranks' load window, and at
        # each rearrangement the replaced and the committed map are scored on
        # that same window.
        eplb_config = connector.vllm_config.parallel_config.eplb_config
        self._log_placement = bool(eplb_config.log_balancedness)
        self._log_interval = (
            int(eplb_config.log_balancedness_interval) if self._log_placement else 0
        )

    def broadcast_initial_state(self) -> None:
        """Collective: tell every rank the EPLB schedule and initial maps.

        Called from every rank's ``init_afd_connector``; FFN world rank 0 is
        the source. The Attention recorder is built from the result.
        """
        schedule: dict[str, int] = {
            "num_moe_layers": int(self._model_state.model.num_moe_layers),
            "num_logical_experts": int(self._model_state.model.num_logical_experts),
            "num_physical_experts": int(self._model_state.model.num_physical_experts),
            "map_slots": int(self._model_state.logical_to_physical_map.shape[2]),
            "window_size": int(self._model_state.expert_load_window.shape[0]),
            "step_interval": int(self._step_interval),
            "initial_rearrangement_step": int(self._rearrange_step),
            "log_interval": int(self._log_interval),
        }
        holder = [schedule]
        dist.broadcast_object_list(holder, src=0, group=self._connector.bootstrap_group)
        map_cpu = self._model_state.logical_to_physical_map.cpu()
        dist.broadcast(map_cpu, src=0, group=self._connector.bootstrap_group)
        count_cpu = self._model_state.logical_replica_count.cpu()
        dist.broadcast(count_cpu, src=0, group=self._connector.bootstrap_group)

    def step(self) -> None:
        """One connector step; logs the map's loss or rearranges when due."""
        self._rearrange_step += 1
        if self._log_interval and self._rearrange_step % self._log_interval == 0:
            self._log_window_loss()
        if self._rearrange_step < self._step_interval:
            return
        self._rearrange_step = 0
        self._rearrange()

    def _log_window_loss(self) -> None:
        """Join the loss-logging exchange and log it on world rank 0.

        Attention ranks contribute their load window aggregated into logical
        experts; FFN ranks hold no load, so they only contribute the zero
        tensor the reduce writes into on rank 0.
        """
        num_logical = self._model_state.logical_to_physical_map.shape[1]
        window_dtype = self._model_state.expert_load_window.dtype
        staging = torch.zeros(
            (int(self._model_state.model.num_moe_layers), num_logical),
            dtype=window_dtype,
        )
        dist.reduce(staging, dst=0, group=self._connector.bootstrap_group)
        if self._connector.world_rank == 0:
            self._log_deployed_loss(staging)

    def _log_deployed_loss(self, logical_load: torch.Tensor) -> None:
        """Log the deployed map's EPLB loss on the recorded window.

        ``logical_load`` is the reduced per-logical-expert load,
        shape ``(num_moe_layers, num_logical)``. The loss is the figure
        ``DefaultEplbPolicy`` optimises, so the series moves when a
        rearrangement changes the placement and stays flat in between.
        """
        summary = eplb_map_loss(
            logical_load,
            self._model_state.physical_to_logical_map.cpu(),
            ffn_size=self._connector.ffn_size,
        )
        if summary.max_tokens <= 0:
            # Dummy steps (warmup, graph capture) record no load.
            return
        window_size = self._model_state.expert_load_window.shape[0]
        recorded_steps = min(self._rearrange_step, window_size)
        logger.info(
            "AFD FFN EPLB step %d: loss=%.4f over %d steps "
            "(avg_tokens=%.2f, max_tokens=%d), "
            "steps until the next rearrangement: %d",
            self._rearrange_step,
            summary.balancedness,
            recorded_steps,
            summary.avg_tokens,
            int(summary.max_tokens),
            self._step_interval - self._rearrange_step,
        )

    def _rearrange(self) -> None:
        window = self._model_state.expert_load_window
        # FFN ranks contribute nothing: rank 0's tensor receives the Attention
        # window sum, the others stay zero for the EP-group all-reduce inside
        # EplbState.rearrange to redistribute exactly one copy.
        staging = torch.zeros(window.shape, dtype=window.dtype)
        dist.reduce(staging, dst=0, group=self._connector.bootstrap_group)
        if self._connector.world_rank == 0:
            window.copy_(staging.to(window.device))
        logs_loss = self._log_placement and self._connector.world_rank == 0
        deployed_maps = (
            (
                self._model_state.physical_to_logical_map.cpu(),
                self._model_state.logical_to_physical_map.cpu(),
                self._model_state.logical_replica_count.cpu(),
            )
            if logs_loss
            else None
        )
        self._state.rearrange()
        if logs_loss:
            # Rearrangement commits the new map in place, so the loss log
            # compares it with the copies saved above.
            self._log_chosen_experts(staging)
            self._log_map_loss(staging, deployed_maps)
        map_cpu = self._model_state.logical_to_physical_map.cpu()
        dist.broadcast(map_cpu, src=0, group=self._connector.bootstrap_group)
        count_cpu = self._model_state.logical_replica_count.cpu()
        dist.broadcast(count_cpu, src=0, group=self._connector.bootstrap_group)

    def _log_chosen_experts(self, load_window: torch.Tensor) -> None:
        """Log the placement EPLB just committed and the load that chose it.

        ``load_window`` is the reduced Attention window (physical expert
        load per recorded step) this rearrangement decided from, shape
        ``(window_size, num_moe_layers, num_physical_experts)``.
        """
        ffn_size = self._connector.ffn_size
        lines = describe_eplb_placement(
            self._model_state.physical_to_logical_map.cpu(),
            ffn_size=ffn_size,
        )
        logger.info(
            "DeepEP M2N EPLB chose experts (recorded load over %d steps: %s):\n%s",
            load_window.shape[0],
            describe_eplb_rank_load(load_window, ffn_size=ffn_size),
            "\n".join(lines),
        )

    def _log_map_loss(
        self,
        load_window: torch.Tensor,
        deployed_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Log what the rearrangement did to the quantity EPLB optimises.

        Both figures use EPLB's own load model (a logical expert's window load
        divided evenly over its replicas) on the same window, so the gain is
        the policy's improvement and not workload drift. ``deployed_maps``
        holds the physical-to-logical map, the logical-to-physical map and the
        replica counts in force while the window was recorded; the state has
        already committed the replacements.
        """
        ffn_size = self._connector.ffn_size
        deployed_map, recording_map, recording_counts = deployed_maps
        logical_load = logical_load_from_window(
            load_window, recording_map, recording_counts
        )
        committed_map = self._model_state.physical_to_logical_map.cpu()
        before = eplb_map_loss(logical_load, deployed_map, ffn_size=ffn_size)
        after = eplb_map_loss(logical_load, committed_map, ffn_size=ffn_size)
        gain = (
            0.0
            if before.balancedness == 0
            else 100.0 * (after.balancedness / before.balancedness - 1.0)
        )
        logger.info(
            "AFD FFN EPLB loss: before=%.4f, after=%.4f, gain=%+.2f%% "
            "(same %d-step window; per-layer mean/max of the load EPLB "
            "estimates by splitting each expert over its replicas)",
            before.balancedness,
            after.balancedness,
            gain,
            load_window.shape[0],
        )


__all__ = [
    "DeepEPM2NAttentionEplb",
    "DeepEPM2NFFNEplb",
    "describe_eplb_placement",
    "real_rows_in_tp_slice",
    "recv_initial_eplb_state",
]
