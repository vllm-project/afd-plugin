# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 eager Mega communication and fused expert backend."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
)

from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.distributed.afd_process_group import (
    DefaultProcessGroupSwitcher,
    init_afd_process_group,
)

from .mega_startup import agree_layer_selection
from .mega_weights import pack_layer_weights

if TYPE_CHECKING:
    from torch_npu.npu import Event, Stream
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig
    from afd_plugin.connectors import AFDA2FTransferPayload
    from afd_plugin.connectors.metadata import AFDTransferContext

    from .mega_layers import LayerWeights, MegaModelSpec
    from .mega_runtime import DispatchTicket, SplitBuffer

STARTUP_TIMEOUT = timedelta(minutes=30)
POLL_SECONDS = 0.001
# A submitted operator may still wait for work on device. Shutdown must
# give up with an explicit error instead of destroying a live ubatch window.
CLOSE_LOCK_TIMEOUT_SECONDS = 5.0
CLOSE_DEVICE_TIMEOUT_SECONDS = 5.0
CLOSE_PEER_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class MegaExtraInfo(ConnectorExtraInfo):
    attn_ranks_per_dp: int | None = None

    def to_mapping(self) -> dict[str, int]:
        if self.attn_ranks_per_dp is None:
            return {}
        return {"attn_ranks_per_dp": self.attn_ranks_per_dp}


@dataclass
class MegaLane:
    """The single fused-FFN lane shared by every ubatch."""

    group: dist.ProcessGroup
    buffer: SplitBuffer
    sequence: int = 0
    stream: Stream | None = None
    completion: Event | None = None


def validate_mega_config(
    config: VllmConfig, afd: AFDConfig, model: MegaModelSpec
) -> None:
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if (
        get_ascend_device_type() != AscendDeviceType.A5
        or config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError("Mega requires A5 with BF16 input")
    attn, ffn = afd.num_attention_ranks, afd.num_ffn_ranks
    world = attn + ffn
    experts = model.num_experts
    # UMDK SplitLayout / arch35 tiling bound the union group and count table.
    if not (
        attn > 0
        and ffn > 0
        and world <= 768
        and ffn <= experts <= 2048
        and experts % ffn == 0
        and experts // ffn <= 1024
        and world * (experts // ffn) <= 16384
    ):
        raise ValueError(f"Invalid Mega split geometry: A={attn}, F={ffn}, E={experts}")
    p = config.parallel_config
    if (
        p.data_parallel_size * p.tensor_parallel_size
        != (attn if afd.role == "attention" else ffn)
        or (
            p.pipeline_parallel_size,
            p.prefill_context_parallel_size,
            p.decode_context_parallel_size,
        )
        != (1, 1, 1)
        or (
            afd.role == "ffn"
            and (not p.enable_expert_parallel or p.use_sequence_parallel_moe)
        )
        or p.expert_placement_strategy != "linear"
        or p.enable_eplb
        or p.enable_elastic_ep
        or p.eplb_config.num_redundant_experts
        or (
            afd.role == "attention"
            and config.scheduler_config.max_num_batched_tokens % p.tensor_parallel_size
        )
    ):
        raise ValueError(
            "Mega requires role DP * TP equal to its rank count, PP/CP=1, "
            "TP-aligned Attention token capacity and static FFN EP without SP"
        )
    if (p.enable_dbo or p.use_ubatching) and (
        afd.role != "attention" or p.num_ubatches != 2
    ):
        raise ValueError(
            "Mega DBO requires two Attention ubatches and an unbatched FFN"
        )
    if (
        not config.model_config.enforce_eager
        or not config.use_v2_model_runner
        or config.speculative_config is not None
        or not afd.compute_gate_on_attention
        or not afd.async_dp
    ):
        raise ValueError(
            "Mega requires eager ModelRunnerV2, async DP, Attention routing"
        )


class CAMAsyncMegaAFDConnector(AFDConnectorBase):
    extra_info: MegaExtraInfo

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> MegaExtraInfo:
        # Match the base interface: values are unvalidated here and may have
        # any type. Validate them before constructing the typed MegaExtraInfo.
        raw = raw or {}
        if set(raw) - {"attn_ranks_per_dp"}:
            raise ValueError("Unknown Mega connector_extra_config field")
        width = raw.get("attn_ranks_per_dp")
        if "attn_ranks_per_dp" in raw and (type(width) is not int or width < 1):
            raise ValueError("attn_ranks_per_dp must be a positive integer")
        return MegaExtraInfo(width)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        from afd_plugin.model_executor.models.npu.deepseek_v4_mega import (
            get_mega_model_spec,
        )

        self.model_spec = get_mega_model_spec(vllm_config)
        self.attn_size = afd_config.num_attention_ranks
        self.ffn_size = afd_config.num_ffn_ranks
        self.world_rank = (
            role_rank if afd_config.role == "attention" else self.attn_size + role_rank
        )
        self.lane: MegaLane | None = None
        # One window serves every ubatch. Attention keeps one in-flight ticket
        # and one cached result per ubatch, and serializes the sends through the
        # native DBO receive hook so the shared window never holds two inputs.
        self.num_ubatches = 1
        self.ubatch_tickets: list[DispatchTicket | None] = []
        self.ubatch_outputs: list[torch.Tensor | None] = []
        self._recv_locks: list[threading.Lock] = []
        self._send_lock = threading.Lock()
        self.layer_ids: list[int] = []
        self.layered_weights: LayerWeights | None = None
        self.layer_slots: dict[int, int] = {}
        # The FFN loop and shutdown run on different threads. A step must finish
        # before close destroys the windows it submits into, and no step may
        # submit once close started.
        self._step_lock = threading.Lock()
        self._closed = False
        self._shutdown_store: dist.Store | None = None

    @property
    def is_initialized(self) -> bool:
        return self.lane is not None

    def prepare_layers(self, weights: Mapping[int, LayerWeights]) -> None:
        """Accept model-extracted, Mega-formatted tensors indexed by layer ID."""
        self.layered_weights = pack_layer_weights(weights)
        self.layer_ids = sorted(weights)

    def init_afd_connector(self, *, preparation_error: Exception | None = None) -> None:
        if self.is_initialized:
            return
        self._closed = False
        from . import mega_runtime as ops

        self.ops = ops

        def initialize_startup(store: dist.Store) -> None:
            self._shutdown_store = dist.PrefixStore("shutdown", store)
            self.layer_slots = agree_layer_selection(
                store,
                self.attn_size,
                self.ffn_size,
                self.world_rank,
                self.layer_ids,
                timeout=STARTUP_TIMEOUT,
                preparation_error=preparation_error,
            )
            store.set_timeout(STARTUP_TIMEOUT)
            stores.append(store)

        stores: list[dist.Store] = []
        group = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.attn_size + self.ffn_size,
            rank=self.world_rank,
            group_name="afd_mega",
            timeout=STARTUP_TIMEOUT,
            store_callback=initialize_startup,
        )
        try:
            store = stores[0]
            # FFN TP can differ from Attention TP. Share actual widths and check
            # all explicit declarations before anyone enters buffer collectives.
            attention = self.afd_config.role == "attention"
            width = (
                self.vllm_config.parallel_config.tensor_parallel_size
                if attention
                else 0
            )
            parallel = self.vllm_config.parallel_config
            dbo = attention and (parallel.enable_dbo or parallel.use_ubatching)
            store.set(
                f"tp/{self.world_rank}",
                f"{width} {self.extra_info.attn_ranks_per_dp or 0} {int(dbo)}",
            )
            widths = [
                tuple(map(int, store.get(f"tp/{i}").split()))
                for i in range(self.attn_size + self.ffn_size)
            ]
            self.attn_tp = widths[0][0]
            if (
                self.attn_size % self.attn_tp
                or any(
                    actual != self.attn_tp for actual, _, _ in widths[: self.attn_size]
                )
                or any(declared not in (0, self.attn_tp) for _, declared, _ in widths)
            ):
                raise ValueError("Inconsistent Attention TP / attn_ranks_per_dp")
            use_ubatching = widths[0][2]
            if any(dbo != use_ubatching for _, _, dbo in widths[: self.attn_size]):
                raise ValueError("All Attention ranks must agree on Mega ubatching")
            # Only Attention schedulers own the token limit. All window offsets
            # must be derived from the same capacity, including on FFN workers.
            if self.afd_config.role == "attention":
                store.set(
                    f"capacity/{self.role_rank}",
                    str(
                        self.vllm_config.scheduler_config.max_num_batched_tokens
                        // self.attn_tp
                    ),
                )
            capacity = max(
                int(store.get(f"capacity/{dp}")) for dp in range(self.attn_size)
            )
            model = self.model_spec
            # One fused FFN operator serves every ubatch. Both ubatches share a
            # single window and a single sequence ring; Attention serializes the
            # two sends through the native DBO receive hook, so one window slot
            # stays correct without a second buffer or HCCL domain.
            with DefaultProcessGroupSwitcher(dist.group.WORLD, group):
                buffer = ops.create_mega_split_buffer(
                    group,
                    self.attn_size,
                    self.ffn_size,
                    model.num_experts,
                    capacity,
                    model.hidden_size,
                    model.intermediate_size,
                    model.top_k,
                    attn_ranks_per_dp=self.attn_tp,
                    shared_experts=0,
                    combine_quant_mode=0,
                    activation_parameters=model.activation_parameters,
                )
            lane = MegaLane(group, buffer)
            self.lane = lane
            self.num_ubatches = 1 + int(use_ubatching)
            self.ubatch_tickets = [None] * self.num_ubatches
            self.ubatch_outputs = [None] * self.num_ubatches
            self._recv_locks = [threading.Lock() for _ in range(self.num_ubatches)]
            if not attention:
                lane.stream = torch.npu.Stream()
                lane.stream.wait_stream(torch.npu.current_stream())
        except Exception:
            if self.lane is not None:
                self.lane.buffer.destroy()
                self.lane = None
            dist.destroy_process_group(group)
            raise

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: torch.Tensor,
    ) -> None:
        stage = context.metadata.stage_idx
        if not 0 <= stage < self.num_ubatches:
            raise RuntimeError("Unexpected Mega Attention ubatch")
        lane = self.lane
        assert lane is not None
        with self._send_lock:
            if self.ubatch_tickets[stage] is not None:
                raise RuntimeError(
                    "Mega eager supports one outstanding ticket per ubatch"
                )
            # Single-window ownership: take back the peer ubatch's result
            # before writing this ubatch's input into the shared window.
            overlapping = dbo_enabled()
            if overlapping:
                dbo_maybe_run_recv_hook()
            dp = self.role_rank // self.attn_tp
            members = self.dp_members(dp)
            layer_slot = self.layer_slots[context.metadata.layer_idx]
            sequence = self.ops.split_sequence(lane.sequence + 1)
            ticket = self.ops.atten_send(
                hidden_states.contiguous(),
                kwargs["topk_ids"].int().contiguous(),
                kwargs["topk_weights"].float().contiguous(),
                lane.buffer,
                members,
                sequence,
                layer_index=layer_slot,
            )
            lane.sequence = sequence
            self.ubatch_tickets[stage] = ticket
            self.ubatch_outputs[stage] = None
            if overlapping:
                dbo_register_recv_hook(lambda: self._receive(stage))

    def dp_members(self, dp: int) -> list[int]:
        start = dp * self.attn_tp
        return list(range(start, start + self.attn_tp))

    def _receive(self, ubatch_idx: int) -> torch.Tensor:
        if not 0 <= ubatch_idx < self.num_ubatches:
            raise RuntimeError("Unexpected Mega Attention ubatch")
        with self._recv_locks[ubatch_idx]:
            output = self.ubatch_outputs[ubatch_idx]
            if output is None:
                ticket = self.ubatch_tickets[ubatch_idx]
                if ticket is None:
                    raise RuntimeError(
                        "Mega combine requires the matching ubatch ticket"
                    )
                assert self.lane is not None
                output = self.ops.atten_recv(ticket, self.lane.buffer)
                self.ubatch_tickets[ubatch_idx] = None
                self.ubatch_outputs[ubatch_idx] = output
        return output

    def recv_ffn_output(
        self, ref_tensor: torch.Tensor | None = None, ubatch_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        return self._receive(ubatch_idx)

    def execute_next_experts(self) -> None:
        with self._step_lock:
            if self._closed:
                return
            lane = self.lane
            assert lane is not None
            if lane.completion is None or lane.completion.query():
                assert self.layered_weights is not None
                args = (*self.layered_weights, lane.buffer)
                if lane.stream is None:
                    self.ops.mega_ffn(*args)
                    torch.npu.synchronize()
                else:
                    with torch.npu.stream(lane.stream):
                        self.ops.mega_ffn(*args)
                        if lane.completion is None:
                            lane.completion = torch.npu.Event()
                        lane.completion.record(lane.stream)
                return
        time.sleep(POLL_SECONDS)

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        raise RuntimeError(
            "Mega FFN runs recv, compute and send in execute_next_experts"
        )

    def send_ffn_output(
        self, ffn_output: torch.Tensor, context: AFDTransferContext, **kwargs: Any
    ) -> None:
        raise RuntimeError("Mega FFN sends results inside fused_moe")

    def close(self) -> None:
        # Wait for an in-flight step and block later ones: this connector is
        # closed by the worker while the FFN loop may still be submitting.
        if not self._step_lock.acquire(timeout=CLOSE_LOCK_TIMEOUT_SECONDS):
            raise RuntimeError(
                "Mega FFN step did not return; retain resources until it exits"
            )
        try:
            self._closed = True
            if self.lane is not None:
                self._wait_device_idle()
                self._wait_peer_drain()
                self.lane.buffer.destroy()
                dist.destroy_process_group(self.lane.group)
                self.lane = None
            self.layer_ids.clear()
            self.layered_weights = None
            self.layer_slots.clear()
        finally:
            self._step_lock.release()

    def _wait_peer_drain(self) -> None:
        """Publish local quiescence, then bound peer acknowledgement on the host."""
        store = self._shutdown_store
        if store is None:
            raise RuntimeError("Mega shutdown store is unavailable; retain resources")
        deadline = time.monotonic() + CLOSE_PEER_TIMEOUT_SECONDS
        try:
            # Bound set() too: the Store server itself may have exited. This
            # connector owns the prefix and uses it only during shutdown.
            store.set_timeout(timedelta(seconds=CLOSE_PEER_TIMEOUT_SECONDS))
            store.set(f"drained/{self.world_rank}", "1")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Mega shutdown publication exceeded its deadline")
            store.wait(
                [f"drained/{rank}" for rank in range(self.attn_size + self.ffn_size)],
                timedelta(seconds=remaining),
            )
        except (RuntimeError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Mega peers did not confirm shutdown; retain resources"
            ) from exc
        # Retain the Store reference after success so this rank does not drop
        # the rendezvous server while peers are still reading acknowledgements.

    def _wait_device_idle(self) -> None:
        """Wait for submitted work with a deadline; keep the windows on timeout."""
        pending = []
        if self.lane is not None and self.lane.completion is not None:
            pending.append(self.lane.completion)
        current = torch.npu.Event()
        current.record()
        pending.append(current)
        deadline = time.monotonic() + CLOSE_DEVICE_TIMEOUT_SECONDS
        while not all(event.query() for event in pending):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Mega FFN device work did not stop; retain resources"
                )
            time.sleep(POLL_SECONDS)
