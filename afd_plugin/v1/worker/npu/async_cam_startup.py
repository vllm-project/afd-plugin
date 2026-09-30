# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-side startup coordination for the Async CAM Attention/FFN world."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import PrefixStore, Store
from vllm.logger import init_logger

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import (
    AFD_ASYNC_CAM_GROUP_NAME,
    AFDAsyncTopology,
)
from afd_plugin.distributed.afd_process_group import ProcessGroupRendezvousContext

STARTUP_TIMEOUT_SECONDS = 300
STARTUP_POLL_SECONDS = 0.25
STARTUP_NONCE_BYTES = 16
STARTUP_WARMUP_STAGE = 0
STARTUP_WARMUP_TOKEN_COUNT = 1
STARTUP_WARMUP_HIDDEN_VALUE = 0.25

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class AsyncCamStartupSpec:
    """Config-derived, immutable shape and topology of one CAM participant."""

    topology: AFDAsyncTopology
    local_rank: int
    tp_size: int
    hidden_size: int
    topk: int
    activation_dtype: torch.dtype

    @property
    def attention_dp_size(self) -> int:
        if self.topology.attn_size % self.tp_size:
            raise ValueError("CAM Attention size must be divisible by TP size")
        return self.topology.attn_size // self.tp_size


@dataclass(frozen=True, slots=True)
class FFNStartupPlan:
    """Selected FFN startup mode and the layer used by graph warmup."""

    use_graph: bool
    first_layer_idx: int | None = None

    def __post_init__(self) -> None:
        if self.use_graph and (
            self.first_layer_idx is None or self.first_layer_idx < 0
        ):
            raise ValueError("CAM FFN graph startup requires a valid first layer")
        if not self.use_graph and self.first_layer_idx is not None:
            raise ValueError("CAM FFN eager startup cannot select a graph layer")


class AsyncCamStartupCoordinator:
    """Run one rank's complete CAM startup without owning its communicator."""

    def __init__(
        self,
        connector: AFDConnectorBase,
        rendezvous_context: ProcessGroupRendezvousContext,
        spec: AsyncCamStartupSpec,
    ) -> None:
        self._connector = connector
        self._rendezvous_context = rendezvous_context
        self._spec = spec
        self._store: Store | None = None
        self._started = False
        self._starting = False
        self._failure: Exception | None = None

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def started(self) -> bool:
        return self._started

    def _begin(self, role: str) -> bool:
        if self._spec.topology.role != role:
            raise RuntimeError(f"CAM {role} startup requires {role} rank")
        if self._failure is not None:
            raise RuntimeError("CAM startup previously failed") from self._failure
        if self._started:
            self._rendezvous_context.borrow()
            return False
        if self._starting:
            raise RuntimeError("CAM startup is already in progress")
        self._starting = True
        return True

    def _get_store(self) -> Store:
        if self._store is not None:
            self._rendezvous_context.borrow()
            return self._store
        store, process_group = self._rendezvous_context.borrow()
        topology = self._spec.topology
        nonce = torch.zeros(
            STARTUP_NONCE_BYTES,
            dtype=torch.uint8,
            device=f"npu:{self._spec.local_rank}",
        )
        if topology.world_rank == 0:
            nonce.copy_(
                torch.tensor(
                    list(os.urandom(STARTUP_NONCE_BYTES)),
                    dtype=torch.uint8,
                    device=nonce.device,
                )
            )
        logger.info("CAM startup nonce broadcast start rank=%d", topology.world_rank)
        dist.broadcast(nonce, src=0, group=process_group)
        logger.info("CAM startup nonce broadcast done rank=%d", topology.world_rank)
        torch.npu.synchronize()
        logger.info("CAM startup nonce synchronize done rank=%d", topology.world_rank)
        namespace = bytes(nonce.cpu().tolist()).hex()
        self._store = PrefixStore(
            f"{AFD_ASYNC_CAM_GROUP_NAME}/startup/{namespace}", store
        )
        return self._store

    def _failure_keys(self) -> list[str]:
        topology = self._spec.topology
        return [f"attn/failure/{rank}" for rank in range(topology.attn_size)] + [
            f"ffn/{rank}" for rank in range(topology.attn_size, topology.world_size)
        ]

    def _check_failures(self, store: Store, stage: str) -> None:
        for key in self._failure_keys():
            if not store.check([key]):
                continue
            status = store.get(key).decode()
            if key.startswith("attn/failure/") or status.startswith("failed:"):
                raise RuntimeError(
                    f"CAM startup rank={self._spec.topology.world_rank} "
                    f"stage={stage} {key} {status}"
                )

    def _wait_for_entries(self, keys: list[str], *, stage: str) -> list[str]:
        store = self._get_store()
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            self._check_failures(store, stage)
            values = []
            pending = []
            for key in keys:
                if store.check([key]):
                    values.append(store.get(key).decode())
                else:
                    pending.append(key)
            if not pending:
                self._check_failures(store, stage)
                return values
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for CAM startup rank="
                    f"{self._spec.topology.world_rank} stage={stage} "
                    f"keys={sorted(pending)}"
                )
            time.sleep(STARTUP_POLL_SECONDS)

    def _wait_for_ffn_modes(self) -> int | None:
        topology = self._spec.topology
        modes = self._wait_for_entries(
            [
                f"ffn/mode/{rank}"
                for rank in range(topology.attn_size, topology.world_size)
            ],
            stage="mode",
        )
        if len(set(modes)) != 1:
            raise RuntimeError(f"CAM FFN startup modes disagree: {modes}")
        mode = modes[0]
        if mode == "eager":
            return None
        if not mode.startswith("graph:"):
            raise RuntimeError(f"Invalid CAM FFN startup mode: {mode}")
        layer_idx = int(mode.removeprefix("graph:"))
        if layer_idx < 0:
            raise RuntimeError(f"Invalid CAM FFN startup layer: {layer_idx}")
        return layer_idx

    def _run_attention_warmup(self, layer_idx: int) -> None:
        spec = self._spec
        topology = spec.topology
        device = f"npu:{spec.local_rank}"
        hidden = torch.full(
            (STARTUP_WARMUP_TOKEN_COUNT, spec.hidden_size),
            STARTUP_WARMUP_HIDDEN_VALUE,
            dtype=spec.activation_dtype,
            device=device,
        )
        tp_rank = topology.world_rank % spec.tp_size
        expert_ids = [
            ((tp_rank * spec.topk + index) % topology.ffn_size)
            * topology.expert_per_rank
            + ((tp_rank * spec.topk + index) // topology.ffn_size)
            % topology.expert_per_rank
            for index in range(spec.topk)
        ]
        ids = torch.tensor([expert_ids], dtype=torch.int32, device=device)
        weights = torch.full(
            (STARTUP_WARMUP_TOKEN_COUNT, spec.topk),
            1.0 / spec.topk,
            dtype=torch.float32,
            device=device,
        )
        store = self._get_store()
        store.set(f"attn/prepared/{topology.world_rank}", "ready")
        self._wait_for_entries(
            [
                f"ffn/warmup/start/{rank}"
                for rank in range(topology.attn_size, topology.world_size)
            ],
            stage="warmup-start",
        )
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=STARTUP_WARMUP_STAGE,
            seq_len=STARTUP_WARMUP_TOKEN_COUNT,
        )
        context = AFDTransferContext(metadata=metadata)
        logger.info("CAM Attention startup warmup DS rank=%d", topology.world_rank)
        self._connector.send_attn_output(
            hidden, context, topk_ids=ids, topk_weights=weights
        )
        # Omit context and routing kwargs so the connector consumes its FIFO.
        warmup_output = self._connector.recv_ffn_output(
            hidden, ubatch_idx=STARTUP_WARMUP_STAGE
        )
        torch.npu.synchronize()
        del warmup_output
        logger.info("CAM Attention startup warmup CR done rank=%d", topology.world_rank)
        store.set(f"attn/done/{topology.world_rank}", "ready")
        self._wait_for_communication_warmup_done()

    def _wait_for_communication_warmup_done(self) -> None:
        topology = self._spec.topology
        self._wait_for_entries(
            [f"attn/done/{rank}" for rank in range(topology.attn_size)]
            + [
                f"ffn/warmup/done/{rank}"
                for rank in range(topology.attn_size, topology.world_size)
            ],
            stage="warmup-done",
        )

    def _wait_for_ffn_ready(self) -> None:
        store = self._get_store()
        topology = self._spec.topology
        keys = [
            f"ffn/{rank}" for rank in range(topology.attn_size, topology.world_size)
        ]
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            self._check_failures(store, "ready")
            pending = []
            for key in keys:
                if not store.check([key]):
                    pending.append(key)
                    continue
                status = store.get(key).decode()
                if status != "ready":
                    raise RuntimeError(f"CAM FFN startup {key} {status}")
            if not pending:
                self._check_failures(store, "ready")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for CAM FFN ranks: {sorted(pending)}"
                )
            time.sleep(STARTUP_POLL_SECONDS)

    def report_failure(self, error: Exception) -> None:
        """Publish a sticky failure only if the nonce Store already exists."""
        if self._failure is not None:
            return
        self._failure = error
        store = self._store
        if store is None:
            return
        topology = self._spec.topology
        key = (
            f"attn/failure/{topology.world_rank}"
            if topology.role == "attention"
            else f"ffn/{topology.world_rank}"
        )
        try:
            store.set(key, f"failed:{error}")
        except Exception:
            logger.warning("CAM startup failure publication failed: %s", key)

    def start_attention(self) -> None:
        if not self._begin("attention"):
            return
        try:
            if not self._connector.is_initialized:
                self._connector.init_afd_connector()
            self._get_store()
            layer_idx = self._wait_for_ffn_modes()
            if layer_idx is not None:
                self._run_attention_warmup(layer_idx)
            self._wait_for_ffn_ready()
            self._started = True
        except Exception as exc:
            self.report_failure(exc)
            raise
        finally:
            self._starting = False

    def start_ffn(
        self,
        *,
        prepare: Callable[[], FFNStartupPlan],
        consume_warmup: Callable[[int], None],
        capture: Callable[[], None],
        start_receiver: Callable[[], None],
    ) -> None:
        if not self._begin("ffn"):
            return
        try:
            plan = prepare()
            if not self._connector.is_initialized:
                self._connector.init_afd_connector()
            store = self._get_store()
            topology = self._spec.topology
            mode = f"graph:{plan.first_layer_idx}" if plan.use_graph else "eager"
            store.set(f"ffn/mode/{topology.world_rank}", mode)
            observed_layer = self._wait_for_ffn_modes()
            if observed_layer != plan.first_layer_idx:
                raise RuntimeError(
                    f"CAM FFN startup rank={topology.world_rank} mode mismatch: "
                    f"local={mode}, all=graph:{observed_layer}"
                )
            if plan.use_graph:
                self._wait_for_entries(
                    [f"attn/prepared/{rank}" for rank in range(topology.attn_size)],
                    stage="warmup-prepared",
                )
                store.set(f"ffn/warmup/start/{topology.world_rank}", "ready")
                consume_warmup(self._spec.attention_dp_size)
                store.set(f"ffn/warmup/done/{topology.world_rank}", "ready")
                self._wait_for_communication_warmup_done()
                capture()
            start_receiver()
            if self._failure is not None:
                raise RuntimeError("CAM FFN receiver failed during startup") from (
                    self._failure
                )
            key = f"ffn/{topology.world_rank}"
            status = store.compare_set(key, "", "ready").decode()
            if status != "ready":
                raise RuntimeError(f"CAM FFN startup {key} {status}")
            self._started = True
        except Exception as exc:
            self.report_failure(exc)
            raise
        finally:
            self._starting = False


__all__ = ["AsyncCamStartupCoordinator", "AsyncCamStartupSpec", "FFNStartupPlan"]
