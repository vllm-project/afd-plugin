# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Push/Pull CAM transactions, communication state and FFN lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

from afd_plugin.connectors.base import FFNExecutionSession

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from afd_plugin.connectors.metadata import AFDTransferContext
    from afd_plugin.connectors.npu.async_cam import AFDAsyncTopology
    from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
        AsyncCAMGraphStartup,
    )

CAM_COMM_ID = 0
PULL_COMM_ID = 0x41464401
PULL_FAILURE_INDEX = 24
PULL_SOURCE_INDEX = 32
PULL_IDLE_BUDGET_INDEX = 64
PULL_IDLE_SCANS = 8


class PushCAMSession(FFNExecutionSession):
    """Push owns per-stage routing queues; FFN uses separate receive/send calls."""

    comm_id = CAM_COMM_ID
    signature_prefix = ""
    allow_row_slicing = False

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.comm_args: Tensor | None = None
        self.placeholder: Tensor | None = None
        self.pending: dict[int, list[tuple[AFDTransferContext, Tensor, Tensor]]] = {}

    @property
    def has_pending_attention(self) -> bool:
        return bool(self.pending)

    def initialize(self, device: str, group_name: str) -> None:
        import torch

        self.comm_args = torch.empty((1,), dtype=torch.float16, device=device)
        self.placeholder = torch.empty((1,), dtype=self.dtype, device=device)

    def pending_for(
        self, stage: int
    ) -> tuple[AFDTransferContext, Tensor, Tensor] | None:
        queue = self.pending.get(stage)
        return queue[0] if queue else None

    def dispatch(
        self,
        payload: tuple[AFDTransferContext, Tensor, Tensor],
        **args: Tensor | int | str | None,
    ) -> None:
        import torch

        torch.ops.afd_ascend.afd_async_dispatch_send(**args)
        self.pending.setdefault(payload[0].metadata.stage_idx, []).append(payload)

    def combine(self, stage: int, **args: Tensor | int | str | None) -> Tensor:
        import torch

        output = torch.ops.afd_ascend.afd_async_combine_recv(**args)
        queue = self.pending[stage]
        queue.pop(0)
        if not queue:
            del self.pending[stage]
        return output

    def attention_step(self) -> AbstractContextManager[None]:
        return nullcontext()

    def prepare_close(self, initialized: bool) -> None:
        pass

    def release(self) -> None:
        self.comm_args = self.placeholder = None
        self.pending.clear()


class PullCAMSession(FFNExecutionSession):
    """One in-flight Attention window; FFN runs the native fused transaction."""

    requires_drain = True
    comm_id = PULL_COMM_ID
    signature_prefix = "pull_v1:"
    allow_row_slicing = True

    def __init__(
        self,
        startup: AsyncCAMGraphStartup,
        topology: AFDAsyncTopology,
        *,
        hidden_size: int,
        topk: int,
        max_num_batched_tokens: int,
        tp_size: int,
        receive_capacity: int,
    ) -> None:
        self.startup = startup
        self.max_num_batched_tokens = max_num_batched_tokens
        self.topk = topk
        self.tp_size = tp_size
        self.receive_capacity = receive_capacity
        self._init_config = [
            max_num_batched_tokens // tp_size,
            hidden_size,
            topk,
            topology.ffn_size,
            topology.attn_size,
            topology.expert_per_rank,
            topology.world_rank,
            tp_size,
        ]
        self.comm_args: Tensor | None = None
        self._pending: tuple[AFDTransferContext, Tensor, Tensor] | None = None
        self._defer_status = False
        self.receiver_finished = False
        self._idle_enabled = False
        self._drained = False

    @property
    def has_pending_attention(self) -> bool:
        return self._pending is not None

    def initialize(self, device: str, group_name: str) -> None:
        import torch

        self.comm_args = torch.ops.afd_ascend.afd_mega_pull_init(
            torch.empty(1, dtype=torch.bfloat16, device=device),
            self._init_config,
            list(self.startup.plan.layer_ids),
            self.startup.session_epoch,
            group_name,
        )
        torch.npu.synchronize()
        self.check_pull_status()
        self.startup.phase("pull-initialized")

    def pending_for(
        self, stage: int
    ) -> tuple[AFDTransferContext, Tensor, Tensor] | None:
        if self._pending is not None and self._pending[0].metadata.stage_idx == stage:
            return self._pending
        return None

    def dispatch(
        self,
        payload: tuple[AFDTransferContext, Tensor, Tensor],
        **args: Tensor | int | str | None,
    ) -> None:
        import torch

        if self._pending is not None:
            raise RuntimeError("Pull window is pending; finish Combine before Dispatch")
        # A launch can enqueue work before raising; retain ownership on failure.
        self._pending = payload
        torch.ops.afd_ascend.afd_mega_dispatch_send(**args)

    def combine(self, stage: int, **args: Tensor | int | str | None) -> Tensor:
        import torch

        output = torch.ops.afd_ascend.afd_mega_combine_recv(
            **args, max_seq_len=self.max_num_batched_tokens, tp_size=self.tp_size
        )
        if not self._defer_status:
            self.check_pull_status()
        self._pending = None
        return output

    @contextmanager
    def attention_step(self) -> Iterator[None]:
        previous = self._defer_status
        self._defer_status = True
        try:
            yield
            if not previous:
                self.check_pull_status()
        finally:
            self._defer_status = previous

    def check_pull_status(self) -> None:
        import torch

        assert self.comm_args is not None
        # The model runner batches this fence outside the DBO receive hook.
        status = int(self.comm_args.view(torch.int64)[PULL_FAILURE_INDEX].item())
        if status:
            raise RuntimeError(f"AFD pull transaction failed with status {status}")

    def enable_pull_idle_returns(self) -> None:
        import torch

        assert self.comm_args is not None
        self.comm_args.view(torch.int64)[PULL_IDLE_BUDGET_INDEX].fill_(PULL_IDLE_SCANS)

    def pull_has_active_source(self) -> bool:
        import torch

        assert self.comm_args is not None
        return int(self.comm_args.view(torch.int64)[PULL_SOURCE_INDEX].item()) >= 0

    def prepare_close(self, initialized: bool) -> None:
        if self.has_pending_attention:
            raise RuntimeError(
                "Pull dispatch is pending; retain communication resources"
            )
        if initialized:
            if self.startup.world_rank < self.startup.attn_size:
                self.close_attention()
            else:
                self.validate_shutdown()

    def release(self) -> None:
        self.comm_args = None
        self._pending = None

    def keep_serving(self, shutdown_requested: bool) -> bool:
        startup = self.startup
        assert startup._store is not None
        startup.check_failure()
        return not startup._store.check(
            startup._keys("pull_quiet", range(startup.attn_size))
        )

    def before_first_step(self) -> None:
        if not self._idle_enabled:
            self.enable_pull_idle_returns()
            self._idle_enabled = True

    def after_step(self) -> None:
        self.check_pull_status()

    def drain(self, run_one: Callable[[], None]) -> None:
        rounds = (
            self.max_num_batched_tokens * self.topk + self.receive_capacity - 1
        ) // self.receive_capacity
        for _ in range(rounds):
            if not self.pull_has_active_source():
                self._drained = True
                return
            run_one()
        if self.pull_has_active_source():
            raise RuntimeError("Pull source did not drain; retain resources")
        self._drained = True

    def finish_receiver(self) -> None:
        if self.receiver_finished:
            return
        if not self._drained:
            raise RuntimeError("Pull receiver has not drained; retain resources")
        startup = self.startup
        startup._wait(startup._keys("pull_quiet", range(startup.attn_size)))
        startup._publish("pull_stopped")
        self.receiver_finished = True

    def close_attention(self) -> None:
        startup = self.startup
        startup._publish("pull_quiet")
        startup._wait(
            startup._keys("pull_stopped", range(startup.attn_size, startup.world_size))
        )

    def validate_shutdown(self) -> None:
        if not self.receiver_finished or self.pull_has_active_source():
            raise RuntimeError("Pull receiver has not drained; retain resources")
        self.check_pull_status()
