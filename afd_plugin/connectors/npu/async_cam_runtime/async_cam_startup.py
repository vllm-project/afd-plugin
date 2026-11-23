# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-only Async CAM graph startup protocol, before and after HCCL setup."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch.distributed import Store

GRAPH_STARTUP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.02
BF16_PUSH_SIGNATURE_PREFIX = "bf16_push:"
# GetMaxWindowSize() in ascend_kernels/utils/op_host/ops_log.h uses MiB.
_DEFAULT_CAM_WINDOW_MIB = 408
_MAX_CAM_WINDOW_MIB = 32 * 1024


def _bf16_push_window_mib(configured_mib: int | None) -> int:
    raw = os.environ.get("HCCL_BUFFSIZE")
    window_mib = _DEFAULT_CAM_WINDOW_MIB
    if raw is not None:
        value = raw.strip(" \t\r\n\v\f").removeprefix("+")
        if not (value.isascii() and value.isdecimal()):
            raise ValueError(
                f"HCCL_BUFFSIZE must be a positive integer in MiB, got {raw!r}"
            )
        window_mib = int(value)
        if not 1 <= window_mib <= _MAX_CAM_WINDOW_MIB:
            raise ValueError(f"HCCL_BUFFSIZE must be in [1, {_MAX_CAM_WINDOW_MIB}] MiB")
    if configured_mib is not None and configured_mib != window_mib:
        raise ValueError(
            f"hccl_buffer_size={configured_mib} MiB must match the tiler's "
            f"HCCL_BUFFSIZE={window_mib} MiB for BF16 Push"
        )
    return window_mib


@dataclass(frozen=True)
class FFNGraphPlan:
    requested: Literal["eager", "full"] = "eager"
    executor_error: str = ""
    signature: str = ""
    layer_ids: tuple[int, ...] = ()


def resolve_ffn_mode(plans: list[FFNGraphPlan]) -> str:
    if not plans or len({plan.requested for plan in plans}) != 1:
        raise ValueError("All FFN ranks must request the same execution mode")
    errors = [plan.executor_error for plan in plans if plan.executor_error]
    if errors:
        raise ValueError(
            "Async FFN layered executor is unsupported: "
            + "; ".join(dict.fromkeys(errors))
        )
    mode = plans[0].requested
    if mode not in ("eager", "full"):
        raise ValueError("Unknown Async FFN execution mode")
    execution = "FULL" if mode == "full" else "eager"
    if len({plan.signature for plan in plans}) > 1:
        raise ValueError(
            f"Async FFN {execution} is unsupported: "
            "FFN executor signatures differ across ranks"
        )
    if not plans[0].signature or not plans[0].layer_ids:
        raise ValueError(f"Async FFN {execution} requires a loaded executor signature")
    if len({plan.layer_ids for plan in plans}) != 1:
        raise ValueError("Async FFN layer tables differ across ranks")
    return mode


class AsyncCAMGraphStartup:
    """Own Store negotiation and barriers, never device or service resources."""

    def __init__(self, world_rank: int, attn_size: int, world_size: int) -> None:
        self.world_rank = world_rank
        self.attn_size = attn_size
        self.world_size = world_size
        self.plan = FFNGraphPlan()
        self.mode = "eager"
        self.warmup_layer_id: int | None = None
        self.session_epoch = 0
        self._store: Store | None = None
        self._prefix = ""
        self._shared_errors = False

    def initialize(
        self,
        store: Store,
        transport_signature: str,
        *,
        hccl_buffer_size_mb: int | None = None,
    ) -> int | None:
        self._store = store
        if self.world_rank == 0:
            store.set("ffn_graph/nonce", uuid.uuid4().hex)
        nonce = store.get("ffn_graph/nonce").decode()
        self.session_epoch = int(nonce[:15], 16) or 1
        self._prefix = f"ffn_graph/{nonce}/"
        return self._negotiate(
            self.plan if self.world_rank >= self.attn_size else None,
            transport_signature,
            hccl_buffer_size_mb,
        )

    def _keys(self, stage: str, ranks: range) -> list[str]:
        return [f"{self._prefix}{stage}/{rank}" for rank in ranks]

    def _publish(self, stage: str, value: str = "ready") -> None:
        assert self._store is not None
        if stage == "error":
            # Publish the shared signal first so new peers still fail if the
            # compatibility publication below loses its Store connection.
            self._store.set(f"{self._prefix}error", value)
        self._store.set(f"{self._prefix}{stage}/{self.world_rank}", value)

    def check_failure(self) -> None:
        assert self._store is not None
        signal = f"{self._prefix}error"
        if self._store.check([signal]):
            raise RuntimeError(self._store.get(signal).decode())
        if self._shared_errors:
            return
        for key in self._keys("error", range(self.world_size)):
            if self._store.check([key]):
                raise RuntimeError(self._store.get(key).decode())

    def fail(self, error: Exception) -> None:
        if self._store is not None:
            self._publish("error", str(error))

    def _wait(self, keys: list[str]) -> None:
        assert self._store is not None
        deadline = time.monotonic() + GRAPH_STARTUP_TIMEOUT_SECONDS
        while True:
            self.check_failure()
            if self._store.check(keys):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Async FFN graph handshake timed out; retain resources"
                )
            time.sleep(POLL_SECONDS)

    def phase(self, stage: str) -> None:
        self._publish(stage)
        self._wait(self._keys(stage, range(self.world_size)))

    def _negotiate(
        self,
        plan: FFNGraphPlan | None,
        transport_signature: str,
        hccl_buffer_size_mb: int | None = None,
    ) -> int | None:
        assert self._store is not None
        try:
            if transport_signature.startswith(BF16_PUSH_SIGNATURE_PREFIX):
                hccl_buffer_size_mb = _bf16_push_window_mib(hccl_buffer_size_mb)
                transport_signature += f":window_mib={hccl_buffer_size_mb}"
            self._publish(
                "plan",
                json.dumps(
                    {
                        "transport": transport_signature,
                        "shared_errors": True,
                        "ffn": asdict(plan) if plan else None,
                    }
                ),
            )
            keys = self._keys("plan", range(self.world_size))
            self._wait(keys)
            entries = [json.loads(self._store.get(key)) for key in keys]
            # Old peers publish only per-rank errors; keep their original check path.
            self._shared_errors = all(
                entry.get("shared_errors", False) for entry in entries
            )
            plans = []
            for entry in entries[self.attn_size :]:
                fields = entry["ffn"]
                fields["layer_ids"] = tuple(fields["layer_ids"])
                plans.append(FFNGraphPlan(**fields))
            self.mode = resolve_ffn_mode(plans)
            ffn_plan = plans[0]
            validate_transport = self.mode == "full" or any(
                entry["transport"].startswith(("pull_v1:", BF16_PUSH_SIGNATURE_PREFIX))
                for entry in entries
            )
            signatures = {entry["transport"] for entry in entries}
            if validate_transport and len(signatures) != 1:
                raise ValueError(
                    f"Async CAM transport signatures differ: {sorted(signatures)}"
                )
            if self.mode == "full":
                self.warmup_layer_id = ffn_plan.layer_ids[0]
            return hccl_buffer_size_mb
        except Exception as exc:
            self.fail(exc)
            raise

    def service_ready(self) -> None:
        self.check_failure()
        self._publish("service")

    def wait_service_ready(self) -> None:
        self._wait(self._keys("service", range(self.attn_size, self.world_size)))
