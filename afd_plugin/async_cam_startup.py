# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-only Async CAM graph startup protocol, before and after HCCL setup."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch.distributed import Store

GRAPH_STARTUP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.02


@dataclass(frozen=True)
class FFNGraphPlan:
    requested: Literal["eager", "full"] = "eager"
    graph_error: str = ""
    signature: str = ""
    layer_ids: tuple[int, ...] = ()


def resolve_ffn_mode(plans: list[FFNGraphPlan]) -> str:
    if not plans or len({plan.requested for plan in plans}) != 1:
        raise ValueError("All FFN ranks must request the same execution mode")
    if plans[0].requested == "eager":
        return "eager"
    if plans[0].requested != "full":
        raise ValueError("Unknown Async FFN execution mode")
    errors = [plan.graph_error for plan in plans if plan.graph_error]
    if len({plan.signature for plan in plans if not plan.graph_error}) > 1:
        errors.append("FFN graph signatures differ across ranks")
    if errors:
        reason = "; ".join(dict.fromkeys(errors))
        raise ValueError(f"Async FFN FULL is unsupported: {reason}")
    if not plans[0].signature or not plans[0].layer_ids:
        raise ValueError("Async FFN FULL requires a loaded executor signature")
    if len({plan.layer_ids for plan in plans}) != 1:
        raise ValueError("Async FFN layer tables differ across ranks")
    return "full"


class AsyncCAMGraphStartup:
    """Own Store negotiation and barriers, never device or service resources."""

    def __init__(self, world_rank: int, attn_size: int, world_size: int) -> None:
        self.world_rank = world_rank
        self.attn_size = attn_size
        self.world_size = world_size
        self.plan = FFNGraphPlan()
        self.mode = "eager"
        self.warmup_layer_id: int | None = None
        self._store: Store | None = None
        self._prefix = ""

    def initialize(self, store: Store, transport_signature: str) -> None:
        self._store = store
        if self.world_rank == 0:
            store.set("ffn_graph/nonce", uuid.uuid4().hex)
        self._prefix = f"ffn_graph/{store.get('ffn_graph/nonce').decode()}/"
        self._negotiate(
            self.plan if self.world_rank >= self.attn_size else None,
            transport_signature,
        )

    def _keys(self, stage: str, ranks: range) -> list[str]:
        return [f"{self._prefix}{stage}/{rank}" for rank in ranks]

    def _publish(self, stage: str, value: str = "ready") -> None:
        assert self._store is not None
        self._store.set(f"{self._prefix}{stage}/{self.world_rank}", value)

    def check_failure(self) -> None:
        assert self._store is not None
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

    def _negotiate(self, plan: FFNGraphPlan | None, transport_signature: str) -> None:
        assert self._store is not None
        try:
            self._publish(
                "plan",
                json.dumps(
                    {
                        "transport": transport_signature,
                        "ffn": asdict(plan) if plan else None,
                    }
                ),
            )
            keys = self._keys("plan", range(self.world_size))
            self._wait(keys)
            entries = [json.loads(self._store.get(key)) for key in keys]
            plans = []
            for entry in entries[self.attn_size :]:
                fields = entry["ffn"]
                fields["layer_ids"] = tuple(fields["layer_ids"])
                plans.append(FFNGraphPlan(**fields))
            self.mode = resolve_ffn_mode(plans)
            if (
                plans[0].requested == "full"
                and len({entry["transport"] for entry in entries}) != 1
            ):
                raise ValueError("Async CAM graph transport signatures differ")
            if self.mode == "full":
                self.warmup_layer_id = plans[0].layer_ids[0]
        except Exception as exc:
            self.fail(exc)
            raise

    def service_ready(self) -> None:
        self.check_failure()
        self._publish("service")

    def wait_service_ready(self) -> None:
        self._wait(self._keys("service", range(self.attn_size, self.world_size)))
