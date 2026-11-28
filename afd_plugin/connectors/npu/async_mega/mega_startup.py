# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host startup agreement before Mega creates its HCCL group or window."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed import Store

PREPARATION_ACK_TIMEOUT = timedelta(seconds=30)


def agree_layer_selection(
    store: Store,
    attn_size: int,
    ffn_size: int,
    world_rank: int,
    layer_ids: list[int],
    *,
    timeout: timedelta,
    preparation_error: Exception | None = None,
) -> dict[int, int]:
    """Publish preparation outcomes and adopt one layer map on all A/F ranks.

    FFN joins even when weight conversion failed. Every rank reads and
    acknowledges the outcomes before raising, so the Store owner cannot exit
    before peers have obtained the original failure.
    """
    # Clients may join while other ranks are still loading/converting weights.
    # Preserve the model-loading allowance until every outcome is available.
    store.set_timeout(timeout)
    store.set(
        f"prepare/{world_rank}",
        json.dumps(
            {
                "layers": sorted(layer_ids),
                "error": (
                    f"{type(preparation_error).__name__}: {preparation_error}"
                    if preparation_error is not None
                    else ""
                ),
            }
        ),
    )
    world_size = attn_size + ffn_size
    keys = [f"prepare/{rank}" for rank in range(world_size)]
    store.wait(keys, timeout)
    entries = [json.loads(store.get(key)) for key in keys]
    errors = [
        f"rank {rank}: {entry['error']}"
        for rank, entry in enumerate(entries)
        if entry["error"]
    ]
    error_message = "Mega weight preparation failed: " + "; ".join(errors)
    store.set_timeout(PREPARATION_ACK_TIMEOUT)
    try:
        store.set(f"prepare_read/{world_rank}", "1")
        store.wait(
            [f"prepare_read/{rank}" for rank in range(world_size)],
            PREPARATION_ACK_TIMEOUT,
        )
    except (RuntimeError, TimeoutError, OSError) as exc:
        # A peer may exit immediately after acknowledging. Preserve the
        # preparation error already read instead of hiding it behind Store EOF.
        if errors:
            raise RuntimeError(error_message) from exc
        raise
    if errors:
        raise RuntimeError(error_message) from preparation_error
    configs = [entry["layers"] for entry in entries[attn_size:]]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("All Mega FFN ranks must agree on layer IDs")
    ids = configs[0]
    if not ids or ids[0] < 0 or len(set(ids)) != len(ids):
        raise ValueError("Mega requires unique nonnegative FFN layer IDs")
    return {layer: slot for slot, layer in enumerate(ids)}
