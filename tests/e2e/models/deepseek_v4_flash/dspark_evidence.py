# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in eager DSpark evidence without adding device reads or fences.

Legacy CAM already exposes a host layer ID in its FFN work item. Count the
submitted returns at that seam, and read the counts only after HTTP requests
complete. This fixture is functional evidence, not a performance benchmark.
"""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


def snapshot(runner) -> dict:
    is_ffn = isinstance(runner, AsyncNPUFFNModelRunner)
    draft = runner.draft_model if is_ffn else runner.speculator.model
    if draft is None:
        return {
            "rank": runner.connector.world_rank,
            "role": "ffn",
            "released": True,
        }
    names = tuple(name for name, _ in draft.named_parameters())
    kv_groups = []
    if not is_ffn:
        tables = runner.speculator.block_tables
        kv_groups = [
            {
                "group": gid,
                "physical_block_size": tables.block_sizes[gid],
                "kernel_block_size": tables.kernel_block_sizes[gid],
            }
            for gid in runner.speculator.draft_kv_cache_group_ids
        ]
    return {
        "rank": runner.connector.world_rank,
        "role": "ffn" if is_ffn else "attention",
        "draft_class": type(draft).__name__,
        "draft_wire_layer_ids": draft.get_experts_layer_indices(),
        "owns_routed_experts": any(".experts." in name for name in names),
        "owns_shared_experts": any(".shared_experts." in name for name in names),
        "owns_attention": any(".self_attn." in name for name in names),
        "submitted_returns": dict(runner.connector.__dict__.get("_dspark_returns", {})),
        "draft_kv_groups": kv_groups,
    }


class DSparkEagerEvidence:
    model_runner: AsyncNPUFFNModelRunner | AFDNPUAttentionModelRunnerV2

    def afd_dspark_snapshot(self) -> dict:
        return snapshot(self.model_runner)


def install_observers() -> None:
    send = CAMAsyncAFDConnector.send_ffn_work_item_output
    check = AFDNPUFFNWorker.raise_ffn_loop_error_if_any

    @wraps(send)
    def sent(self, work_item, ffn_output):
        result = send(self, work_item, ffn_output)
        counters = self.__dict__.setdefault("_dspark_returns", {})
        layer = str(work_item.layer_idx)
        counters[layer] = counters.get(layer, 0) + 1
        return result

    @wraps(check)
    def checked(self):
        result = check(self)
        record = snapshot(self.model_runner)
        directory = Path(os.environ["AFD_DSPARK_EVIDENCE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank-{record['rank']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)
        return result

    CAMAsyncAFDConnector.send_ffn_work_item_output = sent
    AFDNPUFFNWorker.raise_ffn_loop_error_if_any = checked


install_observers()
