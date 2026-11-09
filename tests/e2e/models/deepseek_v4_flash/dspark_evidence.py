# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in DSpark ownership and remote-expert execution evidence.

Layered execution requires reading the device header after each eager return,
adding synchronization. FULL observations use the separate completion-boundary
extension. Disable these fixtures for performance measurements.
"""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.model_executor.ffn.compute import CAM_LAYER_INDEX
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


def snapshot(runner) -> dict:
    is_ffn = isinstance(runner, AsyncNPUFFNModelRunner)
    executor = None
    execution = "uninitialized"
    if is_ffn:
        executor = runner._mega_executor or runner._layered_executor
        if executor is not None:
            execution = "mega" if runner._mega_executor is not None else "layered"
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
        **(
            {
                "ffn_execution": execution,
                "executor_layer_ids": list(executor.layer_ids)
                if executor is not None
                else [],
                "ffn_mode": runner.effective_mode,
                "graph_capture_count": runner._graph_capture_count,
                "graph_replay_count": runner._graph_replay_count,
            }
            if is_ffn
            else {}
        ),
    }


class DSparkEagerEvidence:
    model_runner: AsyncNPUFFNModelRunner | AFDNPUAttentionModelRunnerV2

    def afd_dspark_snapshot(self) -> dict:
        return snapshot(self.model_runner)


def install_observers() -> None:
    send = CAMAsyncAFDConnector.send_ffn_output
    check = AFDNPUFFNWorker.raise_ffn_loop_error_if_any

    @wraps(send)
    def sent(self, ffn_output, context, **kwargs):
        result = send(self, ffn_output, context, **kwargs)
        if self.graph_startup.mode == "eager":
            # Functional observer only: the header copy fences the return
            # before a later receive can reuse this work item's metadata.
            header = context.states.token_nums_rankid_layeridx.cpu().tolist()
            counters = self.__dict__.setdefault("_dspark_returns", {})
            layer = str(header[CAM_LAYER_INDEX])
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

    CAMAsyncAFDConnector.send_ffn_output = sent
    AFDNPUFFNWorker.raise_ffn_loop_error_if_any = checked


install_observers()
