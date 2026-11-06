# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Functional-only DSpark graph observations, one replay per completion fence.

The production runtime submits bounded groups of transactions. This extension
reduces that group to one so each layer's completed device data can be inspected
before the next receive overwrites it. It also computes a Layered eager reference.
Disable the extension for ordinary-group liveness and every performance run.
"""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path

import torch

from afd_plugin.model_executor.npu.async_cam_layered import (
    CAM_LAYER_INDEX,
    AsyncCAMLayeredExecutor,
)
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from tests.e2e.models.deepseek_v4_flash.dspark_evidence import DSparkEagerEvidence


class DSparkFFNGraphEvidence(DSparkEagerEvidence):
    pass


class GraphComparison:
    def __init__(self, runner):
        self.runner = runner
        target, _ = runner.model.get_async_cam_layered_weights()
        draft, _ = runner.draft_model.get_async_cam_layered_weights()
        self.target = AsyncCAMLayeredExecutor(target)
        self.draft = AsyncCAMLayeredExecutor(draft)
        self.records = {}
        self.directory = Path(os.environ["AFD_DSPARK_EVIDENCE_DIR"]) / "graph-proof"
        self.directory.mkdir(parents=True, exist_ok=True)

    def observe(self):
        runner = self.runner
        payload = runner._graph_payload
        state = payload.context.states
        header = state.token_nums_rankid_layeridx.cpu()
        wire = int(header[CAM_LAYER_INDEX])
        counters = runner.connector.__dict__.setdefault("_dspark_returns", {})
        counters[str(wire)] = counters.get(str(wire), 0) + 1
        if wire in self.records or not (self.directory / "enable").exists():
            return
        counts = state.group_list.cpu()
        rows = int(counts.sum())
        if not rows:
            return
        actual = runner._graph_output[:rows].cpu()
        expected = (self.draft if wire in self.draft.layer_ids else self.target)(
            payload.hidden_states,
            state.dynamic_scales,
            state.group_list,
            state.token_nums_rankid_layeridx,
        )[:rows].cpu()
        rank = runner.connector.world_rank
        self.records[wire] = {
            "layer": wire,
            "rows": rows,
            "bitwise_equal": torch.equal(actual, expected),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
        torch.save(
            {
                "header": header,
                "counts": counts,
                "input": payload.hidden_states[:rows].cpu(),
                "scales": state.dynamic_scales[:rows].cpu(),
                "graph": actual,
                "layered_eager": expected,
            },
            self.directory / f"rank-{rank}-layer-{wire}.pt",
        )
        path = self.directory / f"rank-{rank}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "mode": runner.effective_mode,
                    "capture_count": runner._graph_capture_count,
                    "replay_count": runner._graph_replay_count,
                    "submission_unit": 1,
                    "complete": set(self.records) == set(runner.layer_ids),
                    "records": [self.records[k] for k in sorted(self.records)],
                },
                indent=2,
            )
        )
        temporary.replace(path)


def install_observers() -> None:
    submit = AsyncNPUFFNModelRunner._submit_graph
    complete = AsyncNPUFFNModelRunner._complete_graph

    @wraps(submit)
    def submitted(self, count=1):
        return submit(self, 1)

    @wraps(complete)
    def completed(self):
        result = complete(self)
        if result:
            if "_dspark_graph_comparison" not in self.__dict__:
                self._dspark_graph_comparison = GraphComparison(self)
            self._dspark_graph_comparison.observe()
        return result

    AsyncNPUFFNModelRunner._submit_graph = submitted
    AsyncNPUFFNModelRunner._complete_graph = completed


install_observers()
