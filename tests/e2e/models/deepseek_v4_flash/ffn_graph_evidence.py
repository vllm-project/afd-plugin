# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Test-only observations at already fenced Async CAM completion boundaries."""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path

import torch

from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker


def snapshot(runner) -> dict:
    connector = runner.connector
    result = {
        "observed_ns": time.monotonic_ns(),
        "runner": type(runner).__name__,
        "rank": connector.world_rank,
        "transport_live": connector.is_initialized,
    }
    if isinstance(runner, AsyncNPUFFNModelRunner):
        result.update(
            mode=runner.effective_mode,
            requested=runner.requested_mode,
            capture_count=runner._graph_capture_count
            if runner._graph is not None
            else 0,
            replay_count=runner._graph_replay_count if runner._graph is not None else 0,
            receiver_exited=runner._graph_receiver_exited,
            step_pending=runner._graph_pending if runner._graph is not None else False,
            shutdown=runner._is_shutdown,
            model_released=runner.model is None,
            completion=getattr(connector, "_e2e_completion", {}),
            layer_ids=(
                sorted(runner.model.get_experts_layer_indices())
                if runner.model is not None
                else []
            ),
        )
    else:
        result.update(attention=getattr(runner, "_e2e_attention", {}))
    return result


def record(runner, phase: str) -> None:
    directory = Path(os.environ["AFD_FFN_GRAPH_EVIDENCE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    event = {"phase": phase, **snapshot(runner)}
    with (directory / f"rank-{event['rank']}.jsonl").open("a") as output:
        output.write(json.dumps(event) + "\n")


class FFNGraphEvidence:
    """Use vLLM's existing collective_rpc API to snapshot host-only counters."""

    model_runner: AsyncNPUFFNModelRunner | AFDNPUAttentionModelRunnerV2

    def afd_ffn_graph_snapshot(self) -> dict:
        return snapshot(self.model_runner)


def install_observers() -> None:
    complete = AsyncNPUFFNModelRunner._complete_graph
    send_eager = CAMAsyncAFDConnector.send_ffn_output
    execute = AFDNPUAttentionModelRunnerV2.execute_model
    shutdown_attention = AFDNPUAttentionModelRunnerV2.shutdown
    shutdown_ffn = AsyncNPUFFNModelRunner.shutdown
    health_check = AFDNPUFFNWorker.raise_ffn_loop_error_if_any
    stop_ffn = AFDNPUFFNWorker.stop_ffn_server_loop

    def observe(connector, context):
        # Only called after DR/FFN/CS completion on the execution thread.
        # RPC snapshots read this host copy without touching device buffers.
        state = context.states
        header = state.token_nums_rankid_layeridx.cpu().tolist()
        rows = int(state.group_list.sum().item())
        previous = getattr(connector, "_e2e_completion", {})
        layers = dict(previous.get("layers", {}))
        layer = str(header[2])
        layers[layer] = layers.get(layer, 0) + 1
        connector._e2e_completion = {
            "transactions": previous.get("transactions", 0) + 1,
            "layers": layers,
            "last_header": header[:5],
            "max_rows": max(previous.get("max_rows", 0), rows),
            "zero_rows": previous.get("zero_rows", 0) + int(rows == 0),
        }

    @wraps(complete)
    def completed(self):
        result = complete(self)
        if result:
            observe(self.connector, self._graph_payload.context)
        return result

    @wraps(send_eager)
    def sent(self, output, context, **kwargs):
        result = send_eager(self, output, context, **kwargs)
        if self.graph_startup.mode == "eager":
            # Functional coverage only. This fixture adds an eager event fence
            # for observations and must not be used for performance reporting.
            torch.npu.current_stream().synchronize()
            observe(self, context)
        return result

    @wraps(execute)
    def executed(self, scheduler_output, *args, **kwargs):
        result = execute(self, scheduler_output, *args, **kwargs)
        phase = (
            "profile"
            if kwargs.get("is_profile")
            else "dummy"
            if kwargs.get("dummy_run")
            else "live"
        )
        previous = getattr(self, "_e2e_attention", {})
        counts = dict(previous.get("phases", {}))
        counts[phase] = counts.get(phase, 0) + 1
        token_counts = dict(previous.get("scheduled_tokens", {}))
        if phase == "live":
            size = str(scheduler_output.total_num_scheduled_tokens)
            token_counts[size] = token_counts.get(size, 0) + 1
        self._e2e_attention = {"phases": counts, "scheduled_tokens": token_counts}
        return result

    @wraps(shutdown_attention)
    def attention_closed(self):
        record(self, "shutdown_enter")
        result = shutdown_attention(self)
        record(self, "shutdown_complete")
        return result

    @wraps(shutdown_ffn)
    def ffn_closed(self):
        record(self, "shutdown_enter")
        result = shutdown_ffn(self)
        record(self, "shutdown_complete")
        return result

    @wraps(health_check)
    def checked(self):
        result = health_check(self)
        data = snapshot(self.model_runner)
        directory = Path(os.environ["AFD_FFN_GRAPH_EVIDENCE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"latest-ffn-{data['rank']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return result

    @wraps(stop_ffn)
    def ffn_stop_requested(self):
        record(self.model_runner, "stop_rpc_enter")
        result = stop_ffn(self)
        record(self.model_runner, "stop_rpc_complete")
        return result

    AsyncNPUFFNModelRunner._complete_graph = completed
    CAMAsyncAFDConnector.send_ffn_output = sent
    AFDNPUAttentionModelRunnerV2.execute_model = executed
    AFDNPUAttentionModelRunnerV2.shutdown = attention_closed
    AsyncNPUFFNModelRunner.shutdown = ffn_closed
    AFDNPUFFNWorker.raise_ffn_loop_error_if_any = checked
    AFDNPUFFNWorker.stop_ffn_server_loop = ffn_stop_requested


install_observers()
