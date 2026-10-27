# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fused FFN execution for ``FastAFDAFDConnector``.

``FastAFDExpertExecutor`` is the FastAFD execution adapter: it loads this FFN
rank's experts into the kernel layout and turns every step plan into a single
persistent kernel launch over all MoE layers and microbatch lanes. The kernel
receives tokens, computes the experts and writes results back by itself; the
host never splits it into per-layer receive/compute/send. The executor holds
no vLLM model, scheduler or graph state, so the FFN runner below only forwards
step plans to it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.utils.mem_utils import DeviceMemoryProfiler

from afd_plugin.compat.gpu.fastafd import (
    fastafd_ops,
    make_weight_descriptors,
    requant_fp8_block_to_per32,
    transform_weights_for_m2n,
)
from afd_plugin.compat.profiler import step_afd_gpu_profiler, stop_afd_gpu_profiler
from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector, M2NShape
from afd_plugin.model_executor.models.qwen3_5 import fastafd_expert_tensor_name
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import DPMetadata

    from afd_plugin.connectors import AFDDPMetadata

logger = logging.getLogger(__name__)

# The expert kernel serves at most this many microbatch lanes per launch.
_MAX_KERNEL_LANES = 4
# How long shutdown waits for an in-flight step; vLLM force-kills the engine
# after its own 10 s shutdown timeout.
_SHUTDOWN_DRAIN_S = 5.0
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _checkpoint_dir(vllm_config: VllmConfig) -> Path:
    model_config = vllm_config.model_config
    if os.path.isdir(model_config.model):
        return Path(model_config.model)
    return Path(
        download_weights_from_hf(
            model_config.model,
            vllm_config.load_config.download_dir,
            allow_patterns=["*.safetensors", "*.json"],
            revision=model_config.revision,
        )
    )


class FastAFDExpertWeights:
    """This FFN rank's routed experts in the FastAFD kernel layout.

    Loads only the local experts' FP8 tensors from the checkpoint, requantizes
    them to per-32 UE8M0, and keeps the converted tensors alive for the TMA
    descriptors that point at them.
    """

    def __init__(self, vllm_config: VllmConfig, shape: M2NShape, ffn_rank: int) -> None:
        self.num_layers = shape.num_layers
        local = shape.num_experts_per_ffn_rank
        experts = range(ffn_rank * local, (ffn_rank + 1) * local)
        checkpoint = _checkpoint_dir(vllm_config)
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.layers: list[
            tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        ] = []
        descriptors = []
        for layer in range(self.num_layers):
            l1, l2 = _load_layer_experts(checkpoint, weight_map, layer, experts)
            self.layers.append((l1, l2))
            descriptors.append(
                make_weight_descriptors(
                    l1,
                    l2,
                    num_experts_per_rank=local,
                    hidden_size=shape.hidden_size,
                    intermediate_size=shape.intermediate_size,
                ).cuda()
            )
        self.descriptors = torch.stack(descriptors).contiguous()
        self.l1_pointers = torch.tensor(
            [l1[0].data_ptr() for l1, _ in self.layers],
            dtype=torch.int64,
            device="cuda",
        )
        self.l1_nbytes = self.layers[0][0][0].numel()


def _load_layer_experts(
    checkpoint: Path,
    weight_map: dict[str, str],
    layer: int,
    experts: range,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """One layer's local experts as kernel-layout ``(L1, L2)`` weight pairs."""
    tensors = _read_tensors(
        checkpoint,
        weight_map,
        [
            fastafd_expert_tensor_name(layer, expert, projection, kind)
            for expert in experts
            for projection in _PROJECTIONS
            for kind in ("weight", "weight_scale_inv")
        ],
    )

    def stack(projection: str, kind: str) -> torch.Tensor:
        return torch.stack(
            [
                tensors[fastafd_expert_tensor_name(layer, expert, projection, kind)]
                for expert in experts
            ]
        ).cuda()

    gate_up = torch.cat(
        [stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1
    )
    gate_up_scales = torch.cat(
        [stack("gate_proj", "weight_scale_inv"), stack("up_proj", "weight_scale_inv")],
        dim=1,
    )
    down = stack("down_proj", "weight")
    down_scales = stack("down_proj", "weight_scale_inv")
    return transform_weights_for_m2n(
        requant_fp8_block_to_per32(gate_up, gate_up_scales),
        requant_fp8_block_to_per32(down, down_scales),
    )


def _read_tensors(
    checkpoint: Path,
    weight_map: dict[str, str],
    names: list[str],
) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for file_name, file_names in by_file.items():
        with safe_open(str(checkpoint / file_name), framework="pt") as file:
            for name in file_names:
                tensors[name] = file.get_tensor(name)
    return tensors


class FastAFDExpertExecutor:
    """Runs one FFN rank's share of every step with the FastAFD expert kernel.

    Contract with whatever drives it (today the FFN runner):

    - ``submit(step_plan)`` launches one persistent kernel for every MoE layer
      and every stage of the plan, on the current stream, and returns. Plans
      must arrive in the Attention ranks' step order. A plan the connector
      cannot serve (more stages than lanes, or a closed connector) raises
      before anything is launched.
    - ``drain(timeout_s)`` stops accepting steps, then waits for the last
      submitted one and reports whether it finished; a step finishes once
      every Attention rank has run every layer and lane of it. A submit that
      already started completes its launch before draining begins.
    - The caller frees the connector and the executor only after a successful
      drain: a step that never finishes still owns the buffers and weights.
    """

    def __init__(self, vllm_config: VllmConfig, connector: FastAFDAFDConnector) -> None:
        fastafd_ops()
        self.connector = connector
        self.weights = FastAFDExpertWeights(
            vllm_config,
            connector.shape,
            connector.role_rank,
        )
        self._last_step: torch.cuda.Event | None = None
        self._stopped = False
        # Shutdown runs on another thread than the step loop: admission, launch
        # and the completion event are published together under this lock.
        self._lock = threading.Lock()

    def submit(self, dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata]) -> None:
        """Launch the expert kernel for one step: every MoE layer and lane."""
        with self._lock:
            if self._stopped:
                raise RuntimeError("FastAFD expert executor is shutting down")
            self._launch(dp_metadata_list)
            self._last_step = torch.cuda.Event()
            self._last_step.record()

    def _launch(self, dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata]) -> None:
        weights = self.weights
        connector = self.connector
        shape = connector.shape
        num_lanes = len(dp_metadata_list)
        if not 0 < num_lanes <= len(connector.lanes):
            raise RuntimeError(
                f"FastAFD step plan has {num_lanes} stages, the connector "
                f"{len(connector.lanes)} lanes",
            )
        lanes = connector.lanes[:num_lanes]
        extra_lanes = [
            lanes[i] if i < num_lanes else None for i in range(1, _MAX_KERNEL_LANES)
        ]
        expected_tokens = max(
            max(int(count) for count in metadata.num_tokens_across_dp_cpu)
            for metadata in dp_metadata_list.values()
        )
        fastafd_ops().mega_moe_m2n_eg(
            weights.descriptors,
            weights.l1_pointers,
            weights.num_layers,
            shape.num_experts_per_ffn_rank,
            shape.hidden_size,
            shape.intermediate_size,
            weights.l1_nbytes,
            True,
            None,
            lanes[0].buffer,
            *(lane.buffer if lane is not None else None for lane in extra_lanes),
            lanes[0].peer_ptrs,
            *((lane or lanes[0]).peer_ptrs for lane in extra_lanes),
            connector.union_rank,
            shape.num_attention_ranks,
            shape.num_ffn_ranks,
            lanes[0].max_tokens_per_rank,
            shape.num_experts,
            shape.top_k,
            max(1, expected_tokens),
            None,
            True,
            0,
            0,
            None,
        )

    def drain(self, timeout_s: float) -> bool:
        """Stop accepting steps; wait up to ``timeout_s`` for the last one."""
        with self._lock:
            self._stopped = True
            last_step = self._last_step
        if last_step is None:
            return True
        deadline = time.monotonic() + timeout_s
        while not last_step.query():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True


class FastAFDFFNModelRunner(GPUFFNModelRunner):
    """Connector-driven FFN runner that forwards step plans to the executor."""

    connector: FastAFDAFDConnector

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.executor: FastAFDExpertExecutor | None = None

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load this rank's routed experts straight into the kernel layout."""
        if load_dummy_weights:
            raise ValueError("FastAFDAFDConnector needs checkpoint expert weights")
        with DeviceMemoryProfiler() as profiler:
            self.executor = FastAFDExpertExecutor(self.vllm_config, self.connector)
        self.model_memory_usage = profiler.consumed_memory

    def execute_model(
        self,
        scheduler_output: Any = None,
        intermediate_tensors: Any = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
    ) -> None:
        step_afd_gpu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("FastAFDFFNModelRunner requires dp_metadata_list")
        # Attention records its kernels into a graph without running them, so
        # a capture step has no expert work to serve.
        if not is_graph_capturing:
            self._submit(dp_metadata_list)

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if dp_metadata_list is None:
            raise RuntimeError("FastAFDFFNModelRunner requires dp_metadata_list")
        if is_warmup:
            self._submit(dp_metadata_list)
        return 0

    def _submit(self, dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata]) -> None:
        if self.executor is None:
            raise RuntimeError("FastAFD FFN step arrived before load_model")
        self.executor.submit(dp_metadata_list)

    def shutdown(self) -> None:
        executor = self.executor
        if executor is not None and not executor.drain(_SHUTDOWN_DRAIN_S):
            # A peer stopped mid-step, so the kernel is still running and owns
            # the lane buffers and weights. Leave them to process exit.
            logger.warning(
                "FastAFD expert step did not finish within %.0f s; "
                "keeping its buffers until the process exits",
                _SHUTDOWN_DRAIN_S,
            )
            stop_afd_gpu_profiler(self.prof)
            return
        # The base shutdown closes the connector; the experts go last.
        super().shutdown()
        self.executor = None


__all__ = ["FastAFDExpertExecutor", "FastAFDExpertWeights", "FastAFDFFNModelRunner"]
