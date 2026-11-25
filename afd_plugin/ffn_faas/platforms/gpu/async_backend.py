# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone GPU FFN composed with ``GPUAsyncFFNModelRunner``.

Imported inside the externally launched F process, after the adapter has
validated the launch. No vLLM Worker, upstream ModelRunner, EngineCore,
scheduler or KV cache is constructed. Local padded expert graph capture,
arrival polling, producer drain and NVSHMEM window teardown remain in
``GPUAsyncFFNModelRunner`` and ``GpuAsyncAFDConnector``.
"""

from __future__ import annotations

import threading
from argparse import Namespace
from typing import Protocol

import torch
from vllm.config import set_current_vllm_config

from afd_plugin.connectors.gpu.async_nvshmem.connector import GpuAsyncAFDConnector
from afd_plugin.ffn_faas.model_utils import parameter_storage_bytes
from afd_plugin.ffn_faas.platforms.gpu.environment import GPUAsyncFFNEnvironment
from afd_plugin.ffn_faas.platforms.gpu.settings import GPUAsyncFFNConfig
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.async_ffn_model_runner import GPUAsyncFFNModelRunner
from afd_plugin.validation import validate_gpu_model_runner_v2_config


class _ExpertsLayerModel(Protocol):
    """Minimal model contract needed to count local expert layers."""

    def get_experts_layer_indices(self) -> tuple[int, ...]: ...


class AsyncGPUFFNBackend:
    """Own weights and async NVSHMEM transport until producer drain finishes."""

    def __init__(self, settings: GPUAsyncFFNConfig, args: Namespace) -> None:
        """Hold CPU-side launch facts; native initialization happens in ``prepare``."""
        self.settings = settings
        self.args = args
        self.environment = GPUAsyncFFNEnvironment(settings)
        self.runner: GPUAsyncFFNModelRunner | None = None
        self.model: torch.nn.Module | None = None
        self._cancel_requested = threading.Event()

    def prepare(self, service: FFNService) -> None:
        """Load experts, capture padded graphs, then join the async AFD world.

        Capture runs before ``initialize_afd_connector`` because that collective
        rendezvous is the only barrier before Attention begins profiling and
        dispatching work items into the symmetric window.
        """
        environment = self.environment
        config = environment.build_config(self.args)
        cudagraph_policy = validate_gpu_model_runner_v2_config(
            config, expected_role="ffn", device_type="cuda"
        )
        environment.initialize_device(config)
        environment.initialize_compute(config)
        with set_current_vllm_config(config):
            self.runner = GPUAsyncFFNModelRunner(
                config,
                environment.device,
                afd_cudagraph_policy=cudagraph_policy,
            )
            connector = self.runner.connector
            if not isinstance(connector, GpuAsyncAFDConnector):
                raise TypeError(
                    f"standalone async GPU FFN needs GpuAsyncAFDConnector, got "
                    f"{type(connector).__name__}"
                )
            self.runner.load_model()
            model: _ExpertsLayerModel = self.runner.get_model()
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"FFN-only model must be a torch module, got {type(model).__name__}"
                )
            self.model = model
            layer_indices = tuple(model.get_experts_layer_indices())
            total_experts = len(layer_indices) * connector.expert_per_rank
            service.loading_progress(total_experts, total_experts)
            parameter_bytes = parameter_storage_bytes(model)
            local_expert_first = self.settings.rank * connector.expert_per_rank
            local_expert_last = local_expert_first + connector.expert_per_rank - 1
            service.phase(
                ServicePhase.WARMUP,
                model_loaded=True,
                parameter_bytes=parameter_bytes,
                local_expert_first=local_expert_first,
                local_expert_last=local_expert_last,
            )
            graph_bytes = self.runner.capture_padded_ffn_graphs()
            service.phase(ServicePhase.CONNECTING, graph_bytes=graph_bytes)
            self.runner.initialize_afd_connector()
            assert connector.window is not None
            service.phase(
                ServicePhase.CONNECTING,
                async_paired=True,
                window_bytes=connector.window.total_bytes,
                rma_peer_count=len(connector.window.remote_peers),
                compute_ep_world_size=environment.ep_group.world_size,
            )

    def start_receiver(self) -> None:
        """Select the GPU once on the receiving thread."""
        self.environment.activate_thread()

    def execute_step(self) -> bool:
        """Poll and serve one arrived work item through the async runner."""
        assert self.runner is not None
        return self.runner.execute_connector_driven_step(
            cancel_requested=self._cancel_requested.is_set
        )

    def cancel_receive(self) -> None:
        """Signal the arrival loop to stop polling and move into producer drain."""
        self._cancel_requested.set()

    def receiver_exited(self) -> None:
        """Drain remaining Attention arrivals and finalize the FFN session."""
        assert self.runner is not None
        # Also covers host-thread InterruptedError exits before start_receiver().
        self.environment.activate_thread()
        self.runner.drain()
        self.runner.session.finish_receiver()

    def close(self) -> None:
        """Release padded graphs, symmetric window, compute groups and weights."""
        assert self.runner is not None
        if not self.runner.connector.receiver_finished:
            raise RuntimeError(
                "Async GPU FFN receiver has not drained; retain resources"
            )
        self.runner.shutdown()
        self.environment.close()
        self.model = None
