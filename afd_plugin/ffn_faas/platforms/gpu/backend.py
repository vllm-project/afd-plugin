# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-only routed experts composed with the existing GPU M2N runtime.

Imported inside the externally launched F process, after the adapter has
validated the launch. No vLLM Worker, upstream ModelRunner, EngineCore,
scheduler, KV cache or complete CausalLM is constructed. Step, graph, DBO and
transport logic are not re-implemented: they belong to
``GPUFFNRoutedExpertsModelRunner`` and ``DeepEPM2NAFDConnector``.
"""

from __future__ import annotations

import threading

from vllm.config import set_current_vllm_config

from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector
from afd_plugin.ffn_faas.model_utils import parameter_storage_bytes
from afd_plugin.ffn_faas.platforms.gpu.environment import GPUFFNEnvironment
from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNConfig
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
    GPUFFNRoutedExpertsModelRunner,
    RoutedExpertsModel,
)


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/backend.py::CAMFFNBackend.
# Reuse: PARTIAL - the same FFNBackend lifecycle (prepare phases, receiver
#   thread activation, receiver-exit proof, release order runtime ->
#   compute groups -> weights); the native pieces differ (NCCL/DeepEP M2N vs
#   HCCL/CAM, lazy graph capture vs capture inside prepare) and GPU adds
#   cancel_receive because the M2N control plane can end an idle receive.
#   REFACTOR-CANDIDATE(R8): a backend base owning the phase sequence and the
#   release-after-proof ordering, with platform hooks.
class DeepEPM2NFFNBackend:
    """Keep weights and communication owners until receiver exit is proven."""

    def __init__(self, settings: GPUFFNConfig) -> None:
        """Hold the CPU-side launch facts; native work starts in ``prepare``."""
        self.settings = settings
        self.environment = GPUFFNEnvironment(settings)
        self.runner: GPUFFNRoutedExpertsModelRunner | None = None
        self.model: RoutedExpertsModel | None = None
        self._cancel_requested = threading.Event()

    def prepare(self, service: FFNService) -> None:
        """Bring this F rank from launch to paired with Attention.

        Runs on the host thread, in the order that lets a bad deployment fail
        before the expert weights are read:

        1. select and check the GPU and the afdEP/DeepGEMM stack;
        2. build the F-only ``VllmConfig`` (AFD architecture, Triton MoE pin);
        3. create the F-only NCCL world and vLLM's model-parallel groups;
        4. construct the routed-experts runner, which runs the ModelRunnerV2
           and CUDA-graph validation and creates the connector (no
           collective yet);
        5. load the FFN-only routed-experts shell through vLLM's native loader;
        6. start the connector: the collective M2N handshake with Attention,
           the model-info check, the NCCL control group, the lanes' DeepEP
           buffers and the dispatch-kernel precompile.

        Phases: ``loading`` through step 5 (with ``loaded_experts`` and
        ``total_experts`` after the load), ``connecting`` from step 5 on; the
        host marks ``ready`` afterwards. CUDA graphs are not captured here: they
        are captured lazily, driven by Attention's warmup and capture
        payloads, exactly as on the Worker path. Every owner is assigned
        before the next native step, so a failure leaves it reachable.
        """
        environment = self.environment
        environment.initialize_device()
        config = environment.build_config()
        environment.initialize_compute(config)
        with set_current_vllm_config(config):
            self.runner = GPUFFNRoutedExpertsModelRunner(config, environment.device)
            connector = self.runner.connector
            if not isinstance(connector, DeepEPM2NAFDConnector):
                raise TypeError(
                    f"standalone GPU FFN needs DeepEPM2NAFDConnector, got "
                    f"{type(connector).__name__}"
                )
            self.runner.load_model()
            model: RoutedExpertsModel = self.runner.get_model()
            self.model = model
            # vLLM's loader reports no per-expert progress; this rank's experts
            # are all loaded once it returns.
            total_experts = sum(
                len(self.runner.local_experts[layer_idx].expert_ids)
                for layer_idx in self.runner.moe_layer_indices
            )
            service.loading_progress(total_experts, total_experts)
            parameter_bytes = parameter_storage_bytes(model)
            local_expert_ids = self.runner.local_experts[
                self.runner.moe_layer_indices[0]
            ].expert_ids
            service.phase(
                ServicePhase.CONNECTING,
                model_loaded=True,
                parameter_bytes=parameter_bytes,
                local_expert_first=local_expert_ids[0],
                local_expert_last=local_expert_ids[-1],
            )
            self.runner.initialize_afd_connector()
            assert connector.attn_tp_size is not None
            assert connector.capacity_tokens_per_rank is not None
            assert connector.bucket_ladder is not None
            assert connector.network_plan is not None
            service.phase(
                ServicePhase.CONNECTING,
                m2n_paired=True,
                attn_tp_size=connector.attn_tp_size,
                capacity_tokens_per_rank=connector.capacity_tokens_per_rank,
                num_buckets=len(connector.bucket_ladder),
                network_hybrid=connector.network_plan.hybrid,
                compute_ep_world_size=environment.ep_group.world_size,
            )

    # NPU parallel: CAMFFNBackend.start_receiver.
    # Reuse: YES - identical delegation to the environment.
    def start_receiver(self) -> None:
        """Select the GPU once on the receiving thread, as the Worker does."""
        self.environment.activate_thread()

    # NPU parallel: CAMFFNBackend.execute_step.
    # Reuse: PARTIAL - both delegate to the runner's
    #   execute_connector_driven_step; NPU adds its eager fence and never
    #   reports a cancel, GPU's runner already ends each step with a
    #   synchronize.
    def execute_step(self) -> bool:
        """Run one control payload's step through the shared GPU runner method.

        Returns ``False`` only when ``cancel_receive`` closed the control
        plane while the receive was idle. A closed control plane without a
        cancel request, and any other error (including a DeepEP timeout after
        a peer failed), raises and fails the service with owners retained.
        """
        assert self.runner is not None
        return self.runner.execute_connector_driven_step(
            cancel_requested=self._cancel_requested.is_set
        )

    # NPU parallel: CAMFFNBackend.cancel_receive (a no-op there).
    # Reuse: NO - GPU-only: the M2N control plane owns an abortable group.
    def cancel_receive(self) -> None:
        """End an idle control receive so a host stop can exit cleanly.

        Records the request first, then closes the connector's control plane,
        which aborts its dedicated NCCL control group; a blocked
        ``recv_dp_metadata_list`` then raises ``AFDControlPlaneClosedError``
        and ``execute_step`` returns ``False``. The DeepEP lane buffers keep
        their own group, so a step already in flight still completes. Same
        mechanism as ``AFDFFNWorker.stop_ffn_server_loop``.
        """
        self._cancel_requested.set()
        if self.runner is not None:
            control_plane = self.runner.connector.control_plane
            if control_plane is not None:
                control_plane.close()

    # NPU parallel: CAMFFNBackend.receiver_exited.
    # Reuse: YES - identical delegation to the runner.
    def receiver_exited(self) -> None:
        """Record on the receiver thread that its loop ended cleanly."""
        assert self.runner is not None
        self.runner.mark_receiver_exited()

    # NPU parallel: CAMFFNBackend.close.
    # Reuse: PARTIAL - same release order (runtime, compute groups, weights).
    def close(self) -> None:
        """Release CUDA graphs, DeepEP buffers, compute groups and weights.

        Only after the receiver exit was proven. ``runner.shutdown`` resets
        the captured graphs, drops their pool, clears vLLM's forward-context
        registrations and the model reference, and closes the connector
        (control group and lanes); then the environment destroys the F-only
        groups, and the last reference to the FFN-only model is dropped. The
        connector's gloo bootstrap group outlives this, which only matters
        for an in-process restart, which is not supported.
        """
        assert self.runner is not None
        if not self.runner.receiver_exited:
            raise RuntimeError("GPU FFN receiver has not exited; retain resources")
        self.runner.shutdown()
        self.environment.close()
        self.model = None
