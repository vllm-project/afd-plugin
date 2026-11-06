# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Own the F-local GPU, the F-only vLLM config and its compute groups.

The M2N world (bootstrap, control and lane groups) belongs to the connector.
The vLLM world, DP and EP groups created here span only the ``torchrun`` F
ranks; they exist so vLLM can construct FusedMoE and place experts, and carry
no per-step traffic.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNConfig
from afd_plugin.ffn_faas.platforms.launch import reassert_external_launcher

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

# afdEP-comm transport and the DeepGEMM grouped GEMMs the connector runs; both
# are imported lazily by the connector, so a missing package would otherwise
# fail only after 27 GiB of experts were read.
AFDEP_TRANSPORT_MODULE = "afdep.deepep"
DEEPGEMM_MODULE = "deep_gemm"
FUSED_EXPERTS_BACKEND = "fused_experts"
# Set by torchrun: rendezvous in its workers join the agent's TCP store.
TORCHELASTIC_AGENT_STORE_ENV = "TORCHELASTIC_USE_AGENT_STORE"
# vLLM's GPU Worker gives DBO two workspace slots.
DBO_WORKSPACE_UBATCHES = 2


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/environment.py::
#   AscendFFNEnvironment.
# Reuse: PARTIAL - the lifecycle (launcher check, refuse borrowed groups,
#   device, F-only world + EP group == F world, per-thread device, ordered
#   teardown) is the same; NPU initializes HCCL explicitly plus Ascend groups
#   and CPU binding, GPU reuses vLLM's GPU Worker distributed initializer,
#   pins the routed MoE backend and selects the AFD model architecture.
#   REFACTOR-CANDIDATE(R1): a platform-neutral F-only environment base
#   (launcher check, ownership flags, EP-group check, teardown order).
class GPUFFNEnvironment:
    """Never borrow unrelated groups or tear them down on a failed native seam."""

    def __init__(self, settings: GPUFFNConfig) -> None:
        """Record the settings; nothing native happens until ``initialize_device``."""
        self.settings = settings
        self._device: torch.device | None = None
        self._ep_group: GroupCoordinator | None = None
        self._groups_owned = False

    @property
    def device(self) -> torch.device:
        """This rank's GPU, available after ``initialize_device``."""
        if self._device is None:
            raise RuntimeError("GPU FFN device has not been initialized")
        return self._device

    @property
    def ep_group(self) -> GroupCoordinator:
        """vLLM's EP group over the F-only world, after ``initialize_compute``."""
        if self._ep_group is None:
            raise RuntimeError("GPU FFN expert group has not been initialized")
        return self._ep_group

    # NPU parallel: AscendFFNEnvironment.initialize_device.
    # Reuse: PARTIAL - launcher and borrowed-group checks are identical;
    #   the device and native preflight are platform-specific.
    def initialize_device(self) -> None:
        """Select this rank's GPU after checking the launcher and native stack.

        Requires the ``torchrun`` environment to still match the settings and
        no pre-initialized ``torch.distributed`` or vLLM model-parallel state,
        so this process never adopts groups it does not own. Imports afdEP's
        DeepEP transport and, for the DeepGEMM expert backend, DeepGEMM, so a
        missing package fails before any weight is read. Maps ``LOCAL_RANK``
        to the visible GPU the way vLLM's GPU Worker does (honoring
        ``CUDA_VISIBLE_DEVICES``) and requires SM90 or newer.
        """
        if self._groups_owned or self._device is not None:
            raise RuntimeError("GPU FFN environment is already initialized")
        settings = self.settings
        try:
            world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
        except (KeyError, ValueError) as error:
            raise ValueError("GPU FFN requires an external F-only launcher") from error
        if world != settings.ffn_ranks or rank != settings.rank:
            raise ValueError("GPU FFN environment differs from the F-only launcher")
        import torch
        from vllm.distributed.parallel_state import model_parallel_is_initialized
        from vllm.platforms import current_platform

        if torch.distributed.is_initialized() or model_parallel_is_initialized():
            raise RuntimeError("GPU FFN cannot borrow existing native groups")
        importlib.import_module(AFDEP_TRANSPORT_MODULE)
        expert_backend = settings.connector_extra_config.get("expert_backend")
        if expert_backend != FUSED_EXPERTS_BACKEND:
            importlib.import_module(DEEPGEMM_MODULE)
        # Importable only after the afdEP preflight above.
        from afd_plugin.connectors.gpu.deepep_m2n.connector import MIN_SM_MAJOR

        visible_index = current_platform.logical_device_id_to_visible_device_id(
            settings.local_rank
        )
        device = torch.device("cuda", visible_index)
        major, minor = torch.cuda.get_device_capability(device)
        if major < MIN_SM_MAJOR:
            raise RuntimeError(
                f"DeepEP M2N needs SM{MIN_SM_MAJOR}0 or newer, got SM{major}{minor}"
            )
        torch.accelerator.set_device_index(device)
        self._device = device

    # NPU parallel: CAMFFNBackend.prepare (EngineArgs, create_engine_config,
    #   re-asserted external_launcher).
    # Reuse: YES - both use the shared F-only EngineArgs and launcher helpers.
    def build_config(self) -> VllmConfig:
        """Build the F-only ``VllmConfig`` exactly as the GPU FFN Worker sees it.

        Creates the config from ``GPUFFNConfig.engine_kwargs`` through
        ``EngineArgs.create_engine_config`` (with the AFD plugin's config
        patches, which accept DBO without a DeepEP all2all backend), then
        re-asserts the ``external_launcher`` executor, swaps in the
        AFD-prefixed model architecture that ``AFDFFNWorker.init_device``
        selects, and pins vLLM's MoE backend to Triton as
        ``AFDFFNWorker._make_model_runner`` does. The pin must happen before
        any FusedMoE is constructed: FusedMoE picks its weight layout at
        construction, and a reordered layout would be read silently wrong.
        """
        from vllm.engine.arg_utils import EngineArgs

        from afd_plugin.model_executor.models.model_utils import get_afd_model_config
        from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
            select_routed_moe_backend,
        )

        config = EngineArgs(**self.settings.engine_kwargs()).create_engine_config()
        # The F world is the already launched torchrun world, whatever executor
        # platform defaults would pick for an ordinary engine.
        reassert_external_launcher(config)
        config.model_config = get_afd_model_config(
            config.model_config, device_type="cuda"
        )
        select_routed_moe_backend(config)
        return config

    # NPU parallel: AscendFFNEnvironment.initialize_compute.
    # Reuse: PARTIAL - ownership-before-collective and the EP-group check are
    #   identical; GPU calls vLLM's GPU Worker initializer instead of
    #   init_distributed_environment + initialize_model_parallel directly.
    def initialize_compute(self, config: VllmConfig) -> None:
        """Create the F-only NCCL world, model-parallel groups and workspace.

        Uses ``init_worker_distributed_environment``, the same function vLLM's
        GPU Worker calls (batch invariance, custom all-reduce setting, world,
        TP/PP/DP/EP groups), with the ``env://`` rendezvous ``torchrun``
        provides. With ``external_launcher`` vLLM keeps the launcher's world
        size and rank, so EP spans exactly the F ranks; this is re-checked
        because the connector's handshake assumes contiguous expert ranges
        over that world. Ownership is recorded before the first collective so
        a partial initialization is still torn down by ``close``. Then stops
        later rendezvous from joining the torchrun agent's store, so the M2N
        world can host its own, and initializes vLLM's workspace manager with
        the Worker's micro-batch count (two under DBO).
        """
        if self._device is None or self._groups_owned:
            raise RuntimeError("GPU FFN compute initialization requires its device")
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import get_ep_group
        from vllm.platforms import current_platform
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
        from vllm.v1.worker.workspace import init_workspace_manager

        settings = self.settings
        with set_current_vllm_config(config):
            self._groups_owned = True
            init_worker_distributed_environment(
                config,
                settings.rank,
                "env://",
                settings.local_rank,
                current_platform.dist_backend,
            )
            self._ep_group = get_ep_group()
            if (
                self._ep_group.world_size != settings.ffn_ranks
                or self._ep_group.rank_in_group != settings.rank
            ):
                raise ValueError("vLLM expert group differs from the F-only world")
        # torchrun asks every ``tcp://``/``env://`` rendezvous in its workers
        # to join the agent's store. That is right for the F-only world above,
        # but the connector's M2N rendezvous at ``--host:--port`` must be
        # hosted by M2N world rank 0 (F rank 0), as in a ``vllm serve`` FFN;
        # with the flag set, every rank would wait as a client and nobody
        # would listen. vLLM's groups are already created, so drop it.
        os.environ.pop(TORCHELASTIC_AGENT_STORE_ENV, None)
        # Same slot count as Worker._init_workspace_and_model_runner.
        init_workspace_manager(
            self._device,
            DBO_WORKSPACE_UBATCHES if config.parallel_config.enable_dbo else 1,
        )

    # NPU parallel: AscendFFNEnvironment.activate_thread.
    # Reuse: YES (shape) - one call; only the device API differs.
    def activate_thread(self) -> None:
        """Select this rank's GPU on the calling (receiver) thread.

        CUDA's current device is per thread; graph capture and the connector's
        lane streams run on the receiver thread.
        """
        import torch

        torch.cuda.set_device(self.device)

    # NPU parallel: AscendFFNEnvironment.close.
    # Reuse: PARTIAL - same order and ownership guard; NPU also destroys the
    #   Ascend model-parallel groups.
    def close(self) -> None:
        """Destroy the vLLM model-parallel groups and the F-only world.

        Only after the runner released CUDA graphs and DeepEP buffers; a failed
        native seam never reaches here, so groups stay owned until process
        isolation.
        """
        if not self._groups_owned:
            return
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm.v1.worker.workspace import reset_workspace_manager

        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
        self._groups_owned = False
        self._ep_group = None
        self._device = None
