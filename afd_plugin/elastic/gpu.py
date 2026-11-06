# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD worker actions for EEP; the upstream EngineCore state machine is retained."""

from __future__ import annotations

import copy
import gc
from typing import TYPE_CHECKING

import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed import get_world_group
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    unlock_workspace,
)

from afd_plugin.config import afd_config_from_mapping, parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory, AFDControlPayload
from afd_plugin.elastic.config import set_afd_config

if TYPE_CHECKING:
    from afd_plugin.v1.worker.attention_worker import AFDAttentionWorker
    from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker


def send_stop(worker: AFDAttentionWorker) -> None:
    """Only the connector's designated senders emit STOP on the old topology."""
    worker.model_runner.connector.control_plane.send_dp_metadata_list(
        AFDControlPayload({}, is_graph_capturing=False, is_warmup=False, stop=True)
    )
    torch.cuda.synchronize(worker.device)


def release_link(worker: AFDAttentionWorker | AFDFFNWorker) -> None:
    """Called on *all old ranks*, after DP pause and FFN STOP/join."""
    runner = worker.model_runner
    torch.cuda.synchronize(worker.device)
    gc.unfreeze()
    CUDAGraphWrapper.clear_all_graphs()
    if worker.afd_expected_role == "attention":
        worker.afd_runtime_deferred = True
        runner._afd_pending_metadata = None
        # DBO owns a separate cache outside CUDAGraphWrapper's registry.
        # F-only resize retains this wrapper but replaces its A/F connector.
        if isinstance(runner.model, UBatchWrapper):
            runner.model.clear_graphs()
        with set_current_vllm_config(worker.vllm_config):
            reset_compile_wrapper(runner.get_model())
    else:
        for entry in runner._cuda_graphs.values():
            entry["graph"].reset()
        runner._cuda_graphs.clear()
        runner._graph_memory_pool = None
    torch.compiler.reset()
    # Graphs must drop before their communicators are destroyed.
    gc.collect()
    runner.connector.close()
    torch.cuda.empty_cache()


def update_topology(
    worker: AFDAttentionWorker | AFDFFNWorker, raw: dict[str, str | int | bool]
) -> None:
    # EngineCore's nested collective_rpc args are untyped after MessagePack
    # decoding. Use the same explicit mapping contract for A and direct-Ray F.
    afd = afd_config_from_mapping(raw, expected_role=worker.afd_expected_role)
    set_afd_config(worker.vllm_config, afd)
    worker.model_runner.afd_config = afd
    # Model wrappers only retain role/model settings today, but do not leave
    # their config snapshots describing the old topology after a resize.
    for module in worker.model_runner.get_model().modules():
        if "afd_config" in vars(module):
            module.afd_config = afd


def connect(worker: AFDAttentionWorker | AFDFFNWorker) -> None:
    group = get_world_group()
    runner = worker.model_runner
    runner.afd_config = parse_afd_config(worker.vllm_config)
    runner.connector = AFDConnectorFactory.create_connector(
        group.rank, group.local_rank, worker.vllm_config, runner.afd_config
    )
    runner.connector.init_afd_connector()


@torch.inference_mode()
def finish_attention(worker: AFDAttentionWorker) -> None:
    runner = worker.model_runner
    # F-only resize retains weights, while rebuilding KV bindings/metadata.
    # Initial/new/reloaded A already have fresh KV; using this same path keeps
    # the profile/warmup entry point identical on all DP siblings.
    # P1 review (allocation lifetime): stock MRV2 initialize_kv_cache
    # allocates the replacement BEFORE rebinding self.kv_caches and the
    # per-layer views, so re-entering here while the engine's KV is resident
    # peaks two complete budgets -- a --kv-cache-memory-bytes value that
    # fits once OOMs at startup or resize. Every path reaching finish() has
    # already applied this exact snapshot: boot and new engines through the
    # EngineCore initialize_from_config (the worker override re-saves the
    # config it applied) and retained engines through switch_and_prepare.
    # Reuse that fresh cache; rebuild only when this runner never applied
    # the snapshot -- the legacy path stays as the drift fallback. Object
    # identity is the reliable "already applied" signal: the worker's
    # initialize_from_config override stamps the runner it initialized, and
    # Ascend deep-copies and rewrites the layout internally, so deep
    # equality between configs cannot be trusted.
    applied_runner = getattr(worker, "afd_kv_initialized_runner", None)
    if not (getattr(runner, "kv_caches", None) and runner is applied_runner):
        worker.initialize_from_config(copy.deepcopy(worker.afd_kv_cache_config))
    gc.collect()
    unlock_workspace()
    worker.afd_runtime_deferred = False
    with set_current_vllm_config(worker.vllm_config):
        runner.profile_run()
        worker.compile_or_warm_up_model()
    lock_workspace()
    runner._afd_pending_metadata = None
    torch.cuda.synchronize(worker.device)


class AFDElasticGPUExecutor(ElasticEPScalingExecutor):
    """Implement worker actions only; retain EEP states, notifications and KV sync.

    A holds no remote experts. New and retained A load their attention checkpoint
    after changing groups. EEP's paired expert mapping/weight RPCs remain local
    no-ops on both sides, so no stale receive is left in the new-worker loader.
    """

    def load_model(self) -> None:
        self.worker.load_model(load_dummy_weights=False)

    # Source: vLLM 0.30 elastic_execute.py: create_standby_groups.
    # Change: retain group construction, omit EPLB suppression/MoE staging.
    # Signature unchanged; upstreaming target is a role-specific worker executor.
    def create_standby_groups(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        world_size = self.worker.vllm_config.parallel_config.world_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(
            self.worker.vllm_config.parallel_config
        )
        updated_config.parallel_config.data_parallel_size = new_dp_size
        with set_current_vllm_config(updated_config):
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=world_size * new_dp_size,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
                # ### PATCH START: A has no experts: no all2all, no EPLB.
                use_all2all=False,
                enable_eplb=False,
                # ### PATCH END: A has no experts.
            )
        # ### PATCH START: A has no local experts to stage or suppress.
        # The upstream EPLB suppression and MoE staging steps are omitted.
        # ### PATCH END: A has no local experts to stage or suppress.

    def broadcast_expert_mapping(self) -> None:
        pass

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        pass

    def receive_weights(self) -> None:
        pass

    def prepare_new_worker(self) -> None:
        pass

    # Re-anchor main@544f182d: stock prepare_reconfiguration calls this
    # unconditionally and asserts a non-None standby EPLB group, but
    # create_standby_groups builds that group only under enable_eplb. AFD
    # elastic runs EPLB off (experts live on the FFN side; A engines hold no
    # expert weights), so None is the legitimate steady state here.
    def _prepare_eplb_communicator(self, eplb_group) -> None:
        if eplb_group is None:
            return
        super()._prepare_eplb_communicator(eplb_group)

    # Re-anchor main@544f182d: stock commit_scale_up hardwires the expert
    # mapping broadcast/receive plus EPLB setup/reshuffle around the switch.
    # A engines hold no experts and run EPLB off — the new-engine side dies
    # in receive_expert_mapping()/setup_eplb_from_mapping() (observed:
    # A3's commit_scale_up(False) crashed inside the module-level
    # broadcast_expert_mapping helper). Mirror the stock sequence minus the
    # expert/EPLB steps; the paired no-op method overrides above document
    # the same contract since the 0.30 port.
    def commit_scale_up(self, is_existing_worker: bool) -> None:
        from vllm.platforms import current_platform

        retired_groups = None
        if is_existing_worker:
            retired_groups = self.switch_and_prepare()
        if not self._can_reuse_fused_moe_kernel():
            # warm_and_capture's dummy run warms the deferred dp/ep groups.
            self.warm_and_capture()
        elif current_platform.is_rocm():
            from vllm.distributed.parallel_state import get_dp_group, get_ep_group

            self._warm_target_groups(get_dp_group(), get_ep_group())
        if retired_groups:
            self._start_group_cleanup(retired_groups)

    def perform_eplb_reshuffle(self) -> None:
        pass

    def perform_scale_down_eplb_reshuffle(self, new_dp_size: int) -> None:
        pass

    def rewarm_workspace(self) -> None:
        # Public finish_attention performs this after *every* A finishes EEP.
        pass

    def switch_and_remove(self) -> None:
        # The common prelude already released AFD graphs and connectors.
        self.worker.model_runner.shutdown()
        _replace_active_groups(world=None, dp=None, ep=None, eplb=None, node_count=None)

    def switch_and_prepare(self) -> tuple:
        # Lazy import avoids the attention runner -> worker/runtime import cycle.
        from afd_plugin.v1.worker.attention_model_runner_v2 import (
            AFDAttentionModelRunnerV2,
        )

        worker = self.worker
        request = self.reconfig_request
        assert request is not None
        worker.model_runner.shutdown()
        worker.model_runner = None
        gc.collect()
        torch.cuda.empty_cache()
        # Runner.shutdown resets the process workspace. Worker.init_device is
        # deliberately not rerun while EEP switches the existing device groups.
        init_workspace_manager(
            worker.device, num_ubatches=2 if worker.parallel_config.enable_dbo else 1
        )
        # Stock contract: return the retired groups so the caller can
        # schedule their async destruction.
        retired_groups = _replace_active_groups(**pop_standby_groups())
        parallel = worker.parallel_config
        parallel.data_parallel_size = request.new_data_parallel_size
        if request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel.data_parallel_rank = request.new_data_parallel_rank
        if (
            request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel.data_parallel_rank_local = request.new_data_parallel_rank_local
        parallel.data_parallel_master_ip = request.new_data_parallel_master_ip
        parallel.data_parallel_master_port = request.new_data_parallel_master_port
        parallel._data_parallel_master_port_list = (
            request.new_data_parallel_master_port_list
        )
        parallel._coord_store_port = request.coord_store_port
        with set_current_vllm_config(worker.vllm_config):
            worker.model_runner = AFDAttentionModelRunnerV2(
                worker.vllm_config, worker.device
            )
            worker.load_model(load_dummy_weights=False)
            worker.initialize_from_config(copy.deepcopy(worker.afd_kv_cache_config))
        return retired_groups
