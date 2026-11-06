# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD NPU worker actions for EEP; the upstream EngineCore state machine is
retained.

Mirrors :mod:`afd_plugin.elastic.gpu` with Ascend-specific ACL graph
retirement and re-capture. Hardware-neutral actions (topology updates,
connector rebuild) are imported from the GPU module, which stays importable
on NPU because vLLM-Ascend's ``NPUModelRunner`` already extends vLLM's
``GPUModelRunner``.
"""

from __future__ import annotations

import copy
import gc
from typing import TYPE_CHECKING, cast

import torch
import torch_npu  # noqa: F401
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed.elastic_ep.standby_state import pop_standby_groups
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.v1.engine import ReconfigureRankType
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    unlock_workspace,
)
from vllm_ascend.compilation import acl_graph

from afd_plugin.connectors import AFDControlPayload
from afd_plugin.elastic.gpu import (
    AFDElasticGPUExecutor,
    connect,
    update_topology,
)

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.attention_worker import AFDNPUAttentionWorker
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker


def send_stop(worker: AFDNPUAttentionWorker) -> None:
    """Only the connector's designated senders emit STOP on the old topology."""
    control_plane = worker.model_runner.connector.control_plane
    assert control_plane is not None, "STOP requires a live control plane"
    control_plane.send_dp_metadata_list(
        AFDControlPayload({}, is_graph_capturing=False, is_warmup=False, stop=True)
    )
    torch.npu.synchronize(worker.device)


def _bump_acl_graph_pool() -> None:
    """Detach new captures from the retired pool's graph memory."""
    from vllm.platforms import current_platform

    fresh = torch.npu.graph_pool_handle()
    current_platform.__class__._global_graph_pool = fresh
    # Wrappers retained across an F-only resize captured their pool handle at
    # construction; retarget them so the next capture allocates in the new pool.
    for wrapper in list(acl_graph._acl_graph_wrappers):
        wrapper.graph_pool = fresh


def _retire_attention_graphs(worker: AFDNPUAttentionWorker) -> None:
    runner = worker.model_runner
    # V2 DBO captures route through ACLGraphWrapper instances registered in
    # acl_graph._acl_graph_wrappers (NPUUBatchRunner holds no own graphs).
    # vLLM-Ascend has no ACLGraphWrapper.clear_all_graphs; the module's
    # live-instance registry is the equivalent seam.
    for wrapper in list(acl_graph._acl_graph_wrappers):
        wrapper.concrete_aclgraph_entries.clear()
    # FIA graph workspaces belong to the retired capture generation;
    # initialize_from_config re-populates them via set_graph_params before
    # the next capture (NPUModelRunner._check_and_update_cudagraph_mode).
    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None
    with set_current_vllm_config(worker.vllm_config):
        reset_compile_wrapper(runner.get_model())
    _bump_acl_graph_pool()


def _retire_ffn_graphs(worker: AFDNPUFFNWorker) -> None:
    # ACL-graph retirement is the sync FFN runner's protocol; the async
    # runner retires its own graphs elsewhere (runtime behavior unchanged).
    runner = cast("AFDNPUFFNModelRunner", worker.model_runner)
    for entry in runner._acl_graphs.values():
        entry["graph"].reset()
    runner._acl_graphs.clear()
    # Generation bump equivalent of the proven NPU elastic recipe: a fresh
    # pool handle instead of the shared retired one.
    runner.graph_pool = torch.npu.graph_pool_handle() if runner.use_aclgraph else None


def release_link(worker: AFDNPUAttentionWorker | AFDNPUFFNWorker) -> None:
    """Called on *all old ranks*, after DP pause and FFN STOP/join."""
    runner = worker.model_runner
    torch.npu.synchronize(worker.device)
    gc.unfreeze()
    if worker.afd_expected_role == "attention":
        worker.afd_runtime_deferred = True
        attn_runner = cast("AFDNPUAttentionModelRunnerV2", runner)
        attn_runner._afd_pending_metadata = None
        _retire_attention_graphs(worker)
    else:
        _retire_ffn_graphs(worker)
    torch.compiler.reset()
    # Graphs must drop before their communicators are destroyed.
    gc.collect()
    runner.connector.close()
    torch.npu.empty_cache()


@torch.inference_mode()
def finish_attention(worker: AFDNPUAttentionWorker) -> None:
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
    torch.npu.synchronize(worker.device)


class AFDElasticNPUExecutor(AFDElasticGPUExecutor):
    """NPU worker actions only; EEP states, notifications and KV sync retained.

    Everything except runner construction and the DBO workspace width is
    inherited from the GPU executor, whose group construction and no-op
    expert RPCs are hardware neutral.
    """

    def load_model(self) -> None:
        # NPUWorker.load_model takes no weight-loading arguments; A always
        # loads its full attention checkpoint from storage (the GPU executor
        # passes load_dummy_weights=False for the same effect).
        self.worker.load_model()

    # Re-anchor main@544f182d: stock prepare_reconfiguration warms the
    # standby dp/ep device groups with an all-rank all_reduce, but the
    # joining engine never issues a matching collective (its
    # warmup_new_worker skips warm_and_capture without fused-MoE kernel
    # reuse, which AFD A engines never have), so with lazily-initialized
    # HCCL comms the two existing ranks die with EI0015 rootInfo-detect
    # timeout. vLLM already skips this warm on ROCm for the same
    # asymmetry; NPU follows. Deferral is the only structural option:
    # warming inside warmup_new_worker would deadlock instead, since the
    # new engine's PREPARE state is gated on the existing prepare
    # completing (sync_kv_cache_memory_size). The first post-switch
    # serving collective initializes the comms with all ranks present.
    def _warm_target_groups(self, dp_group, ep_group) -> None:
        return

    # Re-anchor main@544f182d: stock warm_and_capture only skips DP
    # coordination when fused-MoE kernels are reusable; AFD A engines have
    # no MoE, so the stock else-branch runs the warm dummy-run's DP sync —
    # which needs the joining engine to dispatch in lockstep while that
    # engine is blocked on this engine's commit broadcast (circular wait,
    # both sides die on 300s store timeouts). The warm is purely local
    # kernel/graph preparation: always skip DP coordination here. (Stock
    # GPU EEP never exercises the else branch because EEP implies MoE.)
    def warm_and_capture(self) -> None:
        from vllm.v1.worker.dp_utils import skip_dp_coordination

        with skip_dp_coordination():
            super().warm_and_capture()

    def switch_and_prepare(self) -> tuple:
        # Lazy import avoids the attention runner -> worker/runtime import cycle.
        from afd_plugin.compat.npu import npu_afd_num_ubatches
        from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
            AFDNPUAttentionModelRunnerV2,
        )

        worker = self.worker
        request = self.reconfig_request
        assert request is not None
        worker.model_runner.shutdown()
        worker.model_runner = None
        gc.collect()
        torch.npu.empty_cache()
        # Runner.shutdown resets the process workspace. Worker.init_device is
        # deliberately not rerun while EEP switches the existing device groups.
        init_workspace_manager(worker.device, npu_afd_num_ubatches(worker.vllm_config))
        # Stock contract: return the retired groups so the caller can schedule
        # their async destruction (commit_scale_up -> _start_group_cleanup).
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
            worker.model_runner = AFDNPUAttentionModelRunnerV2(
                worker.vllm_config, worker.device
            )
            worker.load_model(load_dummy_weights=False)
            worker.initialize_from_config(copy.deepcopy(worker.afd_kv_cache_config))
        return retired_groups


__all__ = [
    "AFDElasticNPUExecutor",
    "connect",
    "finish_attention",
    "release_link",
    "send_stop",
    "update_topology",
]
