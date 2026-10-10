# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vLLM 0.23 scheduler compatibility for AFD DSpark with Mooncake PD.

Upstream: vllm/v1/core/sched/scheduler.py at 0fc695fc6 and
vllm/v1/core/kv_cache_manager.py at the same revision.

Only asynchronous remote-prefill allocation omits lookahead slots. Normal
decode retains its speculative allocation. Hybrid load failures under the
``fail`` policy identify whole requests using Mooncake's group-0 markers;
the flat marker set cannot describe recovery offsets in other cache groups.
Remove this shim when upstream covers DSpark async admission and hybrid fail
handling. It does not change cache layouts or truncate transfer block lists.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version
from types import MethodType
from typing import TYPE_CHECKING

from afd_plugin.config import parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal import MultiModalRegistry
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

_PATCH_ATTR = "_afd_plugin_dspark_pd_scheduler_patch"
_INIT_PARAMETERS = (
    "self",
    "vllm_config",
    "kv_cache_config",
    "structured_output_manager",
    "block_size",
    "hash_block_size",
    "mm_registry",
    "include_finished_set",
    "log_stats",
)
_ALLOCATE_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_computed_tokens",
    "new_computed_blocks",
    "num_lookahead_tokens",
    "num_external_computed_tokens",
    "delay_cache_blocks",
    "num_encoder_tokens",
    "full_sequence_must_fit",
    "reserved_blocks",
)


def _uses_dspark_pd(config: VllmConfig) -> bool:
    afd = parse_optional_afd_config(config.additional_config)
    if afd is None or afd.role != "attention" or afd.connector != "P2pHcclAFDConnector":
        return False
    kv = config.kv_transfer_config
    if (
        kv is None
        or kv.kv_connector != "MooncakeHybridConnector"
        or not kv.is_kv_consumer
    ):
        return False
    if config.speculative_config is None:
        return False
    # The Ascend checkpoint marker selects DSpark behind the v0.23 MTP entry.
    from vllm_ascend.utils import is_dspark_config

    return is_dspark_config(config)


def _install_async_allocation(manager: KVCacheManager) -> None:
    original_allocate = manager.allocate_slots

    # Patch reason: v0.23 limits async PD lookahead only for EAGLE, while DSpark
    # also has a draft model. Extra slots enter Mooncake's unhashed block list.
    # Patch functionality: defer lookahead allocation during remote KV loading.
    # Expansion exception: allocate_slots owns the full cache admission/allocation
    # pipeline; delegate it unchanged with only this argument normalized.
    # Signature: matches the pinned KVCacheManager.allocate_slots exactly.
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
    ) -> KVCacheBlocks | None:
        # ### PATCH START: DSpark asynchronous PD allocation
        if delay_cache_blocks and num_external_computed_tokens > 0:
            num_lookahead_tokens = 0
        # ### PATCH END: DSpark asynchronous PD allocation
        return original_allocate(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
        )

    manager.allocate_slots = MethodType(allocate_slots, manager)


def apply_afd_dspark_pd_scheduler_patch() -> bool:
    """Install only on the pinned v0.23 API, with instance-level AFD scope."""
    if not version("vllm").startswith("0.23.0"):
        return False
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched import scheduler as upstream

    scheduler_cls = upstream.Scheduler
    if hasattr(scheduler_cls, _PATCH_ATTR):
        return True
    if (
        tuple(inspect.signature(scheduler_cls.__init__).parameters) != _INIT_PARAMETERS
        or tuple(inspect.signature(KVCacheManager.allocate_slots).parameters)
        != _ALLOCATE_PARAMETERS
        or tuple(inspect.signature(scheduler_cls._handle_invalid_blocks).parameters)
        != ("self", "invalid_block_ids", "num_scheduled_tokens")
    ):
        return False

    original_init = scheduler_cls.__init__

    # Patch reason: the async allocation shim belongs only to AFD DSpark PD
    # schedulers, not all KVCacheManager instances in the process.
    # Patch functionality: install the per-manager allocation shim after init.
    # Expansion exception: upstream __init__ builds all scheduler subsystems;
    # delegate that construction rather than copying several hundred lines.
    # Signature: matches the pinned Scheduler.__init__ exactly.
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = upstream.MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        original_init(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        # ### PATCH START: scope DSpark PD compatibility to its scheduler
        if _uses_dspark_pd(vllm_config):
            _install_async_allocation(self.kv_cache_manager)
            upstream.logger.info(
                "AFD DSpark PD scheduler compatibility active: "
                "async lookahead deferred; hybrid KV failures use request-level fail"
            )
        # ### PATCH END: scope DSpark PD compatibility to its scheduler

    # Patch reason: upstream single-group recovery cannot unpack hybrid caches.
    # Patch functionality: map group-0 failure markers to whole DSpark PD requests
    # under failure_policy=fail; preserve upstream handling for other policies.
    # Signature: matches Scheduler._handle_invalid_blocks; body copied from v0.23.
    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        # ### PATCH START: fail whole requests for hybrid Mooncake markers
        if (
            _uses_dspark_pd(self.vllm_config)
            and len(self.kv_cache_config.kv_cache_groups) > 1
            and not self.recompute_kv_load_failures
        ):
            failed_req_ids: set[str] = set()
            blocks_to_evict: set[int] = set()
            for evict_blocks, requests in (
                (True, self.running),
                (False, self.skipped_waiting),
            ):
                for request in requests:
                    if (
                        not evict_blocks
                        and request.status
                        != upstream.RequestStatus.WAITING_FOR_REMOTE_KVS
                    ):
                        continue
                    groups = self.kv_cache_manager.get_block_ids(request.request_id)
                    if groups and invalid_block_ids.intersection(groups[0]):
                        failed_req_ids.add(request.request_id)
                        if evict_blocks:
                            blocks_to_evict.update(
                                block for group in groups for block in group
                            )
            if not failed_req_ids:
                return set()
            if blocks_to_evict:
                self.kv_cache_manager.evict_blocks(blocks_to_evict)
            upstream.logger.error(
                "AFD DSpark PD hybrid KV load failed (failure_policy=fail). "
                "Failed request IDs: %s; group-0 markers: %s",
                sorted(failed_req_ids),
                sorted(invalid_block_ids),
            )
            return failed_req_ids
        # ### PATCH END: fail whole requests for hybrid Mooncake markers
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == upstream.RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )
        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )
        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens
        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)
        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            upstream.logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids
        upstream.logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        return sync_failed_req_ids

    scheduler_cls.__init__ = __init__
    scheduler_cls._handle_invalid_blocks = _handle_invalid_blocks
    setattr(scheduler_cls, _PATCH_ATTR, original_init)
    return True
