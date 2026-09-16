# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep AFD settings outside Ascend's strict configuration namespace."""

import importlib.util
import os
import sys

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import (
    AscendConfig,
    KVPPConfig,
    SchedulerConfig,
    SparseKVOffloadConfig,
    _is_ascend_config_initialized,
    logger,
    validate_additional_config_bool,
)

# Exact target module-level imports; lazy callers bind the patched factory later.
_ASCEND_CONFIG_ALIAS_MODULES = (
    "vllm_ascend.platform",
    "vllm_ascend.worker.worker",
    "vllm_ascend.patch.platform.patch_balance_schedule",
    "vllm_ascend.patch.platform.patch_dyntra_lb_core",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker",
)


# Upstream: vllm_ascend/ascend_config.py at
# bd69bad88fc19e1aeeea585416d408df8bda8fef.
# Patch reason: strict AscendConfig construction rejects AFD's shared settings.
# Patch functionality: exclude only AFD-owned keys from native constructor kwargs
# while preserving the original config object, mapping and Ascend-owned cache.
# Signature: matches upstream; no added parameters.
# Removal plan: remove when upstream accepts a separate plugin config namespace.
def init_ascend_config(vllm_config):
    additional_config = (
        vllm_config.additional_config
        if vllm_config.additional_config is not None
        else {}
    )
    if (
        "enable_flashcomm1" in additional_config
        or os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1") is not None
    ):
        logger.warning(
            "FlashComm is deprecated; remove enable_flashcomm1 and "
            "VLLM_ASCEND_ENABLE_FLASHCOMM1 from the configuration. Use upstream configuration instead"
        )
    refresh = validate_additional_config_bool(
        additional_config.get("refresh", False), "additional_config.refresh"
    )
    raw_rl_config = additional_config.get("rl_config", {})
    if isinstance(raw_rl_config, dict):
        refresh = refresh or validate_additional_config_bool(
            raw_rl_config.get("enabled", False), "additional_config.rl_config.enabled"
        )
    elif "rl_config" in additional_config:
        # Do not reuse a cached config: let AscendConfig's normal nested
        # pydantic validation report the invalid sub-config input below.
        refresh = True
    # ### PATCH START: retain native singleton ownership
    if (
        ascend_config._ASCEND_CONFIG is not None
        and not refresh
        and _is_ascend_config_initialized(ascend_config._ASCEND_CONFIG)
        and ascend_config._INIT_VLLM_CONFIG is vllm_config
    ):
        return ascend_config._ASCEND_CONFIG
    # ### PATCH END: retain native singleton ownership

    # Pre-construct sub-configs that need precedence resolution or vllm_config.
    sched = SchedulerConfig.from_additional_config(additional_config)
    sparse_kv = SparseKVOffloadConfig.from_additional_config(
        vllm_config, additional_config.get("sparse_kv_offload_config", {})
    )
    kvpp_config = KVPPConfig.from_vllm_config(vllm_config)
    # dump_config: keep the mutual-exclusion / materialize logic as a factory
    # pre-step; the resolved path is passed as the dump_config_path field.
    dump_config_path = AscendConfig._resolve_dump_config_path(additional_config)

    # Keys that must NOT flow from additional_config into AscendConfig.
    # These are stripped so that only user-configurable keys reach pydantic,
    # where extra="forbid" can reject unknown options.
    _NON_USER_INPUT_KEYS = {
        # ### PATCH START: exclude AFD-owned configuration keys
        "afd",
        "enable_force_load_balance",
        "force_load_balance_topn_per_rank",
        # ### PATCH END: exclude AFD-owned configuration keys
        # control-flow flag (singleton/cache refresh), not a configuration field
        "refresh",
        # Removed upstream option: warn above, but do not pass it into the
        # strict AscendConfig schema where it would be reported as a typo.
        "enable_flashcomm1",
        # injected fields (factory passes explicitly; a copy in additional_config would conflict)
        "scheduler_config",
        "sparse_kv_offload_config",
        # Factory-injected: derived from additional_config.enable_kvpp + TP.
        "enable_kvpp",
        "kvpp_config",
        # Factory-only input: materialized by _resolve_dump_config_path and
        # replaced with the validated dump_config_path field below.
        "dump_config",
        "dump_config_path",
        # pure-derived fields (derive_and_validate computes them; user input would residualize)
        # NOTE: enable_shared_expert_dp/enable_sparse_sfa_c8/enable_sparse_li_c8
        # are NOT here — they are user-input fields that derive_and_validate
        # augments (self.x = self.x and condition), so the user must be able to
        # pass them. Only pure-derived fields (no user input) are stripped.
        "enable_sp_by_pass",
        "pd_tp_ratio",
        "pd_head_ratio",
        "num_head_replica",
        # private derived state (init=False, but listed for safety)
        "_sparse_li_c8_layer_ids",
        "_sparse_li_c8_layer_names",
        "_sparse_li_c8_layer_filter_enabled",
        # SchedulerConfig-internal top-level legacy keys (resolved internally,
        # then replaced by the typed scheduler_config passed above).
        "enable_balance_scheduling",
        "recompute_scheduler_enable",
        "short_request_first_config",
        "profiling_chunk_config",
        "batch_job_sched_config",
    }
    kwargs = {
        k: v for k, v in additional_config.items() if k not in _NON_USER_INPUT_KEYS
    }
    unknown_keys = sorted(set(kwargs) - AscendConfig.__dataclass_fields__.keys())
    # vLLM-Omni shares this mapping with the platform plugin. Preserve its
    # extension keys on VllmConfig while excluding them from Ascend validation.
    if unknown_keys and importlib.util.find_spec("vllm_omni") is not None:
        logger.warning(
            "The following additional_config keys are invalid for vLLM-Ascend: %s. "
            "They may be used by vLLM-Omni or another project. "
            "Please remove them if they are not needed for your use case.",
            unknown_keys,
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in unknown_keys}

    new_config = AscendConfig(  # type: ignore[call-arg]
        scheduler_config=sched,
        sparse_kv_offload_config=sparse_kv,
        kvpp_config=kvpp_config,
        dump_config_path=dump_config_path,
        **kwargs,
    )
    # Business validation (Plan B): pydantic did type/range/enum checks during
    # construction; the cross-config derivations and mutex checks that need
    # vllm_config run here, explicitly, before the instance is usable. This is
    # the single legitimate entry point — bypassing the factory leaves derived
    # fields at their sentinel defaults.
    new_config.derive_and_validate(vllm_config)
    new_config.rl_config.apply(new_config)
    new_config.finegrained_tp_config._validate_preconditions(vllm_config)
    new_config.xlite_graph_config._validate_preconditions(vllm_config)
    if _is_ascend_config_initialized(new_config):
        # ### PATCH START: publish into native singleton
        ascend_config._ASCEND_CONFIG = new_config
        ascend_config._INIT_VLLM_CONFIG = vllm_config
        # ### PATCH END: publish into native singleton
        # Publish the fully validated singleton before invalidating derived
        # process caches. The next runtime read rebuilds them from new_config;
        # failed construction leaves the previous singleton/cache untouched.
        from vllm_ascend.utils import clear_enable_sp

        clear_enable_sp()
    else:
        logger.warning(
            "Ascend config instance is not fully initialized. action: skip singleton cache update. "
        )
    return new_config


def apply_afd_ascend_config_patch() -> None:
    """Replace the factory and known aliases without importing optional callers."""

    ascend_config.init_ascend_config = init_ascend_config
    for module_name in _ASCEND_CONFIG_ALIAS_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.init_ascend_config = init_ascend_config
