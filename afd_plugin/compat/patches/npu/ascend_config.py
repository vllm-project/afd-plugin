# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep AFD settings outside Ascend's strict configuration namespace."""

import importlib.util
import sys

from vllm.config import VllmConfig
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

# Module-level aliases in the pinned target; later lazy imports use the factory.
_ASCEND_CONFIG_ALIAS_MODULES = (
    "vllm_ascend.platform",
    "vllm_ascend.worker.worker",
    "vllm_ascend.patch.platform.patch_engine_core",
    "vllm_ascend.patch.platform.patch_balance_schedule",
    "vllm_ascend.patch.platform.patch_dyntra_lb_core",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker",
)


# Upstream: vllm_ascend/ascend_config.py at
# 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6.
# Patch reason: strict AscendConfig construction rejects AFD's shared settings.
# Patch functionality: exclude the AFD namespace from constructor kwargs,
# preserving the source mapping, config identity and native singleton/cache.
# Signature: matches upstream; no added parameters.
# Removal plan: remove when upstream supports a plugin configuration namespace.
def init_ascend_config(vllm_config: VllmConfig) -> AscendConfig:
    additional_config = (
        vllm_config.additional_config
        if vllm_config.additional_config is not None
        else {}
    )
    # Upstream EngineArgs injects --gdn-prefill-backend / --kda-prefill-backend
    # into additional_config. The generic GDN/KDA model layers consume them
    # (qwen_gdn_linear_attn / kimi_gdn_linear_attn), but on non-CUDA platforms
    # only the triton path is available: the FLA Triton kernels run on Ascend
    # via triton-ascend (the CUDA triton package is replaced in Ascend images).
    # CUDA-only values (flashinfer/cutedsl for GDN, flashkda for KDA) have no
    # kernel on Ascend. Strip the keys here so extra="forbid" does not reject
    # them as typos, and warn only when the user requested an unsupported value.
    _TRITON_COMPATIBLE_VALUES = ("auto", "triton")
    for _prefill_key in ("gdn_prefill_backend", "kda_prefill_backend"):
        _prefill_value = additional_config.get(_prefill_key)
        if (
            _prefill_value is not None
            and str(_prefill_value).strip().lower() not in _TRITON_COMPATIBLE_VALUES
        ):
            logger.warning_once(
                "Ascend does not support %s=%r; only the 'triton' value is "
                "available on Ascend for GDN/KDA prefill (FLA kernels run via "
                "triton-ascend). The option is ignored.",
                _prefill_key,
                _prefill_value,
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
    # ### PATCH START: native singleton ownership
    if (
        ascend_config._ASCEND_CONFIG is not None
        and not refresh
        and _is_ascend_config_initialized(ascend_config._ASCEND_CONFIG)
        and ascend_config._INIT_VLLM_CONFIG is vllm_config
    ):
        return ascend_config._ASCEND_CONFIG
    # ### PATCH END: native singleton ownership

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
        # ### PATCH START: AFD namespace
        "afd",
        # ### PATCH END: AFD namespace
        # control-flow flag (singleton/cache refresh), not a configuration field
        "refresh",
        # Upstream-injected by EngineArgs for the generic GDN/KDA prefill
        # backend selector; Ascend supports only the triton value (FLA kernels
        # run via triton-ascend), and the triton default applies either way
        # (warned above when the user requested a CUDA-only value). Strip
        # instead of letting extra="forbid" report them as typos.
        "gdn_prefill_backend",
        "kda_prefill_backend",
        # Consumed in derive_and_validate as the SP MoE switch. Not an
        # AscendConfig field, so strip it before extra="forbid" validation.
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
        # ### PATCH START: native singleton publication
        ascend_config._ASCEND_CONFIG = new_config
        ascend_config._INIT_VLLM_CONFIG = vllm_config
        # ### PATCH END: native singleton publication
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
    """Replace the factory and loaded aliases without importing optional callers."""
    ascend_config.init_ascend_config = init_ascend_config
    for module_name in _ASCEND_CONFIG_ALIAS_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.init_ascend_config = init_ascend_config


# Upstream: vllm_ascend/patch/platform/patch_engine_core.py at
# 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6.
# Patch reason: DP EngineCore initializes AscendConfig before vLLM loads
# general plugins in the spawned process, so AFD's config namespace fails
# strict validation before register_afd can install the namespace factory.
# Patch functionality: install the AFD namespace factory in the child, then
# run Ascend's complete EngineCore entry point unchanged.
# Signature: matches the upstream process target; no added parameters.
# Removal plan: remove once Ascend initializes plugin config after general
# plugins have loaded in every DP child.
def run_afd_ascend_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    from vllm_ascend.patch.platform import patch_engine_core

    # ### PATCH START: AFD namespace before Ascend child config validation
    apply_afd_ascend_config_patch()
    # ### PATCH END: AFD namespace before Ascend child config validation
    return patch_engine_core._run_engine_core_patch_func(
        *args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs
    )
