# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Select the local (NoDP) MoE prepare/finalize for the AFD FFN role.

An AFD FFN rank is not a vLLM DP rank: it has no scheduler, never forms a
coordinated DP batch, and only ever holds rows the Attention dispatch already
routed to its local experts. vLLM's DP MoE path instead re-assembles the whole
DP token set with an ``all_gatherv`` collective that reads
``dp_metadata`` from the forward context and needs every DP rank in lockstep.
Neither exists on the connector-driven FFN role, so the worker loop dies on
``assert dp_metadata is not None`` (2 FFN ranks would deadlock in the
collective even if the metadata were supplied).

The NoDP prepare/finalize is pure-local: quantize, permute through the layer's
expert_map (our grouped rows carry global expert ids that map onto the local
range), run the experts, combine locally. That is exactly the AFD FFN
execution model.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils_module
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    make_moe_prepare_and_finalize_no_dp_ep,
)

from afd_plugin.config import parse_optional_afd_config

# Patch reason: see module docstring -- the naive DP all-to-all cannot run on
# the connector-driven AFD FFN role.
# Patch functionality: when the active role is the AFD FFN role and no exotic
# all2all kernel is requested, return vLLM's NoDP prepare/finalize instead of
# the naive DP one; every non-AFD caller keeps the upstream selection.
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0,
#           vllm/model_executor/layers/fused_moe/all2all_utils.py
_UPSTREAM_SELECTOR = all2all_utils_module.maybe_make_prepare_finalize


def _is_afd_ffn_role() -> bool:
    try:
        afd_config = parse_optional_afd_config(
            get_current_vllm_config(),
            validate=False,
        )
    except Exception:
        return False
    return afd_config is not None and afd_config.role == "ffn"


def maybe_make_prepare_finalize(*args: Any, **kwargs: Any):
    if not _is_afd_ffn_role():
        return _UPSTREAM_SELECTOR(*args, **kwargs)
    moe = args[0] if args else kwargs["moe"]
    parallel = moe.moe_parallel_config
    # Kernels with their own dispatch own the collective; leave them untouched.
    # Everything else (including the naive DP fallback that use_ep + dp>1
    # selects) must run locally: AFD pre-routes the rows.
    exotic_kernels = (
        parallel.use_deepep_ht_kernels
        or parallel.use_deepep_ll_kernels
        or parallel.use_deepep_v2_kernels
        or parallel.use_fi_nvl_two_sided_kernels
        or parallel.use_fi_nvl_one_sided_kernels
        or parallel.use_nixl_ep_kernels
        or parallel.use_mori_kernels
    )
    if exotic_kernels:
        return _UPSTREAM_SELECTOR(*args, **kwargs)
    return make_moe_prepare_and_finalize_no_dp_ep(
        use_monolithic=bool(kwargs.get("use_monolithic", False)),
    )


def _rebind_source_module() -> None:
    maybe_make_prepare_finalize._afd_installed = True  # type: ignore[attr-defined]
    all2all_utils_module.maybe_make_prepare_finalize = maybe_make_prepare_finalize


def apply_local_moe_prepare() -> None:
    """Install the selector wrapper in every namespace that bound it.

    The plugin loads before vLLM's MoE modules are imported, so re-aliasing
    the source module is what future ``from ... import`` bindings pick up;
    any module already present in ``sys.modules`` that still holds the
    upstream function is re-aliased directly. Idempotent.
    """
    if getattr(maybe_make_prepare_finalize, "_afd_installed", False):
        return
    _rebind_source_module()
    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType) or module is all2all_utils_module:
            continue
        if getattr(module, "maybe_make_prepare_finalize", None) is _UPSTREAM_SELECTOR:
            module.maybe_make_prepare_finalize = maybe_make_prepare_finalize


apply_local_moe_prepare()

__all__ = ["apply_local_moe_prepare", "maybe_make_prepare_finalize"]
