# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_ASYNC_CAM_LAYERED_GMM = "AFD_ASYNC_CAM_LAYERED_GMM"
AFD_MEGAMOE_LAYERED_GMM = "AFD_MEGAMOE_LAYERED_GMM"
AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_MOE_LAYOUT_LOG = "AFD_ASYNC_MOE_LAYOUT_LOG"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_moe_layout_log_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_MOE_LAYOUT_LOG, "").lower() in ENV_TRUE_VALUES


def async_cam_layered_gmm_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


def megamoe_layered_gmm_enabled() -> bool:
    """Opt in to A5 split MegaMoE device layer selection."""
    return os.environ.get(AFD_MEGAMOE_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_ASYNC_CAM_LAYERED_GMM",
    "AFD_MEGAMOE_LAYERED_GMM",
    "async_cam_layered_gmm_enabled",
    "megamoe_layered_gmm_enabled",
    "AFD_ASYNC_MOE_LAYOUT_LOG",
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "async_moe_layout_log_enabled",
    "force_balanced_topk_ids_enabled",
]
