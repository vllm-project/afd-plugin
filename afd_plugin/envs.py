# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_MOE_LAYOUT_LOG = "AFD_ASYNC_MOE_LAYOUT_LOG"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_moe_layout_log_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_MOE_LAYOUT_LOG, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_ASYNC_MOE_LAYOUT_LOG",
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "async_moe_layout_log_enabled",
    "force_balanced_topk_ids_enabled",
]
