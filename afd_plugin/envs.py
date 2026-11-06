# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "force_balanced_topk_ids_enabled",
]
