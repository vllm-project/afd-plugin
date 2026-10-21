# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared startup gate, capture limits and host identity for CAM profiling.

Workers and the coordinator share these device-independent helpers for startup,
metadata layout and clock alignment. See docs/npu/CAM_PROFILING.md for the
control -> endpoint -> worker -> session flow and offline analysis.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROFILE_ENABLE_ENV = "AFD_CAM_PROFILE_ENABLE"
BATCH_INFO_FIELDS = 5
MAX_DURATION_S = 3600
MAX_LIGHTWEIGHT_DURATION_S = 7 * 24 * MAX_DURATION_S


def host_clock_id() -> str | None:
    """Identify Linux's shared CLOCK_REALTIME without relying on hostnames."""
    if sys.platform != "linux":
        return None
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def batch_info_prefix_size(tp_size: int, expert_per_rank: int) -> int:
    """Size of the unmodified CAM dispatch-recv metadata."""
    return BATCH_INFO_FIELDS + tp_size * (1 + expert_per_rank)


def runtime_profiling_enabled() -> bool:
    return os.environ.get(PROFILE_ENABLE_ENV) == "1"
