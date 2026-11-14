# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import pytest

from afd_plugin.envs import (
    AFD_ASYNC_MOE_LAYOUT_LOG,
    AFD_FORCE_BALANCED_TOPK_IDS,
    async_moe_layout_log_enabled,
    force_balanced_topk_ids_enabled,
)


@pytest.mark.parametrize(
    "name,enabled",
    [
        (AFD_FORCE_BALANCED_TOPK_IDS, force_balanced_topk_ids_enabled),
        (AFD_ASYNC_MOE_LAYOUT_LOG, async_moe_layout_log_enabled),
    ],
)
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("1", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
    ],
)
def test_diagnostic_environment_flags(monkeypatch, name, enabled, value, expected):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert enabled() is expected
