# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import pytest

from afd_plugin.envs import (
    async_cam_layered_gmm_enabled,
    force_balanced_topk_ids_enabled,
)


@pytest.mark.parametrize(
    "name,read",
    [
        ("AFD_FORCE_BALANCED_TOPK_IDS", force_balanced_topk_ids_enabled),
        ("AFD_ASYNC_CAM_LAYERED_GMM", async_cam_layered_gmm_enabled),
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
def test_diagnostic_environment_flags(monkeypatch, name, read, value, expected):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert read() is expected
