# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import pytest

from afd_plugin.envs import AFD_FORCE_BALANCED_TOPK_IDS, force_balanced_topk_ids_enabled


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
def test_force_balanced_topk_ids_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(AFD_FORCE_BALANCED_TOPK_IDS, raising=False)
    else:
        monkeypatch.setenv(AFD_FORCE_BALANCED_TOPK_IDS, value)
    assert force_balanced_topk_ids_enabled() is expected
