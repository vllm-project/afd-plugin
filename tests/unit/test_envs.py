# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

from afd_plugin.envs import (
    AFD_FORCE_BALANCED_TOPK_IDS,
    force_balanced_topk_ids_enabled,
)


def test_force_balanced_topk_ids_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_FORCE_BALANCED_TOPK_IDS, raising=False)

    assert force_balanced_topk_ids_enabled() is False


def test_force_balanced_topk_ids_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(AFD_FORCE_BALANCED_TOPK_IDS, value)

        assert force_balanced_topk_ids_enabled() is True
