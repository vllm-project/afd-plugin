# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

from afd_plugin.envs import (
    AFD_ASYNC_CAM_LAYERED_GMM,
    AFD_FORCE_BALANCED_TOPK_IDS,
    async_cam_layered_gmm_enabled,
    force_balanced_topk_ids_enabled,
)


def test_force_balanced_topk_ids_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_FORCE_BALANCED_TOPK_IDS, raising=False)

    assert force_balanced_topk_ids_enabled() is False


def test_force_balanced_topk_ids_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(AFD_FORCE_BALANCED_TOPK_IDS, value)

        assert force_balanced_topk_ids_enabled() is True


def test_async_cam_layered_gmm_defaults_and_boolean_values(monkeypatch):
    monkeypatch.delenv(AFD_ASYNC_CAM_LAYERED_GMM, raising=False)
    assert async_cam_layered_gmm_enabled() is False
    for value in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv(AFD_ASYNC_CAM_LAYERED_GMM, value)
        assert async_cam_layered_gmm_enabled() is True
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(AFD_ASYNC_CAM_LAYERED_GMM, value)
        assert async_cam_layered_gmm_enabled() is False
