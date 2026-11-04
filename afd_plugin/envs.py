# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_CAM_LAYERED_GMM = "AFD_ASYNC_CAM_LAYERED_GMM"
# A2E transport quantization modes the synchronous CAMP2P connector accepts.
# 0 keeps the BF16 activation contract; 4 moves a pre-quantized E4M3 activation
# together with its E8M0 MX scale. The mode is a connector setting, so it is read
# from ``connector_extra_config.quant_mode`` rather than from the environment.
A2E_QUANT_MODE_BF16 = 0
A2E_QUANT_MODE_MXFP8 = 4
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_cam_layered_gmm_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_ASYNC_CAM_LAYERED_GMM",
    "A2E_QUANT_MODE_BF16",
    "A2E_QUANT_MODE_MXFP8",
    "async_cam_layered_gmm_enabled",
    "force_balanced_topk_ids_enabled",
]
