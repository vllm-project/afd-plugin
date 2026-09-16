# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config compatibility adjustments for AFD Ascend workers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FLASHINFER_ALL2ALLV_BACKEND = "flashinfer_all2allv"


def npu_afd_num_ubatches(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        return int(parallel_config.num_ubatches)
    return 1


def fix_all2all_backend_for_afd(vllm_config: VllmConfig) -> None:
    """Normalize the backend before serialization and worker construction.

    Mirror Ascend bd69bad88fc19e1aeeea585416d408df8bda8fef's effective
    FlashComm selector. Use the original DSA-CP option: native derivation can
    clear its finalized value after choosing the backend. A later worker-only
    rewrite otherwise changes the config hash after EngineCore serialization.
    """
    from vllm_ascend.ascend_config import validate_additional_config_bool

    additional_config = vllm_config.additional_config or {}
    flashcomm_explicitly_enabled = validate_additional_config_bool(
        additional_config.get("enable_flashcomm1", False),
        "additional_config.enable_flashcomm1",
    ) or os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower() in (
        "1",
        "true",
    )
    enable_dsa_cp = validate_additional_config_bool(
        additional_config.get("enable_dsa_cp", False),
        "additional_config.enable_dsa_cp",
    )
    if not (flashcomm_explicitly_enabled or enable_dsa_cp):
        vllm_config.parallel_config.all2all_backend = FLASHINFER_ALL2ALLV_BACKEND


__all__ = ["fix_all2all_backend_for_afd", "npu_afd_num_ubatches"]
