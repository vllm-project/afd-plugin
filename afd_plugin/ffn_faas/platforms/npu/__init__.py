# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM device and explicitly supported model selection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel


def get_model(model_path: Path, ffn_ranks: int, ffn_rank: int) -> CAMRoutedModel:
    """Select the one qualified model; incompatible checkpoints fail on CPU.

    This is a loading boundary, not a model registry or multi-model support claim.
    Native CAM still requires its existing HF configuration and W4A8 contract.
    """
    from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel
    from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription
    from afd_plugin.ffn_faas.platforms.npu.v4_runtime import V4ModelAdapter

    adapter = V4ModelAdapter(V4CheckpointDescription.read(model_path))
    return CAMRoutedModel(adapter, ffn_ranks, ffn_rank)
