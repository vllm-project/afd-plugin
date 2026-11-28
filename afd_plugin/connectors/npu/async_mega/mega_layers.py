# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model dimensions and layer-weight contracts for device-selected Mega."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    LayerWeights = tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]


@dataclass(frozen=True)
class MegaModelSpec:
    """Model-independent dimensions and activation parameters for Mega kernels."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation_parameters: tuple[float, float, float]
