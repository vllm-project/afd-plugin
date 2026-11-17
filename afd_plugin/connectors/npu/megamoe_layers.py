# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static layer banks and startup agreement for device-selected MegaMoE."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor
    from torch.distributed import Store

    LayerWeights = tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]


def pack_layer_weights(weights: Mapping[int, LayerWeights]) -> LayerWeights:
    """Build layer-major TensorLists without copying weight storage."""
    if not weights or min(weights) < 0:
        raise ValueError("MegaMoE requires nonempty, nonnegative layer IDs")
    ordered = [weights[layer] for layer in sorted(weights)]
    # expert_weights supplies one stacked tensor per projection per layer.
    if any(
        len(layer) != 4 or any(len(part) != 1 for part in layer) for layer in ordered
    ):
        raise ValueError("MegaMoE layer banks require stacked expert tensors")
    columns = [[layer[part][0] for layer in ordered] for part in range(4)]
    return columns[0], columns[1], columns[2], columns[3]


def agree_layer_selection(
    store: Store,
    ffn_size: int,
    ffn_rank: int | None,
    enabled: bool,
    layer_ids: list[int],
) -> dict[int, int]:
    """FFN owns the switch; all A/F ranks adopt the same model-ID to bank-slot map."""
    if ffn_rank is not None:
        store.set(
            f"layers/{ffn_rank}",
            json.dumps({"enabled": enabled, "layers": sorted(layer_ids)}),
        )
    configs = [json.loads(store.get(f"layers/{rank}")) for rank in range(ffn_size)]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("All MegaMoE FFN ranks must agree on layer selection and IDs")
    ids = configs[0]["layers"]
    if not ids or ids[0] < 0 or len(set(ids)) != len(ids):
        raise ValueError("MegaMoE requires unique nonnegative FFN layer IDs")
    # In legacy mode every call carries one layer, selected as slot zero.
    return {
        layer: slot if configs[0]["enabled"] else 0 for slot, layer in enumerate(ids)
    }
