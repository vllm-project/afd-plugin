# SPDX-License-Identifier: Apache-2.0
"""Layer-bank ordering and the A/F startup contract, without device execution."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

path = (
    Path(__file__).resolve().parents[4] / "afd_plugin/connectors/npu/mega_layers.py"
)
spec = importlib.util.spec_from_file_location("mega_layers", path)
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)


@pytest.fixture
def store():
    records = {}
    return SimpleNamespace(
        set=lambda key, value: records.__setitem__(key, value.encode()),
        get=records.__getitem__,
    )


def test_bank_order_matches_sparse_model_layer_mapping(store):
    weights = {
        7: tuple([object()] for _ in range(4)),
        2: tuple([object()] for _ in range(4)),
    }
    bank = layers.pack_layer_weights(weights)
    slots = layers.agree_layer_selection(store, 1, 0, [7, 2])
    assert slots == {2: 0, 7: 1}
    for layer, slot in slots.items():
        for part in range(4):
            assert bank[part][slot] is weights[layer][part][0]


def test_attention_adopts_ffn_layer_slots(store):
    ffn = layers.agree_layer_selection(store, 1, 0, [7, 2])
    attention = layers.agree_layer_selection(store, 1, None, [])
    assert ffn == attention == {2: 0, 7: 1}


def test_disagreement_between_ffn_ranks_is_rejected(store):
    store.set("layers/1", json.dumps([2, 8]))
    with pytest.raises(ValueError, match="must agree"):
        layers.agree_layer_selection(store, 2, 0, [2, 7])


@pytest.mark.parametrize("ids", [[], [-1], [2, 2]])
def test_invalid_layer_ids_rejected_before_dispatch(store, ids):
    with pytest.raises(ValueError, match="unique nonnegative"):
        layers.agree_layer_selection(store, 1, 0, ids)


def test_bank_rejects_per_expert_or_missing_projection_lists():
    with pytest.raises(ValueError, match="stacked expert tensors"):
        layers.pack_layer_weights({0: ([object(), object()], [], [], [])})
