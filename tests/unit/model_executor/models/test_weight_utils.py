# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared checkpoint mechanics remain independent of torch and model classes."""

import pytest

from afd_plugin.model_executor.models._weight_utils import (
    filter_role_weights,
    layer_weight_path,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.3.mlp.experts.weight", (3, "mlp", ("experts", "weight"))),
        (
            "model.language_model.layers.5.self_attn.weight",
            (5, "self_attn", ("weight",)),
        ),
        ("layers.invalid.layers.2.mlp.weight", (2, "mlp", ("weight",))),
        ("model.layers.2", None),
        ("model.embed_tokens.weight", None),
    ],
)
def test_checkpoint_path_keeps_layer_stage_and_remainder(name, expected):
    assert layer_weight_path(name) == expected


@pytest.mark.parametrize("role", [None, "attention", "ffn"])
def test_filter_streams_once_and_preserves_weights(role):
    first, second = object(), object()
    source = iter([("attention", first), ("ffn", second)])

    def classify(name):
        assert role is not None
        return frozenset((name,))

    actual = list(filter_role_weights(source, role=role, classify=classify))
    expected = [("attention", first), ("ffn", second)]
    if role is not None:
        expected = [item for item in expected if item[0] == role]
    assert actual == expected
    assert list(source) == []
