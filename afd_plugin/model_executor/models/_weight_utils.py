# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Checkpoint path parsing and one-pass filtering; models own role policies."""

from collections.abc import Callable, Iterable, Iterator
from typing import Final, TypeVar

_Weight = TypeVar("_Weight")

ATTENTION_ROLE: Final[frozenset[str]] = frozenset(("attention",))
FFN_ROLE: Final[frozenset[str]] = frozenset(("ffn",))
BOTH_ROLES: Final[frozenset[str]] = ATTENTION_ROLE | FFN_ROLE


def layer_weight_path(name: str) -> tuple[int, str, tuple[str, ...]] | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[index + 1])
        except ValueError:
            continue
        return layer_idx, parts[index + 2], tuple(parts[index + 3 :])
    return None


def filter_role_weights(
    weights: Iterable[tuple[str, _Weight]],
    *,
    role: str | None,
    classify: Callable[[str], frozenset[str]],
) -> Iterator[tuple[str, _Weight]]:
    for name, loaded_weight in weights:
        if role is None or role in classify(name):
            yield name, loaded_weight
