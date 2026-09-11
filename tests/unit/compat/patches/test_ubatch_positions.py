# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Tests for the ubatch positions-propagation patch."""

from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.compat.patches.ubatch_positions import split_attn_metadata


class _Ub:
    def __init__(self, start: int, stop: int):
        self.token_slice = slice(start, stop)


def test_split_propagates_positions_per_token_slice() -> None:
    positions = list(range(10))
    source = SimpleNamespace(positions=positions)
    slices = [_Ub(0, 4), _Ub(4, 10)]

    def fake_split(ubatch_slices, cm):
        return [SimpleNamespace(positions=None) for _ in ubatch_slices]

    import afd_plugin.compat.patches.ubatch_positions as module

    original = module._upstream_split_attn_metadata
    module._upstream_split_attn_metadata = fake_split
    try:
        results = split_attn_metadata(slices, source)
    finally:
        module._upstream_split_attn_metadata = original

    assert [list(cm.positions) for cm in results] == [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]


def test_split_keeps_none_positions() -> None:
    source = SimpleNamespace(positions=None)
    slices = [_Ub(0, 4), _Ub(4, 10)]

    def fake_split(ubatch_slices, cm):
        return [SimpleNamespace(positions=None) for _ in ubatch_slices]

    import afd_plugin.compat.patches.ubatch_positions as module

    original = module._upstream_split_attn_metadata
    module._upstream_split_attn_metadata = fake_split
    try:
        results = split_attn_metadata(slices, source)
    finally:
        module._upstream_split_attn_metadata = original

    assert all(cm.positions is None for cm in results)
