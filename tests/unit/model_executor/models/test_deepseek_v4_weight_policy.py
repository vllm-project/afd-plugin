# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.deepseek_v4 import (
    _checkpoint_weight_roles,
    _iter_mtp_role_weights,
    _iter_role_weights,
    _mtp_checkpoint_weight_roles,
)


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        ("embed.weight", {"attention"}),
        ("head.weight", {"attention"}),
        ("norm.weight", {"attention"}),
        ("hc_head_fn", {"attention"}),
        ("layers.0.attn.wq_a.weight", {"attention"}),
        ("layers.0.attn.wq_a.weight_scale", {"attention"}),
        ("layers.0.attn_norm.weight", {"attention"}),
        ("layers.0.ffn_norm.weight", {"attention"}),
        ("layers.0.hc_attn_fn", {"attention"}),
        ("layers.0.hc_ffn_fn", {"attention"}),
        ("layers.0.ffn.gate.weight", {"ffn"}),
        ("layers.0.ffn.gate.tid2eid", {"ffn"}),
        ("layers.3.ffn.experts.7.w1.weight", {"ffn"}),
        ("layers.3.ffn.experts.7.w1.weight_scale", {"ffn"}),
        ("layers.3.ffn.experts.7.w1.weight_offset", {"ffn"}),
        ("layers.42.ffn.shared_experts.w2.weight", {"ffn"}),
        ("mtp.0.layers.0.ffn.experts.0.w1.weight", set()),
    ],
)
def test_checkpoint_weight_roles(name, roles):
    assert _checkpoint_weight_roles(name) == roles


def test_role_weight_iterator_is_consumed_once():
    class OneShot:
        consumed = False

        def __iter__(self):
            if self.consumed:
                raise AssertionError("checkpoint iterator consumed twice")
            self.consumed = True
            yield "embed.weight", object()
            yield "layers.0.ffn.gate.weight", object()
            yield "mtp.0.weight", object()

    weights = OneShot()

    assert [name for name, _ in _iter_role_weights(weights, role="ffn")] == [
        "layers.0.ffn.gate.weight"
    ]
    assert weights.consumed


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        ("mtp.0.ffn.experts.0.w1.weight", {"ffn"}),
        ("mtp.0.ffn.experts.0.w1.weight_scale", {"ffn"}),
        ("mtp.0.ffn.experts.0.w1.weight_offset", {"ffn"}),
        ("mtp.0.ffn_norm.weight", {"attention"}),
        ("mtp.0.hc_ffn_fn", {"attention"}),
        ("mtp.0.attn.wq_a.weight", {"attention"}),
        ("mtp.0.head.weight", {"attention"}),
        ("layers.0.ffn.gate.weight", set()),
    ],
)
def test_mtp_checkpoint_weight_roles(name, roles):
    assert _mtp_checkpoint_weight_roles(name) == roles


def test_mtp_role_weight_iterator_is_consumed_once():
    class OneShot:
        consumed = False

        def __iter__(self):
            if self.consumed:
                raise AssertionError("MTP checkpoint iterator consumed twice")
            self.consumed = True
            yield "mtp.0.attn.wq_a.weight", object()
            yield "mtp.0.ffn.experts.0.w1.weight", object()
            yield "mtp.0.ffn.experts.0.w1.weight_scale", object()
            yield "mtp.0.ffn.experts.0.w1.weight_offset", object()

    weights = OneShot()

    assert [name for name, _ in _iter_mtp_role_weights(weights, role="ffn")] == [
        "mtp.0.ffn.experts.0.w1.weight",
        "mtp.0.ffn.experts.0.w1.weight_scale",
        "mtp.0.ffn.experts.0.w1.weight_offset",
    ]
    assert weights.consumed
