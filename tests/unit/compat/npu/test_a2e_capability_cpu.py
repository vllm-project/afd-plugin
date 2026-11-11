# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the A2E MXFP8 input-schema probe.

The probe answers three states because the caller needs different messages: a build that
cannot take the arguments, a build whose schema cannot be read, and a build whose schema
matches.

The success case is parsed by the real PyTorch schema parser rather than hand-built: the
rendered type of ``Tensor?`` is ``Optional[Tensor]``, and a fake that spells it the way
the whole-schema text does would pass while every real build failed.
"""

from __future__ import annotations

import types

import pytest

from afd_plugin.compat.npu import ops

torch = pytest.importorskip("torch")

_BASE = (
    "Tensor x, Tensor? expert_ids=None, Tensor? scales=None, int batch_size=0, "
    "int hidden_size=0, int topk=0, int expert_rank_size=0, int attention_rank_size=0, "
    'int rank=0, str group_ep="", int aiv_num=0, int compute_gate=0'
)


def _schema(argument_text: str):
    return torch._C.parse_schema(f"afd_ascend::a2e({argument_text}) -> Tensor[]")


def _install(monkeypatch, schema):
    """Point the probe at a schema without touching the real operator namespace."""

    if schema is None:
        a2e = types.SimpleNamespace()
    else:
        a2e = types.SimpleNamespace(default=types.SimpleNamespace(_schema=schema))
    fake_ops = types.SimpleNamespace(
        afd_ascend=types.SimpleNamespace(a2e=a2e),
    )
    monkeypatch.setattr(torch, "ops", fake_ops)
    monkeypatch.setattr(ops, "ensure_cam_p2p_ops_available", lambda: None)


def test_the_fourteen_argument_shape_is_accepted(monkeypatch):
    _install(
        monkeypatch,
        _schema(_BASE + ", Tensor? x_scale=None, int quant_mode=0"),
    )
    assert ops.a2e_accepts_mxfp8_arguments() is True


def test_a_build_without_the_arguments_is_rejected(monkeypatch):
    _install(monkeypatch, _schema(_BASE))
    assert ops.a2e_accepts_mxfp8_arguments() is False


def test_an_unreadable_schema_answers_unknown(monkeypatch):
    _install(monkeypatch, None)
    assert ops.a2e_accepts_mxfp8_arguments() is None


def test_the_trailing_pair_in_the_wrong_order_is_rejected(monkeypatch):
    _install(
        monkeypatch,
        _schema(_BASE + ", int quant_mode=0, Tensor? x_scale=None"),
    )
    assert ops.a2e_accepts_mxfp8_arguments() is False


def test_a_tensor_list_scale_is_rejected(monkeypatch):
    """List[Tensor] also has getElementType(), so the wrapper must be optional."""

    _install(
        monkeypatch,
        _schema(_BASE + ", Tensor[] x_scale, int quant_mode=0"),
    )
    assert ops.a2e_accepts_mxfp8_arguments() is False


def test_keyword_only_trailing_arguments_are_rejected(monkeypatch):
    _install(
        monkeypatch,
        _schema(_BASE + ", *, Tensor? x_scale=None, int quant_mode=0"),
    )
    assert ops.a2e_accepts_mxfp8_arguments() is False


def test_a_short_schema_with_the_names_is_rejected(monkeypatch):
    """A three-argument schema can still end with the names; the call would fail."""

    _install(monkeypatch, _schema("Tensor x, Tensor? x_scale=None, int quant_mode=0"))
    assert ops.a2e_accepts_mxfp8_arguments() is False


def test_a_missing_extension_answers_false(monkeypatch):
    def explode():
        raise RuntimeError("no custom operators here")

    monkeypatch.setattr(ops, "ensure_cam_p2p_ops_available", explode)
    assert ops.a2e_accepts_mxfp8_arguments() is False
