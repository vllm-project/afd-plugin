# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 A2E MXFP8 contract checks that do not need an NPU or vLLM-Ascend.

The transport itself is exercised on hardware; these cover the pure decisions
AFD makes around it, which is where a wrong answer is silent rather than loud.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.feature_validation import _fail_if_unsupported_a2e_mxfp8
from afd_plugin.envs import A2E_QUANT_MODE_BF16, A2E_QUANT_MODE_MXFP8

#: Mirrors ``afd_plugin.connectors.npu.camp2p.A2E_MX_MAX_TOTAL_RANKS``. The
#: connector needs vLLM at import time, so it cannot be imported during
#: collection on a CPU-only host; ``test_rank_limit_matches_the_connector``
#: asserts the two stay equal wherever vLLM is available.
A2E_MX_MAX_TOTAL_RANKS = 64


def _afd_config(*, attention_ranks: int, ffn_ranks: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_attention_ranks=attention_ranks,
        num_ffn_ranks=ffn_ranks,
    )


class TestA2eMxfp8Topology:
    """The operator carries the A5 restriction; the rank shape is ours to check."""

    def test_accepts_equal_rank_counts(self):
        _fail_if_unsupported_a2e_mxfp8(
            _afd_config(attention_ranks=4, ffn_ranks=4),
            A2E_MX_MAX_TOTAL_RANKS,
        )

    def test_accepts_the_total_rank_boundary(self):
        _fail_if_unsupported_a2e_mxfp8(
            _afd_config(attention_ranks=32, ffn_ranks=32),
            A2E_MX_MAX_TOTAL_RANKS,
        )

    @pytest.mark.parametrize(
        ("attention_ranks", "ffn_ranks"),
        [(4, 2), (2, 4), (1, 8), (8, 1)],
    )
    def test_rejects_unequal_rank_counts(self, attention_ranks, ffn_ranks):
        with pytest.raises(RuntimeError, match="equal Attention and FFN rank counts"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(
                    attention_ranks=attention_ranks,
                    ffn_ranks=ffn_ranks,
                ),
                A2E_MX_MAX_TOTAL_RANKS,
            )

    def test_rejects_more_than_the_total_rank_bound(self):
        with pytest.raises(RuntimeError, match="at most 64 ranks"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(attention_ranks=64, ffn_ranks=64),
                A2E_MX_MAX_TOTAL_RANKS,
            )

    def test_unequal_counts_are_reported_before_the_total_bound(self):
        """A 4A8F config is both unequal and over nothing; name the real mistake."""

        with pytest.raises(RuntimeError, match="equal Attention and FFN rank counts"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(attention_ranks=4, ffn_ranks=8),
                A2E_MX_MAX_TOTAL_RANKS,
            )

    def test_rank_limit_matches_the_connector(self):
        """The asserted bound must be the product's, not a copy that drifts."""

        camp2p = pytest.importorskip("afd_plugin.connectors.npu.camp2p")

        assert A2E_MX_MAX_TOTAL_RANKS == camp2p.A2E_MX_MAX_TOTAL_RANKS


class TestA2eMxfp8QuantMode:
    """``quant_mode`` is the connector's own field; there is no second entry.

    The mode is read from ``connector_extra_config`` only, so a deployment
    cannot end up with one role on MXFP8 and the other on BF16 through two
    configuration surfaces that disagree.
    """

    def test_defaults_to_bf16(self):
        extra_info = _parse_extra_info({})

        assert extra_info.quant_mode == A2E_QUANT_MODE_BF16

    def test_accepts_the_mxfp8_mode(self):
        extra_info = _parse_extra_info({"quant_mode": A2E_QUANT_MODE_MXFP8})

        assert extra_info.quant_mode == A2E_QUANT_MODE_MXFP8

    @pytest.mark.parametrize("mode", [1, 2, 3, 5, -1])
    def test_rejects_a_mode_the_operator_does_not_define(self, mode):
        extra_info = _parse_extra_info({"quant_mode": mode})

        with pytest.raises(RuntimeError, match="quant_mode 0 \\(BF16\\) or 4"):
            extra_info.validate_supported()

    def test_rejects_a_non_integer(self):
        with pytest.raises((TypeError, ValueError)):
            _parse_extra_info({"quant_mode": "mxfp8"})


def _parse_extra_info(extra: dict):
    """Parse ``connector_extra_config`` through the connector's own parser."""

    camp2p = pytest.importorskip("afd_plugin.connectors.npu.camp2p")

    return camp2p.CAMP2PExtraInfo.from_mapping(extra)


class TestMxfp8Dequantization:
    """E8M0 is a biased exponent, so a wrong bias is silent except as drift.

    ``mx_dequantize_a2e_activation`` reconstructs the float32 power of two by
    shifting the scale byte into the exponent field. Pinning that against scales
    whose value is known by construction is the only way to catch an off-by-one
    bias without a device run: on hardware it would only show up as an accuracy
    delta that is easy to attribute to quantization noise.
    """

    @pytest.mark.parametrize("byte, factor", [(127, 1.0), (128, 2.0), (126, 0.5)])
    def test_scale_byte_selects_the_expected_power_of_two(self, byte, factor):
        torch = pytest.importorskip("torch")
        camp2p = pytest.importorskip("afd_plugin.connectors.npu.camp2p")

        hidden = camp2p.A2E_MX_GROUP_SIZE
        activation = torch.ones(1, hidden, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[byte]], dtype=torch.uint8).view(torch.float8_e8m0fnu)

        restored = camp2p.mx_dequantize_a2e_activation(activation, scale)

        assert restored.dtype == torch.bfloat16
        assert torch.equal(restored, torch.full((1, hidden), factor))

    def test_hidden_size_that_is_not_a_multiple_of_the_group(self):
        """The operator sizes the scale row as ceil(hidden / group); so must we."""

        torch = pytest.importorskip("torch")
        camp2p = pytest.importorskip("afd_plugin.connectors.npu.camp2p")

        group = camp2p.A2E_MX_GROUP_SIZE
        hidden = group * 2 - 5
        activation = torch.ones(1, hidden, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[127, 128]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

        restored = camp2p.mx_dequantize_a2e_activation(activation, scale)

        assert restored.shape == (1, hidden)
        # The short last group keeps its own scale, so the tail is doubled while
        # the first group stays at 1.0.
        assert torch.equal(restored[0, :group], torch.ones(group))
        assert torch.equal(restored[0, group:], torch.full((hidden - group,), 2.0))

class TestA2eMxfp8OperatorProbe:
    """The connector refuses mode 4 unless the loaded operator takes the pair.

    Reading the schema is the only way to tell an MXFP8-capable build from an
    older one before the kernel decides it: the Python call shape is accepted
    either way, so without this the failure surfaces inside the operator.
    """

    def _probe_with(self, monkeypatch, arguments):
        torch = pytest.importorskip("torch")
        from afd_plugin.compat.npu.ops import a2e_accepts_mxfp8_arguments

        packet = SimpleNamespace(
            default=SimpleNamespace(
                _schema=SimpleNamespace(
                    arguments=[SimpleNamespace(name=name) for name in arguments],
                ),
            ),
        )
        monkeypatch.setattr(torch.ops.afd_ascend, "a2e", packet, raising=False)
        return a2e_accepts_mxfp8_arguments()

    def test_accepts_an_operator_with_both_arguments(self, monkeypatch):
        assert self._probe_with(
            monkeypatch,
            ["x", "expert_ids", "scales", "compute_gate", "x_scale", "quant_mode"],
        )

    def test_rejects_an_operator_without_them(self, monkeypatch):
        assert not self._probe_with(
            monkeypatch,
            ["x", "expert_ids", "scales", "compute_gate"],
        )

    def test_reports_unsupported_when_the_operator_is_not_loaded(self, monkeypatch):
        """An unloaded extension is 'unknown', which the caller treats as no."""

        torch = pytest.importorskip("torch")
        from afd_plugin.compat.npu.ops import a2e_accepts_mxfp8_arguments

        monkeypatch.setattr(torch.ops.afd_ascend, "a2e", object(), raising=False)

        assert not a2e_accepts_mxfp8_arguments()
