# mypy: disable-error-code="attr-defined, name-defined"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage for the sync gate-on-Attention feature contract.

The public validator is the seam every AFD NPU run passes before execution, so it is
driven directly rather than through its private helpers. Only the two upstream modules
the validator reaches for -- the connector factory and the connector's extra-info type
-- are stand-ins; the validator itself is the real one.

Keeping the assertions on the public entry matters: it fails both when the helper
starts rejecting supported shapes and when the entry stops calling the helper.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import feature_validation
from afd_plugin.config import CAMP2P_CONNECTOR

_DSV4 = "deepseek_v4"


class _FakeExtraInfo:
    """Stands in for the connector extra info the factory would build."""

    def __init__(self) -> None:
        self.validated = 0
        self.ffn_cores = 8

    def aiv_num_for_role(self, role: str) -> int:
        return self.ffn_cores if role == "ffn" else 8

    def validate_supported(self) -> None:
        self.validated += 1


@pytest.fixture
def extra_info(monkeypatch):
    info = _FakeExtraInfo()
    factory = types.ModuleType("afd_plugin.connectors.factory")
    factory.AFDConnectorFactory = SimpleNamespace(
        parse_connector_extra_info=lambda connector, config: info,
    )
    camp2p = types.ModuleType("afd_plugin.connectors.npu.camp2p")
    camp2p.CAMP2PExtraInfo = _FakeExtraInfo
    monkeypatch.setitem(sys.modules, "afd_plugin.connectors.factory", factory)
    monkeypatch.setitem(sys.modules, "afd_plugin.connectors.npu.camp2p", camp2p)
    return info


def _afd_config(*, role: str = "attention"):
    return SimpleNamespace(
        role=role,
        connector=CAMP2P_CONNECTOR,
        compute_gate_on_attention=True,
    )


def _vllm_config(*, tp: int = 1, dbo: bool = False, ubatching: bool = False):
    hf_config = SimpleNamespace(model_type=_DSV4)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=SimpleNamespace(model_type=_DSV4),
        ),
        parallel_config=SimpleNamespace(
            enable_dbo=dbo,
            use_ubatching=ubatching,
            num_ubatches=1,
            tensor_parallel_size=tp,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
        speculative_config=None,
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_the_supported_shape_is_accepted_for_both_roles(extra_info, role):
    """Gate-on-Attention at TP=1 on the sync connector must pass validation."""

    feature_validation.fail_if_unsupported_npu_afd_features(
        _vllm_config(),
        afd_config=_afd_config(role=role),
    )

    assert extra_info.validated == 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dbo": True}, "DBO or ubatching"),
        ({"ubatching": True}, "DBO or ubatching"),
        ({"tp": 2}, "tensor-parallel size 1"),
    ],
)
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_unsupported_shapes_are_rejected(extra_info, kwargs, match, role):
    with pytest.raises(RuntimeError, match=match):
        feature_validation.fail_if_unsupported_npu_afd_features(
            _vllm_config(**kwargs),
            afd_config=_afd_config(role=role),
        )


def test_gate_on_ffn_keeps_its_previous_behaviour(extra_info):
    """Turning the Attention-side gate off must not pick up the new restrictions."""

    afd_config = SimpleNamespace(
        role="ffn",
        connector=CAMP2P_CONNECTOR,
        compute_gate_on_attention=False,
    )

    feature_validation.fail_if_unsupported_npu_afd_features(
        _vllm_config(tp=2),
        afd_config=afd_config,
    )

    assert extra_info.validated == 1


def _real_extra_info():
    """The connector's own type; importing it needs vLLM.

    ``camp2p`` pulls ``vllm.forward_context`` at module scope, so the real
    class is only reachable where vLLM is installed. The validator cases
    above use a stand-in and run everywhere; these two cover the class
    itself and skip on a bare CPU host.
    """

    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo

    return CAMP2PExtraInfo


def test_a_connector_level_gate_flag_does_not_block():
    """The switch is read from ``afd``; the connector-local copy has no say.

    Regression test for a rejection that used to live in ``validate_supported``: a
    supported
        shape was refused at startup whenever the flag also appeared under
        ``connector_extra_config``.
    """

    _real_extra_info().from_mapping(
        {"compute_gate_on_attention": True}
    ).validate_supported()


def test_quant_mode_is_still_rejected():
    """The transport mode really is unsupported, so that check has to stay."""

    with pytest.raises(RuntimeError, match="only quant_mode=0"):
        _real_extra_info().from_mapping({"quant_mode": 4}).validate_supported()


@pytest.mark.parametrize("cores", [4, 8])
@pytest.mark.parametrize("mode", ["normal", "persistent"])
def test_megaffn_communication_cores_fit_resident_kernel(extra_info, cores, mode):
    torch = pytest.importorskip("torch")
    config = _vllm_config()
    model = config.model_config
    model.enforce_eager = True
    model.dtype = torch.bfloat16
    model.hf_config.hidden_size = 4096
    model.hf_config.moe_intermediate_size = 1024
    model.hf_config.swiglu_limit = 0
    model.hf_config.num_hidden_layers = 1
    for name in (
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        setattr(config.parallel_config, name, 1)
    config.quant_config = None
    afd = _afd_config()
    afd.ffn_compute_backend = "mega_ffn"
    afd.mega_ffn_execution_mode = mode
    extra_info.ffn_cores = cores
    if mode == "persistent" and cores != 4:
        with pytest.raises(ValueError, match="ffn_core_num=4"):
            feature_validation.fail_if_unsupported_npu_afd_features(
                config, afd_config=afd
            )
    else:
        feature_validation.fail_if_unsupported_npu_afd_features(config, afd_config=afd)
