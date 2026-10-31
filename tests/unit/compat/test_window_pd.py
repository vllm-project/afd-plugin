# SPDX-License-Identifier: Apache-2.0
"""PD admission keeps the same role, KV, and topology guards for Window."""

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_deepseek_v4_pd,
)
from afd_plugin.config import AFDConfig


@pytest.fixture(params=("P2pHcclAFDConnector", "WindowAFDConnector"))
def pd_config(request):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=2),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeHybridConnector",
            kv_role="kv_consumer",
            engine_id="decode",
            kv_port=31000,
            kv_parallel_size=1,
            kv_connector_extra_config={
                "prefill": {"dp_size": 2, "tp_size": 1},
                "decode": {"dp_size": 2, "tp_size": 1},
            },
        ),
    )
    return config, AFDConfig(role="attention", connector=request.param)


def test_pd_accepts_both_connectors(pd_config):
    _fail_if_unsupported_deepseek_v4_pd(*pd_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kv_connector", "Other", "MooncakeHybridConnector"),
        ("kv_role", "kv_producer", "kv_consumer"),
        ("kv_parallel_size", 2, "kv_parallel_size=1"),
        ("engine_id", "", "non-empty engine_id"),
        ("kv_port", 0, "kv_port"),
    ],
)
def test_window_keeps_existing_kv_guards(pd_config, field, value, message):
    config, afd = pd_config
    setattr(config.kv_transfer_config, field, value)
    with pytest.raises(RuntimeError, match=message):
        _fail_if_unsupported_deepseek_v4_pd(config, afd)


def test_window_keeps_attention_role_and_topology_guards(pd_config):
    config, afd = pd_config
    with pytest.raises(RuntimeError, match="only to Attention"):
        _fail_if_unsupported_deepseek_v4_pd(
            config, AFDConfig(role="ffn", connector=afd.connector)
        )
    config.kv_transfer_config.kv_connector_extra_config["decode"]["dp_size"] = 1
    with pytest.raises(RuntimeError, match="topology must match Attention"):
        _fail_if_unsupported_deepseek_v4_pd(config, afd)


def test_other_connector_still_rejected(pd_config):
    config, _ = pd_config
    with pytest.raises(RuntimeError, match="requires P2pHcclAFDConnector or Window"):
        _fail_if_unsupported_deepseek_v4_pd(
            config, AFDConfig(role="attention", connector="CAMP2pAFDConnector")
        )
