# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 routing and probability contracts at the FaaS transport boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")

from afd_plugin.config import afd_config_from_mapping  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu import ffn_faas  # noqa: E402
from tests.unit.fakes.vllm_config import npu_vllm_config  # noqa: E402


@pytest.fixture
def faas_config():
    config = npu_vllm_config(
        connector="FaaSAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        tensor_parallel_size=4,
        num_attention_ranks=12,
        num_ffn_ranks=4,
        extra_config={"attn_ranks_per_dp": 4},
    )
    config.model_config.hf_config = SimpleNamespace(
        model_type="deepseek_v4", routed_scaling_factor=1.5
    )
    return config


@pytest.fixture
def transport(faas_config, monkeypatch):
    connector = ffn_faas.FaaSAsyncAFDConnector(
        0,
        0,
        faas_config,
        afd_config_from_mapping(faas_config.additional_config["afd"]),
        0,
    )
    packets = []
    result = torch.full((2, 16), 2.0, dtype=torch.bfloat16)

    def send(states, ids, probabilities, *, layer):
        packets.append((states, ids, probabilities, layer))

    monkeypatch.setattr(
        connector, "_session", SimpleNamespace(send=send, receive=lambda: result)
    )
    return connector, packets, result


def test_faas_transport_preserves_native_v4_fp32_combine_probabilities(transport):
    connector, packets, result = transport
    states = torch.ones((2, 16), dtype=torch.bfloat16)
    ids = torch.tensor([[1, 7], [3, 9]], dtype=torch.int32)
    probabilities = torch.tensor([[0.1234567, 0.7654321], [0.2345678, 0.6543219]])
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=2, stage_idx=0, seq_len=2
        )
    )

    connector.send_attn_output(
        states, context, topk_ids=ids, topk_weights=probabilities
    )
    output = connector.recv_ffn_output(ref_tensor=states, context=context)

    assert len(packets) == 1
    sent_states, sent_ids, sent_probabilities, sent_layer = packets[0]
    assert sent_states is states
    assert sent_ids is ids
    assert sent_layer == 2
    assert sent_probabilities.dtype == torch.float32
    assert torch.equal(sent_probabilities, probabilities)
    assert output is result


def test_v4_requires_its_native_hash_and_sqrtsoftplus_router(transport):
    connector, _, _ = transport
    with pytest.raises(RuntimeError, match="native Hash/sqrtsoftplus router"):
        connector.select_experts()


@pytest.mark.parametrize("model_type", ["deepseek_v2", "deepseek_v3"])
def test_faas_connector_rejects_non_v4_models(faas_config, model_type):
    faas_config.model_config.hf_config.model_type = model_type
    with pytest.raises(ValueError, match="requires DeepSeek-V4 W4A8"):
        ffn_faas.FaaSAsyncAFDConnector(
            0,
            0,
            faas_config,
            afd_config_from_mapping(faas_config.additional_config["afd"]),
            0,
        )
