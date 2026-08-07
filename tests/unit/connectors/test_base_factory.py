from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch")

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import (
    AFDA2FTransportSpec,
    AFDA2FTransferPayload,
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDTransferContext,
    AFDTransferMetadata,
)


def test_dummy_connector_is_not_registered():
    with pytest.raises(ValueError, match="unsupported AFD connector type"):
        AFDConnectorFactory.get_connector_class("dummy")


def test_p2p_extra_info_rejects_unknown_fields():
    source = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "attention",
                "connector_extra_config": {"core_num": 8},
            },
        },
    )

    with pytest.raises(ValueError, match="does not support connector_extra_config"):
        AFDConnectorFactory.parse_connector_extra_info(
            "P2pNcclAFDConnector",
            source,
        )


def test_backend_connector_modules_are_registered_by_backend_package():
    pytest.importorskip("vllm")
    pytest.importorskip("torch_npu")

    assert (
        AFDConnectorFactory.get_connector_class("P2pNcclAFDConnector").__module__
        == "afd_plugin.connectors.gpu.p2p"
    )
    assert (
        AFDConnectorFactory.get_connector_class("CAMP2pAFDConnector").__module__
        == "afd_plugin.connectors.npu.camp2p"
    )


def test_connector_metadata_validates_sequence_lengths():
    with pytest.raises(ValueError, match="sequence lengths"):
        AFDTransferMetadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[0],
        )


def test_attn_output_carries_transfer_context():
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=2,
        seq_lens=[3],
    )
    context = AFDTransferContext(metadata=metadata)
    output = AFDA2FTransferPayload(
        hidden_states="hidden",
        context=context,
    )

    assert output.hidden_states == "hidden"
    assert output.context is context
    assert output.context.metadata is metadata
    assert output.context.states is None
    assert repr(output).startswith("AFDA2FTransferPayload(")


def test_a2f_transport_spec_and_payload_preserve_token_id_dtype():
    transport_spec = AFDA2FTransportSpec(
        input_ids_dtype=torch.int64,
    )
    input_ids = torch.tensor([11, 13, 17], dtype=torch.int64)
    output = AFDA2FTransferPayload(
        hidden_states=torch.ones(3, 2),
        context=AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=1,
                stage_idx=2,
                seq_lens=[3],
            ),
        ),
        input_ids=input_ids,
    )

    assert transport_spec.version == 1
    assert transport_spec.input_ids_dtype is torch.int64
    assert output.input_ids is input_ids
    assert output.input_ids.dtype is torch.int64
    assert torch.equal(output.input_ids, input_ids)


class _MinimalConnector(AFDConnectorBase):
    @classmethod
    def parse_extra_config(cls, raw):
        return None

    @property
    def is_initialized(self):
        return True

    def close(self):
        return None

    def init_afd_connector(self):
        return None

    def send_attn_output(self, hidden_states, context, **kwargs):
        return None

    def recv_ffn_output(self, ref_tensor, ubatch_idx=0, **kwargs):
        return ref_tensor

    def recv_attn_output(self, ubatch_idx=0, **kwargs):
        return AFDA2FTransferPayload(
            hidden_states=None,
            context=AFDTransferContext(
                metadata=AFDTransferMetadata.create_ffn_metadata(
                    layer_idx=0,
                    stage_idx=0,
                    seq_lens=[1],
                ),
            ),
        )

    def send_ffn_output(self, ffn_output, context, **kwargs):
        return None


def test_connector_base_contract_can_be_implemented():
    # Instantiation (without running __init__) verifies _MinimalConnector
    # overrides every abstract method of the connector contract, so this test
    # fails when the AFDConnectorBase interface changes.
    connector = object.__new__(_MinimalConnector)

    assert connector.is_initialized is True
    payload = connector.recv_attn_output()
    assert payload.context.metadata.seq_lens == [1]


def test_factory_resolves_role_rank_before_connector_construction(monkeypatch):
    connector_name = "MinimalConnector"
    monkeypatch.setitem(
        AFDConnectorFactory._registry,
        connector_name,
        lambda: _MinimalConnector,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )

    connector = AFDConnectorFactory.create_connector(
        1,
        0,
        vllm_config,
        AFDConfig(
            connector=connector_name,
            role="attention",
            num_attention_ranks=2,
        ),
    )

    assert connector.role_rank == 1
