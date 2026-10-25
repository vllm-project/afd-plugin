# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDRoutedExpertsConnector,
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
            seq_lens=[-1],
        )


def test_connector_metadata_allows_empty_receive():
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_lens=[0],
    )

    assert metadata.total_tokens == 0
    assert metadata.validate_tensor_shape((0, 16))


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


def test_a2f_payload_preserves_token_ids():
    input_ids = torch.tensor([11, 13, 17], dtype=torch.int32)
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

    assert output.input_ids is input_ids
    assert output.input_ids.dtype is torch.int32
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


class _BootstrappingConnector(_MinimalConnector, AFDRoutedExpertsConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    def bootstrap(self):
        self.events.append("bootstrap")
        if self.vllm_config.fail_bootstrap:
            raise RuntimeError("bootstrap failed")

    def init_afd_connector(self, *, moe_layer_indices=None, local_experts=None):
        return None

    def run_experts(self, payload, weights):
        return payload.hidden_states

    def synchronize_lanes(self):
        return None

    def close(self):
        self.events.append("close")


def _register(monkeypatch, name, connector_cls):
    monkeypatch.setitem(AFDConnectorFactory._registry, name, lambda: connector_cls)


def _single_rank_config(
    *, fail_bootstrap=False, data_parallel_size=1, data_parallel_rank=0, tp_size=1
):
    return SimpleNamespace(
        additional_config={},
        fail_bootstrap=fail_bootstrap,
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
    )


def test_create_bootstrapped_connector_skips_other_connectors(monkeypatch):
    """A connector that is not an ``AFDRoutedExpertsConnector`` is not
    created before vLLM's distributed init; its runner creates it later.

    Why: P2P and the NPU connectors keep their existing bring-up order.
    """
    _register(monkeypatch, "MinimalConnector", _MinimalConnector)

    connector = AFDConnectorFactory.create_bootstrapped_connector(
        0,
        0,
        _single_rank_config(),
        AFDConfig(connector="MinimalConnector", role="ffn"),
    )

    assert connector is None


def test_create_bootstrapped_connector_creates_and_bootstraps(monkeypatch):
    """A routed-experts connector is created with the worker's ranks and
    bootstrapped.

    Why: workers call this before vLLM's distributed init so the connector's
    NCCL environment precedes vLLM's NCCL groups.
    """
    _register(monkeypatch, "BootstrappingConnector", _BootstrappingConnector)

    connector = AFDConnectorFactory.create_bootstrapped_connector(
        3,
        1,
        _single_rank_config(),
        AFDConfig(connector="BootstrappingConnector", role="ffn"),
    )

    assert isinstance(connector, _BootstrappingConnector)
    assert (connector.rank, connector.local_rank, connector.role_rank) == (3, 1, 0)
    assert connector.events == ["bootstrap"]


def test_create_bootstrapped_connector_derives_tp_rank_from_worker_rank(
    monkeypatch,
):
    """With TP > 1 the role rank's TP part comes from the worker rank.

    Why: vLLM's TP group does not exist before its distributed init; reading
    it there would fail. TP is vLLM's innermost rank dimension, so the worker
    rank modulo the TP size is the TP rank.
    """
    _register(monkeypatch, "BootstrappingConnector", _BootstrappingConnector)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", None)

    connector = AFDConnectorFactory.create_bootstrapped_connector(
        1,
        1,
        _single_rank_config(data_parallel_size=2, data_parallel_rank=1, tp_size=2),
        AFDConfig(
            connector="BootstrappingConnector",
            role="attention",
            num_attention_ranks=4,
        ),
    )

    assert isinstance(connector, _BootstrappingConnector)
    # DP rank 1, TP rank 1 of 2.
    assert connector.role_rank == 3


def test_create_bootstrapped_connector_closes_on_bootstrap_failure(monkeypatch):
    """A failed bootstrap closes the connector before the error propagates.

    Why: the worker never receives the connector, so nobody else closes it.
    """
    _register(monkeypatch, "BootstrappingConnector", _BootstrappingConnector)
    created: list[_BootstrappingConnector] = []
    original_init = _BootstrappingConnector.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(_BootstrappingConnector, "__init__", recording_init)

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        AFDConnectorFactory.create_bootstrapped_connector(
            0,
            0,
            _single_rank_config(fail_bootstrap=True),
            AFDConfig(connector="BootstrappingConnector", role="ffn"),
        )

    assert created[0].events == ["bootstrap", "close"]
