# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext

import afd_plugin.model_executor.models.forward_context as afd_forward_context
from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDControlPayload
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
    use_afd_metadata_provider,
)
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from tests.unit.fakes.dp_metadata import dp_tokens as _tokens  # noqa: E402


class _UbatchSlice:
    def __init__(self, token_start, token_stop, request_start, request_stop):
        self.token_slice = slice(token_start, token_stop)
        self.request_slice = slice(request_start, request_stop)

    @property
    def num_tokens(self):
        return self.token_slice.stop - self.token_slice.start


def _dp_metadata(tokens):
    return SimpleNamespace(num_tokens_across_dp_cpu=list(tokens))


class _RecordingConnector:
    world_rank = 1

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []
        self.dp_metadata_update_flags = []
        self.sent_dp_metadata_flags = []
        self.closed = False
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_updates.append(payload.dp_metadata_list)
        self.dp_metadata_update_flags.append(
            (
                payload.is_graph_capturing,
                payload.is_warmup,
                payload.is_profile,
            ),
        )

    def send_dp_metadata_list(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.sent_dp_metadata_lists.append(payload.dp_metadata_list)
        self.sent_dp_metadata_flags.append(
            (
                payload.is_graph_capturing,
                payload.is_warmup,
                payload.is_profile,
            ),
        )

    def close(self):
        self.closed = True


def _install_fake_vllm_forward_context(monkeypatch):
    forward_context_module = afd_forward_context.forward_context_module

    def create_forward_context():
        return ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={},
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
            ubatch_slices=None,
            batch_descriptor=BatchDescriptor(num_tokens=1),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )

    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        create_forward_context,
    )
    return forward_context_module, create_forward_context


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "is_moe_model": True,
        "prefill_context_parallel_size": 1,
        "tensor_parallel_size": 1,
        "use_ubatching": False,
        "num_ubatches": 1,
        "dbo_decode_token_threshold": 32,
        "dbo_prefill_token_threshold": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_attention_runner_builds_single_stage_metadata():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.connector = object()
    runner._afd_transaction_counter = 0

    metadata = runner.build_afd_metadata(None, 7)

    assert metadata.tokens_start_loc == [0]
    assert metadata.requests_start_loc == [0]
    assert metadata.tokens_lens == [7]
    assert metadata.tokens_unpadded_lens == [7]
    assert metadata.num_stages == 1
    assert metadata.connector is runner.connector


def test_attention_runner_installs_afd_metadata_on_forward_context():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 5)
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"platform_key": "platform_value"},
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=5),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    runner.install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["platform_key"] == "platform_value"
    assert forward_context.additional_kwargs["afd_metadata"].tokens_lens == [5]
    assert set(runner.connector.dp_metadata_updates[0]) == {0}
    assert _tokens(runner.connector.dp_metadata_updates[0][0]) == [5]
    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0}
    assert _tokens(runner.connector.sent_dp_metadata_lists[0][0]) == [5]
    assert runner.connector.sent_dp_metadata_flags == [(False, False, False)]


def test_attention_runner_initializes_missing_forward_context_kwargs():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_suppress_metadata_send = True
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 5)
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs=None,
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=5),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    runner.install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"].tokens_lens == [5]


def test_attention_runner_uses_padded_full_graph_tokens_for_afd_metadata():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 1)
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=64),
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )

    runner.install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [1]
    sent_metadata = runner.connector.sent_dp_metadata_lists[0][0]
    tokens = sent_metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64]


def test_attention_runner_sends_per_ubatch_dp_metadata():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None
    ubatch_slices = [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)]

    runner.send_dp_metadata(None, ubatch_slices)

    assert set(runner.connector.dp_metadata_updates[0]) == {0, 1}
    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0, 1}
    assert [
        _tokens(value) for value in runner.connector.sent_dp_metadata_lists[0].values()
    ] == [[3], [5]]


@pytest.fixture
def mrv2_child_context():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    slices = [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)]
    contexts = [
        ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={},
            dp_metadata=_dp_metadata([stage.num_tokens]),
            ubatch_slices=None,
            batch_descriptor=BatchDescriptor(num_tokens=stage.num_tokens),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        for stage in slices
    ]
    runner.install_mrv2_ubatch_metadata(slices, contexts, 8)
    # Exercise child-context detection independently of parent send suppression.
    runner._afd_suppress_metadata_send = False
    return runner, contexts[1]


def test_attention_runner_skips_dp_metadata_send_for_ubatch_child_context(
    mrv2_child_context,
):
    runner, forward_context = mrv2_child_context
    child = forward_context.additional_kwargs["afd_metadata"]

    runner.install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"] is child
    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.sent_dp_metadata_lists == []


def test_mrv2_missing_parent_metadata_uses_complete_public_installer():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.vllm_config = SimpleNamespace(parallel_config=_parallel_config())
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_suppress_metadata_send = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    ubatch_slices = [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)]
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=ubatch_slices,
        batch_descriptor=BatchDescriptor(num_tokens=8),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    runner.install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata is not None
    assert metadata.transaction_id == "afd-0"
    assert metadata.tokens_lens == [3, 5]
    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0, 1}


@pytest.mark.parametrize("has_control_plane", [False, True])
def test_forward_context_provider_installs_metadata_before_model_forward(
    monkeypatch, has_control_plane
):
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    if not has_control_plane:
        monkeypatch.setattr(runner.connector, "control_plane", None)
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    with use_afd_metadata_provider(
        runner.install_afd_metadata_on_forward_context,
    ):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is not None
    assert metadata.tokens_lens == [1]
    assert forward_context.additional_kwargs["afd_metadata"] is metadata
    assert bool(runner.connector.sent_dp_metadata_lists) is has_control_plane
    assert bool(runner.connector.dp_metadata_updates) is has_control_plane
    assert fake_forward_context.create_forward_context is original_create


def test_forward_context_provider_can_install_without_sending_metadata(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = True
    runner._afd_suppress_metadata_send = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 1)
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    with use_afd_metadata_provider(
        runner.install_afd_metadata_on_forward_context,
    ):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata.tokens_lens == [1]
    assert runner.connector.sent_dp_metadata_lists == []
    assert fake_forward_context.create_forward_context is original_create


def test_attention_runner_forwards_capture_and_warmup_flags():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    runner.send_dp_metadata(_dp_metadata([1]), None)

    assert runner.connector.dp_metadata_update_flags == [(True, True, False)]
    assert runner.connector.sent_dp_metadata_flags == [(True, True, False)]


def test_attention_runner_forwards_profile_flag():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_is_profile = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    runner.send_dp_metadata(_dp_metadata([1]), None)

    assert runner.connector.dp_metadata_update_flags == [(False, False, True)]
    assert runner.connector.sent_dp_metadata_flags == [(False, False, True)]


def test_attention_runner_builds_capture_dp_metadata_for_native_dp():
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2),
    )

    metadata = runner.build_capture_dp_metadata(64)

    tokens = metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64, 64]
    assert not hasattr(metadata, "max_tokens_across_dp_cpu")
