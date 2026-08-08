from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import afd_plugin.model_executor.models.forward_context as afd_forward_context
from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDControlPayload
from afd_plugin.distributed import resolve_role_rank
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.attention_model_runner import (
    AFDAttentionModelRunner,
    _is_ubatch_child_afd_context,
    fail_if_cuda_graph_enabled,
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
)


class _UbatchSlice:
    def __init__(self, token_start, token_stop, request_start, request_stop):
        self.token_slice = slice(token_start, token_stop)
        self.request_slice = slice(request_start, request_stop)

    @property
    def num_tokens(self):
        return self.token_slice.stop - self.token_slice.start


def _dp_metadata(tokens):
    return SimpleNamespace(num_tokens_across_dp_cpu=list(tokens))


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


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
            (payload.is_graph_capturing, payload.is_warmup),
        )

    def send_dp_metadata_list(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.sent_dp_metadata_lists.append(payload.dp_metadata_list)
        self.sent_dp_metadata_flags.append(
            (payload.is_graph_capturing, payload.is_warmup),
        )

    def close(self):
        self.closed = True


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.stopped = False

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


def _install_fake_vllm_forward_context(monkeypatch):
    forward_context_module = afd_forward_context.forward_context_module

    def create_forward_context():
        return SimpleNamespace(
            additional_kwargs={},
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
            ubatch_slices=None,
            batch_descriptor=SimpleNamespace(num_tokens=1),
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
    runner = object.__new__(AFDAttentionModelRunner)
    runner.connector = object()
    runner._afd_transaction_counter = 0

    metadata = runner._build_afd_metadata(None, 7)

    assert metadata.tokens_start_loc == [0]
    assert metadata.requests_start_loc == [0]
    assert metadata.tokens_lens == [7]
    assert metadata.tokens_unpadded_lens == [7]
    assert metadata.num_stages == 1
    assert metadata.connector is runner.connector


def test_attention_runner_installs_afd_metadata_on_forward_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs={"platform_key": "platform_value"},
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["platform_key"] == "platform_value"
    assert forward_context.additional_kwargs["afd_metadata"].tokens_lens == [5]
    assert set(runner.connector.dp_metadata_updates[0]) == {0}
    assert _tokens(runner.connector.dp_metadata_updates[0][0]) == [5]
    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0}
    assert _tokens(runner.connector.sent_dp_metadata_lists[0][0]) == [5]
    assert runner.connector.sent_dp_metadata_flags == [(False, False)]


def test_attention_runner_initializes_missing_forward_context_kwargs():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_suppress_metadata_send = True
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs=None,
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"].tokens_lens == [5]


def test_attention_runner_uses_padded_full_graph_tokens_for_afd_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 1)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=64),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [1]
    sent_metadata = runner.connector.sent_dp_metadata_lists[0][0]
    tokens = sent_metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64]


def test_attention_runner_sends_per_ubatch_dp_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
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

    runner._send_dp_metadata(None, ubatch_slices)

    assert set(runner.connector.dp_metadata_updates[0]) == {0, 1}
    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0, 1}


def test_attention_runner_skips_dp_metadata_send_for_ubatch_child_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    parent = runner._build_afd_metadata(
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        8,
    )
    child = build_ubatch_afd_metadata(
        parent,
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        1,
    )
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": child},
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
    )

    assert _is_ubatch_child_afd_context(forward_context, child)

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"] is child
    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.sent_dp_metadata_lists == []


def test_attention_runner_does_not_skip_single_stage_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=_dp_metadata([5]),
        ubatch_slices=None,
    )

    assert not _is_ubatch_child_afd_context(
        forward_context,
        runner._afd_pending_metadata,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert set(runner.connector.sent_dp_metadata_lists[0]) == {0}
    assert _tokens(runner.connector.sent_dp_metadata_lists[0][0]) == [5]


def test_ubatch_metadata_clones_parent_and_preserves_additional_kwargs():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.connector = object()
    runner._afd_transaction_counter = 0
    parent = runner._build_afd_metadata(
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        8,
    )
    parent.tokens_unpadded_lens = [3, 4]

    first = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 0
    )
    second = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 1
    )
    child_kwargs = build_ubatch_additional_kwargs(
        {"platform_key": "platform_value", "afd_metadata": parent},
        second,
    )

    assert first is not parent
    assert second is not parent
    assert first is not second
    assert first.stage_idx == 0
    assert first.tokens_lens == [3]
    assert second.stage_idx == 1
    assert second.tokens_start_loc == [3]
    assert second.requests_start_loc == [1]
    assert second.tokens_lens == [5]
    assert second.tokens_unpadded_lens == [4]
    assert child_kwargs["platform_key"] == "platform_value"
    assert child_kwargs["afd_metadata"] is second


def test_phase5_allows_two_way_ubatching_but_rejects_other_counts():
    fail_if_unsupported_ubatching(
        SimpleNamespace(
            parallel_config=_parallel_config(use_ubatching=True, num_ubatches=2),
        ),
    )

    with pytest.raises(RuntimeError, match="exactly two"):
        fail_if_unsupported_ubatching(
            SimpleNamespace(
                parallel_config=_parallel_config(use_ubatching=True, num_ubatches=4),
            ),
        )


def _ubatch_runner(uniform_decode, **parallel_overrides):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(**parallel_overrides),
    )
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **_kwargs: uniform_decode
    return runner


@pytest.mark.parametrize(
    (
        "parallel_overrides",
        "uniform_decode",
        "num_tokens",
        "padded_num_tokens",
        "allow_microbatching",
        "expected",
    ),
    [
        pytest.param(
            {"use_ubatching": False, "num_ubatches": 2},
            True,
            64,
            64,
            True,
            False,
            id="ubatching-disabled",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            True,
            64,
            64,
            False,
            False,
            id="microbatching-disallowed",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            True,
            16,
            16,
            True,
            False,
            id="decode-below-threshold",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            True,
            32,
            32,
            True,
            True,
            id="decode-at-threshold",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            False,
            256,
            256,
            True,
            False,
            id="prefill-below-threshold",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            False,
            512,
            512,
            True,
            True,
            id="prefill-at-threshold",
        ),
        pytest.param(
            {
                "use_ubatching": True,
                "num_ubatches": 2,
                "dbo_decode_token_threshold": 1,
            },
            True,
            1,
            1,
            True,
            False,
            id="fewer-tokens-than-ubatches",
        ),
        pytest.param(
            {
                "use_ubatching": True,
                "num_ubatches": 2,
                "dbo_decode_token_threshold": 2,
            },
            True,
            2,
            64,
            True,
            False,
            id="cudagraph-pad-empties-last-ubatch",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            True,
            32,
            64,
            True,
            False,
            id="cudagraph-pad-split-at-real-boundary",
        ),
        pytest.param(
            {"use_ubatching": True, "num_ubatches": 2},
            True,
            33,
            64,
            True,
            True,
            id="cudagraph-pad-split-inside-real",
        ),
    ],
)
def test_should_ubatch_single_rank(
    parallel_overrides,
    uniform_decode,
    num_tokens,
    padded_num_tokens,
    allow_microbatching,
    expected,
):
    runner = _ubatch_runner(uniform_decode, **parallel_overrides)
    batch_descriptor = SimpleNamespace(num_tokens=padded_num_tokens)

    assert (
        runner._should_ubatch_single_rank(
            batch_descriptor,
            (),
            {
                "num_tokens": num_tokens,
                "num_reqs": num_tokens,
                "num_scheduled_tokens_np": [1] * num_tokens,
                "max_num_scheduled_tokens": 1,
                "use_cascade_attn": False,
                "allow_microbatching": allow_microbatching,
            },
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "dp_size",
        "parent_should_ubatch",
        "num_tokens",
        "padded_num_tokens",
        "expected",
    ),
    [
        pytest.param(1, False, 48, 64, True, id="dp1-enables-local-dbo"),
        pytest.param(1, False, 2, 64, False, id="dp1-rejects-empty-last-ubatch"),
        pytest.param(2, True, 2, 64, True, id="dp2-keeps-coordinated-true"),
        pytest.param(2, False, 48, 64, False, id="dp2-keeps-coordinated-false"),
        pytest.param(2, True, 1, 1, False, id="dp2-rejects-empty-first-ubatch"),
        pytest.param(2, True, 2, 2, True, id="dp2-keeps-minimal-nonempty-split"),
    ],
)
def test_determine_batch_execution_overrides_ubatch_only_for_dp1(
    monkeypatch,
    dp_size,
    parent_should_ubatch,
    num_tokens,
    padded_num_tokens,
    expected,
):
    runner = _ubatch_runner(
        True,
        data_parallel_size=dp_size,
        use_ubatching=True,
        num_ubatches=2,
        dbo_decode_token_threshold=2,
    )
    batch_descriptor = SimpleNamespace(num_tokens=padded_num_tokens)
    parent_result = (
        "cudagraph-mode",
        batch_descriptor,
        parent_should_ubatch,
        "num-tokens-across-dp",
        "cudagraph-stats",
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "_determine_batch_execution_and_padding",
        lambda _self, *_args, **_kwargs: parent_result,
    )

    result = runner._determine_batch_execution_and_padding(
        num_tokens=num_tokens,
        num_reqs=num_tokens,
        num_scheduled_tokens_np=[1] * num_tokens,
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )

    assert result == (
        "cudagraph-mode",
        batch_descriptor,
        expected,
        "num-tokens-across-dp",
        "cudagraph-stats",
    )


def test_attention_runner_inherits_native_dummy_run_microbatching():
    assert "_dummy_run" in AFDAttentionModelRunner.__dict__


def test_attention_runner_steps_gpu_profiler(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.prof = _StepProfiler()

    def execute_model(_self, *args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        AFDAttentionModelRunner.__mro__[1],
        "execute_model",
        execute_model,
    )

    result = runner.execute_model("scheduler", "intermediate")

    assert runner.prof.steps == 1
    assert result == (("scheduler", "intermediate"), {})


def test_attention_runner_preserves_native_shutdown(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.prof = _StepProfiler()
    runner.connector = _RecordingConnector()
    native_shutdowns = []
    monkeypatch.setattr(
        GPUModelRunner,
        "shutdown",
        lambda self: native_shutdowns.append(self),
    )

    runner.shutdown()

    assert native_shutdowns == [runner]
    assert runner.prof.stopped is True
    assert runner.connector.closed is True


def test_attention_warmup_preserves_profile_seq_lens():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner._is_warmup = False
    runner._afd_pending_metadata = None
    runner._afd_suppress_metadata_send = False
    runner._afd_is_graph_capturing = False
    runner._build_afd_metadata = lambda *_args: object()
    runner._build_capture_dp_metadata = lambda *_args: object()
    runner._send_dp_metadata = lambda *_args: None
    dummy_runs = []
    runner._dummy_run = lambda *args, **kwargs: dummy_runs.append((args, kwargs))

    runner._warmup_and_capture(
        SimpleNamespace(num_tokens=8, uniform=True, num_active_loras=0),
        CUDAGraphMode.FULL,
        profile_seq_lens=7,
    )

    assert [kwargs["profile_seq_lens"] for _, kwargs in dummy_runs] == [7, 7]


def test_forward_context_provider_installs_metadata_before_model_forward(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    from afd_plugin.model_executor.models.forward_context import (
        use_afd_metadata_provider,
    )

    with use_afd_metadata_provider(runner):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is not None
    assert metadata.tokens_lens == [1]
    assert forward_context.additional_kwargs["afd_metadata"] is metadata
    assert runner.connector.sent_dp_metadata_lists
    assert fake_forward_context.create_forward_context is original_create


def test_forward_context_provider_can_install_without_sending_metadata(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = True
    runner._afd_suppress_metadata_send = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 1)
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    from afd_plugin.model_executor.models.forward_context import (
        use_afd_metadata_provider,
    )

    with use_afd_metadata_provider(runner):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is runner._afd_pending_metadata
    assert runner.connector.sent_dp_metadata_lists == []
    assert fake_forward_context.create_forward_context is original_create


def test_attention_runtime_rejects_unsupported_cuda_graph_modes():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode="PIECEWISE"),
        parallel_config=_parallel_config(),
    )

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        fail_if_cuda_graph_enabled(vllm_config)


def test_attention_runtime_allows_full_decode_only_cuda_graph():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
        parallel_config=_parallel_config(),
    )

    fail_if_cuda_graph_enabled(vllm_config)


def test_attention_runner_forwards_capture_and_warmup_flags():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    runner._send_dp_metadata(_dp_metadata([1]), None)

    assert runner.connector.dp_metadata_update_flags == [(True, True)]
    assert runner.connector.sent_dp_metadata_flags == [(True, True)]


def test_attention_runner_builds_capture_dp_metadata_for_native_dp():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2),
    )

    metadata = runner._build_capture_dp_metadata(64)

    tokens = metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64, 64]
    assert not hasattr(metadata, "max_tokens_across_dp_cpu")


def test_afd_rank_derives_from_data_parallel_rank():
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2, data_parallel_rank=1),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == 1


# --- TP rank derivation tests ---


def _parallel_config_with_tp(
    *,
    dp_size=1,
    dp_rank=0,
    tp_size=1,
    **overrides,
):
    values = {
        "data_parallel_size": dp_size,
        "data_parallel_rank": dp_rank,
        "prefill_context_parallel_size": 1,
        "tensor_parallel_size": tp_size,
        "use_ubatching": False,
        "num_ubatches": 1,
        "dbo_decode_token_threshold": 32,
        "dbo_prefill_token_threshold": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_afd_rank_derives_from_tp_rank_dp1_tp2(monkeypatch):
    """DP=1, TP=2: each TP worker gets a unique role_rank."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=4,
        num_ffn_ranks=4,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=1, tp_size=2),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == 1


def test_afd_rank_derives_from_pcp_rank_dp1_pcp2(monkeypatch):
    """DP=1, PCP=2, TP=1: each PCP worker gets a unique role_rank."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(
            dp_size=1,
            tp_size=1,
            prefill_context_parallel_size=2,
        ),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == 1


@pytest.mark.parametrize(("pcp_rank", "expected_role_rank"), [(0, 16), (7, 23)])
def test_afd_rank_uses_global_dp_rank_for_dp3_pcp8_node1(
    monkeypatch,
    pcp_rank,
    expected_role_rank,
):
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=pcp_rank),
    )
    config = AFDConfig(
        role="attention",
        connector="CAMAsyncAFDConnector",
        num_attention_ranks=24,
        num_ffn_ranks=8,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(
            dp_size=3,
            dp_rank=2,
            prefill_context_parallel_size=8,
        ),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == expected_role_rank


def test_afd_rank_derives_from_dp_and_tp_ranks_dp2_tp2(monkeypatch):
    """DP=2, TP=2: role_rank = dp_rank * tp_size + tp_rank."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=4,
        num_ffn_ranks=4,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=2, dp_rank=1, tp_size=2),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == 3


def test_afd_rank_is_zero_when_dp1_pcp1_tp1():
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=1,
        num_ffn_ranks=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=1, tp_size=1),
    )

    role_rank = resolve_role_rank(vllm_config, config)

    assert role_rank == 0


def test_afd_rank_raises_for_out_of_range_dp2_tp2(monkeypatch):
    """role_rank must stay within role_size."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    config = AFDConfig(
        role="attention",
        connector="P2pNcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=2, dp_rank=1, tp_size=2),
    )

    with pytest.raises(ValueError, match="out of range"):
        resolve_role_rank(vllm_config, config)
