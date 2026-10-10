# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""CPU-only contracts for the native Ascend ModelRunner V2 wrapper."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.dp_utils import should_skip_dp_coordination  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils  # noqa: E402
from vllm.v1.worker.gpu import model_runner as native_v2  # noqa: E402
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402
from vllm_ascend.worker.v2 import model_runner as native_ascend_v2  # noqa: E402
from vllm_ascend.worker.v2.model_runner import NPUModelRunner  # noqa: E402

from afd_plugin.model_executor.models import (  # noqa: E402
    forward_context as afd_context,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as npu_v2  # noqa: E402

Runner = npu_v2.AFDNPUAttentionModelRunnerV2


def _runner(*, async_dp=False):
    runner = object.__new__(Runner)
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    runner.afd_config = SimpleNamespace(
        async_dp=async_dp, num_attention_ranks=2, num_ffn_ranks=2
    )
    runner.connector = SimpleNamespace(
        extra_info=SimpleNamespace(async_moe_ubatching=False)
    )
    runner.cudagraph_manager = None
    runner.prof = object()
    runner._afd_pending_metadata = object()
    runner._afd_async_moe_ubatch_metadata = None
    runner._afd_suppress_metadata_send = True
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_is_graph_replaying = True
    runner._afd_is_profile = False
    return runner


def _state(runner):
    return (
        runner._afd_pending_metadata,
        runner._afd_async_moe_ubatch_metadata,
        runner._afd_suppress_metadata_send,
        runner._is_warmup,
        runner._afd_is_graph_capturing,
        runner._afd_is_graph_replaying,
        runner._afd_is_profile,
    )


def test_execute_and_capture_match_vllm_030_signatures():
    for method in ("execute_model", "capture_model"):
        assert inspect.signature(
            getattr(Runner, method), eval_str=True
        ) == inspect.signature(getattr(GPUModelRunner, method), eval_str=True)
    assert "context_len" in inspect.signature(NPUModelRunner.execute_model).parameters
    assert (
        "valid_dummy_state_slots"
        in inspect.signature(NPUModelRunner.execute_model).parameters
    )


@pytest.mark.parametrize("async_dp", [False, True])
@pytest.mark.parametrize("raise_in_native", [False, True])
def test_execute_forwards_dummy_profile_and_context_then_restores(
    monkeypatch, raise_in_native, async_dp
):
    runner = _runner(async_dp=async_dp)
    original_state = _state(runner)
    step_calls: list[object] = []
    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", step_calls.append)
    context = SimpleNamespace(additional_kwargs={})
    forward_module = afd_context.forward_context_module
    monkeypatch.setattr(forward_module, "create_forward_context", lambda: context)
    original_factory = forward_module.create_forward_context
    runner.install_afd_metadata_on_forward_context = lambda ctx: (
        ctx.additional_kwargs.update(afd_metadata="installed")
    )
    forwarded = []

    def native_execute(self, *args, **kwargs):
        forwarded.append((args, kwargs))
        assert should_skip_dp_coordination() is async_dp
        assert forward_module.create_forward_context().additional_kwargs == {
            "afd_metadata": "installed"
        }
        assert self._afd_is_profile is True
        self._afd_pending_metadata = None
        self._afd_suppress_metadata_send = False
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_is_graph_replaying = False
        if raise_in_native:
            raise RuntimeError("native execute failed")
        return "native-result"

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=7)
    intermediate = object()
    if raise_in_native:
        with pytest.raises(RuntimeError, match="native execute failed"):
            runner.execute_model(
                scheduler_output, intermediate, True, True, True, 17, True
            )
    else:
        assert (
            runner.execute_model(
                scheduler_output, intermediate, True, True, True, 17, True
            )
            == "native-result"
        )
    assert forwarded == [
        (
            (scheduler_output, intermediate),
            {
                "dummy_run": True,
                "skip_attn_for_dummy_run": True,
                "is_profile": True,
                "context_len": 17,
                "valid_dummy_state_slots": True,
            },
        )
    ]
    assert not should_skip_dp_coordination()
    assert step_calls == [runner.prof]
    assert _state(runner) == original_state
    assert forward_module.create_forward_context is original_factory


@pytest.mark.parametrize("async_dp", [False, True])
@pytest.mark.parametrize("stages_enabled", [False, True])
@pytest.mark.parametrize("raise_in_native", [False, True])
def test_execute_installs_local_stages_without_control_and_restores(
    monkeypatch, async_dp, stages_enabled, raise_in_native
):
    runner = _runner(async_dp=async_dp)
    runner.connector.control_plane = None
    runner.connector.extra_info.async_moe_ubatching = stages_enabled
    previous = object()
    current = object()
    runner._afd_async_moe_ubatch_metadata = previous
    context = SimpleNamespace(additional_kwargs={}, ubatch_slices=None)
    events = []

    @contextmanager
    def stage_scope(self):
        events.append("enter")
        self._afd_async_moe_ubatch_metadata = current
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(npu_v2, "use_async_cam_stage_metadata", stage_scope)
    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", lambda _: None)
    monkeypatch.setattr(
        afd_context.forward_context_module, "create_forward_context", lambda: context
    )

    def native_execute(self, *args, **kwargs):
        ctx = afd_context.forward_context_module.create_forward_context()
        assert ctx.additional_kwargs["afd_metadata"] is self._afd_pending_metadata
        stage_plan = ctx.additional_kwargs.get(npu_v2.ASYNC_MOE_UBATCH_METADATA_KEY)
        assert stage_plan is (current if async_dp and stages_enabled else None)
        if raise_in_native:
            raise RuntimeError("native failed")
        return "result"

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    if raise_in_native:
        with pytest.raises(RuntimeError, match="native failed"):
            runner.execute_model(None)
    else:
        assert runner.execute_model(None) == "result"
    assert events == (["enter", "exit"] if async_dp and stages_enabled else [])
    assert runner._afd_async_moe_ubatch_metadata is previous


@pytest.mark.parametrize("profile_only", [False, True])
@pytest.mark.parametrize("raise_in_native", [False, True])
def test_capture_forwards_new_inputs_and_restores(
    monkeypatch, profile_only, raise_in_native
):
    runner = _runner()
    original_state = _state(runner)
    descs = [
        SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=1, num_tokens=8),
        SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=2, num_tokens=16),
        SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=3, num_tokens=32),
    ]
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: descs},
        _max_full_descs_to_capture=2 if profile_only else None,
    )
    events = []
    runner.build_afd_metadata = lambda _slices, tokens: SimpleNamespace(tokens=tokens)
    runner.build_capture_dp_metadata = lambda tokens: tokens
    runner.send_dp_metadata = lambda tokens, _slices: events.append(
        (tokens, runner._is_warmup, runner._afd_is_graph_capturing)
    )
    runner.install_afd_metadata_on_forward_context = lambda _ctx: None
    original_prepare = cudagraph_utils.prepare_inputs_to_capture
    sentinels = [object() for _ in range(6)]
    prepare_calls = []

    def native_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return "native-state"

    monkeypatch.setattr(cudagraph_utils, "prepare_inputs_to_capture", native_prepare)
    forwarded = []

    def native_capture(self, *, profile_only=False):
        forwarded.append(profile_only)
        native_signature = inspect.signature(original_prepare)
        wrapper_signature = inspect.signature(cudagraph_utils.prepare_inputs_to_capture)
        assert list(wrapper_signature.parameters) == list(native_signature.parameters)
        assert [
            (parameter.kind, parameter.default)
            for parameter in wrapper_signature.parameters.values()
        ] == [
            (parameter.kind, parameter.default)
            for parameter in native_signature.parameters.values()
        ]
        max_descs = self.cudagraph_manager._max_full_descs_to_capture
        for desc in descs[:max_descs]:
            for _ in (True, False):
                assert (
                    cudagraph_utils.prepare_inputs_to_capture(
                        desc.num_reqs,
                        desc.num_tokens,
                        *sentinels[:5],
                        full_cudagraph=True,
                        max_query_len=17,
                        pcp_manager=sentinels[5],
                    )
                    == "native-state"
                )
        if raise_in_native:
            raise RuntimeError("native capture failed")
        return 42

    monkeypatch.setattr(NPUModelRunner, "capture_model", native_capture)
    if raise_in_native:
        with pytest.raises(RuntimeError, match="native capture failed"):
            runner.capture_model(profile_only=profile_only)
    else:
        assert runner.capture_model(profile_only=profile_only) == 42
    assert forwarded == [profile_only]
    expected_shapes = [(1, 8), (1, 8), (2, 16), (2, 16)]
    if not profile_only:
        expected_shapes.extend([(3, 32), (3, 32)])
    assert len(prepare_calls) == len(expected_shapes)
    assert [(args[0], args[1]) for args, _ in prepare_calls] == expected_shapes
    assert all(args[2:] == tuple(sentinels[:5]) for args, _ in prepare_calls)
    assert all(
        kwargs
        == {
            "full_cudagraph": True,
            "max_query_len": 17,
            "pcp_manager": sentinels[5],
        }
        for _, kwargs in prepare_calls
    )
    expected_events = [
        (8, True, False),
        (8, False, True),
        (16, True, False),
        (16, False, True),
    ]
    if not profile_only:
        expected_events.extend([(32, True, False), (32, False, True)])
    assert events == expected_events
    assert _state(runner) == original_state
    assert cudagraph_utils.prepare_inputs_to_capture is native_prepare
    assert original_prepare is not native_prepare


@pytest.mark.parametrize("async_dp", [False, True])
def test_native_dummy_run_reaches_afd_execute_with_context_state(monkeypatch, async_dp):
    runner = _runner(async_dp=async_dp)
    runner.adaptive_verification = None
    runner.max_num_reqs = 2
    runner.lora_config = None
    runner.is_first_pp_rank = True
    runner.is_last_pp_rank = False
    runner.kv_connector = SimpleNamespace(set_disabled=lambda disabled: None)
    runner.maybe_dummy_run_with_lora = lambda *args, **kwargs: nullcontext()
    monkeypatch.setattr(native_ascend_v2, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", lambda _prof: None)
    calls = []

    def native_execute(self, scheduler_output, intermediate_tensors=None, **kwargs):
        calls.append((scheduler_output.total_num_scheduled_tokens, kwargs))
        assert should_skip_dp_coordination() is async_dp
        return None

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    assert runner._dummy_run(
        7,
        context_len=17,
        valid_dummy_state_slots=True,
        is_profile=True,
        skip_eplb=True,
    ) == (None, None)
    assert calls == [
        (
            7,
            {
                "dummy_run": True,
                "skip_attn_for_dummy_run": False,
                "is_profile": True,
                "context_len": 17,
                "valid_dummy_state_slots": True,
            },
        )
    ]


@pytest.mark.parametrize("num_tokens", [0, 1, 7])
@pytest.mark.parametrize("dp_rank", [0, 1])
def test_async_execute_uses_native_local_dp_dispatch(monkeypatch, num_tokens, dp_rank):
    runner = _runner(async_dp=True)
    runner.afd_config.num_ffn_ranks = 1
    original_dispatch = native_v2.dispatch_cg_and_sync_dp

    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", lambda _prof: None)
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )

    def unexpected_collective(*args, **kwargs):
        pytest.fail("Async Attention must not coordinate with an idle DP replica")

    monkeypatch.setattr(dp_utils.dist, "all_reduce", unexpected_collective)

    def native_execute(self, scheduler_output, *args, **kwargs):
        assert native_v2.dispatch_cg_and_sync_dp is original_dispatch
        desc = cudagraph_utils.BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=int(num_tokens > 0),
        )
        return dp_utils.sync_cudagraph_and_dp_padding(
            None, desc, num_tokens, desc.num_reqs, None, 2, dp_rank
        )

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    desc, sync = runner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=num_tokens)
    )
    assert desc.num_tokens == num_tokens
    assert (sync is None) is (num_tokens == 0)
    assert not should_skip_dp_coordination()


@pytest.mark.parametrize(
    ("counts", "ffn_ranks", "attention_tp", "expected_counts", "dp_rank"),
    [
        (counts, ffn_ranks, attention_tp, expected_counts, dp_rank)
        for counts, ffn_ranks, attention_tp, expected_counts in [
            ((3, 7), 1, 1, [7, 7]),
            ((7, 3), 1, 1, [7, 7]),
            ((1, 7), 1, 1, [7, 7]),
            ((8, 8), 1, 1, [8, 8]),
            ((0, 0), 1, 1, [0, 0]),
            ((3, 7), 2, 1, [3, 7]),
            ((3, 7, 5, 2), 2, 1, [5, 7, 5, 7]),
            ((1, 7, 1, 3), 2, 1, [1, 7, 1, 7]),
            ((8, 8, 8, 8), 2, 1, [8, 8, 8, 8]),
            ((3, 7), 2, 2, [7, 7]),
            ((3, 7, 5, 2), 4, 2, [5, 7, 5, 7]),
            ((3, 7, 5), 3, 2, [7, 7, 7]),
        ]
        for dp_rank in range(len(counts))
    ],
)
@pytest.mark.parametrize("execution", ["eager", "graph_miss", "peer_fallback"])
def test_camp2p_dispatch_pads_many_to_one_before_native_inputs(
    monkeypatch, counts, dp_rank, ffn_ranks, attention_tp, expected_counts, execution
):
    runner = _runner()
    dp_size = len(counts)
    runner.afd_config.num_attention_ranks = dp_size * attention_tp
    runner.afd_config.num_ffn_ranks = ffn_ranks
    manager = None
    if execution != "eager":
        runner.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

        def dispatch(num_reqs, num_tokens, uniform_token_count, **kwargs):
            mode = (
                CUDAGraphMode.NONE
                if execution == "graph_miss" or dp_rank == 0
                else CUDAGraphMode.FULL
            )
            return cudagraph_utils.BatchExecutionDescriptor(mode, num_tokens, num_reqs)

        def run_fullgraph(desc):
            pytest.fail("eager fallback must not replay a graph")

        manager = SimpleNamespace(dispatch=dispatch, run_fullgraph=run_fullgraph)
        runner.cudagraph_manager = manager
    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", lambda _prof: None)
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(dp_utils, "should_skip_dp_coordination", lambda: False)
    reductions = []

    def all_reduce(tensor, group):
        reductions.append(tensor)
        tensor[0] = torch.tensor(counts, dtype=torch.int32)
        tensor[1].fill_(
            CUDAGraphMode.FULL.value
            if execution == "peer_fallback"
            else CUDAGraphMode.NONE.value
        )
        tensor[1][0] = CUDAGraphMode.NONE.value
        tensor[5].fill_(1)

    monkeypatch.setattr(dp_utils.dist, "all_reduce", all_reduce)
    original_dispatch = native_v2.dispatch_cg_and_sync_dp

    def native_execute(self, *args, **kwargs):
        desc, sync = native_v2.dispatch_cg_and_sync_dp(
            manager,
            1,
            counts[dp_rank],
            None,
            dp_size,
            dp_rank,
            need_eager=execution == "eager",
        )
        assert desc.num_tokens == expected_counts[dp_rank]
        assert desc.num_reqs == (0 if max(counts) == 0 else 1)
        assert desc.cg_mode == CUDAGraphMode.NONE
        if max(counts) == 0:
            assert sync is None
        else:
            assert sync.num_tokens_across_dp.tolist() == expected_counts
            assert sync.eager
        return "native-result"

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    assert (
        runner.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=counts[dp_rank]),
            dummy_run=counts[dp_rank] == 1,
        )
        == "native-result"
    )
    assert len(reductions) == 1
    assert reductions[0][0].tolist() == list(counts)
    assert native_v2.dispatch_cg_and_sync_dp is original_dispatch


@pytest.mark.parametrize("raise_in_native", [False, True])
@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_camp2p_padding_preserves_graph_descriptor_and_restores_dispatch(
    monkeypatch, raise_in_native, mode
):
    runner = _runner()
    runner.afd_config.num_ffn_ranks = 1
    monkeypatch.setattr(npu_v2, "step_afd_npu_profiler", lambda _prof: None)
    counts = [8, 8] if mode == CUDAGraphMode.FULL else [3, 7]
    desc = cudagraph_utils.BatchExecutionDescriptor(
        mode, counts[0], 3, uniform_token_count=1, max_query_len=1, num_active_loras=2
    )
    sync = dp_utils.DPSyncState(torch.tensor(counts), 1, False, 7)
    dispatch_calls = []

    def native_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return desc, sync

    monkeypatch.setattr(native_v2, "dispatch_cg_and_sync_dp", native_dispatch)

    def native_execute(self, *args, **kwargs):
        actual_desc, actual_sync = native_v2.dispatch_cg_and_sync_dp(
            None, 3, 3, 1, 2, 0, max_query_len=1, need_eager=False
        )
        if mode == CUDAGraphMode.FULL:
            assert actual_desc is desc
            assert actual_sync is sync
        else:
            assert actual_desc.num_tokens == 7
            assert actual_desc.num_reqs == desc.num_reqs
            assert actual_desc.uniform_token_count == desc.uniform_token_count
            assert actual_desc.max_query_len == desc.max_query_len
            assert actual_desc.num_active_loras == desc.num_active_loras
            assert actual_sync.num_tokens_across_dp.tolist() == [7, 7]
            assert actual_sync.eager is False
            assert sync.num_tokens_across_dp.tolist() == [3, 7]
        if raise_in_native:
            raise RuntimeError("native execute failed")
        return "native-result"

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    if raise_in_native:
        with pytest.raises(RuntimeError, match="native execute failed"):
            runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    else:
        assert (
            runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
            == "native-result"
        )
    assert len(dispatch_calls) == 1
    assert native_v2.dispatch_cg_and_sync_dp is native_dispatch


def test_camp2p_dispatch_wrapper_matches_native_signature():
    original = inspect.signature(native_v2.dispatch_cg_and_sync_dp, eval_str=True)
    with npu_v2._use_camp2p_dp_padding(4, 2):
        assert (
            inspect.signature(native_v2.dispatch_cg_and_sync_dp, eval_str=True)
            == original
        )
