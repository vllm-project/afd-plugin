# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""CPU-only contracts for the native Ascend ModelRunner V2 wrapper."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.sequence import IntermediateTensors  # noqa: E402
from vllm.v1.worker.dp_utils import should_skip_dp_coordination  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils, ubatch_utils  # noqa: E402
from vllm.v1.worker.gpu import model_runner as native_v2  # noqa: E402
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402
from vllm_ascend.worker.v2 import model_runner as native_ascend_v2  # noqa: E402
from vllm_ascend.worker.v2.input_batch import (  # noqa: E402
    AscendInputBatch,
    AscendInputBuffers,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner  # noqa: E402

from afd_plugin.compat.patches.npu import mrv2_ubatching as patches  # noqa: E402
from afd_plugin.model_executor.models import (  # noqa: E402
    forward_context as afd_context,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as npu_v2  # noqa: E402
from afd_plugin.v1.worker.npu import ubatch_runner as npu_ubatch  # noqa: E402

Runner = npu_v2.AFDNPUAttentionModelRunnerV2


def _runner(*, async_dp=False):
    runner = object.__new__(Runner)
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        additional_config={
            "afd": {
                "role": "attention",
                "async": async_dp,
                "connector": "CAMAsyncAFDConnector"
                if async_dp
                else "CAMP2pAFDConnector",
            }
        },
        parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
    )
    runner.afd_config = SimpleNamespace(
        async_dp=async_dp, num_attention_ranks=2, num_ffn_ranks=2
    )
    runner.connector = SimpleNamespace(
        extra_info=SimpleNamespace(async_moe_ubatching=False)
    )
    runner.cudagraph_manager = None
    runner.ubatch_runner = None
    runner.intermediate_tensors = None
    runner.prof = object()
    runner._afd_pending_metadata = object()
    runner._afd_suppress_metadata_send = True
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_is_graph_replaying = True
    runner._afd_is_profile = False
    return runner


def _state(runner):
    return (
        runner._afd_pending_metadata,
        runner._afd_suppress_metadata_send,
        runner._is_warmup,
        runner._afd_is_graph_capturing,
        runner._afd_is_graph_replaying,
        runner._afd_is_profile,
    )


@pytest.mark.parametrize("failure", [None, "profiler", "abort", "native"])
def test_shutdown_joins_native_children_and_releases_owned_stage_state(
    monkeypatch, failure
):
    runner = _runner()
    calls = []

    def call(name):
        calls.append(name)
        if failure == name:
            raise RuntimeError(name)

    runner.ubatch_runner = SimpleNamespace(abort_pending_run=lambda: call("abort"))
    runner.connector.close = lambda: call("connector")
    monkeypatch.setattr(npu_v2, "stop_afd_npu_profiler", lambda _: call("profiler"))
    monkeypatch.setattr(NPUModelRunner, "shutdown", lambda _: call("native"))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            runner.shutdown()
    else:
        runner.shutdown()
    assert calls == ["profiler", "abort", "native", "connector"]
    assert (runner.ubatch_runner is None) is (failure != "native")
    assert runner._afd_pending_metadata is None


@pytest.mark.parametrize("fail", [False, True])
def test_native_model_io_scope_slices_every_pp_field_and_merges_aux(monkeypatch, fail):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=True,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v2")
        ),
    )
    hidden = torch.arange(8).reshape(4, 2)
    auxiliary = hidden + 10
    model_inputs = {
        "input_ids": torch.arange(12),
        "positions": torch.arange(24).reshape(2, 12),
        "intermediate_tensors": IntermediateTensors(
            {"hidden_states": hidden, "pp_transport_aux_hidden_states_0": auxiliary}
        ),
        "extra": object(),
    }
    original_slice = ubatch_utils.slice_model_inputs
    original_merge = ubatch_utils.merge_ubatch_outputs

    def exercise():
        with patches.use_npu_model_io(config):
            children = [
                ubatch_utils.slice_model_inputs(model_inputs, stage)
                for stage in (slice(0, 5), slice(5, 12))
            ]
            assert [child["input_ids"].numel() for child in children] == [5, 7]
            assert [child["positions"].shape for child in children] == [(2, 5), (2, 7)]
            assert all(child["extra"] is model_inputs["extra"] for child in children)
            merged = ubatch_utils.merge_ubatch_outputs(
                [child["intermediate_tensors"] for child in children]
            )
            torch.testing.assert_close(merged.tensors["hidden_states"], hidden)
            torch.testing.assert_close(
                merged.tensors["pp_transport_aux_hidden_states_0"], auxiliary
            )
            nested = ubatch_utils.merge_ubatch_outputs(
                [(hidden[:2], [auxiliary[:2]]), (hidden[2:], [auxiliary[2:]])]
            )
            torch.testing.assert_close(nested[0], hidden)
            torch.testing.assert_close(nested[1][0], auxiliary)
            if fail:
                raise RuntimeError("model failed")

    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            exercise()
    else:
        exercise()
    assert ubatch_utils.slice_model_inputs is original_slice
    assert ubatch_utils.merge_ubatch_outputs is original_merge


def test_mla_capture_registry_uses_exact_native_descriptor_and_skips_warmup():
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner._uses_mla_graph_params = True
    runner._mla_capture_params = {}
    desc = cudagraph_utils.BatchExecutionDescriptor(
        CUDAGraphMode.FULL, num_tokens=8, num_reqs=8, num_ubatches=2
    )
    lora_desc = replace(desc, num_active_loras=1)
    first, second = object(), object()
    contexts = [
        SimpleNamespace(additional_kwargs={npu_ubatch.AFD_MLA_GRAPH_PARAMS_KEY: value})
        for value in (first, second)
    ]
    state = SimpleNamespace(forward_contexts=contexts)
    runner.record_capture_graph_params(desc, state, is_warmup=True)
    assert runner._mla_capture_params == {}
    runner.record_capture_graph_params(desc, state, is_warmup=False)
    contexts.reverse()
    runner.record_capture_graph_params(lora_desc, state, is_warmup=False)
    assert runner._mla_capture_params == {
        desc: (first, second),
        lora_desc: (second, first),
    }


@pytest.mark.parametrize("fail_update", [False, True])
def test_mla_replay_updates_captured_stage_records_and_restores_context(
    monkeypatch, fail_update
):
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner._uses_mla_graph_params = True
    runner.device = torch.device("cpu")
    runner.attn_groups = object()
    runner.vllm_config = SimpleNamespace(speculative_config=None)
    desc = cudagraph_utils.BatchExecutionDescriptor(
        CUDAGraphMode.FULL, num_tokens=8, num_reqs=8, num_ubatches=2
    )
    workspace = object()
    params = tuple(npu_ubatch.new_mla_graph_params(4, workspace) for _ in range(2))
    for stage, value in enumerate(params):
        value.events[4] = [f"event{stage}"]
        value.handles[4] = [f"handle{stage}"]
        value.attn_params[4] = [f"capture{stage}"]
    runner._mla_capture_params = {desc: params}
    runner._prepared_state = SimpleNamespace(
        slices=[SimpleNamespace(num_tokens=4), SimpleNamespace(num_tokens=4)],
        forward_contexts=[
            SimpleNamespace(
                attn_metadata={"layer": f"live{stage}"},
                dp_metadata=SimpleNamespace(
                    num_tokens_across_dp_cpu=torch.tensor([4, 4])
                ),
            )
            for stage in range(2)
        ],
    )
    calls: list[str | tuple[str, object]] = []
    current_stream = object()
    manager = SimpleNamespace(
        update_stream=SimpleNamespace(
            wait_stream=lambda stream: calls.append(("wait", stream))
        )
    )
    context = SimpleNamespace(attn_metadata=None, additional_kwargs={"parent": True})
    original_kwargs = context.additional_kwargs
    monkeypatch.setattr(torch.npu, "current_stream", lambda _: current_stream)
    monkeypatch.setattr(npu_ubatch, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(
        npu_ubatch, "set_forward_context", lambda *a, **kw: nullcontext(context)
    )
    monkeypatch.setattr(npu_ubatch, "get_forward_context", lambda: context)
    monkeypatch.setattr(npu_ubatch, "_get_graph_update_backend", lambda _: "mla")

    def replay(self, descriptor):
        assert self is manager and descriptor is desc
        calls.append("replay")
        return "native-output"

    def update(backend, stream, forward, tokens, config, speculative):
        calls.append("update")
        assert (backend, stream, forward, tokens, config, speculative) == (
            "mla",
            manager.update_stream,
            context,
            4,
            runner.vllm_config,
            None,
        )
        assert context.attn_metadata == {("layer", 0): "live0", ("layer", 1): "live1"}
        merged = context.additional_kwargs[npu_ubatch.AFD_MLA_GRAPH_PARAMS_KEY]
        assert merged.attn_params[4] == ["capture0", "capture1"]
        assert merged.events[4] == ["event0", "event1"]
        assert merged.workspaces[4] is workspace
        if fail_update:
            raise RuntimeError("update failed")

    monkeypatch.setattr(npu_ubatch.ModelCudaGraphManager, "run_fullgraph", replay)
    monkeypatch.setattr(npu_ubatch, "update_full_graph_params", update)
    if fail_update:
        with pytest.raises(RuntimeError, match="update failed"):
            runner.run_fullgraph(manager, desc)
    else:
        assert runner.run_fullgraph(manager, desc) == "native-output"
    assert calls == [("wait", current_stream), "replay", "update"]
    assert context.attn_metadata is None
    assert context.additional_kwargs is original_kwargs


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


@pytest.mark.parametrize("profile_only", [False, True])
@pytest.mark.parametrize("raise_in_native", [False, True])
def test_capture_forwards_new_inputs_and_restores(
    monkeypatch, profile_only, raise_in_native
):
    runner = _runner()
    original_state = _state(runner)
    descs = [
        cudagraph_utils.BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_reqs=1, num_tokens=8
        ),
        cudagraph_utils.BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_reqs=2, num_tokens=16
        ),
        cudagraph_utils.BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_reqs=3, num_tokens=32
        ),
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
        bound = inspect.signature(original_prepare).bind(*args, **kwargs).arguments
        prepare_calls.append(
            (
                tuple(
                    bound[key]
                    for key in (
                        "num_reqs",
                        "num_tokens",
                        "model_state",
                        "input_buffers",
                        "block_tables",
                        "attn_groups",
                        "kv_cache_config",
                    )
                ),
                {
                    key: bound[key]
                    for key in (
                        "full_cudagraph",
                        "max_query_len",
                        "pcp_manager",
                    )
                },
            )
        )
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
                assert not self._afd_is_graph_replaying
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


def _ascend_batch(real=5, padded=8):
    buffers = AscendInputBuffers(2, padded, torch.device("cpu"))
    batch = AscendInputBatch.make_dummy(1, real, buffers)
    batch.num_tokens_after_padding = padded
    batch.input_ids = torch.arange(padded)
    batch.positions = torch.arange(padded)
    batch.is_padding = torch.arange(padded) >= real
    batch.seq_lens += 12
    batch.seq_lens_cpu_upper_bound += 12
    batch.seq_lens_np = np.array([real + 12], dtype=np.int32)
    return batch


@pytest.mark.parametrize("real", [2, 5, 7])
def test_native_slices_keep_exact_ascend_lengths_and_padding(real, monkeypatch):
    # make_dummy's RoPE initialization is unrelated to CPU input slicing.
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    batch = _ascend_batch(real)
    slices = ubatch_utils.create_ubatch_slices(batch, 2)
    original = ubatch_utils._slice_input_batch
    with patches.use_npu_input_batch_slicing():
        children = [
            ubatch_utils._slice_input_batch(
                batch,
                stage,
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            )
            for stage in slices
        ]
    assert ubatch_utils._slice_input_batch is original
    assert all(isinstance(child, AscendInputBatch) for child in children)
    assert [child.num_tokens_after_padding for child in children] == [4, 4]
    assert [child.num_tokens for child in children] == [min(real, 4), max(0, real - 4)]
    assert children[0].seq_lens_np.tolist() == [12 + min(real, 4)]
    assert children[1].seq_lens_np.tolist() == [12 + real]
    assert children[1].query_start_loc_np.tolist() == [0, max(0, real - 4)]
    assert all(
        child.is_dummy and child.attn_state == batch.attn_state for child in children
    )
    assert batch.seq_lens_np.tolist() == [12 + real]
    assert children[0].seq_lens_np is not children[1].seq_lens_np


@pytest.mark.parametrize("real", [2, 5])
def test_full_graph_prepares_metadata_for_padded_tail_stage(real, monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    batch = _ascend_batch(real)
    slices = ubatch_utils.create_ubatch_slices(batch, 2)
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner.stage_attn_groups = [object(), object()]
    runner.model_state = SimpleNamespace(max_model_len=128)
    runner.parallel_config = object()
    runner.vllm_config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    runner._uses_mla_graph_params = True
    runner._stage_rope_buffers = [{}, {}]
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {"layer": object()}

    def unexpected_state(*args):
        pytest.fail("FULL must retain the parent's native decode state")

    monkeypatch.setattr(npu_ubatch, "build_attn_metadata", build)
    monkeypatch.setattr(npu_ubatch, "build_attn_state", unexpected_state)
    monkeypatch.setattr(
        npu_ubatch,
        "materialize_deepseek_attention_metadata_by_layer",
        lambda *a, **kw: None,
    )
    with patches.use_npu_input_batch_slicing():
        for stage_index, stage in enumerate(slices):
            child = ubatch_utils._slice_input_batch(
                batch,
                stage,
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            )
            metadata = runner._prepare_attn(
                child,
                CUDAGraphMode.FULL,
                (torch.ones(1, 2),),
                torch.full((1, 4), -1),
                [],
                None,
                ubatch_idx=stage_index,
            )
            assert metadata is not None
    assert [call["num_actual_tokens"] for call in calls] == [
        min(real, 4),
        max(0, real - 4),
    ]
    assert [call["num_input_tokens"] for call in calls] == [4, 4]
    assert [call["num_tokens"] for call in calls] == [4, 4]
    assert all(call["attn_state"] == batch.attn_state for call in calls)
    assert all(not call["for_cudagraph_capture"] for call in calls)


def test_stage_prepare_uses_independent_builders_and_detaches_cache(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    batch = _ascend_batch()
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner.stage_attn_groups = [object(), object()]
    runner.model_state = SimpleNamespace(max_model_len=128)
    runner.parallel_config = object()
    runner.vllm_config = SimpleNamespace(additional_config={})
    runner._uses_mla_graph_params = False
    runner._stage_block_tables = [None, None]
    runner.ubatch_seq_lens = [torch.empty(2), torch.empty(2)]
    calls = []
    reusable = torch.empty(5)

    def build(**kwargs):
        calls.append(kwargs)
        reusable.fill_(len(calls))
        kwargs["block_tables"][0].zero_()
        return {"layer": SimpleNamespace(cos=reusable)}

    def materialize(metadata, positions, count):
        assert positions is batch.positions and count == batch.num_tokens
        metadata["layer"].cos = metadata["layer"].cos.clone()

    monkeypatch.setattr(npu_ubatch, "build_attn_metadata", build)
    monkeypatch.setattr(npu_ubatch, "build_attn_state", lambda *args: "child-state")
    monkeypatch.setattr(
        npu_ubatch, "materialize_deepseek_attention_metadata_by_layer", materialize
    )
    table = torch.ones((1, 2))
    metadata = [
        runner._prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (table,),
            torch.empty(1, 8),
            [],
            None,
            ubatch_idx=i,
        )
        for i in range(2)
    ]
    assert [call["attn_groups"] for call in calls] == runner.stage_attn_groups
    assert all(call["attn_state"] == "child-state" for call in calls)
    assert metadata[0]["layer"].cos.tolist() == [1] * 5
    assert metadata[1]["layer"].cos.tolist() == [2] * 5
    assert table.tolist() == [[1, 1]]
    assert (
        runner._prepare_attn(
            SimpleNamespace(num_tokens=0),
            CUDAGraphMode.NONE,
            (),
            None,
            [],
            None,
            ubatch_idx=1,
        )
        is None
    )
    assert len(calls) == 2


def test_native_child_contexts_preserve_dp_and_own_ascend_kwargs(monkeypatch):
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner.vllm_config = SimpleNamespace(additional_config={})
    slices = [SimpleNamespace(num_tokens=4), SimpleNamespace(num_tokens=4)]
    dp = [object(), object()]
    contexts = [
        SimpleNamespace(attn_metadata={"stage": i}, dp_metadata=dp[i]) for i in range(2)
    ]
    contexts[1].attn_metadata = None
    monkeypatch.setattr(
        ubatch_utils.UBatchRunner, "_make_forward_contexts", lambda *args: contexts
    )
    shared_mask = torch.ones(4, dtype=torch.bool)
    monkeypatch.setattr(
        npu_ubatch.current_platform,
        "set_additional_forward_context",
        lambda *args, **kwargs: {
            "mc2_mask": shared_mask,
            "num_tokens": kwargs["num_tokens"],
        },
    )
    assert runner._make_forward_contexts(slices, [], [], []) is contexts
    assert [ctx.dp_metadata for ctx in contexts] == dp
    assert contexts[1].attn_metadata is None
    from afd_plugin.connectors.npu.camp2p import _get_group_ep

    assert [
        _get_group_ep(getattr(ctx, "ubatch_idx", 0), "stage0", "stage1", "")
        for ctx in contexts
    ] == ["stage0", "stage1"]
    assert contexts[1].additional_kwargs["num_tokens"] == 4
    assert contexts[1].additional_kwargs["dbo_enabled"]
    assert [ctx.additional_kwargs["ubatch_idx"] for ctx in contexts] == [0, 1]
    contexts[1].additional_kwargs["mc2_mask"].zero_()
    assert contexts[0].additional_kwargs["mc2_mask"].all()


def test_scope_restores_native_bindings_after_failure():
    context = patches.ubatching.UBatchContext
    stream = patches.ubatching.current_stream
    runner_stream = patches.ubatch_utils.current_stream
    factory = patches.model_runner.maybe_build_ubatch_runner
    sm_control = patches.ubatch_utils.create_sm_control_context
    with (
        pytest.raises(RuntimeError, match="injected"),
        patches.use_npu_ubatch_factory(SimpleNamespace()),
        patches.use_npu_ubatch_contexts(),
    ):
        raise RuntimeError("injected")
    assert patches.ubatching.UBatchContext is context
    assert patches.ubatching.current_stream is stream
    assert patches.ubatch_utils.current_stream is runner_stream
    assert patches.model_runner.maybe_build_ubatch_runner is factory
    assert patches.ubatch_utils.create_sm_control_context is sm_control


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("tokens,threshold,expected", [(1, 2, 1), (7, 2, 2), (7, 8, 1)])
def test_async_dbo_dispatch_preserves_thresholds_and_falls_back_locally(
    monkeypatch, dp_size, tokens, threshold, expected
):
    from vllm.v1.worker.dp_utils import skip_dp_coordination

    parallel = SimpleNamespace(
        use_ubatching=True,
        enable_dbo=True,
        num_ubatches=2,
        dbo_decode_token_threshold=threshold,
        dbo_prefill_token_threshold=threshold,
    )
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(
        dp_utils.dist, "all_reduce", lambda *a, **k: pytest.fail("Async DP collective")
    )
    native = native_v2.dispatch_cg_and_sync_dp
    with (
        skip_dp_coordination(),
        patches.use_async_cam_dbo_dispatch(),
    ):
        desc, sync = native_v2.dispatch_cg_and_sync_dp(
            None,
            tokens,
            tokens,
            1,
            dp_size,
            dp_size - 1,
            need_eager=True,
            parallel_config=parallel,
            allow_ubatching=True,
            uniform_decode=True,
        )
        assert desc.num_ubatches == expected
        if dp_size == 1:
            assert sync is None
        else:
            assert sync.num_tokens_across_dp.tolist() == [tokens, tokens]
        # Native slicing also supports one request straddling both microbatches.
        desc, _ = native_v2.dispatch_cg_and_sync_dp(
            None,
            1,
            7,
            None,
            dp_size,
            0,
            need_eager=True,
            parallel_config=parallel,
            allow_ubatching=True,
        )
        assert desc.num_ubatches == (2 if threshold <= 7 else 1)
    assert native_v2.dispatch_cg_and_sync_dp is native


def test_async_native_children_keep_request_prefix_and_parent_padding(monkeypatch):

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    buffers = AscendInputBuffers(2, 8, torch.device("cpu"))
    batch = AscendInputBatch.make_dummy(2, 7, buffers)
    batch.num_tokens_after_padding = 8
    batch.input_ids = torch.arange(8)
    batch.positions = torch.arange(8)
    batch.num_scheduled_tokens = np.array([3, 4], dtype=np.int32)
    batch.query_start_loc_np = np.array([0, 3, 7], dtype=np.int32)
    batch.query_start_loc.copy_(torch.from_numpy(batch.query_start_loc_np))
    batch.seq_lens.copy_(torch.tensor([15, 24]))
    batch.seq_lens_cpu_upper_bound.copy_(torch.tensor([15, 24]))
    batch.seq_lens_np = np.array([15, 24], dtype=np.int32)
    boundary = 4
    native_plan = ubatch_utils.create_ubatch_slices
    with patches.use_npu_input_batch_slicing():
        slices = ubatch_utils.create_ubatch_slices(batch, 2)
        children = [
            ubatch_utils._slice_input_batch(
                batch,
                stage,
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
            )
            for stage in slices
        ]
    assert ubatch_utils.create_ubatch_slices is native_plan
    assert [stage.token_slice for stage in slices] == [
        slice(0, boundary),
        slice(boundary, 8),
    ]
    assert [child.num_tokens for child in children] == [boundary, 7 - boundary]
    assert [child.positions.tolist() for child in children] == [
        list(range(boundary)),
        list(range(boundary, 8)),
    ]
    assert children[0].seq_lens_np.tolist() == [15, 21]
    assert children[1].seq_lens_np.tolist() == [24]
    assert children[1].query_start_loc_np.tolist() == [0, 7 - boundary]
    assert batch.seq_lens_np.tolist() == [15, 24]


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("async_dp", [False, True])
def test_native_factory_allows_local_async_dbo_only(monkeypatch, dp_size, async_dp):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True, data_parallel_size=dp_size),
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector"
                if async_dp
                else "CAMP2pAFDConnector",
                "role": "attention",
                "async_dp": async_dp,
            }
        },
    )
    created = []
    instance = object()

    def build(*args, **kwargs):
        created.append((args, kwargs))
        return instance

    monkeypatch.setattr(npu_ubatch, "NPUUBatchRunner", build)
    with patches.use_npu_ubatch_factory(SimpleNamespace(kernel_block_sizes=[128])):
        result = native_v2.maybe_build_ubatch_runner(config, None, None, [], None, 2)
    assert (result is instance) is (async_dp or dp_size > 1)
    assert len(created) == int(async_dp or dp_size > 1)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_async_native_context_creation_without_sp(monkeypatch, dp_size):
    runner = object.__new__(npu_ubatch.NPUUBatchRunner)
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=0,
        use_sequence_parallel_moe=False,
        is_moe_model=True,
        enable_dbo=True,
    )
    runner.vllm_config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "attention",
                "async": True,
                "connector": "CAMAsyncAFDConnector",
            }
        },
        parallel_config=runner.parallel_config,
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )
    monkeypatch.setattr(
        npu_ubatch.current_platform,
        "set_additional_forward_context",
        lambda *a, **k: {},
    )
    native_metadata = ubatch_utils.DPMetadata
    contexts = runner._make_forward_contexts(
        [SimpleNamespace(num_tokens=3)] * 2, [None, None], [{}, {}], [None, None]
    )
    assert all(ctx.dp_metadata is None for ctx in contexts)
    assert [ctx.additional_kwargs["ubatch_idx"] for ctx in contexts] == [0, 1]
    assert ubatch_utils.DPMetadata is native_metadata


def test_async_child_metadata_factory_restores_after_failure():
    native_metadata = ubatch_utils.DPMetadata
    with (
        pytest.raises(RuntimeError, match="injected"),
        patches.use_async_cam_child_dp_metadata(),
    ):
        raise RuntimeError("injected")
    assert ubatch_utils.DPMetadata is native_metadata
