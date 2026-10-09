# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""CPU-only contracts for the native Ascend ModelRunner V2 wrapper."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils  # noqa: E402
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402
from vllm_ascend.worker.v2 import model_runner as native_ascend_v2  # noqa: E402
from vllm_ascend.worker.v2.model_runner import NPUModelRunner  # noqa: E402

from afd_plugin.model_executor.models import (
    forward_context as afd_context,  # noqa: E402
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as npu_v2  # noqa: E402

Runner = npu_v2.AFDNPUAttentionModelRunnerV2


def _runner():
    runner = object.__new__(Runner)
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    )
    runner.cudagraph_manager = None
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


@pytest.mark.parametrize("raise_in_native", [False, True])
def test_execute_forwards_dummy_profile_and_context_then_restores(
    monkeypatch, raise_in_native
):
    runner = _runner()
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


def test_native_dummy_run_reaches_afd_execute_with_context_state(monkeypatch):
    runner = _runner()
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
