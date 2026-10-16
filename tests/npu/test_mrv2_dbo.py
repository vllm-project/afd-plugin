# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real NPU streams/events and native thread handoff, isolated on each fault.

This checks the executor primitives, not checkpoint or CAMP2P correctness.
"""

import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

# Match tests/conftest.py and vllm serve: import ops before device_op to avoid
# Ascend's existing circular import in standalone test subprocesses.
import vllm_ascend.ops  # noqa: E402, F401
from vllm.config import VllmConfig  # noqa: E402
from vllm.forward_context import get_forward_context  # noqa: E402
from vllm.utils.torch_utils import current_stream  # noqa: E402
from vllm.v1.worker import ubatching  # noqa: E402
from vllm.v1.worker.dp_utils import should_skip_dp_coordination  # noqa: E402
from vllm.v1.worker.gpu import model_runner  # noqa: E402, F401
from vllm.v1.worker.gpu.ubatch_utils import UBatchState  # noqa: E402
from vllm.v1.worker.ubatch_utils import UBatchSlice  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    current_workspace_manager,
    init_workspace_manager,
)
from vllm_ascend.ascend_forward_context import (  # noqa: E402
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
)
from vllm_ascend.utils import adapt_patch  # noqa: E402
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper  # noqa: E402

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata  # noqa: E402
from afd_plugin.model_executor.models.npu.async_cam_dbo import (  # noqa: E402
    send_async_cam_remote_call,
)
from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner  # noqa: E402


def _exercise(fault, cam_layers=0, capture=False):
    # Construct only the native execution state; full initialization/metadata
    # and distributed communication are covered by the checkpoint run.
    adapt_patch(is_global_patch=True)
    torch.npu.set_device(0)
    runner = object.__new__(NPUUBatchRunner)
    runner.vllm_config = VllmConfig()
    if cam_layers:
        runner.vllm_config.additional_config = {
            "afd": {
                "role": "attention",
                "async": True,
                "connector": "CAMAsyncAFDConnector",
            }
        }
        runner.vllm_config.parallel_config.enable_dbo = True
    runner.device = torch.device("npu:0")
    runner.num_ubatches = 2
    runner.comm_stream = torch.npu.Stream(device=runner.device)
    runner.capture_stream = torch.npu.Stream(device=runner.device)
    runner.ready_barrier = threading.Barrier(3, timeout=2)
    runner.sm_control = nullcontext()
    runner._pending_finish = None
    init_workspace_manager(runner.device, num_ubatches=2)
    contexts = [SimpleNamespace(stage=i) for i in range(2)]
    state = UBatchState(
        [UBatchSlice(slice(0, 1), slice(i * 4, (i + 1) * 4)) for i in range(2)],
        contexts,
    )
    parent = ubatching.forward_context._forward_context
    inputs = torch.arange(8, dtype=torch.float32, device=runner.device)
    trace: list[tuple[int | str | None, ...]] = []
    injected: list[str] = []
    if fault == "device_init":
        # Native threads initialize before entering UBatchContext/barrier.
        set_device = torch.accelerator.set_device_index

        def fail_device(index):
            if not injected:
                injected.append(fault)
                raise RuntimeError("injected device_init")
            set_device(index)

        torch.accelerator.set_device_index = fail_device

    def fail():
        injected.append(fault)
        raise RuntimeError("injected " + fault)

    def model(input_ids, positions):
        stage = ubatching.dbo_current_ubatch_id()
        assert get_forward_context() is contexts[stage]
        assert torch.npu.current_device() == 0
        assert current_stream() == torch.npu.current_stream()
        assert get_mrv2_in_profile_run()
        trace.append((stage, "send"))
        workspace = current_workspace_manager().get_simultaneous(((4,), torch.float32))[
            0
        ]
        workspace.fill_(stage + 1)
        ubatching.dbo_switch_to_comm_sync()
        value = input_ids * 3 + positions
        if stage == 0 and fault == "after_send":
            fail()
        if stage == 0 and fault == "recv_hook":
            ubatching.dbo_register_recv_hook(fail)
        ubatching.dbo_yield_and_switch_from_comm_to_compute()
        ubatching.dbo_maybe_run_recv_hook()
        trace.append((stage, "consume"))
        if not capture:
            torch.testing.assert_close(workspace.cpu(), torch.full((4,), stage + 1.0))
        if stage == 0 and fault == "owner_consume":
            fail()
        if stage == 0 and fault == "final_exit":
            # Native __exit__ runs this slot after removing registry identity.
            ubatching._CURRENT_CONTEXTS[stage].recv_hook = fail
        return value + workspace if capture else value

    outstanding: list[AFDTransferContext] = []
    if cam_layers:

        def send(value, transfer, **kwargs):
            assert not outstanding, "CAM physical window was overwritten"
            outstanding.append(transfer)
            trace.append(
                (ubatching.dbo_current_ubatch_id(), "send", transfer.metadata.layer_idx)
            )
            if transfer.metadata.stage_idx == 0 and fault == "after_send":
                fail()

        def receive(*, ref_tensor, ubatch_idx, context):
            assert outstanding.pop() is context
            assert ubatch_idx == context.metadata.stage_idx
            # Native exit can drain the final slot after removing its identity.
            actor = ubatching._THREAD_ID_TO_CONTEXT.get(threading.get_ident())
            trace.append((actor, "raw_receive", context.metadata.layer_idx, ubatch_idx))
            assert actor != ubatch_idx
            if ubatch_idx == 0 and fault == "recv_hook":
                fail()
            return ref_tensor * 3

        connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)

        def model(input_ids, positions):
            stage = ubatching.dbo_current_ubatch_id()
            owner = get_forward_context()
            assert owner is contexts[stage]
            assert should_skip_dp_coordination()
            assert current_stream() == torch.npu.current_stream()
            states = input_ids
            for layer in range(cam_layers):
                transfer = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=layer, stage_idx=stage, seq_len=len(states)
                    )
                )
                recv = send_async_cam_remote_call(connector, states, transfer)
                shared = torch.ones_like(states)
                ubatching.dbo_yield()
                assert get_forward_context() is owner
                states = recv() + shared
                trace.append((stage, "consume", layer))
                if stage == 0 and fault == "owner_consume":
                    fail()
            if stage == 0 and fault == "final_exit":
                ubatching._CURRENT_CONTEXTS[stage].recv_hook = fail
            return states

    before = set(threading.enumerate())
    original_blas_handle = torch.cuda.current_blas_handle
    try:
        with override_mrv2_in_profile_run(True), torch_cuda_wrapper():
            model_inputs = {"input_ids": inputs, "positions": inputs}
            if capture:
                # Match the native manager: park threads before graph entry,
                # then release and join them inside the NPU capture scope.
                runner.capture_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(runner.capture_stream):
                    finish = runner.begin_capturable_run(
                        model, model_inputs, state, for_capture=True
                    )
                    if fault == "abort_before_capture":
                        runner.abort_pending_run()
                        injected.append(fault)
                        raise RuntimeError("injected " + fault)
                    finish()
                    trace.clear()
                    finish = runner.begin_capturable_run(
                        model, model_inputs, state, for_capture=True
                    )
                    graph = torch.npu.NPUGraph()
                    try:
                        with torch.npu.graph(graph, stream=runner.capture_stream):
                            result = finish()
                    finally:
                        runner.abort_pending_run()
                torch.npu.current_stream().wait_stream(runner.capture_stream)
                inputs.add_(10)
                graph.replay()
            else:
                result = runner.run(model, model_inputs, state)
        assert fault == "none", "injected failure was lost"
        if cam_layers:
            expected = torch.arange(8, dtype=torch.float32)
            for _ in range(cam_layers):
                expected = expected * 3 + 1
            torch.testing.assert_close(result.cpu(), expected)
            assert not outstanding
            assert (
                len([event for event in trace if event[1] == "raw_receive"])
                == cam_layers * 2
            )
        else:
            expected = torch.arange(8) * 4
            if capture:
                expected = expected + 40 + torch.tensor([1] * 4 + [2] * 4)
            torch.testing.assert_close(result.cpu(), expected, check_dtype=False)
            assert trace == [(0, "send"), (1, "send"), (0, "consume"), (1, "consume")]
    except (RuntimeError, threading.BrokenBarrierError):
        if fault == "none":
            raise
        assert fault in injected, "failure occurred before the injection point"
    # Initialization failures may return from the parent barrier first. Check
    # that all native child threads and registry entries actually disappear.
    deadline = time.monotonic() + 3
    while set(threading.enumerate()) - before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (set(threading.enumerate()) - before)
    assert not ubatching.dbo_enabled()
    assert all(context is None for context in ubatching._CURRENT_CONTEXTS)
    assert ubatching.forward_context._forward_context is parent
    assert torch.cuda.current_blas_handle is original_blas_handle
    assert runner._pending_finish is None
    torch.npu.synchronize()
    print("NPU_DBO_PASS " + fault, flush=True)
    if capture:
        print("NPU_DBO_CAPTURE_PASS " + fault, flush=True)
    if cam_layers:
        print("CAM_TRACE " + repr(trace), flush=True)


@pytest.mark.parametrize(
    "fault",
    ["none", "device_init", "after_send", "recv_hook", "owner_consume", "final_exit"],
)
def test_native_npu_threads_and_bounded_failure(fault):
    if not torch.npu.is_available():
        pytest.skip("requires an allocated NPU")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), fault],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NPU_DBO_PASS " + fault in result.stdout


@pytest.mark.parametrize(
    "fault,layers",
    [("none", 1), ("none", 3)]
    + [
        (fault, 2)
        for fault in (
            "device_init",
            "after_send",
            "recv_hook",
            "owner_consume",
            "final_exit",
        )
    ],
)
def test_native_async_cam_window_and_owner_on_npu(fault, layers):
    if not torch.npu.is_available():
        pytest.skip("requires an allocated NPU")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), fault, str(layers)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NPU_DBO_PASS " + fault in result.stdout


@pytest.mark.parametrize("fault", ["none", "abort_before_capture"])
def test_native_npu_capture_replay_and_parked_thread_abort(fault):
    if not torch.npu.is_available():
        pytest.skip("requires an allocated NPU")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), fault, "0", "capture"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NPU_DBO_CAPTURE_PASS " + fault in result.stdout


if __name__ == "__main__":
    _exercise(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 0,
        len(sys.argv) > 3 and sys.argv[3] == "capture",
    )
