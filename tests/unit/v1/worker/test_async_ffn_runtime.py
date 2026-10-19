# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MRV2 loading, eager isolation and graph-only resource lifetime contracts."""

from __future__ import annotations

import sys
import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
import torch  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.v1.worker.npu import async_ffn_model_runner as runtime  # noqa: E402
from afd_plugin.v1.worker.npu import ffn_model_runner as sync_runtime  # noqa: E402
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker  # noqa: E402


def config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={
            "afd": {
                "role": "ffn",
                "connector": "CAMAsyncAFDConnector",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
                "compute_gate_on_attention": True,
                "async": True,
            }
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=32,
                num_experts_per_tok=2,
                n_routed_experts=4,
                num_hidden_layers=6,
            ),
            dtype=torch.bfloat16,
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
    )


@pytest.fixture
def harness(monkeypatch):
    # These CPU tensor tests exercise execution/loading, while the configuration
    # and real-device contracts validate admission separately.
    monkeypatch.setattr(
        runtime, "validate_npu_model_runner_v2_config", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runtime, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    cfg = config()
    connector = CAMAsyncAFDConnector(
        2,
        0,
        cfg,
        AFDConfig(
            role="ffn",
            connector="CAMAsyncAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=2,
            compute_gate_on_attention=True,
            async_dp=True,
        ),
        0,
    )
    connector._initialized = True
    monkeypatch.setenv("AFD_ASYNC_CAM_LAYERED_GMM", "0")
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    runner = runtime.AsyncNPUFFNModelRunner(cfg, torch.device("cpu"), connector)
    runner.layer_ids = (2, 5)
    calls: list[str | tuple] = []
    payloads: deque[AFDA2FTransferPayload] = deque()

    def receive(**kwargs):
        calls.append("receive")
        return payloads.popleft()

    def compute(**kwargs):
        calls.append(("compute", kwargs["layer_idx"], kwargs["hidden_states"].shape[0]))
        return kwargs["hidden_states"].to(torch.bfloat16)

    def send(output, context, **kwargs):
        calls.append(("send", context.states.token_nums_rankid_layeridx))

    @contextmanager
    def forward(**kwargs):
        yield SimpleNamespace(dp_metadata=None, additional_kwargs={})

    monkeypatch.setattr(connector, "recv_attn_output", receive)
    monkeypatch.setattr(connector, "send_ffn_output", send)
    monkeypatch.setattr(runtime, "ascend_forward_context", forward)
    monkeypatch.setattr(runtime, "_set_moe_layer_index", lambda *args: None)
    runner.model = SimpleNamespace(compute_ffn_output=compute)
    return SimpleNamespace(
        runner=runner, connector=connector, calls=calls, payloads=payloads
    )


def payload(layer, rows):
    state = AFDAsyncTransferState(
        token_nums_rankid_layeridx=torch.tensor([rows, 0, layer, 0, 1]),
        group_list=torch.tensor([rows, 0]),
        dynamic_scales=torch.ones(8),
    )
    return AFDA2FTransferPayload(
        torch.ones(8, 32, dtype=torch.int8),
        AFDTransferContext(AFDTransferMetadata(0, 0, [8]), state),
    )


@pytest.mark.parametrize(
    "connector_name", ["CAMAsyncAFDConnector", "CAMP2pAFDConnector"]
)
def test_separate_constructors_preserve_graph_ownership(
    harness, monkeypatch, connector_name
):
    cfg = config()
    cfg.model_config.enforce_eager = False
    cfg.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(name="FULL")
    )
    is_async = connector_name == "CAMAsyncAFDConnector"
    cfg.additional_config["afd"].update(
        connector=connector_name, compute_gate_on_attention=is_async
    )
    cfg.additional_config["afd"]["async"] = is_async
    pool = object()
    calls = []

    def global_pool():
        assert not is_async, "Async must not acquire the synchronous graph pool"
        calls.append("pool")
        return pool

    monkeypatch.setattr(
        sync_runtime.current_platform, "get_global_graph_pool", global_pool
    )
    if is_async:
        runner = runtime.AsyncNPUFFNModelRunner(
            cfg, harness.runner.device, harness.connector
        )
        assert runner._graph is None and runner._graph_capture_count == 0
    else:
        monkeypatch.setattr(
            sync_runtime, "validate_npu_model_runner_v2_config", lambda *a, **k: None
        )
        monkeypatch.setattr(
            sync_runtime, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
        )
        monkeypatch.setattr(sync_runtime, "create_afd_npu_profiler", lambda role: None)
        runner = sync_runtime.AFDNPUFFNModelRunner(
            cfg,
            harness.runner.device,
            SimpleNamespace(control_plane=object(), is_initialized=False),
        )
        assert runner._acl_graphs == {}
        assert runner.graph_pool is pool
    assert runner.model is None
    assert calls == ([] if is_async else ["pool"])


def test_eager_uses_main_layer_loop_and_original_return_headers(harness):
    h = harness
    first, zero = payload(5, 3), payload(2, 0)
    h.payloads.extend((first, zero))
    assert h.runner.execute_connector_driven_step()
    assert [
        call for call in h.calls if isinstance(call, tuple) and call[0] == "compute"
    ] == [("compute", 5, 3), ("compute", 2, 0)]
    returned = [
        call[1] for call in h.calls if isinstance(call, tuple) and call[0] == "send"
    ]
    assert returned[0] is first.context.states.token_nums_rankid_layeridx
    assert returned[1] is zero.context.states.token_nums_rankid_layeridx
    assert h.runner._graph is None


def test_layered_eager_uses_raw_device_metadata(harness):
    h = harness
    h.payloads.extend((payload(5, 3), payload(2, 0)))
    observed = []

    def layered(hidden, scales, groups, header):
        observed.append((groups, header))
        return hidden.to(torch.bfloat16)

    h.runner._layered_executor = layered
    assert h.runner.execute_connector_driven_step()
    assert len(observed) == 2
    assert [int(header[2]) for _, header in observed] == [5, 2]
    assert not any(isinstance(call, tuple) and call[0] == "compute" for call in h.calls)


@pytest.mark.parametrize("speculative_tokens", [None, 3])
@pytest.mark.parametrize(
    "connector_name", ["CAMAsyncAFDConnector", "CAMP2pAFDConnector"]
)
def test_direct_loader_uses_role_model_config_and_keeps_memory_accounting(
    harness, monkeypatch, speculative_tokens, connector_name
):
    h = harness
    cfg = config()
    cfg.additional_config["afd"].update(
        connector=connector_name,
        compute_gate_on_attention=connector_name == "CAMAsyncAFDConnector",
    )
    cfg.additional_config["afd"]["async"] = connector_name == "CAMAsyncAFDConnector"
    model = h.runner.model
    connector = (
        h.connector
        if connector_name == "CAMAsyncAFDConnector"
        else SimpleNamespace(control_plane=object(), is_initialized=False)
    )
    if connector_name == "CAMAsyncAFDConnector":
        h.runner = runtime.AsyncNPUFFNModelRunner(cfg, h.runner.device, connector)
    else:
        monkeypatch.setattr(
            sync_runtime, "validate_npu_model_runner_v2_config", lambda *a, **k: None
        )
        monkeypatch.setattr(
            sync_runtime, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
        )
        monkeypatch.setattr(sync_runtime, "create_afd_npu_profiler", lambda role: None)
        h.runner = sync_runtime.AFDNPUFFNModelRunner(cfg, h.runner.device, connector)
    h.runner.model = model
    model = h.runner.model
    h.runner.model = None
    h.runner.vllm_config.load_config = object()
    h.runner.vllm_config.scheduler_config.max_num_seqs = 6
    h.runner.vllm_config.speculative_config = (
        None
        if speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=speculative_tokens)
    )
    seen = []
    preparation: list[tuple] = []

    @contextmanager
    def memory_profiler():
        yield SimpleNamespace(consumed_memory=1234)

    def get_loader(load_config):
        assert load_config is h.runner.vllm_config.load_config
        assert preparation == [
            (h.runner.vllm_config, 6, 1 + (speculative_tokens or 0)),
            (h.runner.vllm_config, h.runner.device),
        ]

        def load_model(**kwargs):
            seen.append(kwargs)
            return model

        return SimpleNamespace(load_model=load_model)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(
            set_mc2_tokens_capacity=lambda *args: preparation.append(args),
            set_mc2_mask=lambda *args: preparation.append(args),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader",
        SimpleNamespace(get_model_loader=get_loader),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.mem_utils",
        SimpleNamespace(DeviceMemoryProfiler=memory_profiler),
    )
    h.runner.load_model()
    assert h.runner.get_model() is model
    assert h.runner.model_memory_usage == 1234
    assert seen == [
        {"vllm_config": h.runner.vllm_config, "model_config": h.runner.model_config}
    ]
    with pytest.raises(RuntimeError, match="already loaded"):
        h.runner.load_model()


@pytest.mark.parametrize("state", ["live", "failed", "drained"])
def test_full_stop_only_releases_normal_completed_receiver(harness, monkeypatch, state):
    calls = []
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = harness.runner
    harness.runner.effective_mode = "full"
    harness.runner.shutdown = lambda: calls.append("release")
    worker._ffn_shutdown_event = threading.Event()
    event = worker._ffn_shutdown_event
    worker._ffn_loop_error = (
        RuntimeError("receiver failed") if state == "failed" else None
    )

    def join(timeout):
        assert event.is_set(), "stop must prevent subsequent submissions before joining"
        calls.append("join")

    worker._ffn_thread = SimpleNamespace(join=join, is_alive=lambda: state == "live")
    if state == "drained":
        worker.stop_ffn_server_loop()
        assert calls == ["join", "release"] and worker._ffn_thread is None
    else:
        with pytest.raises(RuntimeError):
            worker.stop_ffn_server_loop()
        assert calls == ["join"] and worker._ffn_thread is not None
        if state == "failed":
            for _ in range(2):
                with pytest.raises(RuntimeError, match="worker loop failed"):
                    worker.raise_ffn_loop_error_if_any()
            with pytest.raises(RuntimeError):
                worker.start_ffn_server_loop()


@pytest.mark.parametrize("fail", [False, True])
def test_full_receiver_exit_requires_success_even_during_shutdown(harness, fail):
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = harness.runner
    harness.runner.effective_mode = "full"
    worker._ffn_thread = None
    worker._ffn_loop_error = None
    worker._bind_cpus_once = lambda: None

    def serve():
        worker._ffn_shutdown_event.set()
        if fail:
            raise RuntimeError("partial device completion")

    worker._run_ffn_server_loop = serve
    worker.start_ffn_server_loop()
    assert isinstance(worker._ffn_thread, threading.Thread)
    worker._ffn_thread.join(3)
    assert not worker._ffn_thread.is_alive()
    assert harness.runner._graph_receiver_exited is not fail
    assert (worker._ffn_loop_error is not None) is fail
