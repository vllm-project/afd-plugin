"""Unit tests for AFD CUDA GPU ModelRunnerV2 support."""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    BatchDescriptor,
    ForwardContext,
)
from vllm.v1.worker import gpu_model_runner as native_v1  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils, dp_utils  # noqa: E402
from vllm.v1.worker.gpu import model_runner as native_v2  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402

from afd_plugin.model_executor.models import (  # noqa: E402
    forward_context as afd_forward_context,
)
from afd_plugin.model_executor.models.model_utils import (  # noqa: E402
    get_afd_model_config,
)
from afd_plugin.v1.worker import (  # noqa: E402
    attention_model_runner_v2 as v2_runner_module,
)
from afd_plugin.v1.worker import (  # noqa: E402
    attention_worker,
    ffn_worker,
)
from afd_plugin.v1.worker.attention_model_runner import (  # noqa: E402
    AFDAttentionModelRunner,
)
from afd_plugin.v1.worker.attention_model_runner_v2 import (  # noqa: E402
    AFDAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker  # noqa: E402
from afd_plugin.validation import (  # noqa: E402
    validate_gpu_model_runner_v2_config,
    validate_npu_model_runner_v2_config,
)

TEST_MODEL_MAX_LEN = 163840
TEST_HIDDEN_SIZE = 2048
TEST_CUDAGRAPH_CAPTURE_SIZE = 8


class _RecordingConnector:
    def __init__(self, events: list[str]):
        self.events = events
        self.control_plane = self
        self.closed = False

    def update_state_from_dp_metadata(self, payload):
        self.events.append("control_update")
        self.last_payload = payload

    def send_dp_metadata_list(self, payload):
        self.events.append("control_send")
        self.last_payload = payload

    def send_attn_output(self, hidden_states, context, **kwargs):
        self.events.append("data")

    def close(self):
        self.closed = True
        self.events.append("close")


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.stopped = False

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


class _RunnerRecorder:
    instances = []

    def __init__(self, vllm_config, device):
        type(self).instances.append((vllm_config, device))


class _NativeRunnerSentinel:
    instances = 0

    def __init__(self, vllm_config, device):
        type(self).instances += 1


def _v2_config(
    *,
    role: str = "attention",
    architecture: str = "DeepseekV2ForCausalLM",
    use_v2: bool = True,
    num_attention_ranks: int = 1,
    num_ffn_ranks: int = 1,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    enforce_eager: bool = True,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int = 0,
):
    hf_config = SimpleNamespace(
        architectures=[architecture],
        model_type="deepseek_v2",
        hidden_size=TEST_HIDDEN_SIZE,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        runner_type="generate",
        is_moe=True,
        is_encoder_decoder=False,
        is_multimodal_model=False,
        enforce_eager=enforce_eager,
        enable_prompt_embeds=False,
        enable_return_routed_experts=False,
        quantization=None,
        quantization_config=None,
        max_model_len=TEST_MODEL_MAX_LEN,
        dtype=torch.bfloat16,
        get_hidden_size=lambda: TEST_HIDDEN_SIZE,
    )
    parallel_config = SimpleNamespace(
        data_parallel_size=data_parallel_size,
        data_parallel_rank=0,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        enable_elastic_ep=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        is_moe_model=True,
        enable_dbo=False,
        use_ubatching=False,
        num_ubatches=1,
    )
    compilation_config = SimpleNamespace(
        mode=None,
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        max_cudagraph_capture_size=max_cudagraph_capture_size,
        compile_sizes=[],
        compile_ranges_endpoints=[model_config.max_model_len],
        pass_config=SimpleNamespace(
            enable_sp=False,
        ),
    )
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        additional_config={
            "afd": {
                "role": role,
                "connector": "P2pNcclAFDConnector",
                "num_attention_ranks": num_attention_ranks,
                "num_ffn_ranks": num_ffn_ranks,
                "compute_gate_on_attention": False,
            },
        },
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            async_scheduling=False,
            max_num_batched_tokens=TEST_MODEL_MAX_LEN,
            max_num_seqs=TEST_CUDAGRAPH_CAPTURE_SIZE,
        ),
        lora_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        diffusion_config=None,
        quant_config=None,
    )


def _runner_for_metadata(
    events: list[str],
    *,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
):
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.vllm_config = _v2_config(
        num_attention_ranks=data_parallel_size,
        num_ffn_ranks=data_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    runner.vllm_config.parallel_config.data_parallel_rank = data_parallel_rank
    runner.connector = _RecordingConnector(events)
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_suppress_metadata_send = False
    runner._afd_transaction_counter = 0
    runner.prof = _StepProfiler()
    runner.cudagraph_manager = SimpleNamespace(run_fullgraph=lambda _desc: None)
    runner.is_encoder_only = False
    return runner


def _scheduler_output(num_tokens: int):
    return SimpleNamespace(total_num_scheduled_tokens=num_tokens)


def _single_rank_graph_config(*, role: str = "attention"):
    config = _v2_config(
        role=role,
        enforce_eager=False,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[TEST_CUDAGRAPH_CAPTURE_SIZE],
        max_cudagraph_capture_size=TEST_CUDAGRAPH_CAPTURE_SIZE,
    )
    return config


def _no_graph_model_manager():
    manager = object.__new__(cudagraph_utils.ModelCudaGraphManager)
    manager._graphs_captured = False
    manager._candidates = {}
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    return manager


def _fake_worker_init(worker):
    worker.device = torch.device("cpu")
    if worker.use_v2_model_runner:
        from vllm.v1.worker.gpu import model_runner as imported_module
    else:
        from vllm.v1.worker import gpu_model_runner as imported_module

    worker.model_runner = imported_module.GPUModelRunner(
        worker.vllm_config,
        worker.device,
    )


def _construct_like_native_worker(worker):
    if worker.use_v2_model_runner:
        from vllm.v1.worker.gpu import model_runner as imported_module
    else:
        from vllm.v1.worker import gpu_model_runner as imported_module

    return imported_module.GPUModelRunner(
        worker.vllm_config,
        worker.device,
    )


def _prepare_worker(worker_cls, config, *, use_v2):
    worker = object.__new__(worker_cls)
    worker.vllm_config = config
    worker.model_config = config.model_config
    worker.use_v2_model_runner = use_v2
    worker.device_config = SimpleNamespace(device_type="cuda")
    return worker


def test_pinned_worker_init_device_signature_and_internal_runner_import():
    upstream_init_device = inspect.unwrap(Worker.init_device)
    assert inspect.signature(upstream_init_device) == inspect.signature(
        attention_worker.AFDAttentionWorker.init_device,
    )
    assert inspect.signature(upstream_init_device) == inspect.signature(
        ffn_worker.AFDFFNWorker.init_device,
    )
    assert inspect.signature(upstream_init_device) == inspect.Signature(
        [
            inspect.Parameter(
                "self",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ),
        ],
    )
    source = inspect.getsource(upstream_init_device)
    assert "from vllm.v1.worker.gpu.model_runner import" in source
    assert "from vllm.v1.worker.gpu_model_runner import" in source
    assert "GPUModelRunner as GPUModelRunnerV2" in source
    assert "GPUModelRunnerV2(" in source
    assert "GPUModelRunnerV1(" in source
    assert "self.model_runner = GPUModelRunnerV1" in source


def test_v2_fullgraph_replay_hook_restores_manager_instance_after_success():
    events: list[str] = []
    runner = _runner_for_metadata(events)
    descriptor = SimpleNamespace(num_tokens=8)

    def native_replay(self, desc):
        events.append("native_replay")
        return desc.num_tokens

    native_bound_replay = MethodType(native_replay, runner.cudagraph_manager)
    runner.cudagraph_manager.run_fullgraph = native_bound_replay
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph

    with v2_runner_module._use_afd_fullgraph_replay_hook(runner, 3):
        assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
        assert runner.cudagraph_manager.run_fullgraph(descriptor) == 8

    assert runner.cudagraph_manager.__dict__["run_fullgraph"] is native_bound_replay
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    assert events == ["control_update", "control_send", "native_replay"]


def test_v2_fullgraph_replay_hook_removes_temporary_instance_override():
    class Manager:
        def run_fullgraph(self, desc):
            return desc.num_tokens

    runner = _runner_for_metadata([])
    runner.cudagraph_manager = Manager()
    descriptor = SimpleNamespace(num_tokens=8)
    assert "run_fullgraph" not in vars(runner.cudagraph_manager)

    with v2_runner_module._use_afd_fullgraph_replay_hook(runner, 3):
        assert runner.cudagraph_manager.run_fullgraph(descriptor) == 8

    assert "run_fullgraph" not in vars(runner.cudagraph_manager)


def test_v2_fullgraph_replay_hooks_on_two_managers_do_not_interfere():
    first_events: list[str] = []
    second_events: list[str] = []
    first = _runner_for_metadata(first_events)
    second = _runner_for_metadata(second_events)
    descriptor = SimpleNamespace(num_tokens=8)

    def first_replay(_self, _desc):
        first_events.append("native_replay")

    def second_replay(_self, _desc):
        second_events.append("native_replay")

    first_original = MethodType(first_replay, first.cudagraph_manager)
    second_original = MethodType(second_replay, second.cudagraph_manager)
    first.cudagraph_manager.run_fullgraph = first_original
    second.cudagraph_manager.run_fullgraph = second_original

    with v2_runner_module._use_afd_fullgraph_replay_hook(first, 3):
        first_hook = first.cudagraph_manager.__dict__["run_fullgraph"]
        with v2_runner_module._use_afd_fullgraph_replay_hook(second, 5):
            assert first.cudagraph_manager.__dict__["run_fullgraph"] is first_hook
            first.cudagraph_manager.run_fullgraph(descriptor)
            second.cudagraph_manager.run_fullgraph(descriptor)
        assert second.cudagraph_manager.__dict__["run_fullgraph"] is second_original
        assert first.cudagraph_manager.__dict__["run_fullgraph"] is first_hook

    assert first.cudagraph_manager.__dict__["run_fullgraph"] is first_original
    assert second.cudagraph_manager.__dict__["run_fullgraph"] is second_original
    assert first_events == ["control_update", "control_send", "native_replay"]
    assert second_events == ["control_update", "control_send", "native_replay"]


def test_v2_fullgraph_replay_hook_rejects_same_manager_reentry():
    runner = _runner_for_metadata([])
    original = runner.cudagraph_manager.__dict__["run_fullgraph"]

    with v2_runner_module._use_afd_fullgraph_replay_hook(runner, 3):
        active_hook = runner.cudagraph_manager.__dict__["run_fullgraph"]
        with (
            pytest.raises(RuntimeError, match="already active"),
            v2_runner_module._use_afd_fullgraph_replay_hook(runner, 3),
        ):
            pass
        assert runner.cudagraph_manager.__dict__["run_fullgraph"] is active_hook

    assert runner.cudagraph_manager.__dict__["run_fullgraph"] is original


@pytest.mark.parametrize(
    (
        "role",
        "architecture",
        "num_attention_ranks",
        "num_ffn_ranks",
        "data_parallel_size",
        "tensor_parallel_size",
        "prefill_context_parallel_size",
    ),
    [
        ("attention", "DeepseekV2ForCausalLM", 1, 1, 1, 1, 1),
        ("ffn", "AFDDeepseekV2ForCausalLM", 1, 1, 1, 1, 1),
        ("attention", "DeepseekV3ForCausalLM", 6, 3, 2, 3, 1),
        ("ffn", "Qwen3MoeForCausalLM", 6, 3, 3, 1, 1),
    ],
)
def test_v2_validator_accepts_structurally_valid_topologies(
    role,
    architecture,
    num_attention_ranks,
    num_ffn_ranks,
    data_parallel_size,
    tensor_parallel_size,
    prefill_context_parallel_size,
):
    config = _v2_config(
        role=role,
        architecture=architecture,
        num_attention_ranks=num_attention_ranks,
        num_ffn_ranks=num_ffn_ranks,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
    )

    validate_gpu_model_runner_v2_config(
        config,
        expected_role=role,
        device_type="cuda",
    )


def test_v2_validator_has_shared_owner_and_v2_export_boundary():
    assert validate_gpu_model_runner_v2_config.__module__ == "afd_plugin.validation"
    assert v2_runner_module.__all__ == ["AFDAttentionModelRunnerV2"]


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_v2_graph_validator_accepts_general_capture_configuration(role):
    config = _v2_config(
        role=role,
        architecture="DeepseekV3ForCausalLM",
        num_attention_ranks=3,
        num_ffn_ranks=3,
        data_parallel_size=3,
        enforce_eager=False,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[4, 16],
        max_cudagraph_capture_size=16,
    )
    config.model_config.hf_config.hidden_size = 5120
    config.scheduler_config.max_num_seqs = 16
    config.compilation_config.compile_sizes = [16]
    config.compilation_config.compile_ranges_endpoints = [16, TEST_MODEL_MAX_LEN]

    validate_gpu_model_runner_v2_config(
        config,
        expected_role=role,
        device_type="cuda",
    )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_v2_graph_validator_rejects_non_full_decode_only_modes(cudagraph_mode):
    config = _single_rank_graph_config()
    config.compilation_config.cudagraph_mode = cudagraph_mode

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        validate_gpu_model_runner_v2_config(
            config,
            expected_role="attention",
            device_type="cuda",
        )


def test_v2_validator_rejects_role_mismatch():
    with pytest.raises(ValueError, match="role mismatch"):
        validate_gpu_model_runner_v2_config(
            _v2_config(role="attention"),
            expected_role="ffn",
            device_type="cuda",
        )


def test_v2_validator_rejects_non_cuda_device():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        validate_gpu_model_runner_v2_config(
            _v2_config(),
            expected_role="attention",
            device_type="npu",
        )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY],
)
def test_npu_v2_validator_allows_full_acl_graph(cudagraph_mode):
    config = _v2_config(
        enforce_eager=False,
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=[TEST_CUDAGRAPH_CAPTURE_SIZE],
        max_cudagraph_capture_size=TEST_CUDAGRAPH_CAPTURE_SIZE,
    )
    config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"

    validate_npu_model_runner_v2_config(
        config,
        expected_role="attention",
        device_type="npu",
    )


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_npu_v2_validator_rejects_non_full_acl_graph(cudagraph_mode):
    config = _v2_config(enforce_eager=False, cudagraph_mode=cudagraph_mode)
    config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"

    with pytest.raises(RuntimeError, match="ACL graph modes FULL"):
        validate_npu_model_runner_v2_config(
            config,
            expected_role="attention",
            device_type="npu",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c.additional_config["afd"].update(
                connector="CAMAsyncAFDConnector",
            ),
            "P2pNcclAFDConnector",
        ),
        (
            lambda c: c.additional_config["afd"].update(
                compute_gate_on_attention=True,
            ),
            "compute_gate_on_attention=false",
        ),
        (
            lambda c: c.additional_config["afd"].update(num_attention_ranks=2),
            "ranks must match DP",
        ),
        (lambda c: setattr(c.parallel_config, "enable_dbo", True), "DBO"),
        (lambda c: setattr(c.parallel_config, "use_ubatching", True), "ubatching"),
        (
            lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
            "PP or CP",
        ),
        (
            lambda c: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
            "PP or CP",
        ),
        (
            lambda c: setattr(c.parallel_config, "decode_context_parallel_size", 2),
            "PP or CP",
        ),
        (
            lambda c: setattr(c.parallel_config, "enable_elastic_ep", True),
            "static expert",
        ),
        (
            lambda c: setattr(c.parallel_config, "enable_eplb", True),
            "static expert",
        ),
        (
            lambda c: setattr(c.parallel_config, "use_sequence_parallel_moe", True),
            "static expert",
        ),
        (
            lambda c: setattr(c.parallel_config, "enable_expert_parallel", False),
            "static expert",
        ),
        (
            lambda c: setattr(c.compilation_config.pass_config, "enable_sp", True),
            "static expert",
        ),
        (
            lambda c: setattr(
                c.model_config.hf_config,
                "architectures",
                ["UnsupportedForCausalLM"],
            ),
            "registered AFD model",
        ),
    ],
)
def test_v2_validator_rejects_unsupported_execution_constraints(
    mutation,
    message,
):
    config = _v2_config()
    mutation(config)

    with pytest.raises(RuntimeError, match=message):
        validate_gpu_model_runner_v2_config(
            config,
            expected_role="attention",
            device_type="cuda",
        )


def test_v2_metadata_provider_has_complete_transaction_and_dp_dependencies():
    runner = _runner_for_metadata([])

    first = runner.build_afd_metadata(None, 7)
    second = runner.build_afd_metadata(None, 7)

    assert first.transaction_id == "afd-0"
    assert second.transaction_id == "afd-1"
    assert first.tokens_lens == [7]
    assert first.tokens_unpadded_lens == [7]
    assert runner.build_capture_dp_metadata(9).num_tokens_across_dp_cpu.tolist() == [9]


def _capture_descriptor(num_reqs: int, num_tokens: int):
    return SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
    )


@pytest.mark.parametrize(
    ("data_parallel_size", "expected_dp_counts"),
    [
        (1, [[8], [8], [16], [16]]),
        (2, [[8, 8], [8, 8], [16, 16], [16, 16]]),
    ],
    ids=["dp1", "dp2"],
)
def test_v2_capture_publishes_two_descriptor_events_outside_graph_body(
    monkeypatch,
    data_parallel_size,
    expected_dp_counts,
):
    events: list[str] = []
    runner = _runner_for_metadata(
        events,
        data_parallel_size=data_parallel_size,
    )
    descriptors = [_capture_descriptor(1, 8), _capture_descriptor(2, 16)]
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: descriptors},
    )
    in_graph_body = False
    payloads = []
    captured_metadata = []
    prepare_calls = []

    class CaptureConnector(_RecordingConnector):
        def update_state_from_dp_metadata(self, payload):
            assert not in_graph_body
            super().update_state_from_dp_metadata(payload)

        def send_dp_metadata_list(self, payload):
            assert not in_graph_body
            payloads.append(payload)
            super().send_dp_metadata_list(payload)

    runner.connector = CaptureConnector(events)

    def original_prepare(num_reqs, num_tokens, *args, **kwargs):
        prepare_calls.append((num_reqs, num_tokens))
        return f"attention-state-{len(prepare_calls)}"

    original_create = afd_forward_context.forward_context_module.create_forward_context

    def native_create_forward_context():
        return SimpleNamespace(
            additional_kwargs={},
            ubatch_slices=None,
            dp_metadata=None,
            batch_descriptor=None,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )

    def native_capture(self):
        nonlocal in_graph_body
        for desc in descriptors:
            for _ in (True, False):
                state = cudagraph_utils.prepare_inputs_to_capture(
                    desc.num_reqs,
                    desc.num_tokens,
                    None,
                    None,
                    None,
                    [],
                    None,
                    False,
                )
                assert state == f"attention-state-{len(prepare_calls)}"
                events.append("graph_enter")
                in_graph_body = True
                context = (
                    afd_forward_context.forward_context_module.create_forward_context()
                )
                captured_metadata.append(
                    context.additional_kwargs["afd_metadata"].clone(),
                )
                in_graph_body = False
                events.append("graph_exit")
        return 23

    monkeypatch.setattr(
        cudagraph_utils,
        "prepare_inputs_to_capture",
        original_prepare,
    )
    monkeypatch.setattr(
        afd_forward_context.forward_context_module,
        "create_forward_context",
        native_create_forward_context,
    )
    monkeypatch.setattr(native_v2.GPUModelRunner, "capture_model", native_capture)

    assert AFDAttentionModelRunnerV2.capture_model(runner) == 23

    assert prepare_calls == [(1, 8), (1, 8), (2, 16), (2, 16)]
    assert [(p.is_warmup, p.is_graph_capturing) for p in payloads] == [
        (True, False),
        (False, True),
        (True, False),
        (False, True),
    ]
    assert [
        p.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist() for p in payloads
    ] == expected_dp_counts
    assert [m.tokens_lens for m in captured_metadata] == [[8], [8], [16], [16]]
    assert [m.tokens_unpadded_lens for m in captured_metadata] == [
        [8],
        [8],
        [16],
        [16],
    ]
    assert events == [
        "control_update",
        "control_send",
        "graph_enter",
        "graph_exit",
        "control_update",
        "control_send",
        "graph_enter",
        "graph_exit",
        "control_update",
        "control_send",
        "graph_enter",
        "graph_exit",
        "control_update",
        "control_send",
        "graph_enter",
        "graph_exit",
    ]
    assert cudagraph_utils.prepare_inputs_to_capture is original_prepare
    assert (
        afd_forward_context.forward_context_module.create_forward_context
        is native_create_forward_context
    )
    assert native_create_forward_context is not original_create


@pytest.mark.parametrize("failure", ["prepare", "control", "forward"])
def test_v2_capture_restores_symbol_and_sidecars_on_failure(monkeypatch, failure):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    descriptor = _capture_descriptor(1, 8)
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: [descriptor]},
    )
    previous_metadata = object()
    runner._afd_pending_metadata = previous_metadata
    runner._afd_suppress_metadata_send = False
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False

    def original_prepare(*args, **kwargs):
        if failure == "prepare":
            raise RuntimeError("prepare failed")
        return "attention-state"

    if failure == "control":

        def fail_control(payload):
            raise RuntimeError("control failed")

        runner.connector.send_dp_metadata_list = fail_control

    def native_create_forward_context():
        return SimpleNamespace(
            additional_kwargs={},
            ubatch_slices=None,
            dp_metadata=None,
            batch_descriptor=None,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )

    def native_capture(self):
        cudagraph_utils.prepare_inputs_to_capture(
            descriptor.num_reqs,
            descriptor.num_tokens,
            None,
            None,
            None,
            [],
            None,
            False,
        )
        afd_forward_context.forward_context_module.create_forward_context()
        if failure == "forward":
            raise RuntimeError("forward failed")
        return 1

    monkeypatch.setattr(
        cudagraph_utils,
        "prepare_inputs_to_capture",
        original_prepare,
    )
    monkeypatch.setattr(
        afd_forward_context.forward_context_module,
        "create_forward_context",
        native_create_forward_context,
    )
    monkeypatch.setattr(native_v2.GPUModelRunner, "capture_model", native_capture)

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        AFDAttentionModelRunnerV2.capture_model(runner)

    assert cudagraph_utils.prepare_inputs_to_capture is original_prepare
    assert runner._afd_pending_metadata is previous_metadata
    assert runner._afd_suppress_metadata_send is False
    assert runner._is_warmup is False
    assert runner._afd_is_graph_capturing is False


@pytest.mark.parametrize(
    ("drift", "expected_prepare_count"),
    [
        ("descriptor", 0),
        ("missing", 3),
        ("extra", 5),
        ("order", 2),
        ("shape", 1),
    ],
)
def test_v2_capture_source_drift_fails_loud_and_restores(
    monkeypatch,
    drift,
    expected_prepare_count,
):
    runner = _runner_for_metadata([])
    descriptors = [_capture_descriptor(1, 8), _capture_descriptor(2, 16)]
    if drift == "descriptor":
        descriptors[0].cg_mode = CUDAGraphMode.PIECEWISE
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: descriptors},
    )
    calls = [(1, 8), (1, 8), (2, 16), (2, 16)]
    if drift == "missing":
        calls.pop()
    elif drift == "extra":
        calls.append((2, 16))
    elif drift == "order":
        calls[1], calls[2] = calls[2], calls[1]
    elif drift == "shape":
        calls[0] = (1, 9)
    original_calls = []

    def original_prepare(num_reqs, num_tokens, *args, **kwargs):
        original_calls.append((num_reqs, num_tokens))
        return "attention-state"

    def native_capture(self):
        for num_reqs, num_tokens in calls:
            cudagraph_utils.prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                None,
                None,
                None,
                [],
                None,
                False,
            )
        return 1

    monkeypatch.setattr(
        cudagraph_utils,
        "prepare_inputs_to_capture",
        original_prepare,
    )
    monkeypatch.setattr(native_v2.GPUModelRunner, "capture_model", native_capture)

    with pytest.raises(RuntimeError, match="descriptor|call count|extra"):
        AFDAttentionModelRunnerV2.capture_model(runner)

    assert cudagraph_utils.prepare_inputs_to_capture is original_prepare
    assert original_calls == calls[:expected_prepare_count]


@pytest.mark.parametrize(
    ("data_parallel_rank", "real_tokens"),
    [(0, 3), (1, 1)],
    ids=["rank0-real3", "rank1-idle-real1"],
)
def test_v2_dp2_repeated_fullgraph_replay_sends_local_real_and_padded_tokens(
    monkeypatch,
    data_parallel_rank,
    real_tokens,
):
    events: list[str] = []
    runner = _runner_for_metadata(
        events,
        data_parallel_size=2,
        data_parallel_rank=data_parallel_rank,
    )
    runner.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
    )
    manager = SimpleNamespace()
    runner.cudagraph_manager = manager
    descriptor = SimpleNamespace(num_tokens=8)
    payloads = []
    metadata_seen = []
    replay_returns = []

    class ReplayConnector(_RecordingConnector):
        def send_dp_metadata_list(self, payload):
            payloads.append(payload)
            super().send_dp_metadata_list(payload)

    runner.connector = ReplayConnector(events)

    def native_replay(self, desc):
        metadata = runner._afd_pending_metadata
        metadata_seen.append(
            (list(metadata.tokens_lens), list(metadata.tokens_unpadded_lens)),
        )
        events.append("native_replay")
        return f"replay-{len(metadata_seen)}"

    def native_execute(self, *args, **kwargs):
        assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
        for _ in range(2):
            replay_returns.append(manager.run_fullgraph(descriptor))
        return "native-result"

    native_bound_replay = MethodType(native_replay, manager)
    manager.run_fullgraph = native_bound_replay
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert (
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(real_tokens),
        )
        == "native-result"
    )

    assert replay_returns == ["replay-1", "replay-2"]
    assert metadata_seen == [
        ([8], [real_tokens]),
        ([8], [real_tokens]),
    ]
    assert [(p.is_warmup, p.is_graph_capturing) for p in payloads] == [
        (False, False),
        (False, False),
    ]
    assert [
        p.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist() for p in payloads
    ] == [[8, 8], [8, 8]]
    assert events == [
        "control_update",
        "control_send",
        "native_replay",
        "control_update",
        "control_send",
        "native_replay",
    ]
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    assert manager.__dict__["run_fullgraph"] is native_bound_replay
    assert runner._afd_pending_metadata is None


def test_v2_dp2_zero_work_without_native_replay_sends_no_control(monkeypatch):
    events: list[str] = []
    runner = _runner_for_metadata(
        events,
        data_parallel_size=2,
        data_parallel_rank=1,
    )
    runner.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
    )
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph

    def native_execute(self, *args, **kwargs):
        return "idle-result"

    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert (
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(0),
        )
        == "idle-result"
    )
    assert events == []
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    assert runner._afd_pending_metadata is None


def test_v2_replay_exception_restores_manager_and_sidecars(monkeypatch):
    runner = _runner_for_metadata([])
    runner.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
    )
    manager = SimpleNamespace()
    runner.cudagraph_manager = manager
    descriptor = SimpleNamespace(num_tokens=8)
    previous_metadata = object()
    runner._afd_pending_metadata = previous_metadata

    def native_replay(self, desc):
        raise RuntimeError("replay failed")

    def native_execute(self, *args, **kwargs):
        return manager.run_fullgraph(descriptor)

    native_bound_replay = MethodType(native_replay, manager)
    manager.run_fullgraph = native_bound_replay
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph
    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    with pytest.raises(RuntimeError, match="replay failed"):
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(3),
        )

    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    assert manager.__dict__["run_fullgraph"] is native_bound_replay
    assert runner._afd_pending_metadata is previous_metadata
    assert runner._afd_suppress_metadata_send is False
    assert runner._is_warmup is False
    assert runner._afd_is_graph_capturing is False


def test_v2_graph_miss_uses_provider_once_without_replay_control(monkeypatch):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    runner.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
    )
    replay_calls = []
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=3),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    def original_create_forward_context():
        return context

    def native_execute(self, *args, **kwargs):
        native_context = (
            afd_forward_context.forward_context_module.create_forward_context()
        )
        native_context.additional_kwargs["afd_metadata"].connector.send_attn_output(
            None,
            None,
        )
        return "eager-result"

    monkeypatch.setattr(
        afd_forward_context.forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph
    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert (
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(3),
        )
        == "eager-result"
    )
    assert replay_calls == []
    assert events == ["control_update", "control_send", "data"]
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay


def test_v2_eager_execute_does_not_replace_fullgraph_replay(monkeypatch):
    runner = _runner_for_metadata([])
    replay_symbol_during_execute = []

    def native_replay(self, desc):
        return None

    def native_execute(self, *args, **kwargs):
        replay_symbol_during_execute.append(
            cudagraph_utils.CudaGraphManager.run_fullgraph,
        )
        return "eager-result"

    monkeypatch.setattr(
        cudagraph_utils.CudaGraphManager,
        "run_fullgraph",
        native_replay,
    )
    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert (
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(3),
        )
        == "eager-result"
    )
    assert replay_symbol_during_execute == [native_replay]
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is native_replay


def test_v2_profile_before_graph_manager_uses_provider_without_replay_hook(
    monkeypatch,
):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    runner.vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY
    )
    runner.cudagraph_manager = None
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=7),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    forward_context_module = afd_forward_context.forward_context_module

    def original_create_forward_context():
        return context

    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )

    def unexpected_replay_hook(*_args, **_kwargs):
        raise AssertionError("profile execution must not install the replay hook")

    monkeypatch.setattr(
        v2_runner_module,
        "_use_afd_fullgraph_replay_hook",
        unexpected_replay_hook,
    )
    class_replay = cudagraph_utils.CudaGraphManager.run_fullgraph
    execute_calls = []

    def native_execute(self, *args, **kwargs):
        execute_calls.append(kwargs)
        assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
        native_context = forward_context_module.create_forward_context()
        native_context.additional_kwargs["afd_metadata"].connector.send_attn_output(
            None,
            None,
        )
        return "profile-result"

    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert (
        AFDAttentionModelRunnerV2.execute_model(
            runner,
            _scheduler_output(7),
            is_profile=True,
        )
        == "profile-result"
    )

    assert execute_calls == [
        {
            "dummy_run": False,
            "skip_attn_for_dummy_run": False,
            "is_profile": True,
            "context_len": 0,
        },
    ]
    assert events == ["control_update", "control_send", "data"]
    assert cudagraph_utils.CudaGraphManager.run_fullgraph is class_replay
    assert runner._afd_pending_metadata is None
    assert runner._afd_suppress_metadata_send is False
    assert runner._is_warmup is False
    assert runner._afd_is_graph_capturing is False
    assert (
        forward_context_module.create_forward_context is original_create_forward_context
    )


@pytest.mark.parametrize("need_eager", [False, True])
def test_dp1_native_token_seams_preserve_real_token_count(need_eager):
    cudagraph_manager = None if need_eager else _no_graph_model_manager()
    descriptor, dp_metadata = dp_utils.dispatch_cg_and_sync_dp(
        cudagraph_manager=cudagraph_manager,
        num_reqs=2,
        num_tokens=7,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=need_eager,
    )
    assert dp_metadata is None
    assert descriptor.cg_mode is CUDAGraphMode.NONE
    assert descriptor.num_tokens == 7


def test_dp1_ordinary_native_path_binds_unpadded_afd_metadata():
    manager = _no_graph_model_manager()
    descriptor, dp_metadata = dp_utils.dispatch_cg_and_sync_dp(
        cudagraph_manager=manager,
        num_reqs=2,
        num_tokens=7,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=False,
    )
    assert dp_metadata is None

    runner = _runner_for_metadata([])
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=descriptor.num_tokens),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    runner.install_afd_metadata_on_forward_context(context)
    metadata = context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [7]
    assert metadata.tokens_unpadded_lens == [7]
    payload = runner.connector.last_payload
    assert payload.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist() == [7]


@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, True)],
)
def test_native_v2_dummy_profile_thin_path_uses_afd_execute_wrapper(
    monkeypatch,
    is_profile,
    skip_attn,
):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    runner.max_num_reqs = 2
    runner.lora_config = None
    runner.is_first_pp_rank = True
    runner.is_last_pp_rank = False

    class KVConnector:
        def set_disabled(self, disabled):
            self.disabled = disabled

    runner.kv_connector = KVConnector()
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=7),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    def original_create_forward_context():
        return context

    forward_context_module = afd_forward_context.forward_context_module
    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )
    execute_calls: list[tuple[bool, bool, bool]] = []

    def native_execute(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len=0,
    ):
        execute_calls.append(
            (dummy_run, skip_attn_for_dummy_run, is_profile),
        )
        native_context = forward_context_module.create_forward_context()
        native_context.additional_kwargs["afd_metadata"].connector.send_attn_output(
            None,
            None,
        )

    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    assert AFDAttentionModelRunnerV2._dummy_run(
        runner,
        7,
        skip_attn=skip_attn,
        is_profile=is_profile,
        skip_eplb=True,
    ) == (None, None)

    assert execute_calls == [(True, skip_attn, is_profile)]
    assert events == ["control_update", "control_send", "data"]
    assert runner._afd_pending_metadata is None
    assert runner.kv_connector.disabled is False
    assert (
        forward_context_module.create_forward_context is original_create_forward_context
    )


def test_v2_execute_and_capture_signatures_are_pinned():
    assert inspect.signature(
        AFDAttentionModelRunnerV2.execute_model,
        eval_str=True,
    ) == inspect.signature(
        native_v2.GPUModelRunner.execute_model,
        eval_str=True,
    )
    assert inspect.signature(
        AFDAttentionModelRunnerV2.__init__,
        eval_str=True,
    ) == inspect.signature(
        native_v2.GPUModelRunner.__init__,
        eval_str=True,
    )
    assert inspect.signature(
        AFDAttentionModelRunnerV2.capture_model,
        eval_str=True,
    ) == inspect.signature(
        native_v2.GPUModelRunner.capture_model,
        eval_str=True,
    )


@pytest.mark.parametrize("raise_in_native", [False, True])
def test_v2_provider_orders_control_before_data_and_always_restores(
    monkeypatch,
    raise_in_native,
):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=BatchDescriptor(num_tokens=7),
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    def original_create_forward_context():
        return context

    forward_context_module = afd_forward_context.forward_context_module
    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )

    def native_execute(self, *args, **kwargs):
        native_context = forward_context_module.create_forward_context()
        native_context.additional_kwargs["afd_metadata"].connector.send_attn_output(
            None,
            None,
        )
        if raise_in_native:
            raise RuntimeError("native execute failed")
        return "native-result"

    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    if raise_in_native:
        with pytest.raises(RuntimeError, match="native execute failed"):
            AFDAttentionModelRunnerV2.execute_model(runner, _scheduler_output(7))
    else:
        assert (
            AFDAttentionModelRunnerV2.execute_model(
                runner,
                _scheduler_output(7),
            )
            == "native-result"
        )

    assert events == ["control_update", "control_send", "data"]
    assert runner._afd_pending_metadata is None
    assert (
        forward_context_module.create_forward_context is original_create_forward_context
    )
    assert runner.prof.steps == 1


def test_v2_repeated_execute_reinstalls_provider_and_advances_transactions(
    monkeypatch,
):
    events: list[str] = []
    runner = _runner_for_metadata(events)
    contexts: list[ForwardContext] = []
    transaction_ids: list[str] = []

    def original_create_forward_context():
        context = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={},
            dp_metadata=None,
            ubatch_slices=None,
            batch_descriptor=BatchDescriptor(num_tokens=7),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        contexts.append(context)
        return context

    forward_context_module = afd_forward_context.forward_context_module
    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )

    def native_execute(self, *args, **kwargs):
        context = forward_context_module.create_forward_context()
        metadata = context.additional_kwargs["afd_metadata"]
        transaction_ids.append(metadata.transaction_id)
        metadata.connector.send_attn_output(None, None)
        return "native-result"

    monkeypatch.setattr(native_v2.GPUModelRunner, "execute_model", native_execute)

    for _ in range(2):
        assert (
            AFDAttentionModelRunnerV2.execute_model(
                runner,
                _scheduler_output(7),
            )
            == "native-result"
        )
        assert runner._afd_pending_metadata is None
        assert (
            forward_context_module.create_forward_context
            is original_create_forward_context
        )

    assert len(contexts) == 2
    assert transaction_ids == ["afd-0", "afd-1"]
    assert events == [
        "control_update",
        "control_send",
        "data",
        "control_update",
        "control_send",
        "data",
    ]
    assert runner.prof.steps == 2


def test_v1_execute_does_not_install_v2_provider(monkeypatch):
    context = SimpleNamespace(additional_kwargs={})

    def original_create_forward_context():
        return context

    forward_context_module = afd_forward_context.forward_context_module
    monkeypatch.setattr(
        forward_context_module,
        "create_forward_context",
        original_create_forward_context,
    )

    def native_execute(self, *args, **kwargs):
        return forward_context_module.create_forward_context()

    monkeypatch.setattr(native_v1.GPUModelRunner, "execute_model", native_execute)
    runner = object.__new__(AFDAttentionModelRunner)
    runner.prof = _StepProfiler()

    result = AFDAttentionModelRunner.execute_model(runner, None)

    assert result is context
    assert "afd_metadata" not in context.additional_kwargs
    assert (
        forward_context_module.create_forward_context is original_create_forward_context
    )


@pytest.mark.parametrize("failure", ["profiler", "native", "connector"])
def test_v2_shutdown_runs_all_cleanup_layers_when_each_layer_fails(
    monkeypatch,
    failure,
):
    events: list[str] = []
    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.prof = _StepProfiler()
    runner._afd_pending_metadata = object()
    runner.connector = _RecordingConnector(events)

    def stop_profiler(profiler):
        events.append("profiler")
        if failure == "profiler":
            raise RuntimeError("profiler failed")

    def native_shutdown(self):
        events.append("native")
        if failure == "native":
            raise RuntimeError("native failed")

    def close_connector():
        events.append("connector")
        if failure == "connector":
            raise RuntimeError("connector failed")

    runner.connector.close = close_connector
    monkeypatch.setattr(
        "afd_plugin.v1.worker.attention_model_runner_v2.stop_afd_gpu_profiler",
        stop_profiler,
    )
    monkeypatch.setattr(native_v2.GPUModelRunner, "shutdown", native_shutdown)

    with pytest.raises(RuntimeError, match=failure):
        AFDAttentionModelRunnerV2.shutdown(runner)

    assert events == ["profiler", "native", "connector"]
    assert runner._afd_pending_metadata is None


def _patch_v2_constructor_dependencies(monkeypatch, connector):
    config = _v2_config()

    monkeypatch.setattr(
        (
            "afd_plugin.v1.worker.attention_model_runner_v2."
            "validate_gpu_model_runner_v2_config"
        ),
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.attention_model_runner_v2._resolve_world_ranks",
        lambda: (0, 0),
    )
    monkeypatch.setattr(
        (
            "afd_plugin.v1.worker.attention_model_runner_v2."
            "AFDConnectorFactory.create_connector"
        ),
        lambda *args, **kwargs: connector,
    )
    return config


def test_v2_load_model_initializes_connector_after_native_load(monkeypatch):
    events: list[str] = []

    class Connector:
        is_initialized = False

        def init_afd_connector(self):
            events.append("connector_init")
            self.is_initialized = True

    def native_load(self, load_dummy_weights=False):
        events.append(f"native_load:{load_dummy_weights}")

    runner = object.__new__(AFDAttentionModelRunnerV2)
    runner.connector = Connector()
    monkeypatch.setattr(native_v2.GPUModelRunner, "load_model", native_load)

    runner.load_model(load_dummy_weights=True)

    assert events == ["native_load:True", "connector_init"]
    assert runner.connector.is_initialized


def test_v2_constructor_rejects_missing_control_plane_and_cleans_native(
    monkeypatch,
):
    events: list[str] = []

    class Connector:
        control_plane = None

        def close(self):
            events.append("connector")

    def native_init(self, vllm_config, device):
        events.append("native_init")
        self.vllm_config = vllm_config

    def native_shutdown(self):
        events.append("native_shutdown")

    monkeypatch.setattr(native_v2.GPUModelRunner, "__init__", native_init)
    monkeypatch.setattr(native_v2.GPUModelRunner, "shutdown", native_shutdown)
    config = _patch_v2_constructor_dependencies(monkeypatch, Connector())

    with pytest.raises(RuntimeError, match="control-plane"):
        AFDAttentionModelRunnerV2(config, torch.device("cpu"))

    assert events == ["native_init", "connector", "native_shutdown"]


def test_v2_constructor_defers_connector_initialization(monkeypatch):
    events: list[str] = []

    class Connector:
        control_plane = object()

        def close(self):
            events.append("connector")

    def native_init(self, vllm_config, device):
        events.append("native_init")
        self.vllm_config = vllm_config

    monkeypatch.setattr(native_v2.GPUModelRunner, "__init__", native_init)
    monkeypatch.setattr(
        native_v2.GPUModelRunner,
        "shutdown",
        lambda self: events.append("native_shutdown"),
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.attention_model_runner_v2.create_afd_gpu_profiler",
        lambda role: _StepProfiler(),
    )
    connector = Connector()
    config = _patch_v2_constructor_dependencies(monkeypatch, connector)

    runner = AFDAttentionModelRunnerV2(config, torch.device("cpu"))

    assert runner.connector is connector
    assert events == ["native_init"]


@pytest.mark.parametrize(
    ("worker_module", "worker_cls", "runner_name", "native_module"),
    [
        (
            attention_worker,
            attention_worker.AFDAttentionWorker,
            "AFDAttentionModelRunnerV2",
            native_v2,
        ),
        (ffn_worker, AFDFFNWorker, "GPUFFNModelRunner", native_v2),
    ],
)
def test_v2_worker_directly_constructs_one_plugin_runner(
    monkeypatch,
    worker_module,
    worker_cls,
    runner_name,
    native_module,
):
    config = _v2_config(role="ffn" if worker_cls is AFDFFNWorker else "attention")
    worker = _prepare_worker(worker_cls, config, use_v2=True)
    _RunnerRecorder.instances.clear()
    _NativeRunnerSentinel.instances = 0

    monkeypatch.setattr(worker_module, runner_name, _RunnerRecorder)
    monkeypatch.setattr(native_module, "GPUModelRunner", _NativeRunnerSentinel)
    monkeypatch.setattr(worker_module.Worker, "init_device", _fake_worker_init)
    monkeypatch.setattr(
        worker_module,
        "assert_compatible_afd_stack",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(worker_module.torch.accelerator, "empty_cache", lambda: None)

    worker.init_device()

    assert len(_RunnerRecorder.instances) == 1
    assert _NativeRunnerSentinel.instances == 0
    assert native_module.GPUModelRunner is _NativeRunnerSentinel
    assert worker.model_runner.__class__ is _RunnerRecorder
    assert worker.model_config is worker.vllm_config.model_config
    assert worker.model_config.hf_config.architectures == [
        "AFDDeepseekV2ForCausalLM",
    ]
    _construct_like_native_worker(worker)
    assert _NativeRunnerSentinel.instances == 1


@pytest.mark.parametrize(
    ("worker_module", "worker_cls", "runner_name"),
    [
        (
            attention_worker,
            attention_worker.AFDAttentionWorker,
            "AFDAttentionModelRunner",
        ),
        (ffn_worker, AFDFFNWorker, "GPUFFNModelRunner"),
    ],
)
def test_v1_worker_preserves_role_specific_construction_path(
    monkeypatch,
    worker_module,
    worker_cls,
    runner_name,
):
    config = _v2_config(
        role="ffn" if worker_cls is AFDFFNWorker else "attention",
        use_v2=False,
    )
    worker = _prepare_worker(worker_cls, config, use_v2=False)
    _RunnerRecorder.instances.clear()
    _NativeRunnerSentinel.instances = 0

    native_module = native_v1
    monkeypatch.setattr(worker_module, runner_name, _RunnerRecorder)
    monkeypatch.setattr(native_module, "GPUModelRunner", _NativeRunnerSentinel)
    monkeypatch.setattr(worker_module.Worker, "init_device", _fake_worker_init)
    monkeypatch.setattr(
        worker_module,
        "assert_compatible_afd_stack",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(worker_module.torch.accelerator, "empty_cache", lambda: None)

    worker.init_device()

    assert len(_RunnerRecorder.instances) == 1
    expected_native_instances = 1 if worker_cls is AFDFFNWorker else 0
    assert _NativeRunnerSentinel.instances == expected_native_instances
    assert native_module.GPUModelRunner is _NativeRunnerSentinel
    assert worker.model_runner.__class__ is _RunnerRecorder
    _construct_like_native_worker(worker)
    assert _NativeRunnerSentinel.instances == expected_native_instances + 1


@pytest.mark.parametrize(
    ("worker_module", "worker_cls", "runner_name"),
    [
        (
            attention_worker,
            attention_worker.AFDAttentionWorker,
            "AFDAttentionModelRunnerV2",
        ),
        (ffn_worker, AFDFFNWorker, "GPUFFNModelRunner"),
    ],
)
def test_worker_runner_substitution_restores_on_constructor_exception(
    monkeypatch,
    worker_module,
    worker_cls,
    runner_name,
):
    config = _v2_config(role="ffn" if worker_cls is AFDFFNWorker else "attention")
    worker = _prepare_worker(worker_cls, config, use_v2=True)
    _RunnerRecorder.instances.clear()
    _NativeRunnerSentinel.instances = 0

    native_module = native_v2

    def raising_worker_init(current_worker):
        current_worker.device = torch.device("cpu")
        from vllm.v1.worker.gpu import model_runner as imported_module

        imported_module.GPUModelRunner(
            current_worker.vllm_config,
            current_worker.device,
        )
        raise RuntimeError("runner construction failed")

    monkeypatch.setattr(worker_module, runner_name, _RunnerRecorder)
    monkeypatch.setattr(native_module, "GPUModelRunner", _NativeRunnerSentinel)
    monkeypatch.setattr(worker_module.Worker, "init_device", raising_worker_init)
    monkeypatch.setattr(
        worker_module,
        "assert_compatible_afd_stack",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(worker_module.torch.accelerator, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="runner construction failed"):
        worker.init_device()

    assert len(_RunnerRecorder.instances) == 1
    assert _NativeRunnerSentinel.instances == 0
    assert native_module.GPUModelRunner is _NativeRunnerSentinel
    _construct_like_native_worker(worker)
    assert _NativeRunnerSentinel.instances == 1


@pytest.mark.parametrize("use_v2", [False, True])
def test_non_afd_model_config_and_native_runner_identity_are_unchanged(
    monkeypatch,
    use_v2,
):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["OtherForCausalLM"]),
    )

    assert get_afd_model_config(model_config, device_type="cuda") is model_config

    native_module = native_v2 if use_v2 else native_v1
    _NativeRunnerSentinel.instances = 0
    monkeypatch.setattr(native_module, "GPUModelRunner", _NativeRunnerSentinel)
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2,
        vllm_config=SimpleNamespace(model_config=model_config),
        device=torch.device("cpu"),
    )
    _construct_like_native_worker(worker)

    assert _NativeRunnerSentinel.instances == 1
    assert native_module.GPUModelRunner is _NativeRunnerSentinel
