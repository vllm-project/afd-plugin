from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.compat.npu import (
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)


@contextmanager
def _temporarily_reimport_module(module_name: str) -> Iterator[ModuleType]:
    """Reimport a module without leaking it through its parent package."""
    package_name, _, module_attribute = module_name.rpartition(".")
    package = importlib.import_module(package_name)
    missing_attribute = object()
    original_package_attribute = package.__dict__.get(
        module_attribute,
        missing_attribute,
    )
    original_module = sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if original_package_attribute is missing_attribute:
            package.__dict__.pop(module_attribute, None)
        else:
            package.__dict__[module_attribute] = original_package_attribute


def _ffn_payload(hidden_states, metadata, states=None):
    return AFDA2FTransferPayload(
        hidden_states=hidden_states,
        context=AFDTransferContext(
            metadata=metadata,
            states=states if states is not None else AFDTransferState(),
        ),
    )


@contextmanager
def _fake_ffn_ascend_forward_context(**_kwargs):
    """Stand in for the vLLM-Ascend forward context in FFN runner unit tests.

    The real ``ascend_forward_context`` delegates to vLLM-Ascend internals that
    read many ``vllm_config`` fields; these unit tests only exercise the FFN
    runner's recv/compute/send orchestration, so a minimal fake context is used.
    """
    yield SimpleNamespace(additional_kwargs={}, dp_metadata=None, all_moe_layers={})


def _patch_ffn_forward_context(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )


class _RecordingConnector:
    world_rank = 0

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_updates.append(
            (
                payload.dp_metadata_list,
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )

    def send_dp_metadata_list(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.sent_dp_metadata_lists.append(
            (
                payload.dp_metadata_list,
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )


class _AsyncRecordingConnector(_RecordingConnector):
    def __init__(self):
        super().__init__()
        self.control_plane = None


class _FakeFFNConnector:
    def __init__(self, *, attn_size=1, ffn_size=1, role_rank=0, world_rank=0):
        self.dp_metadata_list = {}
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.updates = []
        self.attn_size = attn_size
        self.ffn_size = ffn_size
        self.world_rank = world_rank
        self.topology = SimpleNamespace(role_rank=role_rank)
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_list = dict(payload.dp_metadata_list)
        self.updates.append(
            (
                dict(payload.dp_metadata_list),
                {
                    "is_graph_capturing": payload.is_graph_capturing,
                    "is_warmup": payload.is_warmup,
                },
            ),
        )

    def recv_attn_output(self, ubatch_idx=None, **kwargs):
        for item in tuple(self.attn_outputs):
            payload = (
                item
                if isinstance(item, AFDA2FTransferPayload)
                else _ffn_payload(item[0], item[1])
            )
            if payload.context.metadata.stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return payload
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, context, **kwargs):
        self.ffn_outputs.append((ffn_output, context.metadata, kwargs))

    def close(self):
        return None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _RecordingFakeModel:
    def __init__(self):
        self.calls = []

    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        self.calls.append((hidden_states, layer_idx, kwargs))
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeStructuredFFNModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **_kwargs):
        return AFDF2ATransferPayload(
            routed_output=f"routed({hidden_states}, layer={layer_idx})",
            shared_output=f"shared({hidden_states}, layer={layer_idx})",
        )


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "enable_dbo": False,
        "use_ubatching": False,
        "num_ubatches": 1,
        "ubatch_size": 0,
        "tensor_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(
    *,
    role="attention",
    connector="CAMP2pAFDConnector",
    extra_config=None,
    use_mla=False,
    cudagraph_mode="FULL",
    speculative_config=None,
    **parallel_overrides,
):
    async_dp = bool(parallel_overrides.pop("async_dp", False))
    compute_gate_on_attention = bool(
        parallel_overrides.pop("compute_gate_on_attention", False),
    )
    return SimpleNamespace(
        additional_config={
            "afd": {
                "role": role,
                "connector": connector,
                "async": async_dp,
                "compute_gate_on_attention": compute_gate_on_attention,
                "connector_extra_config": extra_config or {},
            },
        },
        parallel_config=_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_text_config=SimpleNamespace(),
            use_mla=use_mla,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                name=cudagraph_mode,
                has_full_cudagraphs=lambda: (
                    cudagraph_mode in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
                ),
            ),
            fast_moe_cold_start=False,
        ),
        speculative_config=speculative_config,
    )


def _async_moe_config(
    *,
    role="attention",
    compute_gate_on_attention=True,
    tensor_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    **extra_config,
):
    return _vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=compute_gate_on_attention,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
        extra_config={
            "async_moe_ubatching": True,
            **extra_config,
        },
    )


def _require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")


def _new_attention_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

    return object.__new__(AFDNPUAttentionModelRunner)


def _new_ffn_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

    # object.__new__ bypasses __init__, which is where the runner would set up
    # the profiler and device; provide inert defaults the runtime paths expect.
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.prof = None
    runner.device = SimpleNamespace(type="npu")
    return runner


def _new_ffn_worker():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

    return object.__new__(AFDNPUFFNWorker)


@pytest.mark.parametrize(
    ("wrapper_owns_update", "expected_updates"),
    [(True, 0), (False, 1)],
)
def test_npu_attention_runner_skips_outer_update_only_for_owned_graph(
    monkeypatch,
    wrapper_owns_update,
    expected_updates,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    class FakeUBatchWrapper:
        def owns_full_graph_update(self, _forward_context):
            return wrapper_owns_update

        def __call__(self, **_model_inputs):
            return "hidden_states"

    forward_context = SimpleNamespace(
        dbo_enabled=False,
        flash_comm_v1_enabled=False,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "AscendUBatchWrapper",
        FakeUBatchWrapper,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "get_forward_context",
        lambda: forward_context,
    )

    runner = object.__new__(
        attention_model_runner.AFDNPUAttentionModelRunner,
    )
    runner.enable_enpu = False
    runner.model = FakeUBatchWrapper()
    runner.ubatch_slices = None
    runner._install_afd_metadata_on_forward_context = lambda _context: None
    runner._install_async_moe_ubatch_metadata_on_forward_context = lambda _context: None
    updates = []
    runner._update_full_graph_params_if_needed = lambda *args: updates.append(args)

    result = runner._model_forward(
        8,
        input_ids=None,
        positions=object(),
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "hidden_states"
    assert len(updates) == expected_updates


def test_npu_attention_runner_installs_mla_graph_wrapper(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    captured = {}

    class RecordingUBatchWrapper:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        attention_model_runner,
        "AscendUBatchWrapper",
        RecordingUBatchWrapper,
    )
    runner = object.__new__(
        attention_model_runner.AFDNPUAttentionModelRunner,
    )
    runner.model = "model"
    runner.device = "npu"
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
    )
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True),
    )
    runner.use_sparse = False
    runner.enable_enpu = False

    runner._install_ascend_ubatch_wrapper()

    assert captured["args"][:2] == ("model", runner.vllm_config)
    assert captured["kwargs"]["mla_full_graph_enabled"] is True
    assert captured["kwargs"]["enable_enpu"] is False
    updater = captured["kwargs"]["full_graph_params_updater"]
    assert updater.__self__ is runner


def test_npu_attention_runner_builds_and_sets_metadata():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=5),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [1]
    assert len(runner.connector.dp_metadata_updates) == 1
    assert len(runner.connector.sent_dp_metadata_lists) == 1


def test_npu_attention_async_connector_skips_dp_metadata_control_plane():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        data_parallel_size=2,
    )
    runner.connector = _AsyncRecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=3),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [3]
    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.sent_dp_metadata_lists == []


def test_npu_attention_runner_builds_dp_fallback():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = object()
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 7)

    dp_metadata = runner._ensure_dp_metadata(None)

    tokens = dp_metadata.num_tokens_across_dp_cpu
    if not isinstance(tokens, list):
        tokens = tokens.tolist()
    assert tokens == [7]


def test_npu_attention_runner_sends_graph_flags():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 3)

    runner._send_dp_metadata(SimpleNamespace(num_tokens_across_dp_cpu=[3]), None)

    assert runner.connector.dp_metadata_updates[0][1:] == (True, True)
    assert runner.connector.sent_dp_metadata_lists[0][1:] == (True, True)


def test_npu_attention_runner_sends_per_ubatch_dp_metadata():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False

    ubatch_slices = [
        SimpleNamespace(
            request_slice=slice(0, 2),
            token_slice=slice(0, 4),
            num_tokens=4,
        ),
        SimpleNamespace(
            request_slice=slice(2, 3),
            token_slice=slice(4, 7),
            num_tokens=3,
        ),
    ]

    runner._send_dp_metadata(None, ubatch_slices)

    dp_metadata_list = runner.connector.dp_metadata_updates[0][0]
    assert sorted(dp_metadata_list) == [0, 1]
    assert _tokens(dp_metadata_list[0]) == [4]
    assert _tokens(dp_metadata_list[1]) == [3]
    sent_dp_metadata_list = runner.connector.sent_dp_metadata_lists[0][0]
    assert sorted(sent_dp_metadata_list) == [0, 1]
    assert _tokens(sent_dp_metadata_list[0]) == [4]
    assert _tokens(sent_dp_metadata_list[1]) == [3]


def test_npu_attention_capture_microbatch_also_captures_single_stage():
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    runner = _new_attention_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner.connector = SimpleNamespace(control_plane=object())
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_suppress_metadata_send = False
    runner._afd_pending_metadata = "original"
    dummy_calls = []
    sent_metadata = []

    def dummy_run(num_tokens, **kwargs):
        dummy_calls.append(
            (
                num_tokens,
                kwargs.copy(),
                runner._is_warmup,
                runner._afd_is_graph_capturing,
                runner._afd_suppress_metadata_send,
            ),
        )
        return kwargs["allow_microbatching"]

    runner._dummy_run = dummy_run
    runner._build_afd_metadata = lambda ubatch_slices, num_tokens: SimpleNamespace(
        ubatch_slices=ubatch_slices,
        num_tokens=num_tokens,
    )
    runner._build_capture_dp_metadata = lambda num_tokens: SimpleNamespace(
        num_tokens_across_dp_cpu=[num_tokens],
    )

    def send_dp_metadata(dp_metadata, ubatch_slices):
        sent_metadata.append(
            (
                dp_metadata,
                ubatch_slices,
                runner._afd_is_graph_capturing,
                runner._is_warmup,
            ),
        )

    runner._send_dp_metadata = send_dp_metadata
    desc = SimpleNamespace(num_tokens=12, uniform=True, num_active_loras=0)

    result = runner._warmup_and_capture(
        desc,
        CUDAGraphMode.FULL,
        allow_microbatching=True,
    )

    assert result is True
    assert [call[1]["allow_microbatching"] for call in dummy_calls] == [
        False,
        False,
        True,
        True,
    ]
    assert [call[1]["cudagraph_runtime_mode"] for call in dummy_calls] == [
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
    ]
    assert [call[1].get("is_graph_capturing", False) for call in dummy_calls] == [
        False,
        True,
        False,
        True,
    ]
    assert [call[2] for call in dummy_calls] == [True, False, True, False]
    assert [call[3] for call in dummy_calls] == [False, True, False, True]
    assert len(sent_metadata) == 1
    dp_metadata, ubatch_slices, is_graph_capturing, is_warmup = sent_metadata[0]
    assert _tokens(dp_metadata) == [12]
    assert ubatch_slices is None
    assert is_graph_capturing is True
    assert is_warmup is False
    assert runner._is_warmup is False
    assert runner._afd_is_graph_capturing is False
    assert runner._afd_suppress_metadata_send is False
    assert runner._afd_pending_metadata == "original"


def test_npu_attention_metadata_positional_args_and_padded_slices():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ubatch_utils import (
        UBatchSlice,
        pad_out_ubatch_slices,
    )

    ubatch_slices = [
        UBatchSlice(slice(0, 1), slice(0, 4)),
        UBatchSlice(slice(1, 2), slice(4, 8)),
    ]

    normalized = pad_out_ubatch_slices(ubatch_slices, 8, 4)

    assert normalized[-1].request_slice == slice(1, 4)
    assert normalized[-1].token_slice == slice(4, 8)


def test_npu_request_boundary_ubatch_slices_balance_tokens(monkeypatch):
    np = pytest.importorskip("numpy")
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_vllm = ModuleType("vllm")
    fake_vllm_config = ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = object
    fake_vllm_v1 = ModuleType("vllm.v1")
    fake_vllm_worker = ModuleType("vllm.v1.worker")
    fake_vllm_ubatch_utils = ModuleType("vllm.v1.worker.ubatch_utils")

    class UBatchSlice:
        def __init__(self, request_slice, token_slice):
            self.request_slice = request_slice
            self.token_slice = token_slice

        @property
        def num_tokens(self):
            return self.token_slice.stop - self.token_slice.start

        def is_empty(self):
            return self.num_tokens <= 0

    fake_vllm_ubatch_utils.UBatchSlice = UBatchSlice
    fake_vllm_ubatch_utils.UBatchSlices = list
    fake_vllm_ubatch_utils.check_ubatch_thresholds = lambda *_args, **_kwargs: False
    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_forward_context = ModuleType("vllm_ascend.ascend_forward_context")
    fake_forward_context.MoECommType = type("MoECommType", (), {})
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention_utils = ModuleType("vllm_ascend.attention.utils")
    fake_attention_utils.AscendCommonAttentionMetadata = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", fake_vllm_worker)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.ubatch_utils",
        fake_vllm_ubatch_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_forward_context,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", fake_attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.utils",
        fake_attention_utils,
    )

    with _temporarily_reimport_module(
        "afd_plugin.v1.worker.npu.ubatch_utils",
    ) as ubatch_utils:
        slices = ubatch_utils.create_request_boundary_ubatch_slices(
            np.array([2, 3, 5, 7], dtype=np.int32),
        )

        assert slices[0].request_slice == slice(0, 3)
        assert slices[0].token_slice == slice(0, 10)
        assert slices[1].request_slice == slice(3, 4)
        assert slices[1].token_slice == slice(10, 17)

        slices = ubatch_utils.create_request_boundary_ubatch_slices(
            np.array([824, 846, 16], dtype=np.int32),
        )

        assert slices[0].request_slice == slice(0, 1)
        assert slices[0].token_slice == slice(0, 824)
        assert slices[1].request_slice == slice(1, 3)
        assert slices[1].token_slice == slice(824, 1686)
        assert (
            ubatch_utils.create_request_boundary_ubatch_slices(
                np.array([17], dtype=np.int32),
            )
            is None
        )


def test_npu_async_moe_metadata_tracks_stage_padding_separately(monkeypatch):
    _require_npu_runtime()
    torch = pytest.importorskip("torch")
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    from afd_plugin.async_moe import AsyncMoeStage
    from afd_plugin.v1.worker.npu import ubatch_utils

    monkeypatch.setattr(
        ubatch_utils,
        "AscendCommonAttentionMetadata",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    parent = SimpleNamespace(
        query_start_loc=torch.tensor([0, 40], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 40], dtype=torch.int32),
        seq_lens=torch.tensor([40], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([40], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=40,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(40),
        causal=True,
        num_input_tokens=112,
        actual_seq_lengths_q=list(range(1, 41)),
        positions=torch.arange(40),
        attn_state=object(),
        graph_pad_size=0,
        decode_token_per_req=1,
        kvcomp_metadata=None,
        encoder_seq_lens=None,
        encoder_seq_lens_cpu=None,
        logits_indices_padded=None,
        num_logits_indices=0,
    )

    metadata = ubatch_utils.split_async_moe_attn_metadata(
        (
            AsyncMoeStage(
                request_slice=slice(0, 1),
                token_slice=slice(37, 40),
                input_tokens=4,
            ),
        ),
        parent,
    )[0]

    assert metadata.num_input_tokens == 4
    assert metadata.num_actual_tokens == 3
    assert metadata.query_start_loc_cpu.tolist() == [0, 3]
    assert metadata.seq_lens.tolist() == [40]
    assert metadata.slot_mapping.tolist() == [37, 38, 39]
    assert metadata.positions.tolist() == [37, 38, 39]
    assert metadata.actual_seq_lengths_q == [38, 39, 40]

    mixed_parent = SimpleNamespace(
        **{
            **vars(parent),
            "query_start_loc": torch.tensor([0, 1, 1070], dtype=torch.int32),
            "query_start_loc_cpu": torch.tensor(
                [0, 1, 1070],
                dtype=torch.int32,
            ),
            "seq_lens": torch.tensor([10, 1069], dtype=torch.int32),
            "seq_lens_cpu": torch.tensor([10, 1069], dtype=torch.int32),
            "num_computed_tokens_cpu": torch.tensor([9, 0], dtype=torch.int32),
            "num_reqs": 2,
            "num_actual_tokens": 1070,
            "max_query_len": 1069,
            "max_seq_len": 1069,
            "block_table_tensor": torch.zeros((2, 1), dtype=torch.int32),
            "slot_mapping": torch.arange(1070),
            "num_input_tokens": 1070,
            "actual_seq_lengths_q": list(range(1, 1071)),
            "positions": torch.arange(1070),
        },
    )
    mixed_stages = ubatch_utils.split_async_moe_attn_metadata(
        (
            AsyncMoeStage(slice(0, 2), slice(0, 534), input_tokens=534),
            AsyncMoeStage(slice(1, 2), slice(534, 1070), input_tokens=536),
        ),
        mixed_parent,
    )

    assert mixed_stages[0].query_start_loc_cpu.tolist() == [0, 1, 534]
    assert mixed_stages[0].seq_lens.tolist() == [10, 533]
    assert mixed_stages[0].num_actual_tokens == 534
    assert mixed_stages[1].query_start_loc_cpu.tolist() == [0, 536]
    assert mixed_stages[1].seq_lens.tolist() == [1069]
    assert mixed_stages[1].num_actual_tokens == 536

    native_metadata = ubatch_utils.split_attn_metadata(
        [UBatchSlice(slice(0, 1), slice(0, 40))],
        parent,
    )[0]
    assert native_metadata.num_input_tokens == 40
    assert native_metadata.num_actual_tokens == 40


@pytest.mark.parametrize(
    (
        "split_mode",
        "use_sequence_parallel",
        "tp_size",
        "scheduled_tokens",
        "num_tokens",
        "num_tokens_padded",
        "expected_actual_tokens",
        "expected_input_tokens",
        "expected_request_slices",
        "expected_token_slices",
    ),
    [
        pytest.param(
            "token",
            True,
            2,
            [1099],
            1099,
            1100,
            (550, 549),
            (550, 550),
            [slice(0, 1), slice(0, 1)],
            [slice(0, 550), slice(550, 1099)],
            id="token-with-sp",
        ),
        pytest.param(
            "request",
            False,
            1,
            [4, 4, 4, 4],
            16,
            20,
            (8, 8),
            (8, 8),
            [slice(0, 2), slice(2, 4)],
            [slice(0, 8), slice(8, 16)],
            id="request-without-sp",
        ),
        pytest.param(
            "request",
            True,
            2,
            [5, 6, 7],
            18,
            18,
            (11, 7),
            (12, 8),
            [slice(0, 2), slice(2, 3)],
            [slice(0, 11), slice(11, 18)],
            id="request-with-sp",
        ),
    ],
)
def test_npu_attention_runner_builds_stage_metadata(
    monkeypatch,
    split_mode,
    use_sequence_parallel,
    tp_size,
    scheduled_tokens,
    num_tokens,
    num_tokens_padded,
    expected_actual_tokens,
    expected_input_tokens,
    expected_request_slices,
    expected_token_slices,
):
    _require_npu_runtime()
    import numpy as np
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo
    from afd_plugin.model_executor.models import AsyncMoeUbatchMetadata
    from afd_plugin.v1.worker.npu import attention_model_runner
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        tensor_parallel_size=tp_size,
        extra_config={
            "async_moe_ubatching": True,
            "async_moe_split": split_mode,
        },
    )
    runner.connector = _AsyncRecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    runner.afd_async_extra_info = AFDAsyncExtraInfo(
        async_moe_ubatching=True,
        async_moe_split=split_mode,
    )

    full_metadata = SimpleNamespace(name="full")
    stage_attn_metadata = [{"layer": "stage-0"}, {"layer": "stage-1"}]
    monkeypatch.setattr(
        NPUModelRunner,
        "_build_attention_metadata",
        lambda self, *args, **kwargs: full_metadata,
    )
    stage_build_calls = []

    def build_stage_metadata(self, *args, **kwargs):
        stage_build_calls.append((args, kwargs))
        return stage_attn_metadata, None

    monkeypatch.setattr(
        AFDNPUAttentionModelRunner,
        "_build_attention_metadata_with_ubatches",
        build_stage_metadata,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "enable_sp",
        lambda _config: use_sequence_parallel,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )

    result = runner._build_attention_metadata_with_async_moe_ubatches(
        (),
        {},
        {
            "num_tokens": num_tokens,
            "num_tokens_padded": num_tokens_padded,
            "num_reqs_padded": len(scheduled_tokens),
            "num_scheduled_tokens_np": np.array(
                scheduled_tokens,
                dtype=np.int32,
            ),
        },
    )

    assert result is full_metadata
    metadata = runner._afd_async_moe_ubatch_metadata
    assert isinstance(metadata, AsyncMoeUbatchMetadata)
    assert metadata.attn_metadata is stage_attn_metadata
    assert metadata.use_sequence_parallel is use_sequence_parallel
    assert metadata.parent_input_tokens == num_tokens_padded
    assert tuple(stage.actual_tokens for stage in metadata.stages) == (
        expected_actual_tokens
    )
    assert (
        tuple(stage.input_tokens for stage in metadata.stages) == expected_input_tokens
    )
    assert [stage.request_slice for stage in metadata.stages] == expected_request_slices
    assert [stage.token_slice for stage in metadata.stages] == expected_token_slices
    assert len(stage_build_calls) == 1
    assert stage_build_calls[0][1]["metadata_builder_offset"] == 1
    assert stage_build_calls[0][1]["async_moe_stages"] == metadata.stages


def test_npu_attention_runner_async_moe_allocates_three_metadata_builders(
    monkeypatch,
):
    _require_npu_runtime()
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        tensor_parallel_size=2,
        extra_config={
            "async_moe_ubatching": True,
            "async_moe_split": "token",
        },
    )
    runner.afd_async_extra_info = AFDAsyncExtraInfo(
        async_moe_ubatching=True,
        async_moe_split="token",
    )
    runner.device = object()
    create_calls = []
    attn_group = SimpleNamespace(metadata_builders=[object()])

    def create_metadata_builders(
        vllm_config,
        device,
        *,
        num_metadata_builders,
    ):
        assert vllm_config is runner.vllm_config
        assert device is runner.device
        create_calls.append(num_metadata_builders)
        attn_group.metadata_builders = [object()] * num_metadata_builders

    attn_group.create_metadata_builders = create_metadata_builders
    runner.attn_groups = [[attn_group]]
    initialized = object()
    monkeypatch.setattr(
        NPUModelRunner,
        "initialize_attn_backend",
        lambda self, *args, **kwargs: initialized,
    )

    assert runner.initialize_attn_backend() is initialized
    assert create_calls == [3]
    assert len(attn_group.metadata_builders) == 3


def test_npu_create_ascend_forward_context_marks_current_ubatch(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import forward_context as forward_context_module

    monkeypatch.setattr(
        forward_context_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        forward_context_module,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        forward_context_module,
        "get_moe_comm_method",
        lambda moe_comm_type: f"method:{moe_comm_type}",
    )
    afd_metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0, 4],
        requests_start_loc=[0, 1],
        stage_idx=0,
        connector=object(),
        tokens_lens=[4, 3],
        num_stages=2,
        tokens_unpadded_lens=[4, 3],
    )
    cur_forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": afd_metadata},
        all_moe_layers={},
        moe_comm_type="mc2",
        in_profile_run=False,
        capturing=False,
        mmrs_fusion=False,
        flash_comm_v1_enabled=False,
        flashcomm_v2_enabled=False,
        is_first_layer=True,
        layer_idx=0,
        prefetch_mlp_gate_up_proj=False,
        prefetch_mlp_down_proj=False,
        model_instance=None,
        is_draft_model=False,
        is_draft_model_prefill=False,
        draft_attn_metadatas=None,
        max_tokens_across_pcp=None,
        mc2_mask=None,
    )
    ubatch_slices = [
        SimpleNamespace(
            request_slice=slice(0, 1),
            token_slice=slice(0, 4),
            num_tokens=4,
        ),
        SimpleNamespace(
            request_slice=slice(1, 2),
            token_slice=slice(4, 7),
            num_tokens=3,
        ),
    ]
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    new_forward_context = forward_context_module.create_ascend_forward_context(
        cur_forward_context,
        attn_metadata=None,
        vllm_config=vllm_config,
        ubatch_slices=ubatch_slices,
        ubatch_num=1,
    )

    child_metadata = new_forward_context.additional_kwargs["afd_metadata"]
    assert new_forward_context.ubatch_idx == 1
    assert new_forward_context.num_ubatches == 2
    assert new_forward_context.num_tokens == 3
    assert child_metadata.stage_idx == 1


def test_npu_ffn_runner_executes_eager_ffn_step(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert len(runner.connector.updates) == 1
    update_metadata, update_flags = runner.connector.updates[0]
    assert update_metadata == {0: runner.connector.dp_metadata_list[0]}
    assert update_flags == {"is_graph_capturing": False, "is_warmup": False}
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_dp_path_invokes_model_with_hidden_states_and_layer(monkeypatch):
    from afd_plugin.connectors.npu.async_cam import AFDAsyncTransferState

    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _RecordingFakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    # The DP-metadata FFN path forwards only hidden states and the layer index
    # to the model; backend transfer state stays on the context and is consumed
    # by the connector, not spread into compute_ffn_output kwargs.
    runner.connector.attn_outputs.append(
        _ffn_payload(
            "hidden",
            metadata,
            states=AFDAsyncTransferState(
                group_list="groups",
                dynamic_scales="scales",
                expand_x_shared="shared-hidden",
                dynamic_scales_shared="shared-scales",
            ),
        ),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.model.calls == [("hidden", 0, {})]


def test_npu_ffn_connector_driven_uses_cam_layer_and_token_metadata(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.connectors.npu.async_cam import (
        AFDAsyncFFNWorkItem,
        AFDAsyncTransferState,
    )
    from afd_plugin.v1.worker.npu import ffn_model_runner

    context_calls = []
    sent_outputs = []

    @contextmanager
    def fake_ascend_forward_context(**kwargs):
        context_calls.append(kwargs)
        yield SimpleNamespace(
            additional_kwargs={},
            dp_metadata="dp",
            all_moe_layers={},
        )

    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        fake_ascend_forward_context,
    )
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = SimpleNamespace(control_plane=None)
    runner.model = _RecordingFakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 16
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=7,
        stage_idx=0,
        seq_lens=[5],
    )
    states = AFDAsyncTransferState(
        batch_size=5,
        hidden_size=16,
        topk=2,
        layer_idx=7,
        group_list="groups",
        dynamic_scales="scales[:5]",
        expand_x_shared="shared-hidden[:2]",
        dynamic_scales_shared="shared-scales[:2]",
    )
    context = AFDTransferContext(metadata=metadata, states=states)
    recv_output = AFDA2FTransferPayload(
        hidden_states="recv-hidden",
        context=context,
    )
    work_item = AFDAsyncFFNWorkItem(
        hidden_states="hidden[:5]",
        context=context,
        recv_output=recv_output,
        layer_idx=7,
        stage_idx=0,
        num_tokens=5,
        total_num_tokens=7,
        shared_num_tokens=2,
    )

    def recv_ffn_work_item(*, stage_idx, max_num_tokens):
        assert stage_idx == 0
        assert max_num_tokens == 16
        return work_item

    def send_ffn_work_item_output(sent_work_item, ffn_output):
        sent_outputs.append((sent_work_item, ffn_output))
        return ffn_output

    runner.connector.recv_ffn_work_item = recv_ffn_work_item
    runner.connector.send_ffn_work_item_output = send_ffn_work_item_output

    runner._ffn_forward_connector_driven()

    assert runner.model.calls == [
        (
            "hidden[:5]",
            7,
            {
                "group_list": "groups",
                "dynamic_scales": "scales[:5]",
                "expand_x_shared": "shared-hidden[:2]",
                "dynamic_scales_shared": "shared-scales[:2]",
            },
        ),
    ]
    assert sent_outputs == [(work_item, "npu-ffn(hidden[:5], layer=7)")]
    assert context_calls[0]["num_tokens"] == 5
    assert context_calls[0]["afd_metadata"].tokens_lens == [5]


def test_npu_ffn_runner_sends_structured_shared_output(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeStructuredFFNModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        (
            "routed(hidden, layer=0)",
            metadata,
            {
                "ubatch_idx": 0,
                "expand_x_shared": "shared(hidden, layer=0)",
            },
        ),
    ]


def test_npu_ffn_runner_filters_dense_layers_when_gate_runs_on_attention():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_model_runner import _ffn_layer_indices

    runner = _new_ffn_runner()
    runner.num_layers = 5
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=True)
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            n_routed_experts=8,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        ),
    )

    assert _ffn_layer_indices(runner) == [2, 4]


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


def test_npu_ffn_runner_replays_acl_graph_when_key_exists():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([1])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_npu_ffn_runner_graph_key_uses_ffn_aggregated_token_counts():
    runner = _new_ffn_runner()
    runner.connector = _FakeFFNConnector(attn_size=8, ffn_size=4)
    runner.max_num_tokens = 24

    assert runner._make_graph_key({0: _FakeDPMetadata([12] * 8)}) == (
        (0, (24, 24, 24, 24)),
    )


def test_npu_ffn_runner_falls_back_to_eager_on_acl_graph_miss(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_warmup_uses_eager_forward_without_graph(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    capture_flags = []

    def fail_graph_capture_context(device):
        raise AssertionError("warmup must not enter graph_capture context")

    monkeypatch.setattr(ffn_model_runner, "graph_capture", fail_graph_capture_context)
    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "set_cudagraph_capturing_enabled",
        capture_flags.append,
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "mem_get_info", lambda: (0, 0))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_warmup=True,
    )

    assert runner._acl_graphs == {}
    assert capture_flags == [True, False]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_capture_stores_acl_graph_and_skips_duplicate_state_update(
    monkeypatch,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    runner.graph_pool = None
    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )
    monkeypatch.setattr(ffn_model_runner, "graph_capture", lambda device: nullcontext())
    monkeypatch.setattr(
        ffn_model_runner.torch.npu,
        "graph",
        lambda graph, pool: nullcontext(),
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "NPUGraph", _FakeGraph)
    monkeypatch.setattr(
        ffn_model_runner,
        "set_cudagraph_capturing_enabled",
        lambda enabled: None,
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "mem_get_info", lambda: (0, 0))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list=dp_metadata,
        is_graph_capturing=True,
    )

    assert runner._make_graph_key(dp_metadata) in runner._acl_graphs
    assert len(runner.connector.updates) == 1
    update_metadata, update_flags = runner.connector.updates[0]
    assert sorted(update_metadata) == [0]
    assert _tokens(update_metadata[0]) == [1]
    assert update_flags == {"is_graph_capturing": True, "is_warmup": False}


def test_npu_ffn_runner_requires_compute_hook(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = SimpleNamespace()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    with pytest.raises(AttributeError, match="compute_ffn_output"):
        runner.execute_ffn_step(dp_metadata_list={0: _FakeDPMetadata([1])})


def test_npu_ffn_worker_scheduler_execute_model_fails_fast():
    worker = _new_ffn_worker()

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_npu_ffn_worker_loop_error_is_propagated(caplog):
    worker = _new_ffn_worker()
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True),
    )

    expected_error = RuntimeError("boom")

    def fail_loop():
        raise expected_error

    worker._run_ffn_server_loop = fail_loop

    with caplog.at_level(
        logging.ERROR,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker.start_ffn_server_loop()
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

    with pytest.raises(RuntimeError, match="AFD NPU FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()

    assert exc.value.__cause__ is expected_error
    assert "AFD NPU FFN worker loop failed" in caplog.text


def test_npu_ffn_worker_uses_connector_driven_loop_for_async_connector():
    worker = _new_ffn_worker()
    event = threading.Event()
    calls = []

    def execute_connector_driven_step():
        calls.append("step")
        event.set()

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = SimpleNamespace(
        connector=_AsyncRecordingConnector(),
        execute_connector_driven_step=execute_connector_driven_step,
    )

    worker._run_ffn_server_loop()

    assert calls == ["step"]


def test_npu_feature_validation_rejects_unsupported_switches():
    for extra_config, message in [
        ({"compute_gate_on_attention": True}, "compute_gate_on_attention"),
        ({"quant_mode": 1}, "quant_mode=0"),
    ]:
        with pytest.raises((RuntimeError, ValueError), match=message):
            fail_if_unsupported_npu_afd_features(
                _vllm_config(extra_config=extra_config),
            )


def test_npu_feature_validation_uses_selected_connector_extra_info_parser():
    with pytest.raises(ValueError, match="does not support connector_extra_config"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="P2pNcclAFDConnector",
                extra_config={"core_num": 8},
            ),
        )


def test_npu_feature_validation_allows_two_ubatches_only():
    config = _vllm_config(
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    fail_if_unsupported_npu_afd_features(config)
    assert npu_afd_num_ubatches(config) == 2

    with pytest.raises(RuntimeError, match="exactly two ubatches"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                enable_dbo=True,
                use_ubatching=True,
                num_ubatches=4,
                ubatch_size=4,
            ),
        )

    config = _vllm_config()
    config.model_config.enforce_eager = False
    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("cudagraph_mode", ["FULL", "FULL_AND_PIECEWISE"])
def test_npu_feature_validation_requires_decode_only_full_graph_for_mla_dbo(
    cudagraph_mode,
):
    config = _vllm_config(
        use_mla=True,
        cudagraph_mode=cudagraph_mode,
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        fail_if_unsupported_npu_afd_features(config)

    fail_if_unsupported_npu_afd_features(
        _vllm_config(
            use_mla=True,
            cudagraph_mode="FULL_DECODE_ONLY",
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
            ubatch_size=4,
        ),
    )

    sparse_config = _vllm_config(
        use_mla=True,
        cudagraph_mode="FULL",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    sparse_config.model_config.hf_text_config = SimpleNamespace(index_topk=8)
    fail_if_unsupported_npu_afd_features(sparse_config)


def test_npu_feature_validation_rejects_speculative_mla_dbo_full_graph():
    config = _vllm_config(
        use_mla=True,
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=object(),
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )

    with pytest.raises(RuntimeError, match="does not support speculative decoding"):
        fail_if_unsupported_npu_afd_features(config)


def test_npu_async_feature_validation_requires_async_config_and_eager():
    with pytest.raises(RuntimeError, match="async=true"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(connector="CAMAsyncAFDConnector", async_dp=False),
        )

    config = _vllm_config(connector="CAMAsyncAFDConnector", async_dp=True)
    config.model_config.enforce_eager = False
    with pytest.raises(RuntimeError, match="only eager"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("parallel_override", "error"),
    [
        ({"use_ubatching": True}, "ubatching"),
        ({"enable_dbo": True}, "DBO"),
    ],
)
def test_npu_async_feature_validation_rejects_native_ubatching(
    parallel_override,
    error,
):
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                **parallel_override,
            ),
        )


def test_npu_async_feature_validation_allows_dynamic_quant_zero_or_one():
    fail_if_unsupported_npu_afd_features(
        _vllm_config(
            connector="CAMAsyncAFDConnector",
            async_dp=True,
            extra_config={"dynamicQuant": "1"},
        ),
    )

    with pytest.raises(RuntimeError, match="dynamicQuant"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                extra_config={"dynamicQuant": 2},
            ),
        )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_async_moe_config(), id="request-tp1"),
        pytest.param(
            _async_moe_config(
                tensor_parallel_size=2,
                async_moe_split="token",
            ),
            id="token-attention-tp2",
        ),
        pytest.param(
            _async_moe_config(
                tensor_parallel_size=2,
                async_moe_split="request",
            ),
            id="request-attention-tp2",
        ),
        pytest.param(
            _async_moe_config(role="ffn", async_moe_split="token"),
            id="token-ffn-tp1",
        ),
        pytest.param(
            _async_moe_config(prefill_context_parallel_size=2),
            id="request-pcp",
        ),
    ],
)
def test_npu_async_moe_ubatching_validation_accepts_supported_shape(config):
    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            _async_moe_config(compute_gate_on_attention=False),
            "compute_gate_on_attention",
            id="missing-attention-gate",
        ),
        pytest.param(
            _async_moe_config(async_moe_num_ubatches=3),
            "exactly two",
            id="three-stages",
        ),
        pytest.param(
            _async_moe_config(async_moe_split="token"),
            "TP/SP",
            id="token-attention-tp1",
        ),
        pytest.param(
            _async_moe_config(decode_context_parallel_size=2),
            "decode context parallel",
            id="decode-context-parallel",
        ),
    ],
)
def test_npu_async_moe_ubatching_validation_rejects_unsupported_shape(
    config,
    error,
):
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


def test_npu_ubatch_allows_mc2_comm_when_thresholds_are_met(monkeypatch):
    fake_numpy = ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_vllm = ModuleType("vllm")
    fake_vllm_config = ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = object
    fake_vllm_v1 = ModuleType("vllm.v1")
    fake_vllm_worker = ModuleType("vllm.v1.worker")
    fake_vllm_ubatch_utils = ModuleType("vllm.v1.worker.ubatch_utils")
    fake_vllm_ubatch_utils.UBatchSlice = object
    fake_vllm_ubatch_utils.UBatchSlices = list

    def check_ubatch_thresholds(config, num_tokens, uniform_decode):
        if not config.use_ubatching:
            return False
        if uniform_decode:
            return num_tokens >= config.dbo_decode_token_threshold
        return num_tokens >= config.dbo_prefill_token_threshold

    fake_vllm_ubatch_utils.check_ubatch_thresholds = check_ubatch_thresholds

    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_forward_context = ModuleType("vllm_ascend.ascend_forward_context")

    class MoECommType:
        MC2 = object()
        FUSED_MC2 = object()

    fake_forward_context.MoECommType = MoECommType
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention_utils = ModuleType("vllm_ascend.attention.utils")
    fake_attention_utils.AscendCommonAttentionMetadata = object

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", fake_vllm_worker)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.ubatch_utils",
        fake_vllm_ubatch_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_forward_context,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", fake_attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.utils",
        fake_attention_utils,
    )

    with _temporarily_reimport_module(
        "afd_plugin.v1.worker.npu.ubatch_utils",
    ) as ubatch_utils:
        config = _vllm_config(
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
            ubatch_size=4,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=12,
        )

        assert ubatch_utils.check_enable_ubatch(
            num_tokens_unpadded=12,
            num_tokens_padded=12,
            uniform_decode=True,
            vllm_config=config,
            moe_comm_type=ubatch_utils.MoECommType.MC2,
        )
        assert ubatch_utils.check_enable_ubatch(
            num_tokens_unpadded=12,
            num_tokens_padded=12,
            uniform_decode=True,
            vllm_config=config,
            moe_comm_type=ubatch_utils.MoECommType.FUSED_MC2,
        )


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
