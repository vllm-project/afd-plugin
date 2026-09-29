from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.compat.npu import (  # noqa: E402
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)


def _ffn_payload(hidden_states, metadata, states=None, input_ids=None):
    return AFDA2FTransferPayload(
        hidden_states=hidden_states,
        context=AFDTransferContext(
            metadata=metadata,
            states=states if states is not None else AFDTransferState(),
        ),
        input_ids=input_ids,
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
    def __init__(
        self,
        *,
        attn_size=1,
        ffn_size=1,
        role_rank=0,
        world_rank=0,
        requires_input_ids=False,
        requires_mtp=True,
    ):
        self.dp_metadata_list = {}
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.updates = []
        self.attn_size = attn_size
        self.ffn_size = ffn_size
        self.world_rank = world_rank
        self.requires_input_ids = requires_input_ids
        self.requires_mtp = requires_mtp
        self.received_input_ids = []
        self.mtp_phase_ready = deque()
        self.mtp_phase_control_enabled = True
        self.mtp_phase_graph_replay = False
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
                preloaded_input_ids = kwargs.get("input_ids")
                if preloaded_input_ids is not None and payload.input_ids is None:
                    payload = _ffn_payload(
                        payload.hidden_states,
                        payload.context.metadata,
                        states=payload.context.states,
                        input_ids=preloaded_input_ids,
                    )
                return payload
        raise IndexError(ubatch_idx)

    def recv_input_ids(self, num_tokens, *, ubatch_idx):
        input_ids = torch.arange(num_tokens, dtype=torch.int32) + ubatch_idx * 10
        self.received_input_ids.append((num_tokens, ubatch_idx, input_ids))
        return input_ids

    def send_ffn_output(self, ffn_output, context, **kwargs):
        self.ffn_outputs.append((ffn_output, context.metadata, kwargs))

    def recv_mtp_phase_ready(self):
        return self.mtp_phase_ready.popleft() if self.mtp_phase_ready else True

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
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (SimpleNamespace(use_dcp=True), True),
        (SimpleNamespace(use_dcp=False), False),
        (SimpleNamespace(use_cp=False), False),
        (SimpleNamespace(), False),
    ],
)
def test_npu_attention_legacy_dcp_manager_compatibility(runner, expected):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    assert attention_model_runner._uses_legacy_dcp_manager(runner) is expected


def test_npu_attention_common_metadata_abi_is_supported():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    fields = attention_model_runner._ASCEND_COMMON_METADATA_FIELDS
    legacy_fields = {
        "context_parallel_metadata",
        "group_len",
        "group_key_idx",
        "group_key_cache_idx",
    }
    vllm_cann_fields = {
        "prefill_context_parallel_metadata",
        "slot_mapping_cpu",
    }

    assert legacy_fields <= fields or vllm_cann_fields <= fields


def test_npu_ubatch_metadata_split_supports_runtime_abi():
    _require_npu_runtime()
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    from afd_plugin.v1.worker.npu import ubatch_utils

    fields = ubatch_utils._ASCEND_COMMON_METADATA_FIELDS
    metadata_kwargs = {
        "query_start_loc": torch.tensor([0, 2, 4], dtype=torch.int32),
        "query_start_loc_cpu": torch.tensor([0, 2, 4], dtype=torch.int32),
        "seq_lens": torch.tensor([2, 2], dtype=torch.int32),
        "seq_lens_cpu": torch.tensor([2, 2], dtype=torch.int32),
        "num_computed_tokens_cpu": torch.tensor([0, 0], dtype=torch.int32),
        "num_reqs": 2,
        "num_actual_tokens": 4,
        "max_query_len": 2,
        "max_seq_len": 2,
        "block_table_tensor": torch.zeros((2, 1), dtype=torch.int32),
        "slot_mapping": torch.arange(4, dtype=torch.int32),
        "actual_seq_lengths_q": [1, 2, 3, 4],
        "positions": torch.arange(4, dtype=torch.int64),
        "positions_cpu": torch.arange(4, dtype=torch.int64),
        "num_input_tokens": 4,
    }
    if "slot_mapping_cpu" in fields:
        metadata_kwargs["slot_mapping_cpu"] = torch.arange(4, dtype=torch.int32)
    if "prefill_context_parallel_metadata" in fields:
        metadata_kwargs["prefill_context_parallel_metadata"] = None
    if "context_parallel_metadata" in fields:
        metadata_kwargs["context_parallel_metadata"] = None
    if "kvcomp_metadata" in fields:
        metadata_kwargs["kvcomp_metadata"] = None
    for field_name in (
        "mm_req_doc_ranges",
        "rswa_prefix_lens",
        "group_len",
        "group_key_idx",
        "group_key_cache_idx",
    ):
        if field_name in fields:
            metadata_kwargs[field_name] = None

    metadata = AscendCommonAttentionMetadata(**metadata_kwargs)
    result = ubatch_utils._make_metadata_with_slice(
        UBatchSlice(slice(0, 1), slice(0, 2)),
        metadata,
    )

    assert result.num_reqs == 1
    assert result.num_actual_tokens == 2
    assert result.slot_mapping.tolist() == [0, 1]
    if "slot_mapping_cpu" in fields:
        assert result.slot_mapping_cpu.tolist() == [0, 1]


def _vllm_config(
    *,
    role="attention",
    connector="CAMP2pAFDConnector",
    extra_config=None,
    use_mla=False,
    cudagraph_mode="FULL",
    speculative_config=None,
    architecture=None,
    kv_transfer_config=None,
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
            hf_config=SimpleNamespace(
                architectures=[] if architecture is None else [architecture],
            ),
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
        kv_transfer_config=kv_transfer_config,
    )


def _require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")
    # The Ascend platform plugin loads the ops package before worker modules.
    # Preserve that order when this test file runs in isolation.
    importlib.import_module("vllm_ascend.ops")


def _new_attention_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

    runner = object.__new__(AFDNPUAttentionModelRunner)
    runner._afd_transaction_counter = 0
    runner.prof = None
    return runner


def test_npu_ubatch_dsa_ratio_metadata_is_stage_local():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        _new_ubatch_dsa_ratio_metadata,
    )

    stage_metadata = _new_ubatch_dsa_ratio_metadata(2)
    stage_metadata[0][0]["num_prefill"] = 3
    stage_metadata[0][1]["num_decode"] = 1
    stage_metadata[0][2]["block_table_rows"] = 3

    assert stage_metadata[1] == ({}, {}, {})
    assert all(
        id(stage_metadata[0][index]) != id(stage_metadata[1][index])
        for index in range(3)
    )


def test_npu_dspark_uses_its_cache_group_without_step35_metadata(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    class FakeDSparkProposer:
        kv_cache_gid = 2

        def set_per_group_attn_metadata(self, *_args):
            pytest.fail("DSpark does not implement Step3.5 per-group metadata")

    monkeypatch.setattr(
        attention_model_runner,
        "AscendDSparkProposer",
        FakeDSparkProposer,
    )
    drafter = FakeDSparkProposer()

    attention_model_runner._set_speculative_group_metadata(
        drafter,
        2,
        object(),
        object(),
    )

    assert attention_model_runner._speculative_drafter_matches_cache_group(
        drafter,
        2,
        {"draft.swa_cache"},
    )
    assert not attention_model_runner._speculative_drafter_matches_cache_group(
        drafter,
        1,
        {"target.mla_cache"},
    )


@pytest.mark.parametrize(
    ("request_slice", "num_reqs", "expected"),
    [
        (slice(0, 2), 3, 2),
        (slice(1, 3), 3, 2),
        (slice(2, 8), 3, 1),
        (slice(4, 8), 3, 0),
    ],
)
def test_npu_ubatch_actual_request_count_excludes_padding(
    request_slice,
    num_reqs,
    expected,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        _num_actual_requests_for_ubatch,
    )

    assert _num_actual_requests_for_ubatch(request_slice, num_reqs) == expected


def test_npu_attention_live_execution_scope_restores_on_success_and_error(
    monkeypatch,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    runner = _new_attention_runner()
    runner.prof = None
    runner._afd_live_execution = False
    observed_live_state = []

    monkeypatch.setattr(
        attention_model_runner,
        "step_afd_npu_profiler",
        lambda _prof: None,
    )

    def execute_success(_runner, _scheduler_output, _intermediate_tensors=None):
        observed_live_state.append(_runner._afd_live_execution)
        return "success"

    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "execute_model",
        execute_success,
    )
    assert runner.execute_model(object()) == "success"
    assert observed_live_state == [True]
    assert runner._afd_live_execution is False

    def execute_failure(_runner, _scheduler_output, _intermediate_tensors=None):
        observed_live_state.append(_runner._afd_live_execution)
        raise RuntimeError("execute failure")

    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "execute_model",
        execute_failure,
    )
    with pytest.raises(RuntimeError, match="execute failure"):
        runner.execute_model(object())
    assert observed_live_state == [True, True]
    assert runner._afd_live_execution is False


def test_npu_attention_logs_execute_and_sample_boundaries(monkeypatch, caplog):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    runner = _new_attention_runner()
    runner.connector = SimpleNamespace(stage_diagnostics_enabled=True)
    runner._afd_live_execution = False
    monkeypatch.setattr(
        attention_model_runner,
        "step_afd_npu_profiler",
        lambda _prof: None,
    )
    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "execute_model",
        lambda _runner, _scheduler_output, _intermediate_tensors=None: "executed",
    )
    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "sample_tokens",
        lambda _runner, _grammar_output: "sampled",
    )

    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.npu.attention_model_runner",
    ):
        assert runner.execute_model(object()) == "executed"
        assert runner.sample_tokens(None) == "sampled"

    assert "step=1 event=execute_begin" in caplog.text
    assert "step=1 event=execute_returned" in caplog.text
    assert "step=1 event=sample_begin" in caplog.text
    assert "step=1 event=sample_returned" in caplog.text


def test_npu_attention_non_live_execution_disables_microbatching(monkeypatch):
    _require_npu_runtime()
    import numpy as np
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from afd_plugin.v1.worker.npu import attention_model_runner

    runner = _new_attention_runner()
    runner._afd_live_execution = False
    runner._afd_in_mtp_proposal = False
    runner._afd_mtp_phase_announced = False
    runner._pad_for_sequence_parallelism = lambda num_tokens: num_tokens
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.ones(4, dtype=np.int32),
        lora_id_to_lora_request={},
    )
    runner.speculative_config = None
    runner.uniform_decode_query_len = 1
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
        ),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda **kwargs: (
            CUDAGraphMode.NONE,
            BatchDescriptor(kwargs["num_tokens"]),
        ),
    )
    monkeypatch.setattr(attention_model_runner, "enable_sp", lambda _config: False)
    monkeypatch.setattr(
        attention_model_runner,
        "check_enable_ubatch",
        lambda *_args, **_kwargs: True,
    )

    result = runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=4,
        num_scheduled_tokens_np=np.ones(4, dtype=np.int32),
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
        allow_microbatching=False,
    )
    assert result[2] is False

    runner._afd_live_execution = True
    result = runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=4,
        num_scheduled_tokens_np=np.ones(4, dtype=np.int32),
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
        allow_microbatching=False,
    )
    assert result[2] is True


@pytest.mark.parametrize("live_tokens", [1, 2, 3])
@pytest.mark.parametrize("force_dp_padding", [False, True])
def test_npu_attention_mixed_graph_modes_share_dp_token_counts(
    monkeypatch,
    live_tokens,
    force_dp_padding,
):
    _require_npu_runtime()
    import numpy as np
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    from afd_plugin.v1.worker.npu import attention_model_runner

    counts = [live_tokens, 4, 4, 4]

    def all_reduce(packed_tensor, *, group):
        assert group == "cpu-group"
        assert tuple(packed_tensor.shape) == (6, 4)
        packed_tensor[0, :] = torch.tensor(counts)
        packed_tensor[1, :] = torch.tensor(counts)
        packed_tensor[2, :] = torch.tensor(
            [
                CUDAGraphMode.NONE.value,
                *([CUDAGraphMode.FULL.value] * 3),
            ]
        )
        packed_tensor[3, :] = torch.tensor([0, 1, 1, 1])
        packed_tensor[5, :] = torch.tensor([0, 1, 1, 1])

    def dispatch(*, num_tokens, uniform_decode, valid_modes=None, **_kwargs):
        mode = CUDAGraphMode.FULL if uniform_decode else CUDAGraphMode.NONE
        if valid_modes is not None:
            mode = next(iter(valid_modes))
        return mode, BatchDescriptor(num_tokens)

    monkeypatch.setattr(attention_model_runner.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        attention_model_runner,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group="cpu-group"),
    )
    monkeypatch.setattr(
        attention_model_runner,
        "should_skip_allreduce_across_dp_group",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "check_enable_ubatch",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(attention_model_runner, "enable_sp", lambda _config: False)
    monkeypatch.setattr(
        attention_model_runner,
        "oproj_tp_enable",
        lambda: force_dp_padding,
    )
    monkeypatch.setattr(attention_model_runner, "embedding_tp_enable", lambda: False)

    observed_counts = []
    for rank, num_tokens in enumerate(counts):
        runner = _new_attention_runner()
        runner.dp_size = 4
        runner.dp_rank = rank
        runner._afd_live_execution = rank == 0
        runner._afd_engine_idle_dummy = rank != 0
        runner.connector = SimpleNamespace(
            control_plane=object(),
            is_window_connector=False,
            requires_lockstep_dp_sync=False,
        )
        runner.vllm_config = _vllm_config(
            data_parallel_size=4,
            data_parallel_rank=rank,
            tensor_parallel_size=1,
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
        )
        runner.vllm_config.observability_config = SimpleNamespace(
            cudagraph_metrics=False
        )
        runner.parallel_config = runner.vllm_config.parallel_config
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.speculative_config = None
        runner.uniform_decode_query_len = 4
        runner.input_batch = SimpleNamespace(
            num_computed_tokens_cpu=np.ones(1, dtype=np.int32),
            lora_id_to_lora_request={},
        )
        runner._pad_for_sequence_parallelism = lambda tokens: tokens
        runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)

        mode, descriptor, _, token_counts, _ = (
            runner._determine_batch_execution_and_padding(
                num_tokens=num_tokens,
                num_reqs=1,
                num_scheduled_tokens_np=np.array([num_tokens]),
                max_num_scheduled_tokens=num_tokens,
                use_cascade_attn=False,
                force_uniform_decode=rank != 0,
            )
        )

        assert mode is CUDAGraphMode.NONE
        assert descriptor.num_tokens == (4 if force_dp_padding else num_tokens)
        observed_counts.append(token_counts.tolist())

    expected = [4, 4, 4, 4] if force_dp_padding else counts
    assert observed_counts == [expected] * 4


def _new_ffn_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

    # object.__new__ bypasses __init__, which is where the runner would set up
    # the profiler and device; provide inert defaults the runtime paths expect.
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.prof = None
    runner.device = SimpleNamespace(type="npu")
    runner._is_shutdown = False
    runner.afd_config = AFDConfig(role="ffn")
    runner._ffn_input_ids_cache = {}
    runner._mtp_acl_graphs = {}
    runner.ffn_stream_overlap_enabled = False
    runner.max_num_tokens = 1
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

    positions = object()
    result = runner._model_forward(
        8,
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "hidden_states"
    assert len(updates) == expected_updates
    if updates:
        assert updates[0][2] is positions


@pytest.mark.parametrize(
    ("graph_exists", "expected_mode", "expected_updates"),
    [(False, "NONE", 1), (True, "FULL", 0)],
)
def test_npu_attention_runner_live_graph_u2_miss_falls_back_eager(
    monkeypatch,
    graph_exists,
    expected_mode,
    expected_updates,
):
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    from afd_plugin.v1.worker.npu import attention_model_runner

    observed_modes = []

    class FakeUBatchWrapper:
        def has_ubatch_full_graph(self, _forward_context):
            return graph_exists

        def owns_full_graph_update(self, forward_context):
            return forward_context.cudagraph_runtime_mode is CUDAGraphMode.FULL

        def __call__(self, **_model_inputs):
            observed_modes.append(forward_context.cudagraph_runtime_mode.name)
            return "hidden_states"

    ubatch_slices = [object(), object()]
    forward_context = SimpleNamespace(
        dbo_enabled=False,
        flash_comm_v1_enabled=False,
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
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

    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.enable_enpu = False
    runner.model = FakeUBatchWrapper()
    runner.ubatch_slices = ubatch_slices
    runner._afd_live_execution = True
    runner._install_afd_metadata_on_forward_context = lambda _context: None
    runner._install_async_moe_ubatch_metadata_on_forward_context = lambda _context: None
    updates = []
    runner._update_full_graph_params_if_needed = lambda *args: updates.append(args)

    result = runner._model_forward(
        6,
        input_ids=None,
        positions=object(),
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "hidden_states"
    assert observed_modes == [expected_mode]
    assert len(updates) == expected_updates


def test_npu_attention_runner_pretransfers_dsv4_ids_outside_model(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    events = []
    forward_context = SimpleNamespace(
        dbo_enabled=False,
        flash_comm_v1_enabled=False,
    )

    class FakeModel:
        def __call__(self, **_model_inputs):
            events.append(
                (
                    "model",
                    forward_context.afd_input_ids_pretransferred,
                )
            )
            return "hidden_states"

    class FakeConnector:
        requires_input_ids = True

        def send_input_ids(self, input_ids, *, ubatch_idx):
            events.append(("ids", input_ids.clone(), ubatch_idx))

    monkeypatch.setattr(
        attention_model_runner,
        "get_forward_context",
        lambda: forward_context,
    )
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.enable_enpu = False
    runner.model = FakeModel()
    runner.connector = FakeConnector()
    runner.ubatch_slices = None
    runner._install_afd_metadata_on_forward_context = lambda _context: None
    runner._install_async_moe_ubatch_metadata_on_forward_context = lambda _context: None
    runner._update_full_graph_params_if_needed = lambda *_args: None
    input_ids = torch.tensor([-1, 7], dtype=torch.int32)

    result = runner._model_forward(2, input_ids=input_ids)

    assert result == "hidden_states"
    assert events[0][0] == "ids"
    assert events[0][1].tolist() == [-1, 7]
    assert events[0][2] == 0
    assert events[1] == ("model", True)
    assert forward_context.afd_input_ids_pretransferred is False


def test_npu_attention_runner_defers_dsv4_ids_to_each_ubatch_layer_zero(
    monkeypatch,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    events = []
    forward_context = SimpleNamespace(
        dbo_enabled=False,
        flash_comm_v1_enabled=False,
    )

    class FakeModel:
        def __call__(self, **_model_inputs):
            events.append(
                ("model", forward_context.afd_input_ids_pretransferred),
            )
            return "hidden_states"

    class FakeConnector:
        requires_input_ids = True

        def send_input_ids(self, input_ids, *, ubatch_idx):
            events.append(("ids", input_ids.clone(), ubatch_idx))

    monkeypatch.setattr(
        attention_model_runner,
        "get_forward_context",
        lambda: forward_context,
    )
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.enable_enpu = False
    runner.model = FakeModel()
    runner.connector = FakeConnector()
    runner.ubatch_slices = [
        SimpleNamespace(token_slice=slice(0, 2), num_tokens=2),
        SimpleNamespace(token_slice=slice(2, 5), num_tokens=3),
    ]
    runner._install_afd_metadata_on_forward_context = lambda _context: None
    runner._install_async_moe_ubatch_metadata_on_forward_context = lambda _context: None
    runner._update_full_graph_params_if_needed = lambda *_args: None
    input_ids = torch.tensor([10, 11, 20, 21, -1], dtype=torch.int32)

    result = runner._model_forward(5, input_ids=input_ids)

    assert result == "hidden_states"
    assert events == [("model", False)]
    assert forward_context.afd_input_ids_pretransferred is False


def test_npu_attention_runner_pretransfers_hccl_stream_ids_by_stage(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    events = []
    forward_context = SimpleNamespace(
        dbo_enabled=False,
        flash_comm_v1_enabled=False,
    )

    class FakeModel:
        def __call__(self, **_model_inputs):
            events.append(
                ("model", forward_context.afd_input_ids_pretransferred),
            )
            return "hidden_states"

    connector = object.__new__(attention_model_runner.P2pHcclAFDConnector)
    connector.stream_overlap_enabled = True
    connector.afd_config = SimpleNamespace(role="attention")
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(0, 0): object()}
    connector.requires_input_ids = True
    connector.send_input_ids = lambda input_ids, *, ubatch_idx: events.append(
        ("ids", input_ids.clone(), ubatch_idx),
    )

    monkeypatch.setattr(
        attention_model_runner,
        "get_forward_context",
        lambda: forward_context,
    )
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.enable_enpu = False
    runner.model = FakeModel()
    runner.connector = connector
    runner.ubatch_slices = [
        SimpleNamespace(token_slice=slice(0, 2), num_tokens=2),
        SimpleNamespace(token_slice=slice(2, 5), num_tokens=3),
    ]
    runner._install_afd_metadata_on_forward_context = lambda _context: None
    runner._install_async_moe_ubatch_metadata_on_forward_context = lambda _context: None
    runner._update_full_graph_params_if_needed = lambda *_args: None
    input_ids = torch.tensor([10, 11, 20, 21, -1], dtype=torch.int32)

    result = runner._model_forward(5, input_ids=input_ids)

    assert result == "hidden_states"
    assert events[0][0] == "ids"
    assert events[0][1].tolist() == [10, 11]
    assert events[0][2] == 0
    assert events[1][0] == "ids"
    assert events[1][1].tolist() == [20, 21, -1]
    assert events[1][2] == 1
    assert events[2] == ("model", True)
    assert forward_context.afd_input_ids_pretransferred is False


@pytest.mark.parametrize(
    ("model_fields", "use_compress", "expected_mla_full_graph"),
    [
        ({}, False, True),
        ({"hf_config": SimpleNamespace(index_topk=8)}, False, False),
        ({"hf_text_config": SimpleNamespace(index_topk=8)}, False, False),
        ({}, True, False),
    ],
)
def test_npu_attention_runner_installs_mla_graph_wrapper(
    monkeypatch,
    model_fields,
    use_compress,
    expected_mla_full_graph,
):
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
        model_config=SimpleNamespace(use_mla=True, **model_fields),
    )
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True),
    )
    runner.use_sparse = False
    runner.use_compress = use_compress
    runner.enable_enpu = False

    runner._install_ascend_ubatch_wrapper()

    assert captured["args"][:2] == ("model", runner.vllm_config)
    assert captured["kwargs"]["mla_full_graph_enabled"] is expected_mla_full_graph
    assert captured["kwargs"]["enable_enpu"] is False
    assert captured["kwargs"]["enable_layer_major_eager_u2"] is False
    updater = captured["kwargs"]["full_graph_params_updater"]
    assert updater.__self__ is runner


@pytest.mark.parametrize("has_full_cudagraphs", [False, True])
@pytest.mark.parametrize("requires_mtp", [False, True])
@pytest.mark.parametrize("eager_stream_overlap", [False, True])
def test_npu_attention_runner_enables_p2p_layer_major_eager_u2_warmup(
    monkeypatch,
    has_full_cudagraphs,
    requires_mtp,
    eager_stream_overlap,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    captured = {}

    class RecordingUBatchWrapper:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    class LayerMajorModel:
        def forward_ubatches_layer_major(self, _metadata):
            raise AssertionError("installation must not execute the model")

    monkeypatch.setattr(
        attention_model_runner,
        "AscendUBatchWrapper",
        RecordingUBatchWrapper,
    )
    connector = object.__new__(attention_model_runner.P2pHcclAFDConnector)
    connector.stream_overlap_enabled = True
    connector.eager_u2_stream_overlap_enabled = eager_stream_overlap
    connector.requires_mtp = requires_mtp
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.model = LayerMajorModel()
    runner.connector = connector
    runner.device = "npu"
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
    )
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(
            has_full_cudagraphs=lambda: has_full_cudagraphs,
        ),
    )
    runner.use_sparse = False
    runner.enable_enpu = False

    runner._install_ascend_ubatch_wrapper()

    assert captured["kwargs"]["enable_layer_major_eager_u2"] is True


@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("execution", ["startup", "live", "dummy"])
def test_npu_attention_runner_builds_and_sets_metadata(
    monkeypatch, graph_replay, execution
):
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_unpadded_tokens_across_dp = None
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    runner._afd_live_execution = execution == "live"
    runner._afd_live_dummy_execution = execution == "dummy"
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=5),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL" if graph_replay else "NONE"),
    )
    sent = []
    monkeypatch.setattr(runner.connector, "send_dp_metadata_list", sent.append)

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [1]
    assert len(runner.connector.dp_metadata_updates) == 1
    assert len(sent) == 1
    assert sent[0].target_graph_replay is graph_replay
    assert sent[0].mtp_phase_control_enabled is (execution != "startup")


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
    runner._afd_unpadded_tokens_across_dp = None

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

    assert result is None
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


def test_npu_attention_target_graph_u2_capture_omits_eager_mtp_drafter():
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    runner = _new_attention_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner.connector = SimpleNamespace(control_plane=None)
    runner.speculative_config = _mtp_speculative_config(enforce_eager=True)
    eager_drafter = object()
    runner.drafter = eager_drafter
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_suppress_metadata_send = False
    runner._afd_pending_metadata = None
    calls = []

    def dummy_run(num_tokens, **kwargs):
        calls.append(
            (
                num_tokens,
                kwargs["cudagraph_runtime_mode"],
                runner._is_warmup,
                runner.drafter,
            )
        )

    runner._dummy_run = dummy_run
    desc = SimpleNamespace(num_tokens=8, uniform=True, num_active_loras=0)

    runner._warmup_and_capture(
        desc,
        CUDAGraphMode.FULL,
        allow_microbatching=True,
    )

    assert calls == [
        (8, CUDAGraphMode.NONE, True, eager_drafter),
        (8, CUDAGraphMode.FULL, False, None),
        (8, CUDAGraphMode.NONE, True, eager_drafter),
        (8, CUDAGraphMode.FULL, False, None),
    ]
    assert runner.drafter is eager_drafter


def test_npu_attention_full_draft_graph_capture_keeps_mtp_drafter():
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    runner = _new_attention_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner.connector = SimpleNamespace(control_plane=None)
    runner.speculative_config = _mtp_speculative_config(enforce_eager=False)
    graph_drafter = object()
    runner.drafter = graph_drafter
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_suppress_metadata_send = False
    runner._afd_pending_metadata = None
    calls = []

    def dummy_run(num_tokens, **kwargs):
        calls.append(
            (
                num_tokens,
                kwargs["cudagraph_runtime_mode"],
                runner.drafter,
            )
        )

    runner._dummy_run = dummy_run
    runner._warmup_and_capture(
        SimpleNamespace(num_tokens=8, uniform=True, num_active_loras=0),
        CUDAGraphMode.FULL,
        allow_microbatching=False,
    )

    assert calls == [
        (8, CUDAGraphMode.NONE, graph_drafter),
        (8, CUDAGraphMode.FULL, graph_drafter),
    ]


def test_npu_attention_load_model_keeps_dspark_drafter_local(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import attention_model_runner

    calls = []

    class FakeDSparkProposer:
        pass

    monkeypatch.setattr(
        attention_model_runner,
        "AscendDSparkProposer",
        FakeDSparkProposer,
    )
    monkeypatch.setattr(
        attention_model_runner.NPUModelRunner,
        "load_model",
        lambda self: calls.append("parent"),
    )
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunner)
    runner.speculative_config = _dspark_speculative_config()
    runner.drafter = FakeDSparkProposer()
    runner.connector = object()
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    runner._configure_mtp_draft_graph = lambda: calls.append("configure_mtp")
    runner._install_ascend_ubatch_wrapper = lambda: calls.append("install_ubatch")

    runner.load_model()

    assert calls == ["parent"]


def test_npu_ffn_load_model_keeps_dspark_drafter_off_ffn(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    calls = []
    monkeypatch.setattr(
        ffn_model_runner.NPUModelRunner,
        "load_model",
        lambda self: calls.append(("parent", self.drafter)),
    )
    runner = object.__new__(ffn_model_runner.AFDNPUFFNModelRunner)
    runner.speculative_config = _dspark_speculative_config()
    runner.connector = SimpleNamespace(requires_mtp=False)
    runner.drafter = SimpleNamespace(
        _get_model=lambda: pytest.fail("FFN must not load the DSpark draft model"),
    )
    runner.mtp_ffn_model = None

    drafter = runner.drafter
    runner.load_model()

    assert calls == [("parent", None)]
    assert runner.drafter is drafter
    assert runner.mtp_ffn_model is None


def test_npu_attention_full_draft_graph_enables_replay_synchronization():
    _require_npu_runtime()
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    runner = _new_attention_runner()
    runner.speculative_config = _mtp_speculative_config(enforce_eager=False)
    draft_graph = object.__new__(ACLGraphWrapper)
    draft_graph.use_eagle = True
    runner.drafter = SimpleNamespace(_runnable=draft_graph)

    runner._configure_mtp_draft_graph()

    assert draft_graph.use_eagle is False
    assert runner.drafter._runnable._runnable is draft_graph


def test_npu_attention_eager_draft_gets_phase_boundary_wrapper():
    runner = _new_attention_runner()
    runner.speculative_config = _mtp_speculative_config(enforce_eager=True)

    def eager_runnable():
        return None

    runner.drafter = SimpleNamespace(_runnable=eager_runnable)

    runner._configure_mtp_draft_graph()

    assert runner.drafter._runnable._runnable is eager_runnable


@pytest.mark.parametrize("online_dummy", [False, True])
def test_npu_attention_announces_only_executed_live_mtp_phase(
    monkeypatch, online_dummy
):
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    from afd_plugin.v1.worker.npu import attention_model_runner
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        _AFDMTPPhaseAwareRunnable,
    )

    runner = _new_attention_runner()
    runner._afd_live_execution = False
    calls = []
    runner.connector = SimpleNamespace(
        control_plane=SimpleNamespace(
            send_mtp_phase_ready=lambda **kwargs: calls.append(
                ("ready", kwargs["graph_replay"]),
            ),
        ),
    )

    class CallableDraftGraph:
        runtime_mode = CUDAGraphMode.FULL
        concrete_aclgraph_entries = {}

        def __call__(self, value):
            calls.append(("runnable", value))
            return value + 1

    graph_wrapper = CallableDraftGraph()
    runnable = _AFDMTPPhaseAwareRunnable(
        graph_wrapper,
        runner._announce_live_mtp_phase,
    )

    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor="draft-batch",
    )
    monkeypatch.setattr(
        attention_model_runner,
        "get_forward_context",
        lambda: forward_context,
    )

    assert runnable(3) == 4
    runner._afd_in_mtp_proposal = not online_dummy
    runner._afd_live_dummy_execution = online_dummy
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    assert runnable(5) == 6
    assert forward_context.cudagraph_runtime_mode is CUDAGraphMode.FULL
    graph_wrapper.concrete_aclgraph_entries["draft-batch"] = SimpleNamespace(
        aclgraph=object(),
    )
    assert runnable(7) == 8

    assert calls == [
        ("runnable", 3),
        ("ready", False),
        ("runnable", 5),
        ("ready", True),
        ("runnable", 7),
    ]
    assert runner._afd_mtp_phase_announced is True
    assert runner._afd_mtp_graph_replayed is True


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

    module_name = "afd_plugin.v1.worker.npu.ubatch_utils"
    original_module = sys.modules.pop(module_name, None)
    try:
        ubatch_utils = importlib.import_module(module_name)
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
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


@pytest.mark.parametrize(
    ("parent_graph_mode", "expected_graph_ubatching"),
    [
        (None, False),
        (SimpleNamespace(name="FULL"), True),
    ],
)
def test_npu_create_ascend_forward_context_marks_current_ubatch(
    monkeypatch,
    parent_graph_mode,
    expected_graph_ubatching,
):
    _require_npu_runtime()
    import torch

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
        is_first_layer=True,
        layer_idx=0,
        prefetch_mlp_gate_up_proj=False,
        prefetch_mlp_down_proj=False,
        model_instance=None,
        is_draft_model=False,
        is_draft_model_prefill=False,
        draft_attn_metadatas=None,
        max_tokens_across_pcp=None,
        sinks=False,
        input_ids=torch.tensor([10, 11, 12, 13, 20, 21, 22]),
        afd_input_ids_pretransferred=True,
        cudagraph_runtime_mode=parent_graph_mode,
        eplb_heat_collection_status=False,
        is_padding=None,
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
    assert new_forward_context.input_ids.tolist() == [20, 21, 22]
    assert new_forward_context.afd_input_ids_pretransferred is True
    assert new_forward_context.afd_graph_ubatching is expected_graph_ubatching
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


def test_npu_ffn_runner_executes_decoder_then_mtp_phase(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=1,
            enforce_eager=True,
        ),
    )
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.mtp_ffn_model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 2
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    decoder_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=2,
    )
    mtp_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=2,
        phase="mtp",
        speculative_step=0,
    )
    runner.connector.attn_outputs.extend(
        [
            ("decoder-hidden", decoder_metadata),
            ("mtp-hidden", mtp_metadata),
        ]
    )
    runner.connector.recv_mtp_header = lambda *, stage_idx: SimpleNamespace(
        num_tokens=2,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([2], dtype=torch.int32),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([2])})

    assert runner.connector.ffn_outputs == [
        ("npu-ffn(decoder-hidden, layer=0)", decoder_metadata, {"ubatch_idx": 0}),
        ("npu-ffn(mtp-hidden, layer=0)", mtp_metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_dspark_does_not_enter_remote_mtp_phase(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_dspark_speculative_config(),
    )
    runner.connector = _FakeFFNConnector(requires_mtp=False)
    runner.model = _FakeModel()
    runner.mtp_ffn_model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    decoder_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("decoder-hidden", decoder_metadata))
    runner.connector.recv_mtp_header = lambda **_kwargs: pytest.fail(
        "DSpark keeps its draft on Attention and must not receive an AFD MTP header",
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("npu-ffn(decoder-hidden, layer=0)", decoder_metadata, {"ubatch_idx": 0}),
    ]
    assert runner._recv_mtp_phase_ready() is False


def test_npu_ffn_runner_executes_all_configured_mtp_steps(monkeypatch):
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(
            enforce_eager=True,
            num_speculative_tokens=3,
        ),
    )
    runner.connector = _FakeFFNConnector()
    runner.use_aclgraph = False
    calls = []
    monkeypatch.setattr(
        runner,
        "_mtp_ffn_forward",
        lambda stage_ids, *, expected_speculative_step=None, **_kwargs: calls.append(
            (tuple(stage_ids), expected_speculative_step)
        ),
    )

    runner._execute_mtp_after_target({0: _FakeDPMetadata([8])})

    assert calls == [((0,), 0), ((0,), 1), ((0,), 2)]


def test_npu_ffn_runner_executes_u2_decoder_then_one_merged_mtp_phase(
    monkeypatch,
):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
    )
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.mtp_ffn_model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 3
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    decoder_metadata = [
        AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=stage_idx,
            seq_len=stage_idx + 1,
        )
        for stage_idx in (0, 1)
    ]
    mtp_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=3,
        phase="mtp",
        speculative_step=0,
    )
    runner.connector.attn_outputs.extend(
        [
            ("decoder-stage-0", decoder_metadata[0]),
            ("decoder-stage-1", decoder_metadata[1]),
            ("mtp-merged", mtp_metadata),
        ]
    )
    header_stages = []

    def recv_mtp_header(*, stage_idx):
        header_stages.append(stage_idx)
        return SimpleNamespace(
            num_tokens=3,
            speculative_step=0,
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
        )

    runner.connector.recv_mtp_header = recv_mtp_header

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([1]),
            1: _FakeDPMetadata([2]),
        }
    )

    assert header_stages == [0]
    assert [item[1] for item in runner.connector.ffn_outputs] == [
        *decoder_metadata,
        mtp_metadata,
    ]


def test_npu_ffn_runner_builds_forward_context_for_each_dbo_stage(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    context_calls = []

    @contextmanager
    def fake_ascend_forward_context(**kwargs):
        context_calls.append(kwargs)
        yield SimpleNamespace(
            additional_kwargs={},
            dp_metadata=None,
            all_moe_layers={},
        )

    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        fake_ascend_forward_context,
    )
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector(attn_size=2, ffn_size=2)
    runner.model = _FakeModel()
    runner.num_layers = 3
    runner.max_num_tokens = 16
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    for layer_idx in range(runner.num_layers):
        for stage_idx, num_tokens in enumerate((6, 7)):
            metadata = AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx,
                stage_idx=stage_idx,
                seq_len=num_tokens,
            )
            runner.connector.attn_outputs.append(
                (f"hidden-{layer_idx}-{stage_idx}", metadata),
            )

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([6]),
            1: _FakeDPMetadata([7]),
        },
    )

    assert [call["num_tokens"] for call in context_calls] == [6, 7]
    assert [call["num_tokens_across_dp"].tolist() for call in context_calls] == [
        [6],
        [7],
    ]
    assert len(runner.connector.ffn_outputs) == 6


def test_npu_ffn_runner_projects_attention_counts_to_ffn_fanout():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    connector = SimpleNamespace(attn_size=2, ffn_size=4)
    counts = ffn_model_runner._ffn_token_counts_across_ranks(
        connector,
        {0: _FakeDPMetadata([5, 1])},
        0,
        fallback=16,
    )

    assert counts.tolist() == [3, 2, 1, 1]


@pytest.mark.parametrize(
    (
        "is_graph_capturing",
        "graph_runtime",
        "graph_overlap_enabled",
        "graph_recv_enabled",
        "graph_cross_layer_enabled",
        "eager_overlap_enabled",
    ),
    [
        (False, False, True, True, True, True),
        (False, False, True, True, True, False),
        (True, True, True, True, True, True),
        (True, True, True, True, False, True),
        (True, True, True, False, False, True),
        (True, True, False, True, True, True),
    ],
    ids=[
        "eager-streamed",
        "eager-serial",
        "graph-full-pipeline",
        "graph-recv-layer-barrier",
        "graph-parent-recv-layer-barrier",
        "graph-capture-baseline",
    ],
)
def test_npu_ffn_runner_stream_pipeline_orders_each_layer_and_stage(
    monkeypatch,
    is_graph_capturing,
    graph_runtime,
    graph_overlap_enabled,
    graph_recv_enabled,
    graph_cross_layer_enabled,
    eager_overlap_enabled,
):
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    from afd_plugin.v1.worker.npu import ffn_model_runner

    _patch_ffn_forward_context(monkeypatch)
    calls = []
    active_stream = [None]
    recv_stream = object()
    compute_stream = object()
    send_stream = object()
    default_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    class StreamConnector(_FakeFFNConnector):
        stream_overlap_enabled = True

        def recv_attn_output(self, *, ubatch_idx, **kwargs):
            if active_stream[0] is not recv_stream:
                calls.append(
                    (
                        "recv",
                        kwargs["layer_idx"],
                        ubatch_idx,
                        ffn_model_runner.torch.npu.current_stream(),
                    )
                )
            return super().recv_attn_output(ubatch_idx=ubatch_idx, **kwargs)

        def send_ffn_output(self, ffn_output, context, *, ubatch_idx, **kwargs):
            if active_stream[0] is not send_stream:
                calls.append(
                    (
                        "send",
                        context.metadata.layer_idx,
                        ubatch_idx,
                        active_stream[0] or ffn_model_runner.torch.npu.current_stream(),
                    )
                )
            return super().send_ffn_output(
                ffn_output,
                context,
                ubatch_idx=ubatch_idx,
                **kwargs,
            )

        def recv_attn_output_streamed(
            self,
            *,
            ubatch_idx,
            recv_stream,
            wait_event,
            done_event,
            **kwargs,
        ):
            with use_stream(recv_stream):
                if wait_event is not None:
                    wait_event.wait(recv_stream)
                calls.append(("recv", kwargs["layer_idx"], ubatch_idx, recv_stream))
                payload = self.recv_attn_output(ubatch_idx=ubatch_idx, **kwargs)
                done_event.record(recv_stream)
            return payload, done_event

        def send_ffn_output_streamed(
            self,
            ffn_output,
            context,
            *,
            ubatch_idx,
            send_stream,
            wait_event,
            done_event,
        ):
            with use_stream(send_stream):
                wait_event.wait(send_stream)
                calls.append(
                    ("send", context.metadata.layer_idx, ubatch_idx, send_stream),
                )
                self.send_ffn_output(
                    ffn_output,
                    context,
                    ubatch_idx=ubatch_idx,
                )
                done_event.record(send_stream)
            return done_event

    class StreamModel:
        def compute_ffn_output(self, hidden_states, layer_idx, **_kwargs):
            stage_idx = int(hidden_states[0, 0].item())
            calls.append(("compute", layer_idx, stage_idx, active_stream[0]))
            return hidden_states + 1

    monkeypatch.setattr(ffn_model_runner, "P2pHcclAFDConnector", StreamConnector)
    monkeypatch.setattr(ffn_model_runner.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        ffn_model_runner.torch.npu,
        "current_stream",
        lambda: default_stream,
    )

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
    )
    runner.connector = StreamConnector(attn_size=2, ffn_size=2)
    runner.connector.eager_u2_stream_overlap_enabled = eager_overlap_enabled
    runner.connector.graph_u2_compute_overlap_enabled = graph_overlap_enabled
    runner.connector.graph_u2_ffn_recv_stream_enabled = graph_recv_enabled
    runner.connector.graph_u2_ffn_cross_layer_enabled = graph_cross_layer_enabled
    runner.model = StreamModel()
    runner.num_layers = 2
    runner.max_num_tokens = 2
    runner.ffn_stream_overlap_enabled = True
    runner.ffn_recv_stream = recv_stream
    runner.ffn_compute_stream = compute_stream
    runner.ffn_send_stream = send_stream
    runner.ffn_graph_recv_ready_event = FakeEvent("graph-recv-ready")
    event_keys = [
        (layer_idx, stage_idx) for layer_idx in range(2) for stage_idx in range(2)
    ]
    runner.ffn_recv_events = {
        key: FakeEvent(f"recv-{key[0]}-{key[1]}") for key in event_keys
    }
    runner.ffn_compute_events = {
        key: FakeEvent(f"compute-{key[0]}-{key[1]}") for key in event_keys
    }
    runner.ffn_send_events = {
        key: FakeEvent(f"send-{key[0]}-{key[1]}") for key in event_keys
    }
    for layer_idx in range(2):
        for stage_idx in range(2):
            metadata = AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx,
                stage_idx=stage_idx,
                seq_len=1,
            )
            runner.connector.attn_outputs.append(
                _ffn_payload(
                    torch.full((1, 1), stage_idx, dtype=torch.float32),
                    metadata,
                ),
            )

    runner._ffn_forward(
        dp_metadata_list={
            0: _FakeDPMetadata([1]),
            1: _FakeDPMetadata([1]),
        },
        is_graph_capturing=is_graph_capturing,
        aclgraph_runtime_mode=CUDAGraphMode.FULL if graph_runtime else None,
    )

    eager_stream_overlap = not is_graph_capturing and eager_overlap_enabled
    for layer_idx in range(2):
        for stage_idx in range(2):
            recv_marker = (
                "recv",
                layer_idx,
                stage_idx,
                (
                    recv_stream
                    if eager_stream_overlap
                    or (
                        is_graph_capturing
                        and graph_overlap_enabled
                        and graph_recv_enabled
                    )
                    else default_stream
                ),
            )
            expected_compute_stream = (
                compute_stream
                if eager_stream_overlap
                or (is_graph_capturing and graph_overlap_enabled)
                else None
            )
            compute_marker = (
                "compute",
                layer_idx,
                stage_idx,
                expected_compute_stream,
            )
            send_marker = (
                "send",
                layer_idx,
                stage_idx,
                (
                    send_stream
                    if eager_stream_overlap
                    or (is_graph_capturing and graph_overlap_enabled)
                    else default_stream
                ),
            )
            assert calls.index(recv_marker) < calls.index(compute_marker)
            assert calls.index(compute_marker) < calls.index(send_marker)
            if layer_idx > 0 and (
                eager_stream_overlap
                or (is_graph_capturing and graph_overlap_enabled and graph_recv_enabled)
            ):
                previous_send_wait = (
                    f"send-{layer_idx - 1}-{stage_idx}",
                    "wait",
                    recv_stream,
                )
                assert calls.index(previous_send_wait) < calls.index(recv_marker)
            if (
                layer_idx == 0
                and is_graph_capturing
                and graph_overlap_enabled
                and graph_recv_enabled
            ):
                assert calls.index(
                    ("graph-recv-ready", "wait", recv_stream)
                ) < calls.index(recv_marker)
            if is_graph_capturing and graph_overlap_enabled:
                compute_wait = (
                    f"compute-{layer_idx}-{stage_idx}",
                    "wait",
                    send_stream,
                )
                assert calls.index(compute_wait) < calls.index(send_marker)
    if is_graph_capturing:
        graph_schedule = [
            call[:3] for call in calls if call[0] in {"recv", "compute", "send"}
        ]
        if graph_overlap_enabled:
            assert graph_schedule == [
                ("recv", 0, 0),
                ("compute", 0, 0),
                ("send", 0, 0),
                ("recv", 0, 1),
                ("compute", 0, 1),
                ("send", 0, 1),
                ("recv", 1, 0),
                ("compute", 1, 0),
                ("send", 1, 0),
                ("recv", 1, 1),
                ("compute", 1, 1),
                ("send", 1, 1),
            ]
        else:
            assert graph_schedule == [
                (op, layer_idx, stage_idx)
                for layer_idx in range(2)
                for stage_idx in range(2)
                for op in ("recv", "compute", "send")
            ]
        if graph_overlap_enabled:
            cross_layer_active = graph_recv_enabled and graph_cross_layer_enabled
            joined_layers = (1,) if cross_layer_active else (0, 1)
            for layer_idx in joined_layers:
                for stage_idx in range(2):
                    assert (
                        f"send-{layer_idx}-{stage_idx}",
                        "wait",
                        default_stream,
                    ) in calls
            if cross_layer_active:
                for stage_idx in range(2):
                    assert (
                        f"send-0-{stage_idx}",
                        "wait",
                        default_stream,
                    ) not in calls
            else:
                for layer_idx in range(2):
                    recv_stream_for_layer = (
                        recv_stream if graph_recv_enabled else default_stream
                    )
                    assert calls.index(
                        ("recv", layer_idx, 1, recv_stream_for_layer)
                    ) < calls.index((f"send-{layer_idx}-0", "wait", default_stream))
    else:
        if eager_overlap_enabled:
            assert ("send-1-0", "wait", default_stream) in calls
            assert ("send-1-1", "wait", default_stream) in calls
        else:
            assert ("send-1-0", "wait", default_stream) not in calls
            assert ("send-1-1", "wait", default_stream) not in calls


def test_dsv4_ffn_eager_receives_ids_and_hidden_stage_by_stage(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hash_layers=1),
    )
    runner.model = _RecordingFakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 8
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    events = []

    class StageOrderedConnector(_FakeFFNConnector):
        requires_input_ids = True

        def update_state_from_dp_metadata(self, payload):
            events.append(("update", tuple(sorted(payload.dp_metadata_list))))
            return super().update_state_from_dp_metadata(payload)

        def recv_input_ids(self, num_tokens, *, ubatch_idx):
            events.append(("ids", ubatch_idx))
            return super().recv_input_ids(num_tokens, ubatch_idx=ubatch_idx)

        def recv_attn_output(self, ubatch_idx=None, **kwargs):
            events.append(("hidden", ubatch_idx))
            return super().recv_attn_output(ubatch_idx=ubatch_idx, **kwargs)

    runner.connector = StageOrderedConnector(
        attn_size=2,
        ffn_size=2,
        requires_input_ids=True,
    )
    for stage_idx, num_tokens in enumerate((3, 2)):
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=stage_idx,
            seq_len=num_tokens,
        )
        runner.connector.attn_outputs.append((f"hidden-{stage_idx}", metadata))

    runner.execute_ffn_step(
        dp_metadata_list={
            0: _FakeDPMetadata([3]),
            1: _FakeDPMetadata([2]),
        },
    )

    assert events == [
        ("update", (0, 1)),
        ("ids", 0),
        ("ids", 1),
        ("hidden", 0),
        ("hidden", 1),
    ]


def test_dsv4_ffn_runner_clears_ids_cache_after_exception(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hash_layers=3),
    )
    runner.connector = _FakeFFNConnector()
    runner.num_layers = 2
    runner.max_num_tokens = 2

    class FailingModel:
        def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
            del hidden_states, kwargs
            if layer_idx == 1:
                raise RuntimeError("injected FFN failure")
            return "layer-0-output"

    runner.model = FailingModel()
    for layer_idx in range(2):
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=0,
            seq_len=2,
        )
        runner.connector.attn_outputs.append(
            _ffn_payload(
                f"hidden-{layer_idx}",
                metadata,
                input_ids=(
                    torch.tensor([-1, 31], dtype=torch.int32)
                    if layer_idx == 0
                    else None
                ),
            )
        )

    with pytest.raises(RuntimeError, match="injected FFN failure"):
        runner._ffn_forward(dp_metadata_list={0: _FakeDPMetadata([2])})

    assert runner._ffn_input_ids_cache == {}


def test_dsv4_ffn_runner_logs_u1_stage_boundaries(monkeypatch, caplog):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hash_layers=0),
    )
    runner.connector = _FakeFFNConnector()
    runner.connector.stage_diagnostics_enabled = True
    runner.model = _RecordingFakeModel()
    runner.num_layers = 2
    runner.max_num_tokens = 1
    for layer_idx in range(2):
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=0,
            seq_len=1,
        )
        runner.connector.attn_outputs.append(
            _ffn_payload(f"hidden-{layer_idx}", metadata),
        )

    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.npu.ffn_model_runner",
    ):
        runner._ffn_forward(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert "event=recv_begin mode=serial layer=0 stage=0" in caplog.text
    assert "event=compute_begin mode=serial layer=0 stage=0" in caplog.text
    assert "event=send_returned mode=serial layer=1 stage=0" in caplog.text


def test_dsv4_ffn_runner_compute_sync_diagnostics_run_before_send(
    monkeypatch,
    caplog,
):
    _patch_ffn_forward_context(monkeypatch)
    from afd_plugin.v1.worker.npu import ffn_model_runner

    events = []

    class SyncDiagnosticConnector(_FakeFFNConnector):
        def send_ffn_output(self, ffn_output, context, **kwargs):
            events.append(("send", context.metadata.layer_idx))
            return super().send_ffn_output(ffn_output, context, **kwargs)

    class SyncDiagnosticModel:
        def compute_ffn_output(self, hidden_states, layer_idx, **_kwargs):
            events.append(("compute", layer_idx))
            return f"npu-ffn({hidden_states}, layer={layer_idx})"

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hash_layers=0),
    )
    runner.connector = SyncDiagnosticConnector()
    runner.connector.ffn_compute_sync_diagnostics_enabled = True
    runner.connector.is_warmup = False
    runner.model = SyncDiagnosticModel()
    runner.num_layers = 2
    runner.max_num_tokens = 1
    for layer_idx in range(2):
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=0,
            seq_len=1,
        )
        runner.connector.attn_outputs.append(
            _ffn_payload(f"hidden-{layer_idx}", metadata),
        )

    monkeypatch.setattr(
        ffn_model_runner.torch.npu,
        "synchronize",
        lambda: events.append(("sync",)),
    )
    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.npu.ffn_model_runner",
    ):
        runner._ffn_forward(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert events == [
        ("compute", 0),
        ("sync",),
        ("send", 0),
        ("compute", 1),
        ("sync",),
        ("send", 1),
    ]
    assert "event=compute_sync_begin mode=serial layer=0 stage=0" in caplog.text
    assert "event=compute_sync_complete mode=serial layer=1 stage=0" in caplog.text


@pytest.mark.parametrize(
    ("is_graph_capturing", "is_warmup"),
    [(True, False), (False, True)],
)
def test_dsv4_ffn_runner_compute_sync_diagnostics_skip_startup_paths(
    monkeypatch,
    is_graph_capturing,
    is_warmup,
):
    _patch_ffn_forward_context(monkeypatch)
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hash_layers=0),
    )
    runner.connector = _FakeFFNConnector()
    runner.connector.ffn_compute_sync_diagnostics_enabled = True
    runner.connector.is_warmup = is_warmup
    runner.model = _RecordingFakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(_ffn_payload("hidden", metadata))
    sync_calls = []
    monkeypatch.setattr(
        ffn_model_runner.torch.npu,
        "synchronize",
        lambda: sync_calls.append(True),
    )

    runner._ffn_forward(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_graph_capturing=is_graph_capturing,
    )

    assert sync_calls == []


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


def test_npu_ffn_runner_graph_key_uses_exact_attention_peer_token_counts():
    runner = _new_ffn_runner()
    runner.connector = _FakeFFNConnector(attn_size=8, ffn_size=4)
    runner.max_num_tokens = 24

    assert runner._make_graph_key({0: _FakeDPMetadata([12] * 8)}) == (
        "decoder",
        "off",
        0,
        False,
        (0, (12, 12, 12, 12, 12, 12, 12, 12)),
    )


def test_npu_ffn_runner_graph_key_separates_mtp_configuration():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
    )
    runner.connector = _FakeFFNConnector(attn_size=2, ffn_size=2)
    runner.max_num_tokens = 16
    assert runner._make_graph_key({0: _FakeDPMetadata([8, 12])}) == (
        "decoder",
        "mtp",
        1,
        True,
        (0, (8, 12)),
    )


def test_npu_ffn_runner_replays_target_graph_then_runs_mtp_eager(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
    )
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.mtp_ffn_model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 2
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([2])}
    decoder_graph = _FakeGraph()
    runner._acl_graphs = {
        runner._make_graph_key(dp_metadata): {
            "graph": decoder_graph,
        },
    }
    header = SimpleNamespace(
        num_tokens=2,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([2], dtype=torch.int32),
    )
    runner.connector.recv_mtp_header = lambda *, stage_idx: header
    mtp_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=2,
        phase="mtp",
        speculative_step=0,
    )
    runner.connector.attn_outputs.append(("mtp-hidden", mtp_metadata))

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert decoder_graph.replay_count == 1
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(mtp-hidden, layer=0)", mtp_metadata, {"ubatch_idx": 0}),
    ]


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
@pytest.mark.parametrize("control_enabled", [False, True])
def test_npu_ffn_runner_replays_target_then_full_draft_graph(
    num_speculative_tokens,
    control_enabled,
):
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(
            enforce_eager=False,
            num_speculative_tokens=num_speculative_tokens,
        ),
    )
    runner.connector = _FakeFFNConnector(attn_size=2, ffn_size=2)
    runner.connector.mtp_phase_control_enabled = control_enabled
    runner.connector.mtp_phase_graph_replay = control_enabled
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 8
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([3, 5])}
    target_graph = _FakeGraph()
    mtp_graph = _FakeGraph()
    runner._acl_graphs = {
        runner._make_graph_key(dp_metadata): {"graph": target_graph},
    }
    runner._mtp_acl_graphs = {
        runner._make_mtp_graph_key(dp_metadata): {"graph": mtp_graph},
    }

    runner.execute_model(dp_metadata_list=dp_metadata, input_ids_by_stage={})

    assert target_graph.replay_count == 1
    assert mtp_graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_npu_ffn_runner_replays_target_graph_u2_then_runs_one_mtp_phase(
    monkeypatch,
):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
    )
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.mtp_ffn_model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 4
    runner.use_aclgraph = True
    dp_metadata = {
        0: _FakeDPMetadata([2]),
        1: _FakeDPMetadata([2]),
    }
    decoder_graph = _FakeGraph()
    runner._acl_graphs = {
        runner._make_graph_key(dp_metadata): {
            "graph": decoder_graph,
        },
    }
    header = SimpleNamespace(
        num_tokens=4,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([4], dtype=torch.int32),
    )
    received_stages = []

    def recv_mtp_header(*, stage_idx):
        received_stages.append(stage_idx)
        return header

    runner.connector.recv_mtp_header = recv_mtp_header
    mtp_metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=4,
        phase="mtp",
        speculative_step=0,
    )
    runner.connector.attn_outputs.append(("mtp-hidden", mtp_metadata))

    runner.execute_model(
        dp_metadata_list=dp_metadata,
        input_ids_by_stage={},
    )

    assert decoder_graph.replay_count == 1
    assert received_stages == [0]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(mtp-hidden, layer=0)", mtp_metadata, {"ubatch_idx": 0}),
    ]


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


def test_npu_ffn_runner_shutdown_is_idempotent(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    close_calls = []
    profiler_calls = []
    parent_calls = []
    runner.connector = SimpleNamespace(
        is_initialized=True,
        close=lambda: close_calls.append(True),
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "stop_afd_npu_profiler",
        lambda profiler: profiler_calls.append(profiler),
    )
    monkeypatch.setattr(
        ffn_model_runner.NPUModelRunner,
        "shutdown",
        lambda self: parent_calls.append(self),
    )

    runner.shutdown()
    runner.shutdown()

    assert close_calls == [True]
    assert profiler_calls == [None]
    assert parent_calls == [runner]


def test_npu_ffn_worker_scheduler_execute_model_fails_fast():
    worker = _new_ffn_worker()

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_npu_ffn_worker_reports_zero_compilation_times():
    worker = _new_ffn_worker()

    compilation_times = worker.compile_or_warm_up_model()

    assert compilation_times.language_model == 0.0
    assert compilation_times.encoder == 0.0


def test_npu_ffn_worker_stops_on_attention_shutdown_payload(monkeypatch, caplog):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker as ffn_worker_module

    event = threading.Event()
    worker = _new_ffn_worker()
    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="npu")
    control_plane = SimpleNamespace(
        recv_dp_metadata_list=lambda: AFDControlPayload(
            dp_metadata_list={},
            is_graph_capturing=False,
            is_warmup=False,
            shutdown=True,
        )
    )
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(control_plane=control_plane),
    )
    monkeypatch.setattr(ffn_worker_module.torch.npu, "set_device", lambda _device: None)

    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker._run_ffn_server_loop()

    assert event.is_set()
    assert "AFD NPU FFN received Attention shutdown payload" in caplog.text


def test_npu_ffn_worker_preserves_complete_control_payload(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker as ffn_worker_module

    event = threading.Event()
    payload = AFDControlPayload(
        dp_metadata_list={0: _FakeDPMetadata([3, 5])},
        is_graph_capturing=True,
        is_warmup=True,
        tensor_parallel_size=2,
        mtp_phase_control_enabled=True,
    )
    updates = []
    executions = []

    def execute_ffn_step(**kwargs):
        executions.append(kwargs)
        event.set()

    control_plane = SimpleNamespace(
        recv_dp_metadata_list=lambda: payload,
        update_state_from_dp_metadata=updates.append,
    )
    worker = _new_ffn_worker()
    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="npu")
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(control_plane=control_plane),
        execute_ffn_step=execute_ffn_step,
    )
    monkeypatch.setattr(ffn_worker_module.torch.npu, "set_device", lambda _device: None)
    monkeypatch.setattr(ffn_worker_module.torch.npu, "synchronize", lambda: None)

    worker._run_ffn_server_loop()

    assert updates == [payload]
    assert executions == [
        {
            "dp_metadata_list": payload.dp_metadata_list,
            "is_graph_capturing": True,
            "is_warmup": True,
            "connector_state_prepared": True,
        }
    ]


def test_npu_ffn_worker_logs_u1_execute_and_sync_boundaries(monkeypatch, caplog):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker as ffn_worker_module

    event = threading.Event()
    payload = AFDControlPayload(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_graph_capturing=False,
        is_warmup=False,
        target_graph_replay=False,
    )

    def execute_ffn_step(**_kwargs):
        event.set()

    connector = SimpleNamespace(
        control_plane=SimpleNamespace(
            recv_dp_metadata_list=lambda: payload,
            update_state_from_dp_metadata=lambda _payload: None,
        ),
        stage_diagnostics_enabled=True,
    )
    worker = _new_ffn_worker()
    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="npu")
    worker.model_runner = SimpleNamespace(
        connector=connector,
        execute_ffn_step=execute_ffn_step,
    )
    monkeypatch.setattr(ffn_worker_module.torch.npu, "set_device", lambda _device: None)
    monkeypatch.setattr(ffn_worker_module.torch.npu, "synchronize", lambda: None)

    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker._run_ffn_server_loop()

    assert "stages=(0,)" in caplog.text
    assert "target_graph_replay=False" in caplog.text
    for event_name in (
        "execute_begin",
        "execute_returned",
        "device_sync_begin",
        "device_sync_complete",
    ):
        assert f"event={event_name}" in caplog.text


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


def test_npu_ffn_worker_ignores_receive_error_during_shutdown(caplog):
    worker = _new_ffn_worker()
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True),
    )

    def stop_while_receiving():
        worker._ffn_shutdown_event.set()
        raise RuntimeError("CAM recv interrupted by connector close")

    worker._run_ffn_server_loop = stop_while_receiving

    with caplog.at_level(
        logging.ERROR,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker.start_ffn_server_loop()
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

    worker.raise_ffn_loop_error_if_any()
    assert "AFD NPU FFN worker loop failed" not in caplog.text


def test_npu_ffn_worker_recognizes_control_plane_close_only():
    from afd_plugin.connectors import AFDControlPlaneClosedError
    from afd_plugin.v1.worker.npu.ffn_worker import (
        _is_attention_control_plane_shutdown,
    )

    assert _is_attention_control_plane_shutdown(
        AFDControlPlaneClosedError("peer closed")
    )
    assert not _is_attention_control_plane_shutdown(ValueError("bad payload"))


def test_npu_ffn_worker_stops_loop_before_parent_shutdown(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    worker._ffn_shutdown_event = threading.Event()
    calls = []

    class _StoppingThread:
        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return False

    connector = SimpleNamespace(close=lambda: calls.append(("close", None)))
    worker._ffn_thread = _StoppingThread()
    worker.model_runner = SimpleNamespace(connector=connector)
    monkeypatch.setattr(
        ffn_worker.NPUWorker,
        "shutdown",
        lambda self: calls.append(("parent", None)),
    )

    worker.shutdown()

    assert calls == [
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("parent", None),
    ]


def test_npu_ffn_worker_preserves_live_thread_after_shutdown_timeout(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    shutdown_event = threading.Event()
    worker._ffn_shutdown_event = shutdown_event
    calls = []

    class _StoppingThread:
        alive = True

        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return self.alive

    thread = _StoppingThread()
    worker._ffn_thread = thread
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(close=lambda: calls.append(("close", None))),
    )
    monkeypatch.setattr(
        ffn_worker.NPUWorker,
        "shutdown",
        lambda self: calls.append(("parent", None)),
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        worker.shutdown()

    assert worker._ffn_thread is thread
    assert worker._ffn_shutdown_event is shutdown_event
    assert shutdown_event.is_set()
    assert ("parent", None) not in calls

    thread.alive = False
    worker.shutdown()

    assert worker._ffn_thread is None
    assert worker._ffn_shutdown_event is None
    assert calls == [
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("parent", None),
    ]


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


def _dsv4_config(**kwargs):
    parallel_defaults = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "use_sequence_parallel_moe": False,
    }
    parallel_defaults.update(kwargs)
    config = _vllm_config(
        architecture="DeepseekV4ForCausalLM",
        **parallel_defaults,
    )
    config.additional_config["afd"].update(
        num_attention_ranks=1,
        num_ffn_ranks=1,
    )
    return config


def _mtp_speculative_config(**overrides):
    values = {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "enforce_eager": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dspark_speculative_config(*, block_size=4, **overrides):
    values = {
        "method": "mtp",
        "num_speculative_tokens": block_size,
        "enforce_eager": True,
        "draft_model_config": SimpleNamespace(
            hf_config=SimpleNamespace(dspark_block_size=block_size),
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _mooncake_pd_config(**overrides):
    values = {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_consumer",
        "engine_id": "dsv4-afd-decode",
        "kv_port": 30100,
        "kv_parallel_size": 1,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 2, "tp_size": 4},
            "decode": {"dp_size": 1, "tp_size": 1},
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dsv4_feature_validation_accepts_eager_u1_camp2p():
    fail_if_unsupported_npu_afd_features(_dsv4_config())


def test_dsv4_feature_validation_accepts_eager_u2_camp2p():
    fail_if_unsupported_npu_afd_features(
        _dsv4_config(
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
            ubatch_size=2,
        )
    )


@pytest.mark.parametrize("use_ubatching", [False, True])
def test_dsv4_feature_validation_accepts_eager_hccl_p2p(use_ubatching):
    config = _dsv4_config(
        enable_dbo=use_ubatching,
        use_ubatching=use_ubatching,
        num_ubatches=2 if use_ubatching else 1,
        ubatch_size=2,
    )
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_eager_hccl_p2p_a2f1():
    config = _dsv4_config()
    config.additional_config["afd"].update(
        connector="P2pHcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=1,
    )

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("enforce_eager", "use_ubatching", "with_mtp"),
    [
        (True, False, False),
        (True, True, False),
        (False, False, False),
        (False, True, False),
        (True, False, True),
        (True, True, True),
        (False, False, True),
        (False, True, True),
    ],
)
def test_dsv4_feature_validation_accepts_mooncake_pd_combinations(
    enforce_eager,
    use_ubatching,
    with_mtp,
):
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        data_parallel_size=1,
        enable_dbo=use_ubatching,
        use_ubatching=use_ubatching,
        num_ubatches=2 if use_ubatching else 1,
        ubatch_size=2,
        kv_transfer_config=_mooncake_pd_config(),
    )
    config.model_config.enforce_eager = enforce_eager
    if with_mtp:
        config.speculative_config = _mtp_speculative_config(enforce_eager=True)
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("cudagraph_mode", "enforce_eager"),
    [("FULL", True), ("FULL_DECODE_ONLY", False)],
)
def test_dsv4_feature_validation_accepts_equal_hccl_tp2(
    cudagraph_mode,
    enforce_eager,
):
    config = _dsv4_config(
        cudagraph_mode=cudagraph_mode,
        tensor_parallel_size=2,
        data_parallel_size=1,
    )
    config.model_config.enforce_eager = enforce_eager
    config.additional_config["afd"].update(
        connector="P2pHcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
    )

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.additional_config["afd"].update(
                connector="CAMP2pAFDConnector"
            ),
            "TP2 supports only P2pHcclAFDConnector",
        ),
        (
            lambda config: config.additional_config["afd"].update(num_ffn_ranks=1),
            "requires equal Attention and FFN ranks",
        ),
        (
            lambda config: setattr(config.parallel_config, "data_parallel_size", 2),
            "ranks to equal DP x TP",
        ),
    ],
)
def test_dsv4_feature_validation_rejects_unvalidated_tp2_topology(
    mutation,
    message,
):
    config = _dsv4_config(tensor_parallel_size=2, data_parallel_size=1)
    config.additional_config["afd"].update(
        connector="P2pHcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
    )
    mutation(config)

    with pytest.raises(RuntimeError, match=message):
        fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_hccl_p2p_full_decode_only_u2():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=2,
    )
    config.model_config.enforce_eager = False
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
def test_dsv4_feature_validation_accepts_mtp_m1_eager_u1_hccl_p2p(
    num_speculative_tokens,
):
    config = _dsv4_config(
        speculative_config=_mtp_speculative_config(
            num_speculative_tokens=num_speculative_tokens,
        )
    )
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_mtp_m3_eager_u2_hccl_p2p():
    config = _dsv4_config(
        speculative_config=_mtp_speculative_config(),
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=2,
    )
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_matching_dspark_block_size():
    config = _dsv4_config(
        speculative_config=_dspark_speculative_config(block_size=4),
    )
    config.model_config.hf_config.num_nextn_predict_layers = 2
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: setattr(
                config.speculative_config,
                "num_speculative_tokens",
                3,
            ),
            "num_speculative_tokens to match dspark_block_size",
        ),
        (
            lambda config: setattr(
                config.speculative_config,
                "enforce_eager",
                False,
            ),
            "DSpark eager execution requires draft enforce_eager=true",
        ),
    ],
)
def test_dsv4_feature_validation_rejects_invalid_dspark_modes(mutation, message):
    config = _dsv4_config(
        speculative_config=_dspark_speculative_config(block_size=4),
    )
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"
    mutation(config)

    with pytest.raises(RuntimeError, match=message):
        fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_graph_target_with_eager_mtp_draft():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
    )
    config.model_config.enforce_eager = False
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_full_draft_mtp_graph():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=_mtp_speculative_config(enforce_eager=False),
    )
    config.model_config.enforce_eager = False
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_hccl_p2p_graph_a2f1():
    config = _dsv4_config(cudagraph_mode="FULL_DECODE_ONLY")
    config.model_config.enforce_eager = False
    config.additional_config["afd"].update(
        connector="P2pHcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=1,
    )

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.additional_config["afd"].update(
                connector="P2pNcclAFDConnector"
            ),
            "only CAMP2pAFDConnector or P2pHcclAFDConnector",
        ),
        (
            lambda config: config.additional_config["afd"].update(num_ffn_ranks=2),
            "equal Attention and FFN",
        ),
        (
            lambda config: setattr(config.parallel_config, "tensor_parallel_size", 3),
            "tensor_parallel_size=1 or 2",
        ),
        (
            lambda config: setattr(config.parallel_config, "pipeline_parallel_size", 2),
            "pipeline_parallel_size=1",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "prefill_context_parallel_size", 2
            ),
            "prefill context parallel",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "decode_context_parallel_size", 2
            ),
            "decode context parallel",
        ),
        (
            lambda config: setattr(
                config.parallel_config, "use_sequence_parallel_moe", True
            ),
            "sequence-parallel MoE",
        ),
        (
            lambda config: config.additional_config["afd"].update(
                compute_gate_on_attention=True
            ),
            "FFN-side gate",
        ),
        (
            lambda config: setattr(config.model_config, "enforce_eager", False),
            "FULL_DECODE_ONLY",
        ),
        (
            lambda config: setattr(config, "speculative_config", object()),
            "MTP supports only P2pHcclAFDConnector",
        ),
    ],
)
def test_dsv4_feature_validation_rejects_unvalidated_modes(mutation, message):
    config = _dsv4_config()
    mutation(config)

    with pytest.raises(RuntimeError, match=message):
        fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_graph_mtp_a2f1():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=_mtp_speculative_config(enforce_eager=True),
    )
    config.model_config.enforce_eager = False
    config.additional_config["afd"].update(
        connector="P2pHcclAFDConnector",
        num_attention_ranks=2,
        num_ffn_ranks=1,
    )

    fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_accepts_mtp_target_graph_u2():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=_mtp_speculative_config(),
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=2,
    )
    config.model_config.enforce_eager = False
    config.additional_config["afd"]["connector"] = "P2pHcclAFDConnector"

    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("cudagraph_mode", ["NONE", "FULL", "PIECEWISE"])
def test_dsv4_feature_validation_rejects_other_graph_modes(cudagraph_mode):
    config = _dsv4_config(cudagraph_mode=cudagraph_mode)
    config.model_config.enforce_eager = False

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        fail_if_unsupported_npu_afd_features(config)


def test_dsv4_feature_validation_rejects_camp2p_graph_u2():
    config = _dsv4_config(
        cudagraph_mode="FULL_DECODE_ONLY",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=2,
    )
    config.model_config.enforce_eager = False

    with pytest.raises(RuntimeError, match="graph U2 supports only P2pHccl"):
        fail_if_unsupported_npu_afd_features(config)


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


def test_npu_ubatch_output_merge_preserves_aux_hidden_states():
    _require_npu_runtime()
    import torch

    from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import _cat_ubatch_outputs

    merged = _cat_ubatch_outputs(
        [
            (torch.tensor([[1.0]]), [torch.tensor([[2.0]])]),
            (torch.tensor([[3.0]]), [torch.tensor([[4.0]])]),
        ],
    )

    assert isinstance(merged, tuple)
    assert merged[0].tolist() == [[1.0], [3.0]]
    assert len(merged[1]) == 1
    assert merged[1][0].tolist() == [[2.0], [4.0]]


def test_npu_ubatch_all_gather_preserves_aux_outputs_and_trims_padding(
    monkeypatch,
):
    _require_npu_runtime()
    import torch

    from afd_plugin.v1.worker.npu import npu_ubatch_wrapper

    gathered_inputs = []

    def fake_all_gather(output, dim):
        assert dim == 0
        gathered_inputs.append(output.clone())
        return torch.cat((output, output + 10), dim=0)

    monkeypatch.setattr(
        npu_ubatch_wrapper,
        "tensor_model_parallel_all_gather",
        fake_all_gather,
    )
    output = (
        torch.tensor([[1.0], [2.0]]),
        [
            torch.tensor([[3.0], [4.0]]),
            torch.tensor([[5.0], [6.0]]),
        ],
    )

    gathered = npu_ubatch_wrapper._all_gather_ubatch_output(output, pad_size=1)

    assert isinstance(gathered, tuple)
    assert gathered[0].tolist() == [[1.0], [2.0], [11.0]]
    assert [tensor.tolist() for tensor in gathered[1]] == [
        [[3.0], [4.0], [13.0]],
        [[5.0], [6.0], [15.0]],
    ]
    assert len(gathered_inputs) == 3


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


def test_npu_async_feature_validation_rejects_ubatching():
    with pytest.raises(RuntimeError, match="ubatching"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                use_ubatching=True,
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


def test_npu_async_moe_ubatching_validation_requires_supported_shape():
    fail_if_unsupported_npu_afd_features(
        _vllm_config(
            connector="CAMAsyncAFDConnector",
            async_dp=True,
            compute_gate_on_attention=True,
            extra_config={
                "async_moe_ubatching": True,
            },
        ),
    )

    with pytest.raises(RuntimeError, match="compute_gate_on_attention"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                extra_config={"async_moe_ubatching": True},
            ),
        )

    with pytest.raises(RuntimeError, match="exactly two"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                compute_gate_on_attention=True,
                extra_config={
                    "async_moe_ubatching": True,
                    "async_moe_num_ubatches": 3,
                },
            ),
        )

    with pytest.raises(RuntimeError, match="request-boundary"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                compute_gate_on_attention=True,
                extra_config={
                    "async_moe_ubatching": True,
                    "async_moe_split": "token",
                },
            ),
        )

    with pytest.raises(RuntimeError, match="decode context parallel"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                compute_gate_on_attention=True,
                decode_context_parallel_size=2,
                extra_config={
                    "async_moe_ubatching": True,
                },
            ),
        )


def test_npu_ubatch_enabled_when_thresholds_are_met(monkeypatch):
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

    module_name = "afd_plugin.v1.worker.npu.ubatch_utils"
    original_module = sys.modules.pop(module_name, None)
    try:
        ubatch_utils = importlib.import_module(module_name)
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
        )
        assert ubatch_utils.check_enable_ubatch(
            num_tokens_unpadded=12,
            num_tokens_padded=12,
            uniform_decode=True,
            vllm_config=config,
        )
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
