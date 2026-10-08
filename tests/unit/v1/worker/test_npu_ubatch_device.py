# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")

ubatching = importlib.import_module("afd_plugin.v1.worker.npu.ubatching")
ubatch_utils = importlib.import_module("afd_plugin.v1.worker.npu.ubatch_utils")


def test_split_partial_request_preserves_seq_len_bounds(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        ubatch_utils,
        "AscendCommonAttentionMetadata",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    query_start = torch.tensor([0, 3, 8], dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc_cpu=query_start,
        query_start_loc=query_start,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        _seq_lens_cpu=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        max_seq_len=10,
        max_query_len=5,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(8, dtype=torch.int32),
        causal=True,
        actual_seq_lengths_q=[],
        positions=torch.arange(8, dtype=torch.int32),
        positions_cpu=None,
        attn_state=None,
        graph_pad_size=0,
        decode_token_per_req=1,
        dcp_local_seq_lens=None,
        dcp_local_seq_lens_cpu=None,
        is_prefilling=None,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
        context_parallel_metadata=None,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
        encoder_seq_lens=None,
        encoder_seq_lens_cpu=None,
        logits_indices_padded=None,
        num_logits_indices=0,
    )
    slices = ubatch_utils.create_ubatch_slices(np.array([3, 5], dtype=np.int32), [5])

    first, second = ubatch_utils.split_attn_metadata(slices, metadata)

    assert first.seq_lens_cpu_upper_bound.tolist() == [3, 2]
    assert first._seq_lens_cpu.tolist() == [3, 2]
    assert first.max_seq_len == 10
    assert second.seq_lens_cpu_upper_bound.tolist() == [5]
    assert second.query_start_loc_cpu.tolist() == [0, 3]
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [3, 5]

    metadata.seq_lens_cpu_upper_bound = torch.tensor([4, 7], dtype=torch.int32)
    metadata.max_seq_len = 1
    first_with_upper_bound, _ = ubatch_utils.split_attn_metadata(slices, metadata)
    assert first_with_upper_bound.seq_lens_cpu_upper_bound.tolist() == [4, 4]
    assert first_with_upper_bound._seq_lens_cpu.tolist() == [3, 2]
    assert first_with_upper_bound.max_seq_len == 4
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [4, 7]


@pytest.mark.parametrize("device_index", [0, 1])
@pytest.mark.parametrize("same_stream", [False, True])
def test_new_thread_binds_device_before_barrier_and_stream_query(
    monkeypatch, device_index, same_stream
):
    device = torch.device("npu", device_index)
    compute_stream = SimpleNamespace(device=device, stream_id=1)
    default_stream = (
        compute_stream if same_stream else SimpleNamespace(device=device, stream_id=0)
    )
    calls = []
    local = threading.local()

    def set_device(target):
        local.device = target
        calls.append(("device", target))

    def current_stream():
        calls.append(("query", local.device))
        assert local.device == device
        return default_stream

    def set_stream(stream):
        assert local.device == stream.device
        calls.append(("stream", stream))

    def ready():
        # Graph capture starts after this barrier, so device setup must precede it.
        assert calls == [("device", device)]
        calls.append(("ready", device))

    monkeypatch.setattr(torch.npu, "set_device", set_device)
    monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    monkeypatch.setattr(torch.npu, "set_stream", set_stream)
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [])
    monkeypatch.setattr(ubatching, "_DBO_CURRENT_STREAM", threading.local())
    monkeypatch.setattr(ubatching.forward_context, "_forward_context", None)
    contexts = ubatching.make_ubatch_contexts(
        2, compute_stream, [object(), object()], SimpleNamespace(wait=ready)
    )
    context = contexts[0]
    context.cpu_wait_event.set()

    def run():
        local.device = torch.device("npu", 0)
        with context:
            assert ubatching.dbo_current_stream() is compute_stream
            assert ubatching.forward_context._forward_context is context.forward_context
            assert ubatching.dbo_enabled()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run).result(timeout=5)

    expected = [("device", device), ("ready", device), ("query", device)]
    if not same_stream:
        expected.append(("stream", compute_stream))
    assert calls == expected
    assert context.cpu_signal_event.is_set()
    assert not context.cpu_wait_event.is_set()
    assert not ubatching.dbo_enabled()
    assert ubatching._CURRENT_CONTEXTS == [None, None]
