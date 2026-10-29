# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dependency-free policy checks for the scoped GPU DP1 DBO seams."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_PATH = REPO_ROOT / "afd_plugin/compat/patches/gpu_mrv2_ubatching.py"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/attention_model_runner_v2.py"


class _CUDAGraphMode(Enum):
    NONE = 0
    FULL = 1


@dataclass(frozen=True)
class _BatchDescriptor:
    cg_mode: _CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


def _vllm_config(
    *,
    speculative=False,
    inner_state=False,
    mla=False,
    pcp_size=1,
    dcp_size=1,
    parallel=None,
):
    parallel = SimpleNamespace(
        **{
            **vars(parallel or _parallel()),
            "prefill_context_parallel_size": pcp_size,
            "decode_context_parallel_size": dcp_size,
        }
    )
    return SimpleNamespace(
        speculative_config=object() if speculative else None,
        model_config=SimpleNamespace(has_inner_state=inner_state, use_mla=mla),
        parallel_config=parallel,
    )


UNSUPPORTED_SPLIT_CONFIGS: dict[str, dict[str, bool | int]] = {
    "speculative": {"speculative": True},
    "inner-state": {"inner_state": True},
    "mla": {"mla": True},
    "pcp": {"pcp_size": 2},
    "dcp": {"dcp_size": 2},
}


class _GraphManager:
    def __init__(self, capture_tokens: int | None = None, vllm_config=None):
        self.capture_tokens = capture_tokens
        self.vllm_config = vllm_config or _vllm_config()
        self.calls: list[int] = []

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        *,
        num_active_loras=0,
        max_query_len=None,
        num_ubatches=1,
    ):
        self.calls.append(num_ubatches)
        graph_matches = (
            self.capture_tokens is not None
            and num_tokens <= self.capture_tokens
            and uniform_token_count == 1
        )
        return _BatchDescriptor(
            cg_mode=_CUDAGraphMode.FULL if graph_matches else _CUDAGraphMode.NONE,
            num_tokens=self.capture_tokens if graph_matches else num_tokens,
            num_reqs=self.capture_tokens if graph_matches else num_reqs,
            uniform_token_count=uniform_token_count,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )


def _parallel(*, dp_size=1, use_ubatching=True):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=0,
        tensor_parallel_size=2,
        use_ubatching=use_ubatching,
        num_ubatches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )


@dataclass(frozen=True)
class _UBatchSlice:
    request_slice: slice
    token_slice: slice


def _bounds(slices):
    return [
        (
            s.request_slice.start,
            s.request_slice.stop,
            s.token_slice.start,
            s.token_slice.stop,
        )
        for s in slices
    ]


@dataclass(frozen=True)
class _DecodeBatch:
    """The InputBatch fields prepare reads for a one-token-per-request batch."""

    num_reqs: int
    num_tokens: int
    num_reqs_after_padding: int
    num_tokens_after_padding: int
    max_query_len: int | None = None


def _native_slices(input_batch, num_ubatches):
    # Native vLLM 0.30 slicing clamps a trailing microbatch without real
    # requests onto the last real request; only that outcome matters here.
    split = input_batch.num_tokens_after_padding // num_ubatches
    last_start = min(split, input_batch.num_reqs - 1)
    return [
        _UBatchSlice(slice(0, split), slice(0, split)),
        _UBatchSlice(
            slice(last_start, input_batch.num_reqs_after_padding),
            slice(split, input_batch.num_tokens_after_padding),
        ),
    ]


def _slice_decode_batch(input_batch, ubatch_slice, *args, **kwargs):
    start, stop = ubatch_slice.request_slice.start, ubatch_slice.request_slice.stop
    return SimpleNamespace(
        num_reqs=max(0, min(stop, input_batch.num_reqs) - start),
        max_query_len=input_batch.max_query_len,
        seq_lens_cpu_upper_bound=[0] * (stop - start),
        is_padding=ubatch_slice.token_slice,
    )


class _ModelState:
    def prepare_attn(self, ubatch, cg_mode, block_tables, slot_mappings, *args, **kw):
        return {
            "max_query_len": ubatch.max_query_len,
            "max_seq_len": max(ubatch.seq_lens_cpu_upper_bound),
            "block_tables": block_tables,
        }


class _SlotMappings:
    def __getitem__(self, key):
        return key[1]


@pytest.fixture
def patches():
    # Execute the patch's functions with small native API doubles. This keeps
    # the real eligibility/control policy under test without importing CUDA.
    tree = ast.parse(PATCH_PATH.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id.isupper()
                for target in node.targets
            )
        )
    ]

    class NativeUBatchRunner:
        def __init__(
            self,
            vllm_config,
            device,
            model_state,
            attn_groups,
            kv_cache_config,
            max_reqs,
        ):
            self.vllm_config = vllm_config
            self.parallel_config = vllm_config.parallel_config
            self.inputs = (device, model_state, attn_groups, kv_cache_config, max_reqs)

        def _make_forward_contexts(self, *args):
            raise AssertionError("native DP1 metadata factory is ineligible")

    @dataclass
    class NativeDPMetadata:
        num_tokens_across_dp_cpu: list[int]
        local_sizes: list[int] | None = None

        @staticmethod
        def make(*args):
            raise AssertionError("DPMetadata.make rejects DP1 without SP")

    def native_factory(*args):
        return None

    def native_dispatch(*args, **kwargs):
        raise AssertionError("dispatch was not scoped")

    def check_thresholds(config, num_tokens, uniform_decode):
        threshold = (
            config.dbo_decode_token_threshold
            if uniform_decode
            else config.dbo_prefill_token_threshold
        )
        return config.use_ubatching and num_tokens >= threshold

    native_module = SimpleNamespace(
        maybe_build_ubatch_runner=native_factory,
        dispatch_cg_and_sync_dp=native_dispatch,
    )
    namespace = {
        "contextmanager": contextmanager,
        "replace": replace,
        "MethodType": MethodType,
        "CUDAGraphMode": _CUDAGraphMode,
        "BatchExecutionDescriptor": _BatchDescriptor,
        "UBatchRunner": NativeUBatchRunner,
        "DPMetadata": NativeDPMetadata,
        "model_runner": native_module,
        "logger": SimpleNamespace(info_once=lambda *args: None),
        "torch": SimpleNamespace(
            int32="int32",
            full=lambda shape, value, **kwargs: [value] * shape[0],
            full_like=lambda values, value: [value] * len(values),
        ),
        "create_forward_context": lambda attn, config, **kwargs: SimpleNamespace(
            attn_metadata=attn, vllm_config=config, **kwargs
        ),
        "check_ubatch_thresholds": check_thresholds,
        "get_num_ubatches": lambda config: (
            config.num_ubatches if config.use_ubatching else 1
        ),
        "pairwise": pairwise,
        "UBatchSlice": _UBatchSlice,
        "UBatchState": SimpleNamespace,
        "create_ubatch_slices": _native_slices,
        "_slice_input_batch": _slice_decode_batch,
        "build_slot_mappings_by_layer": lambda slot_mappings, kv_cache_config: (
            slot_mappings
        ),
    }
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + functions, type_ignores=[])
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace, namespace=namespace)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_factory_keeps_native_runner_and_changes_only_dp1_instance(patches, dp_size):
    parallel = _parallel(dp_size=dp_size)
    config = _vllm_config(parallel=parallel)
    native_context_factory = patches.UBatchRunner._make_forward_contexts
    inputs = ("device", "model-state", [["attn-group"]], "kv-cache", 32)

    runner = patches.maybe_build_ubatch_runner(config, *inputs)

    assert type(runner) is patches.UBatchRunner
    assert runner.inputs == inputs
    assert patches.UBatchRunner._make_forward_contexts is native_context_factory
    assert ("_make_forward_contexts" in vars(runner)) is (dp_size == 1)


@pytest.mark.parametrize("unsupported", [None, *UNSUPPORTED_SPLIT_CONFIGS])
def test_factory_binds_captured_split_prepare_only_where_supported(
    patches, unsupported
):
    overrides = UNSUPPORTED_SPLIT_CONFIGS.get(unsupported, {})
    config = _vllm_config(parallel=_parallel(dp_size=2), **overrides)

    runner = patches.maybe_build_ubatch_runner(config, None, None, [], None, 32)

    assert ("prepare" in vars(runner)) is (unsupported is None)


def test_disabled_dbo_does_not_construct_runner(patches):
    config = _vllm_config(parallel=_parallel(use_ubatching=False))
    assert patches.maybe_build_ubatch_runner(config, None, None, [], None, 32) is None


def test_dp1_native_child_metadata_keeps_stage_counts_and_context_fields(patches):
    config = _vllm_config(parallel=_parallel())
    runner = patches.maybe_build_ubatch_runner(config, None, None, [], None, 32)
    slices = [SimpleNamespace(num_tokens=3), SimpleNamespace(num_tokens=5)]
    attention = [{"layer": object()}, {"layer": object()}]
    slot_mappings = [{"layer": object()}, {"layer": object()}]
    padding = [object(), object()]

    contexts = runner._make_forward_contexts(slices, attention, slot_mappings, padding)

    assert [context.dp_metadata.num_tokens_across_dp_cpu for context in contexts] == [
        [3],
        [5],
    ]
    for index, context in enumerate(contexts):
        assert type(context.dp_metadata) is patches.DPMetadata
        assert context.dp_metadata.local_sizes is None
        assert context.attn_metadata is attention[index]
        assert context.slot_mapping is slot_mappings[index]
        assert context.is_padding is padding[index]
        assert context.vllm_config is config
        assert context.cudagraph_runtime_mode is _CUDAGraphMode.NONE


@pytest.mark.parametrize(
    "num_tokens, uniform_decode, allow_ubatching, expected_ubatches",
    [
        (0, True, True, 1),
        (1, True, True, 1),
        (2, True, True, 2),
        (11, False, True, 1),
        (12, False, True, 2),
        (12, False, False, 1),
    ],
)
def test_dp1_thresholds_select_eager_microbatches_without_handshake(
    patches, num_tokens, uniform_decode, allow_ubatching, expected_ubatches
):
    desc, sync = patches.dispatch_cg_and_sync_dp(
        None,
        1,
        num_tokens,
        1 if uniform_decode else None,
        1,
        0,
        need_eager=True,
        parallel_config=_parallel(),
        allow_ubatching=allow_ubatching,
        uniform_decode=uniform_decode,
    )
    assert sync is None
    assert desc.cg_mode is _CUDAGraphMode.NONE
    assert desc.num_tokens == num_tokens
    assert desc.num_ubatches == expected_ubatches


@pytest.mark.parametrize(
    "num_tokens, capture_tokens, split_supported, expected",
    [
        # (cg_mode, padded tokens, microbatches)
        (9, 16, False, (_CUDAGraphMode.FULL, 16, 2)),
        # At most half of the graph: the captured split replays it ...
        (3, 16, True, (_CUDAGraphMode.FULL, 16, 2)),
        (8, 16, True, (_CUDAGraphMode.FULL, 16, 2)),
        # ... and without it the regular graph does, never eager.
        (3, 16, False, (_CUDAGraphMode.FULL, 16, 1)),
        (8, 16, False, (_CUDAGraphMode.FULL, 16, 1)),
        (8, None, True, (_CUDAGraphMode.NONE, 8, 2)),
    ],
)
def test_dp1_replays_a_graph_for_an_underfilled_microbatched_step(
    patches, num_tokens, capture_tokens, split_supported, expected
):
    vllm_config = _vllm_config() if split_supported else _vllm_config(mla=True)
    manager = _GraphManager(capture_tokens, vllm_config)
    desc, sync = patches.dispatch_cg_and_sync_dp(
        manager,
        num_tokens,
        num_tokens,
        1,
        1,
        0,
        parallel_config=_parallel(),
        allow_ubatching=True,
        uniform_decode=True,
    )
    assert sync is None
    assert (desc.cg_mode, desc.num_tokens, desc.num_ubatches) == expected


@pytest.mark.parametrize(
    "graph_tokens, expected",
    [
        (8, [(0, 4, 0, 4), (4, 8, 4, 8)]),
        (32, [(0, 16, 0, 16), (16, 32, 16, 32)]),
        (96, [(0, 48, 0, 48), (48, 96, 48, 96)]),
    ],
)
def test_captured_split_halves_requests_and_tokens_alike(
    patches, graph_tokens, expected
):
    batch = _DecodeBatch(3, 3, graph_tokens, graph_tokens)
    assert _bounds(patches.captured_ubatch_slices(batch, 2)) == expected


@pytest.mark.parametrize(
    "unsupported, min_tokens, max_tokens, capture_tokens, falls_back",
    [
        (None, 15, 17, 32, False),
        ("mla", 15, 17, 32, True),
        ("mla", 16, 17, 32, True),
        ("mla", 17, 17, 32, False),
        ("speculative", 2, 3, 8, True),
        ("mla", 15, 17, None, False),
    ],
)
def test_regular_graph_fallback_only_without_the_captured_split(
    patches, unsupported, min_tokens, max_tokens, capture_tokens, falls_back
):
    config = _vllm_config(**UNSUPPORTED_SPLIT_CONFIGS.get(unsupported, {}))
    manager = _GraphManager(capture_tokens, config)
    assert (
        patches.needs_regular_graph_fallback(
            manager, max_tokens, min_tokens, max_tokens, 1, 0, 2
        )
        is falls_back
    )


def _prepare(patches, input_batch, cg_mode):
    runner = SimpleNamespace(
        num_ubatches=2,
        ubatch_query_start_loc=[None, None],
        ubatch_seq_lens=[None, None],
        ubatch_dcp_local_seq_lens=[None, None],
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        model_state=_ModelState(),
        attn_groups=[],
        kv_cache_config=None,
        _make_forward_contexts=lambda slices, attn, slot_mappings, padding: attn,
    )
    block_tables = (list(range(input_batch.num_reqs_after_padding)),)
    return patches.prepare(runner, input_batch, block_tables, _SlotMappings(), cg_mode)


def test_underfilled_full_decode_replays_the_captured_split(patches):
    # 15 real requests padded into a 32 graph: native slicing would start the
    # trailing microbatch at request 14 instead of the captured 16.
    state = _prepare(patches, _DecodeBatch(15, 15, 32, 32), _CUDAGraphMode.FULL)

    assert _bounds(state.slices) == [(0, 16, 0, 16), (16, 32, 16, 32)]
    trailing = state.forward_contexts[1]
    assert trailing["block_tables"] == (list(range(16, 32)),)
    assert trailing["max_query_len"] == 1
    assert trailing["max_seq_len"] == 1


@pytest.mark.parametrize("num_reqs", [17, 24, 32])
def test_filled_full_decode_split_matches_native_slicing(patches, num_reqs):
    input_batch = _DecodeBatch(num_reqs, num_reqs, 32, 32)
    state = _prepare(patches, input_batch, _CUDAGraphMode.FULL)
    assert state.slices == _native_slices(input_batch, 2)


@pytest.mark.parametrize(
    "input_batch, cg_mode",
    [
        (_DecodeBatch(15, 15, 32, 32), _CUDAGraphMode.NONE),
        (_DecodeBatch(15, 20, 32, 32), _CUDAGraphMode.FULL),
    ],
    ids=["eager", "multi-token-request"],
)
def test_other_microbatched_steps_keep_native_slicing(patches, input_batch, cg_mode):
    state = _prepare(patches, input_batch, cg_mode)

    assert state.slices == _native_slices(input_batch, 2)
    assert [context["max_query_len"] for context in state.forward_contexts] == [
        None,
        None,
    ]


def test_profile_dispatch_does_not_select_a_graph(patches):
    manager = _GraphManager(16)
    desc, _ = patches.dispatch_cg_and_sync_dp(
        manager,
        8,
        8,
        1,
        1,
        0,
        need_eager=True,
        parallel_config=_parallel(),
        allow_ubatching=True,
        uniform_decode=True,
    )
    assert desc.cg_mode is _CUDAGraphMode.NONE
    assert desc.num_ubatches == 2
    assert manager.calls == []


def test_dp2_keeps_native_coordination_and_all_dispatch_arguments(patches):
    calls = []
    expected = (object(), object())

    def sync(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    patches.namespace["sync_cudagraph_and_dp_padding"] = sync
    manager = _GraphManager()
    parallel = _parallel(dp_size=2)
    actual = patches.dispatch_cg_and_sync_dp(
        manager,
        3,
        12,
        None,
        2,
        1,
        max_query_len=6,
        num_active_loras=0,
        parallel_config=parallel,
        allow_ubatching=True,
        uniform_decode=False,
    )
    assert actual is expected
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is manager
    assert args[2:] == (12, 3, None, 2, 1)
    assert kwargs == {
        "max_query_len": 6,
        "num_active_loras": 0,
        "parallel_config": parallel,
        "allow_ubatching": True,
        "uniform_decode": False,
    }
    assert manager.calls == [1]


@pytest.mark.parametrize(
    "scope_name, symbol",
    [
        ("use_gpu_mrv2_ubatch_factory", "maybe_build_ubatch_runner"),
        ("use_gpu_mrv2_dbo_dispatch", "dispatch_cg_and_sync_dp"),
    ],
)
def test_scopes_restore_native_alias_after_exception_and_nested_call(
    patches, scope_name, symbol
):
    scope = getattr(patches, scope_name)
    original = getattr(patches.model_runner, symbol)
    replacement = getattr(patches, symbol)
    with pytest.raises(RuntimeError, match="native failure"), scope():
        assert getattr(patches.model_runner, symbol) is replacement
        with scope():
            assert getattr(patches.model_runner, symbol) is replacement
        assert getattr(patches.model_runner, symbol) is replacement
        raise RuntimeError("native failure")
    assert getattr(patches.model_runner, symbol) is original


@pytest.mark.parametrize("fail", [False, True])
def test_attention_kv_initialization_scopes_factory_and_preserves_native_arguments(
    patches, fail
):
    calls = []
    original = patches.model_runner.maybe_build_ubatch_runner

    class NativeRunner:
        def initialize_kv_cache(self, *args, **kwargs):
            assert (
                patches.model_runner.maybe_build_ubatch_runner
                is patches.maybe_build_ubatch_runner
            )
            calls.append((args, kwargs))
            if fail:
                raise RuntimeError("KV allocation failed")

    tree = ast.parse(RUNNER_PATH.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AFDAttentionModelRunnerV2"
    )
    initialize = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache"
    )
    runner.bases = [ast.Name(id="NativeRunner", ctx=ast.Load())]
    runner.body = [initialize]
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + [runner], type_ignores=[])
    )
    namespace = {**patches.namespace, "NativeRunner": NativeRunner}
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    attention_runner = namespace["AFDAttentionModelRunnerV2"]()
    kv_config, allocation_context = object(), object()
    if fail:
        with pytest.raises(RuntimeError, match="KV allocation failed"):
            attention_runner.initialize_kv_cache(kv_config, True, allocation_context)
    else:
        attention_runner.initialize_kv_cache(kv_config, True, allocation_context)
    assert calls == [
        (
            (kv_config,),
            {
                "is_profiling": True,
                "kv_cache_allocation_context": allocation_context,
            },
        )
    ]
    assert patches.model_runner.maybe_build_ubatch_runner is original


@pytest.fixture
def native_patch():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from afd_plugin.compat.patches import gpu_mrv2_ubatching

    return gpu_mrv2_ubatching


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("graph_tokens", [8, 16, 32, 64, 96, 128])
def test_captured_split_is_native_slicing_once_real_requests_reach_it(
    native_patch, graph_tokens
):
    import numpy as np
    from vllm.v1.worker.gpu.ubatch_utils import create_ubatch_slices

    for num_reqs in range(1, graph_tokens + 1):
        input_batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
            num_tokens_after_padding=graph_tokens,
            num_reqs_after_padding=graph_tokens,
        )
        native = _bounds(create_ubatch_slices(input_batch, 2))
        captured = _bounds(native_patch.captured_ubatch_slices(input_batch, 2))
        assert (native == captured) is (2 * num_reqs > graph_tokens), num_reqs


class _NativeGraphManager:
    CAPTURE_SIZES = (8, 16, 32, 64)

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        *,
        num_active_loras=0,
        max_query_len=None,
        num_ubatches=1,
    ):
        from vllm.config import CUDAGraphMode
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        size = next(size for size in self.CAPTURE_SIZES if size >= num_tokens)
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=size,
            num_reqs=size,
            uniform_token_count=uniform_token_count,
            num_ubatches=num_ubatches,
        )


@pytest.mark.vllm_runtime
@pytest.mark.parametrize(
    "tokens_across_dp, split_supported, expected",
    [
        # (padded tokens, microbatches); every case replays a FULL graph.
        ((15, 17), True, (32, 2)),
        ((15, 17), False, (32, 1)),
        ((16, 32), True, (32, 2)),
        ((16, 32), False, (32, 1)),
        ((16, 16), False, (16, 2)),
    ],
)
def test_dp2_underfilled_rank_replays_a_graph_on_every_rank(
    native_patch, monkeypatch, tokens_across_dp, split_supported, expected
):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    def all_reduce(tensor, group=None):
        # Peers vote like the calling rank except for their own batch size.
        local_vote = tensor.sum(dim=1)
        for rank, tokens in enumerate(tokens_across_dp):
            tensor[:, rank] = local_vote
            tensor[0, rank] = tokens
            tensor[5, rank] = tokens

    monkeypatch.setattr(native_patch.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        native_patch, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    vllm_config = _vllm_config() if split_supported else _vllm_config(mla=True)
    manager = _NativeGraphManager(vllm_config)

    results = []
    for rank, tokens in enumerate(tokens_across_dp):
        desc, sync = native_patch.sync_cudagraph_and_dp_padding(
            manager,
            BatchExecutionDescriptor(CUDAGraphMode.FULL, tokens, tokens, 1),
            tokens,
            tokens,
            1,
            len(tokens_across_dp),
            rank,
            parallel_config=_parallel(dp_size=len(tokens_across_dp)),
            allow_ubatching=True,
            uniform_decode=True,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert sync is not None and not sync.eager
        assert torch.equal(
            sync.num_tokens_across_dp, torch.full((2,), expected[0], dtype=torch.int32)
        )
        results.append((desc.num_tokens, desc.num_ubatches))
    assert results == [expected] * len(tokens_across_dp)
