# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the async GPU connector's wire format and routing math."""

from __future__ import annotations

import inspect
from functools import partial
from itertools import pairwise
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from afd_plugin.connectors.factory import AFDConnectorFactory  # noqa: E402
from afd_plugin.connectors.gpu.async_gpu import (  # noqa: E402
    FLAG_REPLY_READY,
    GpuAsyncAFDConnector,
    GpuAsyncExtraInfo,
    GpuAsyncTransferState,
    _PendingDispatch,
    plan_dispatch,
)
from afd_plugin.connectors.gpu.symm_window import (  # noqa: E402
    HEADER_FIXED_WORDS,
    HEADER_HOST_WORDS,
    SlotLayout,
    SymmWindow,
    decode_header,
)
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDTransferContext,
    AFDTransferMetadata,
)


@pytest.fixture
def layout() -> SlotLayout:
    # shared_cap is deliberately not token_cap: the shared field is sized by the
    # per-rank split, so a layout that quietly reused token_cap for it would
    # otherwise still satisfy every assertion below.
    return SlotLayout.build(
        expert_per_rank=4,
        partial_cap=100,
        token_cap=32,
        shared_cap=8,
        hidden_size=8,
        payload_itemsize=2,
    )


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_connector_is_registered_and_has_no_control_plane():
    connector_cls = AFDConnectorFactory.get_connector_class("GpuAsyncAFDConnector")
    assert connector_cls is GpuAsyncAFDConnector
    assert connector_cls.control_plane is None


def test_ring_depth_defaults_to_the_number_of_live_stages():
    assert GpuAsyncExtraInfo.from_mapping(None).ring_depth == 1
    assert GpuAsyncExtraInfo.from_mapping({"async_moe_ubatching": True}).ring_depth == 2
    assert GpuAsyncExtraInfo.from_mapping({"ring_depth": 4}).ring_depth == 4


def test_unknown_extra_config_field_is_rejected():
    with pytest.raises(ValueError, match="unknown AFD async GPU"):
        GpuAsyncExtraInfo.from_mapping({"nope": 1})


@pytest.mark.parametrize("ffn_size", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("num_tokens", [0, 1, 5, 7, 64, 513])
def test_shared_split_tiles_the_batch_within_its_capacity(
    ffn_size: int,
    num_tokens: int,
):
    # _shared_slice only reads these two attributes, so the bound can be checked
    # without a device or a vLLM config behind it.
    connector = SimpleNamespace(has_shared_experts=True, ffn_size=ffn_size)
    slices = [
        GpuAsyncAFDConnector._shared_slice(connector, rank, num_tokens)
        for rank in range(ffn_size)
    ]

    # Every shared token is computed exactly once: the slices tile [0, n).
    assert slices[0].start == 0
    assert slices[-1].stop == num_tokens
    for earlier, later in pairwise(slices):
        assert earlier.stop == later.start

    # And none of them can overflow the field the slot reserves for them, which
    # is what lets shared_cap be a fraction of the batch rather than all of it.
    shared_cap = -(-num_tokens // ffn_size)
    assert max(s.stop - s.start for s in slices) <= shared_cap


def test_shared_split_is_empty_without_shared_experts():
    connector = SimpleNamespace(has_shared_experts=False, ffn_size=4)
    for rank in range(4):
        assert GpuAsyncAFDConnector._shared_slice(connector, rank, 64) == slice(0, 0)


def test_shutdown_announcement_matches_the_window_write_signature(layout: SlotLayout):
    # Nothing calls announce_shutdown yet, so no runtime path would notice it
    # passing a keyword write_slot does not take. Binding the arguments it
    # actually sends against the real signature is what catches that.
    calls: list[dict] = []
    window = SimpleNamespace(write_slot=lambda **kwargs: calls.append(kwargs))
    connector = SimpleNamespace(
        _require_initialized=lambda: window,
        attn_size=2,
        ffn_size=3,
        is_attention=True,
        # A device tensor in the real thing; the CPU one behaves the same here
        # and keeps the flag stamp on the same code path as a dispatch.
        _seq_device=torch.zeros((), dtype=torch.int32),
        layout=layout,
        role_rank=1,
        topk=6,
        expert_per_rank=layout.header_words - HEADER_FIXED_WORDS,
    )

    GpuAsyncAFDConnector.announce_shutdown(connector)

    # One message per opposite-role peer, each carrying the shutdown bit.
    assert len(calls) == 3
    signature = inspect.signature(SymmWindow.write_slot)
    for kwargs in calls:
        signature.bind(window, **kwargs)
        assert decode_header(kwargs["header"]).is_shutdown


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------

_NUM_TOKENS = 7
_TOPK = 3
_FFN_SIZE = 2
_EXPERT_PER_RANK = 4
_HIDDEN = 5


@pytest.fixture
def routing_inputs():
    generator = torch.Generator().manual_seed(0)
    num_experts = _FFN_SIZE * _EXPERT_PER_RANK
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, generator=generator)[:_TOPK]
            for _ in range(_NUM_TOKENS)
        ],
    ).to(torch.int32)
    hidden_states = torch.randn(_NUM_TOKENS, _HIDDEN, generator=generator)
    topk_weights = torch.rand(_NUM_TOKENS, _TOPK, generator=generator)
    return topk_ids, hidden_states, topk_weights


def _destination_slices(plan, ffn_size, expert_per_rank):
    """Walk the plan the way an FFN rank does: one run of partials each.

    A destination is told where its run starts and how long it is, and reads the
    index arrays the sender shipped whole -- there is no per-destination slicing
    on the send side any more, which is what removed the readback.
    """
    routed = plan.routed_per_rank.tolist()
    starts = plan.segment_start.tolist()
    for ffn_rank in range(ffn_size):
        yield ffn_rank, slice(starts[ffn_rank], starts[ffn_rank] + routed[ffn_rank])


def test_every_partial_is_routed_exactly_once(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    assert plan.expand_idx.shape == (_NUM_TOKENS * _TOPK,)
    assert plan.weights.shape == (_NUM_TOKENS * _TOPK,)
    assert int(plan.counts.sum()) == _NUM_TOKENS * _TOPK
    assert int(plan.routed_per_rank.sum()) == _NUM_TOKENS * _TOPK
    # The runs must tile the array end to end, or a partial is read twice or not
    # at all: they are the only thing a destination gets to locate itself by.
    cursor = 0
    for _, partials in _destination_slices(plan, _FFN_SIZE, _EXPERT_PER_RANK):
        assert partials.start == cursor
        cursor = partials.stop
    assert cursor == _NUM_TOKENS * _TOPK


def test_each_destination_segment_is_grouped_by_local_expert(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    counts = plan.counts.tolist()
    for ffn_rank, partials in _destination_slices(
        plan,
        _FFN_SIZE,
        _EXPERT_PER_RANK,
    ):
        base = ffn_rank * _EXPERT_PER_RANK
        # The token behind each partial, in the order the receiver sees them.
        tokens = plan.expand_idx[partials]
        cursor = 0
        for local_expert in range(_EXPERT_PER_RANK):
            for _ in range(counts[base + local_expert]):
                token_idx = int(tokens[cursor])
                assert base + local_expert in topk_ids[token_idx].tolist()
                cursor += 1
        assert cursor == partials.stop - partials.start


def test_every_partial_names_a_token_of_this_batch(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    # Destinations read the whole batch out of the slot, so an index is only in
    # range if it names a row of it.
    assert int(plan.expand_idx.min()) >= 0
    assert int(plan.expand_idx.max()) < _NUM_TOKENS
    for ffn_rank, partials in _destination_slices(plan, _FFN_SIZE, _EXPERT_PER_RANK):
        base = ffn_rank * _EXPERT_PER_RANK
        expected = {
            token
            for token in range(_NUM_TOKENS)
            for expert in topk_ids[token].tolist()
            if base <= expert < base + _EXPERT_PER_RANK
        }
        assert set(plan.expand_idx[partials].tolist()) == expected


def test_identity_experts_recombine_to_the_weighted_sum(routing_inputs):
    """The full chain: ship the batch, expand, weight, reduce, add."""
    topk_ids, hidden_states, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    accumulator = torch.zeros(_NUM_TOKENS, _HIDDEN, dtype=torch.float32)
    for _, partials in _destination_slices(plan, _FFN_SIZE, _EXPERT_PER_RANK):
        expand = plan.expand_idx[partials].to(torch.int64)
        # What the FFN side does with the batch it was sent.
        expanded = hidden_states.index_select(0, expand)
        reduced = torch.zeros(_NUM_TOKENS, _HIDDEN, dtype=torch.float32)
        reduced.index_add_(0, expand, expanded * plan.weights[partials].unsqueeze(1))
        # A reply is a whole batch, so combine adds it without an index.
        accumulator += reduced
    expected = hidden_states * topk_weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(accumulator, expected.to(torch.float32))


def test_routing_handles_experts_not_divisible_by_ffn_size():
    # expert_per_rank is a ceiling division, so the padded tail must stay empty
    # instead of silently absorbing real partials.
    ffn_size, expert_per_rank, num_experts = 3, 2, 5
    topk_ids = torch.tensor([[0, 4], [1, 3], [2, 4]], dtype=torch.int32)
    plan = plan_dispatch(
        topk_ids,
        torch.ones(3, 2),
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    assert plan.counts.numel() == ffn_size * expert_per_rank
    assert int(plan.counts.sum()) == topk_ids.numel()
    assert int(plan.counts[num_experts:].sum()) == 0


def test_routing_can_leave_one_destination_empty():
    """A single decode token's topk can land entirely on one FFN rank.

    The peer that gets nothing must still see a well-formed, empty segment --
    zero-length windows are what crashed a 2A2F decode step.
    """
    ffn_size, expert_per_rank = 2, 4
    # Every partial targets experts owned by FFN rank 0.
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    plan = plan_dispatch(
        topk_ids,
        torch.ones(1, 3),
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    slices = list(_destination_slices(plan, ffn_size, expert_per_rank))

    _, busy_partials = slices[0]
    assert busy_partials.stop - busy_partials.start == 3
    # One token, three of its partials: it is sent once, read three times.
    assert plan.expand_idx[busy_partials].tolist() == [0, 0, 0]

    _, empty_partials = slices[1]
    assert empty_partials.stop - empty_partials.start == 0

    # Reducing an empty destination must be a no-op, not an error.
    accumulator = torch.zeros(1, 4, dtype=torch.float32)
    empty = plan.expand_idx[empty_partials].to(torch.int64)
    accumulator.index_add_(0, empty, torch.zeros(0, 4))
    assert torch.count_nonzero(accumulator) == 0


# ----------------------------------------------------------------------
# Flag protocol
#
# The Attention side runs inside a CUDA graph, which records kernels once and
# replays them without running any Python. Two properties keep that correct and
# neither is visible from the eager path alone, so they are pinned here:
# a dispatch's sequence number must come off a device tensor the graph advances,
# and a reply must be waited for by a constant that the same graph resets.
# ----------------------------------------------------------------------


class _RecordingWindow:
    """Window stand-in that records the flag traffic in the order it is issued."""

    def __init__(self, hidden_size: int, payload_dtype: torch.dtype) -> None:
        self.hidden_size = hidden_size
        self.payload_dtype = payload_dtype
        self.calls: list[tuple] = []

    def write_slot(self, **kwargs) -> None:
        self.calls.append(
            ("write_slot", kwargs["peer"], kwargs["flag_value"], kwargs["header"]),
        )

    def stream_wait(self, region: int, ring: int, value) -> None:
        self.calls.append(("stream_wait", region, ring, value))

    def clear_flag(self, region: int, ring: int) -> None:
        self.calls.append(("clear_flag", region, ring))

    def local_routed(self, region: int, ring: int, count: int) -> torch.Tensor:
        self.calls.append(("local_routed", region, ring))
        return torch.zeros(count, self.hidden_size, dtype=self.payload_dtype)

    def local_shared(self, region: int, ring: int, count: int) -> torch.Tensor:
        self.calls.append(("local_shared", region, ring))
        return torch.zeros(count, self.hidden_size, dtype=self.payload_dtype)


def _combining_connector(window, *, ffn_size: int, has_shared_experts: bool):
    """Minimal stand-in carrying only what recv_ffn_output reads."""
    return SimpleNamespace(
        _require_initialized=lambda: window,
        _pending={},
        _free_rings={},
        has_shared_experts=has_shared_experts,
        hidden_size=window.hidden_size,
        payload_dtype=window.payload_dtype,
        ffn_size=ffn_size,
        role_rank=0,
    )


def _pending(*, ffn_size: int, num_tokens: int, ring: int) -> _PendingDispatch:
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=num_tokens,
    )
    return _PendingDispatch(
        context=AFDTransferContext(metadata=metadata),
        shared_slices=[
            slice(r * num_tokens // ffn_size, (r + 1) * num_tokens // ffn_size)
            for r in range(ffn_size)
        ],
        num_tokens=num_tokens,
        ring=ring,
        expected_ffn=list(range(ffn_size)),
    )


def test_combine_waits_on_the_constant_reply_marker():
    # A per-dispatch number here would be baked into the captured graph, and
    # every later replay would find the flag already at or above it.
    window = _RecordingWindow(hidden_size=8, payload_dtype=torch.float32)
    connector = _combining_connector(window, ffn_size=2, has_shared_experts=False)
    connector._pending[0] = [_pending(ffn_size=2, num_tokens=4, ring=1)]

    GpuAsyncAFDConnector.recv_ffn_output(
        connector,
        ref_tensor=torch.zeros(4, 8),
        ubatch_idx=0,
    )

    waits = [call for call in window.calls if call[0] == "stream_wait"]
    assert waits == [
        ("stream_wait", 0, 1, FLAG_REPLY_READY),
        ("stream_wait", 1, 1, FLAG_REPLY_READY),
    ]


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_combine_clears_each_flag_only_after_it_has_read_the_slot(has_shared_experts):
    # The reset re-arms the slot for the next dispatch. Issued before the reads
    # it would race the peer's next reply; left out it would let the following
    # replay fall straight through a flag that is still raised.
    window = _RecordingWindow(hidden_size=8, payload_dtype=torch.float32)
    connector = _combining_connector(
        window,
        ffn_size=2,
        has_shared_experts=has_shared_experts,
    )
    connector._pending[0] = [_pending(ffn_size=2, num_tokens=4, ring=0)]

    GpuAsyncAFDConnector.recv_ffn_output(
        connector,
        ref_tensor=torch.zeros(4, 8),
        ubatch_idx=0,
    )

    for ffn_rank in range(2):
        own = [call for call in window.calls if call[1] == ffn_rank]
        assert own[0][0] == "stream_wait"
        assert own[-1] == ("clear_flag", ffn_rank, 0)
        assert "local_routed" in {call[0] for call in own}
        if has_shared_experts:
            assert "local_shared" in {call[0] for call in own}


def test_combine_releases_the_ring_it_consumed():
    window = _RecordingWindow(hidden_size=8, payload_dtype=torch.float32)
    connector = _combining_connector(window, ffn_size=1, has_shared_experts=False)
    connector._pending[0] = [_pending(ffn_size=1, num_tokens=4, ring=3)]

    GpuAsyncAFDConnector.recv_ffn_output(
        connector,
        ref_tensor=torch.zeros(4, 8),
        ubatch_idx=0,
    )

    assert connector._free_rings[0] == [3]


def _replying_connector(window):
    layout = SlotLayout.build(
        expert_per_rank=2,
        partial_cap=16,
        token_cap=4,
        shared_cap=0,
        hidden_size=8,
        payload_itemsize=4,
    )
    return SimpleNamespace(
        _require_initialized=lambda: window,
        layout=layout,
        role_rank=1,
        topk=2,
        hidden_size=8,
        payload_dtype=torch.float32,
    )


def test_reply_stamps_the_constant_marker_not_a_sequence_number():
    window = _RecordingWindow(hidden_size=8, payload_dtype=torch.float32)
    connector = _replying_connector(window)
    states = GpuAsyncTransferState(
        region=0,
        ring=2,
        src_role_rank=0,
        layer_idx=0,
        stage_idx=0,
        num_tokens=4,
        routed_tokens=2,
        shared_tokens=0,
        expand_idx=torch.tensor([0, 1]),
        weights=torch.ones(2),
    )

    GpuAsyncAFDConnector.send_ffn_output(
        connector,
        torch.zeros(2, 8),
        AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_lens=[2],
            ),
            states=states,
        ),
    )

    assert [call[:3] for call in window.calls] == [
        ("write_slot", 0, FLAG_REPLY_READY),
    ]


def test_reply_carries_no_header():
    # The Attention rank knows a reply's shape before it exists -- that is the
    # premise of waiting on a stream instead of polling -- so it never reads
    # one. Writing a header here was a copy per peer per MoE layer that nothing
    # consumed.
    window = _RecordingWindow(hidden_size=8, payload_dtype=torch.float32)
    connector = _replying_connector(window)
    states = GpuAsyncTransferState(
        region=0,
        ring=2,
        src_role_rank=0,
        layer_idx=0,
        stage_idx=0,
        num_tokens=4,
        routed_tokens=2,
        shared_tokens=0,
        expand_idx=torch.tensor([0, 1]),
        weights=torch.ones(2),
    )

    GpuAsyncAFDConnector.send_ffn_output(
        connector,
        torch.zeros(2, 8),
        AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_lens=[2],
            ),
            states=states,
        ),
    )

    assert [call[3] for call in window.calls] == [None]
    # write_slot has to accept that, not just tolerate it by luck.
    assert (
        inspect.signature(SymmWindow.write_slot).parameters["header"].annotation
        == "torch.Tensor | None"
    )


# ----------------------------------------------------------------------
# Dispatch header assembly
#
# The prefix is constant for a dispatch shape, so it belongs off the layer
# path: one buffer per shape, written once. Rebuilding it per dispatch cost a
# strided host-to-device copy every MoE layer, and inside a captured graph it
# cost a copy node that re-shipped that constant on every replay.
# ----------------------------------------------------------------------


def _dispatch_plan(ffn_size: int, expert_per_rank: int, fill: int):
    experts = ffn_size * expert_per_rank
    return SimpleNamespace(
        counts=torch.full((experts,), fill, dtype=torch.int32),
        routed_per_rank=torch.full(
            (ffn_size,), fill * expert_per_rank, dtype=torch.int32
        ),
        segment_start=torch.arange(ffn_size, dtype=torch.int32),
        expand_idx=torch.zeros(1, dtype=torch.int32),
        weights=torch.zeros(1, dtype=torch.float32),
    )


def _dispatching_connector(layout: SlotLayout, *, ffn_size: int):
    connector = SimpleNamespace(
        window=SimpleNamespace(device=torch.device("cpu")),
        layout=layout,
        ffn_size=ffn_size,
        expert_per_rank=layout.header_words - HEADER_FIXED_WORDS,
        _header_device={},
    )
    # _headers_for reaches back through self for the per-shape buffer.
    connector._headers_for_shape = partial(
        GpuAsyncAFDConnector._headers_for_shape,
        connector,
    )
    return connector


def test_dispatch_headers_are_built_once_per_shape(layout: SlotLayout):
    connector = _dispatching_connector(layout, ffn_size=2)
    first = GpuAsyncAFDConnector._headers_for_shape(
        connector, layer_idx=3, num_tokens=16
    )
    again = GpuAsyncAFDConnector._headers_for_shape(
        connector, layer_idx=3, num_tokens=16
    )
    other_layer = GpuAsyncAFDConnector._headers_for_shape(
        connector, layer_idx=4, num_tokens=16
    )
    other_size = GpuAsyncAFDConnector._headers_for_shape(
        connector, layer_idx=3, num_tokens=32
    )

    assert again is first, "same shape must reuse the buffer, not rebuild it"
    assert other_layer is not first
    assert other_size is not first
    assert len(connector._header_device) == 3

    decoded = decode_header(first[0].cpu())
    assert decoded.layer_idx == 3
    assert decoded.num_tokens == 16
    assert not decoded.is_shutdown


def test_dispatch_writes_only_the_routing_tail(layout: SlotLayout):
    # The point of the per-shape buffer: a dispatch must leave the prefix
    # alone. A sentinel there survives if -- and only if -- nothing rewrites it.
    connector = _dispatching_connector(layout, ffn_size=2)
    expert_per_rank = connector.expert_per_rank
    headers = GpuAsyncAFDConnector._headers_for_shape(
        connector, layer_idx=1, num_tokens=8
    )
    sentinel = torch.arange(HEADER_HOST_WORDS, dtype=torch.int32)
    headers[:, :HEADER_HOST_WORDS] = sentinel

    for fill in (1, 2):
        out = GpuAsyncAFDConnector._headers_for(
            connector,
            layer_idx=1,
            num_tokens=8,
            plan=_dispatch_plan(2, expert_per_rank, fill),
        )
        assert out is headers
        assert torch.equal(out[0, :HEADER_HOST_WORDS], sentinel)
        assert torch.equal(
            out[:, HEADER_FIXED_WORDS:],
            torch.full((2, expert_per_rank), fill, dtype=torch.int32),
        )


def test_a_shape_first_seen_during_capture_is_refused(layout: SlotLayout, monkeypatch):
    # Allocating there would come from the graph's private pool and the prefix
    # copy would be recorded against a host buffer freed before the first
    # replay -- silent corruption. Warmup runs the shapes capture runs.
    connector = _dispatching_connector(layout, ffn_size=2)
    connector.window = SimpleNamespace(device=torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        GpuAsyncAFDConnector._headers_for_shape(connector, layer_idx=0, num_tokens=8)


# ----------------------------------------------------------------------
# Ring allocation across stages
#
# A ring names a window slot. Two stages sharing one means the second dispatch
# overwrites the first's payload and flag, and the reply the first waits for
# never arrives -- which deadlocked both DBO ubatch threads, one inside
# recv_ffn_output and one waiting to be yielded to.
# ----------------------------------------------------------------------


def _stage_config(*, use_ubatching, num_ubatches=2, extra=None):
    afd_raw = {
        "connector": "GpuAsyncAFDConnector",
        "role": "attention",
        "num_attention_ranks": 1,
        "num_ffn_ranks": 1,
        "compute_gate_on_attention": True,
        "connector_extra_config": extra or {},
    }
    return SimpleNamespace(
        additional_config={"afd": afd_raw},
        parallel_config=SimpleNamespace(
            use_ubatching=use_ubatching,
            num_ubatches=num_ubatches,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=0,
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )


def _connector_for(config):
    from afd_plugin.config import afd_config_from_mapping

    return GpuAsyncAFDConnector(
        rank=0,
        local_rank=0,
        vllm_config=config,
        afd_config=afd_config_from_mapping(
            config.additional_config["afd"],
            validate=False,
        ),
        role_rank=0,
    )


def test_vllm_ubatching_gets_a_ring_per_stage():
    # DBO drives two forwards at once and stamps each with its ubatch index,
    # which arrives here as the stage. Counting only this connector's own
    # splitter left both on ring 0.
    connector = _connector_for(_stage_config(use_ubatching=True))

    assert connector.num_stages == 2
    assert connector.ring_depth >= 2
    assert set(connector._rings_for_stage(0)).isdisjoint(
        connector._rings_for_stage(1),
    )


def test_a_single_stage_still_needs_only_one_ring():
    connector = _connector_for(_stage_config(use_ubatching=False))

    assert connector.num_stages == 1
    assert connector.ring_depth == 1


def test_a_pinned_ring_depth_too_small_for_the_stages_is_refused():
    # Silently growing it would overrule a deliberate choice; deadlocking on it
    # is worse. Fail at construction with the arithmetic in the message.
    with pytest.raises(ValueError, match="cannot serve"):
        _connector_for(
            _stage_config(use_ubatching=True, extra={"ring_depth": 1}),
        )


def test_header_cache_survives_being_first_filled_in_inference_mode():
    """A cached header must stay writable after the call that created it.

    The cache outlives its creating call, and the first dispatch for a shape
    can land inside vLLM's inference-mode forward. A tensor allocated there is
    an inference tensor, and the next dispatch's write to the routing tail
    raises "Inplace update to inference tensor outside InferenceMode" -- which
    is what a compiled prefill hit, because AOT compilation moves the first
    touch of each shape inside the compiled region.
    """
    with torch.inference_mode():
        headers = torch.empty((2, 8), dtype=torch.int32)
    assert headers.is_inference(), "guard premise: this is what we must avoid"

    with torch.inference_mode(), torch.inference_mode(False):
        safe = torch.empty((2, 8), dtype=torch.int32)
    assert not safe.is_inference()
    # Writable afterwards, which is all the dispatch path needs.
    safe[:, 0] = 1
    assert int(safe[0, 0]) == 1
