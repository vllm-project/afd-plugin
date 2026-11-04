# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Wire protocol of the sync Attention-side gate shell.

The shell's design claim is an ordering: send the routing, run the shared experts
while the FFN side works, then receive and merge. These tests assert that ordering
and the payload, using a fake connector. They prove the Python protocol; they do not
prove that the shared-expert work actually overlaps the remote work on a device,
which needs an NPU timeline.

Like the other model-layer tests, this needs vLLM and is skipped on a CPU-only host.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# The adapter module needs this specific vLLM module, and a CPU-only host can have a
# partial vLLM install where `import vllm` succeeds but this one does not.
pytest.importorskip("vllm.forward_context")

# The NPU adapter; the CUDA module of the same name has no sync gate shell.
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402

TOP_K = 2
HIDDEN = 4
LAYER_IDX = 3


class _RecordingConnector:
    """Records the wire calls and the shared-expert run in one ordered list."""

    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ubatch_idx))
        return ref_tensor * 0.5


def _fake_gate_topk(monkeypatch, events: list[tuple], topk_ids, topk_weights):
    """Replace the gate helper with one that records and returns fixed routing.

    The forward does ``from package import deepseek_v4_attention_gate``. Two states
    are possible: the submodule was already imported once, so the package carries a
    real attribute, or it was not, so the import falls back to ``sys.modules``.
    Setting both covers either one; ``raising=False`` because the attribute may not
    exist yet.
    """

    import afd_plugin.model_executor.models.npu as npu_package

    helper = ModuleType("afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate")

    def compute_attention_gate_topk(moe, hidden_states, *, input_ids=None,
                                    hidden_states_fp32=None):
        del moe, input_ids, hidden_states_fp32
        events.append(("gate", int(hidden_states.shape[0])))
        return topk_weights, topk_ids

    helper.compute_attention_gate_topk = compute_attention_gate_topk
    monkeypatch.setitem(
        sys.modules,
        "afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate",
        helper,
    )
    monkeypatch.setattr(
        npu_package,
        "deepseek_v4_attention_gate",
        helper,
        raising=False,
    )


def _shell(*, events, connector, shared_experts):
    """Build the shell without its weight construction.

    The forward only needs the layer index and the shared-expert module, so the
    linear layers the constructor would build are irrelevant here.

    ``shared_experts`` is a presence flag: True installs a recorder-backed module,
    False leaves the slot empty.
    """

    del connector  # the connector arrives through the forward metadata
    shell = object.__new__(adapter.AFDDeepseekV4SyncAttentionGateRemoteMoE)
    torch.nn.Module.__init__(shell)
    shell.layer_idx = LAYER_IDX
    shell.shared_experts = None
    if shared_experts:

        class _Shared(torch.nn.Module):
            def forward(self, hidden_states):
                events.append(("shared", int(hidden_states.shape[0])))
                return torch.full_like(hidden_states, 2.0)

        shell.shared_experts = _Shared()
    return shell


def _install_metadata(monkeypatch, connector, events):
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=7)
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    return afd_metadata


def test_forward_sends_routing_then_runs_shared_then_receives(monkeypatch):
    """The shared experts must run between the send and the receive."""

    events: list[tuple] = []
    connector = _RecordingConnector(events)
    _install_metadata(monkeypatch, connector, events)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.full((2, TOP_K), 0.5, dtype=torch.float32)
    _fake_gate_topk(monkeypatch, events, topk_ids, topk_weights)
    shell = _shell(events=events, connector=connector, shared_experts=True)
    hidden_states = torch.full((2, HIDDEN), 8.0, dtype=torch.bfloat16)

    output = shell(hidden_states)

    assert [event[0] for event in events] == ["gate", "send", "shared", "recv"]
    # The routing travels on the wire slot, not as token ids.
    sent = events[1]
    assert sent[3]["routing"][0] is topk_ids
    assert sent[3]["routing"][1] is topk_weights
    # The context identifies the layer and the stage the routing belongs to.
    assert sent[2].metadata.layer_idx == LAYER_IDX
    assert sent[2].metadata.seq_lens == [2]
    assert events[3][1] == 7  # recv uses the metadata stage index
    # routed (0.5 * input) plus shared (2.0).
    assert torch.equal(output, hidden_states * 0.5 + 2.0)


def test_forward_without_shared_experts_returns_routed_only(monkeypatch):
    """With no shared experts the output is the received routed tensor alone."""

    events: list[tuple] = []
    connector = _RecordingConnector(events)
    _install_metadata(monkeypatch, connector, events)
    topk_ids = torch.zeros((2, TOP_K), dtype=torch.int32)
    topk_weights = torch.ones((2, TOP_K), dtype=torch.float32)
    _fake_gate_topk(monkeypatch, events, topk_ids, topk_weights)
    shell = _shell(events=events, connector=connector, shared_experts=False)
    hidden_states = torch.full((2, HIDDEN), 8.0, dtype=torch.bfloat16)

    output = shell(hidden_states)

    assert [event[0] for event in events] == ["gate", "send", "recv"]
    assert torch.equal(output, hidden_states * 0.5)


def test_forward_refuses_a_context_without_afd_metadata(monkeypatch):
    """A missing forward metadata is an error, not a silent activations-only send."""

    events: list[tuple] = []
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )
    _fake_gate_topk(
        monkeypatch,
        events,
        torch.zeros((2, TOP_K), dtype=torch.int32),
        torch.ones((2, TOP_K), dtype=torch.float32),
    )
    shell = _shell(events=events, connector=_RecordingConnector(events), shared_experts=False)

    with pytest.raises(RuntimeError, match="AFD forward metadata"):
        shell(torch.zeros(2, HIDDEN, dtype=torch.bfloat16))
