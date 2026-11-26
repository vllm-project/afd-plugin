# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-role MoE runners built through vLLM's native FusedMoEFactory."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer  # noqa: E402
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)
from vllm.v1.worker import ubatching  # noqa: E402

from afd_plugin.config import DEEPEP_M2N_CONNECTOR  # noqa: E402
from afd_plugin.connectors import AFDConnectorFactory  # noqa: E402
from afd_plugin.model_executor.ffn import remote  # noqa: E402
from tests.unit.fakes.connector import (  # noqa: E402
    ProxyConnector,
    install_proxy_context,
)
from tests.unit.fakes.model_layers import GateLinear  # noqa: E402
from tests.unit.fakes.vllm_config import attention_moe_vllm_config  # noqa: E402

LAYER_IDX = 3
PREFIX = f"model.layers.{LAYER_IDX}.mlp.experts"
HIDDEN_SIZE = 7
NUM_EXPERTS = 4
TOP_K = 2
INTERMEDIATE_SIZE = 11
# ``ProxyConnector.recv_ffn_output`` returns the reference rows times this.
FFN_OUTPUT_SCALE = 0.25
STAGE_IDX = 1
# The in-place combine below writes this into the receive buffer.
COMBINED_VALUE = 1.0
SHARED_EXPERTS_SCALE = 3.0
MICROBATCH_HANDOFF_TIMEOUT_S = 30
# FusedMoEFactory arguments the AFD factory fixes for a weightless runner.
RESERVED_FACTORY_ARGUMENTS = (
    "quant_config",
    "shared_expert_gate",
    "routed_input_transform",
    "routed_output_transform",
    "n_shared_experts",
    "fuse_shared_experts",
    "apply_routed_scale_to_output",
    "enable_eplb",
    "num_redundant_experts",
    "is_sequence_parallel",
    "tp_size",
    "dp_size",
    "pcp_size",
    "runner_cls",
    "runner_args",
    "routed_experts_cls",
    "routed_experts_args",
)


def _build_runner(monkeypatch, config, *, device_type="cuda", gate=None, **kwargs):
    monkeypatch.setattr(
        remote, "current_platform", SimpleNamespace(device_type=device_type)
    )
    with set_current_vllm_config(config):
        return remote.build_expert(
            config,
            gate=gate,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            prefix=PREFIX,
            **kwargs,
        )


class _RegisteredRunner(remote.AFDRemoteMoERunner):
    """A runner a test connector registers; it marks its output."""

    def forward(
        self,
        hidden_states,
        router_logits,
        input_ids=None,
        shared_experts_input=None,
    ):
        return hidden_states + 1


class _InPlaceCombineConnector(ProxyConnector):
    """Combine writes the routed output into the buffer it is given, as DeepEP
    M2N does."""

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        # Record a copy: combine may be given the sent buffer.
        super().send_attn_output(hidden_states.clone(), context, **kwargs)

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor.fill_(COMBINED_VALUE)


class _ScaledSharedExperts(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * SHARED_EXPERTS_SCALE


def _gate():
    return GateLinear(HIDDEN_SIZE, NUM_EXPERTS, out_dtype=torch.float32)


def _hidden_states(num_tokens: int):
    return torch.arange(num_tokens * HIDDEN_SIZE, dtype=torch.float32).view(
        num_tokens, HIDDEN_SIZE
    )


@pytest.mark.parametrize(
    "compute_gate_on_attention", [False, True], ids=["ffn-gate", "attention-gate"]
)
def test_factory_registers_native_runner_without_expert_weights(
    monkeypatch, compute_gate_on_attention
):
    """The runner registers like a native MoE layer and holds no expert weights.

    Why: the Attention role must allocate no expert memory; the only
    parameters its MoE may hold are those of the model's own gate.
    """
    config = attention_moe_vllm_config(
        compute_gate_on_attention=compute_gate_on_attention
    )
    gate = _gate() if compute_gate_on_attention else None

    runner = _build_runner(monkeypatch, config, gate=gate)

    assert config.compilation_config.static_forward_context == {PREFIX: runner}
    assert config.compilation_config.static_all_moe_layers == [PREFIX]
    gate_parameters = [] if gate is None else list(gate.parameters())
    assert [id(p) for p in runner.parameters()] == [id(p) for p in gate_parameters]
    assert list(runner.buffers()) == []


def test_runner_stays_weightless_and_fails_on_expert_checkpoints(monkeypatch):
    """Post-load processing keeps the runner weightless, and an expert weight
    that reaches it fails the load.

    Why: Attention role filtering must keep FFN expert weights away; a leaked
    weight must fail loudly instead of being dropped.
    """
    runner = _build_runner(monkeypatch, attention_moe_vllm_config())
    model_config = SimpleNamespace(
        quantization=None, word_embeddings_untied_by_checkpoint=False
    )

    process_weights_after_loading(runner, model_config, torch.device("cpu"))

    assert list(runner.parameters()) == []
    expert_weight = ("0.gate_proj.weight", torch.ones(INTERMEDIATE_SIZE, HIDDEN_SIZE))
    with pytest.raises(AttributeError, match="w13_weight"):
        list(runner.load_weights([expert_weight]))


@pytest.mark.parametrize(
    ("device_type", "connector"),
    [
        ("cuda", "CAMAsyncAFDConnector"),
        ("npu", "CAMP2pAFDConnector"),
        ("cpu", "P2pNcclAFDConnector"),
    ],
)
def test_factory_rejects_unsupported_placement_before_registration(
    monkeypatch, device_type, connector
):
    """Only a CUDA connector whose registration names a runner builds one.

    Why: the other connectors run on NPU through their own model paths, which
    expect a different payload.
    """
    config = attention_moe_vllm_config(connector=connector)

    with pytest.raises(ValueError, match="unsupported Attention remote MoE"):
        _build_runner(monkeypatch, config, device_type=device_type)

    assert config.compilation_config.static_forward_context == {}


def test_factory_builds_the_runner_its_connector_registers(monkeypatch):
    """The factory builds the runner class the configured connector names.

    Why: each connector picks its Attention-side MoE runner, so every model
    builds its experts with the same call whatever the connector.
    """
    monkeypatch.setattr(
        AFDConnectorFactory, "_registry", dict(AFDConnectorFactory._registry)
    )
    AFDConnectorFactory.register_connector(
        "RegisteredRunnerConnector",
        "afd_plugin.connectors.gpu.p2p",
        "P2pNcclAFDConnector",
        attention_moe_runner=(__name__, _RegisteredRunner.__name__),
    )
    config = attention_moe_vllm_config(connector="RegisteredRunnerConnector")
    runner = _build_runner(monkeypatch, config)
    hidden_states = _hidden_states(2)

    output = runner(hidden_states=hidden_states, router_logits=hidden_states)

    assert torch.equal(output, hidden_states + 1)


@pytest.mark.parametrize(
    "compute_gate_on_attention", [False, True], ids=["ffn-gate", "attention-gate"]
)
def test_factory_rejects_gate_that_contradicts_gate_placement(
    monkeypatch, compute_gate_on_attention
):
    """The model passes its gate exactly when Attention computes the gate.

    Why: the gate decides whether the FFN receives router logits; a mismatch
    would send a payload the FFN role does not expect.
    """
    config = attention_moe_vllm_config(
        compute_gate_on_attention=compute_gate_on_attention
    )
    gate = None if compute_gate_on_attention else _gate()

    with pytest.raises(ValueError, match="gate if and only if"):
        _build_runner(monkeypatch, config, gate=gate)

    assert config.compilation_config.static_forward_context == {}


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("parallel_config.enable_eplb", True, "Attention-local EPLB"),
        (
            "parallel_config.eplb_config.num_redundant_experts",
            1,
            "Attention-local redundant experts",
        ),
        (
            "aux_output_config.enable_return_routed_experts",
            True,
            "Attention-local routed_experts capture",
        ),
    ],
)
def test_factory_rejects_attention_local_expert_features(
    monkeypatch, setting, value, message
):
    """EPLB, redundant experts and routed-expert capture need local experts.

    Why: the Attention runner owns no experts to balance, replicate or
    capture; vLLM would otherwise wire these features to a runner that never
    routes.
    """
    config = attention_moe_vllm_config()
    *parents, field = setting.split(".")
    target = config
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, field, value)

    with pytest.raises(RuntimeError, match=message):
        _build_runner(monkeypatch, config)

    assert config.compilation_config.static_forward_context == {}


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), -float("inf")])
def test_factory_rejects_nonfinite_routed_scaling_factor(monkeypatch, factor):
    config = attention_moe_vllm_config()

    with pytest.raises(ValueError, match="routed_scaling_factor must be finite"):
        _build_runner(monkeypatch, config, routed_scaling_factor=factor)

    assert config.compilation_config.static_forward_context == {}


@pytest.mark.parametrize("argument", RESERVED_FACTORY_ARGUMENTS)
def test_factory_rejects_reserved_native_arguments(monkeypatch, argument):
    """The factory alone sets the arguments that describe local experts.

    Why: a caller-provided shared expert, quantization or sharding argument
    would allocate or compute local experts on the Attention role.
    """
    config = attention_moe_vllm_config()

    with pytest.raises(ValueError, match=f"reserves these arguments: {argument}"):
        _build_runner(monkeypatch, config, **{argument: None})

    assert config.compilation_config.static_forward_context == {}


@pytest.mark.parametrize(
    "attention_tp_size", [1, 4], ids=["attention-tp1", "attention-tp4"]
)
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_runner_without_gate_sends_hidden_states_only(
    monkeypatch, num_tokens, attention_tp_size
):
    """With the gate on FFN, the runner sends only the hidden states, tagged
    with its layer and stage, and returns the FFN output.

    Why: the FFN role runs its native internal-router MoE on the rows it
    receives; the send, DBO yield and receive order is the stage contract.
    Attention TP must neither slice those rows nor reduce the output, because
    the experts live on the FFN role.
    """
    monkeypatch.setattr(
        fused_moe_layer,
        "get_tensor_model_parallel_world_size",
        lambda: attention_tp_size,
    )
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events, stage_idx=STAGE_IDX)
    runner = _build_runner(monkeypatch, attention_moe_vllm_config())
    hidden_states = _hidden_states(num_tokens)

    output = runner(hidden_states=hidden_states, router_logits=hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    _, sent_states, context, send_kwargs = events[0]
    assert sent_states is hidden_states
    assert send_kwargs == {}
    assert context.metadata.layer_idx == LAYER_IDX
    assert context.metadata.stage_idx == STAGE_IDX
    assert context.metadata.seq_lens == [num_tokens]
    assert events[2][2] == STAGE_IDX
    assert torch.equal(output, hidden_states * FFN_OUTPUT_SCALE)


def test_runner_with_gate_sends_its_router_logits(monkeypatch):
    """With the gate on Attention, the runner applies its gate once to the
    hidden states and sends the logits with them, ignoring its router_logits
    argument.

    Why: native DeepseekV2MoE passes the hidden states as router_logits and
    lets the runner's gate route, as native MoERunner does; the FFN role
    needs the real logits for its external-router experts.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events, stage_idx=STAGE_IDX)
    gate = _gate()
    gate_inputs: list = []
    gate.register_forward_hook(
        lambda _module, args, _output: gate_inputs.append(args[0])
    )
    config = attention_moe_vllm_config(compute_gate_on_attention=True)
    runner = _build_runner(monkeypatch, config, gate=gate)
    hidden_states = _hidden_states(3)

    output = runner(hidden_states=hidden_states, router_logits=hidden_states)

    assert len(gate_inputs) == 1
    assert gate_inputs[0] is hidden_states
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    _, sent_states, context, send_kwargs = events[0]
    assert sent_states is hidden_states
    assert context.states is None
    assert set(send_kwargs) == {"router_logits"}
    assert torch.equal(send_kwargs["router_logits"], hidden_states @ gate.weight.t())
    assert torch.equal(output, hidden_states * FFN_OUTPUT_SCALE)


def test_routed_experts_send_routes_and_combine_into_a_new_tensor(monkeypatch):
    """Under DeepEP M2N, the routed experts send the routes vLLM's runner
    selected with the hidden states and return the combined output in a new
    tensor; their input and the shared experts' input stay unchanged.

    Why: the FFN runs only the routed experts on the routes it receives.
    Combine writes into the tensor it is given, and vLLM's MoE op declares
    that it does not mutate its inputs. The send, DBO yield and receive order
    is the stage contract.
    """
    events: list[tuple] = []
    afd_metadata = install_proxy_context(monkeypatch, events, stage_idx=STAGE_IDX)
    afd_metadata.connector = _InPlaceCombineConnector(events)
    config = attention_moe_vllm_config(
        connector=DEEPEP_M2N_CONNECTOR, compute_gate_on_attention=True
    )
    runner = _build_runner(
        monkeypatch, config, gate=_gate(), shared_experts=_ScaledSharedExperts()
    )
    hidden_states = _hidden_states(2)
    dispatched_states = hidden_states.clone()
    topk_weights = torch.full((2, TOP_K), 0.5)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    output = runner.routed_experts.forward_modular(
        x=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=runner.shared_experts,
        shared_experts_input=hidden_states,
    )

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    _, sent_states, context, send_kwargs = events[0]
    assert torch.equal(sent_states, dispatched_states)
    assert send_kwargs.keys() == {"topk_ids", "topk_weights"}
    assert send_kwargs["topk_ids"] is topk_ids
    assert send_kwargs["topk_weights"] is topk_weights
    assert context.metadata.layer_idx == LAYER_IDX
    assert context.metadata.stage_idx == STAGE_IDX
    assert events[2][2] == STAGE_IDX
    assert torch.equal(output, torch.full_like(dispatched_states, COMBINED_VALUE))
    assert torch.equal(hidden_states, dispatched_states)
    assert torch.equal(
        runner.shared_experts.output, dispatched_states * SHARED_EXPERTS_SCALE
    )


@pytest.mark.parametrize(
    "tokens_lens", [[3], [5, 3]], ids=["stage-metadata", "batch-metadata"]
)
@pytest.mark.parametrize(
    ("tp_rank", "padding_rows"), [(0, []), (1, [1])], ids=["rank0", "rank1"]
)
def test_routed_experts_unroute_tp_padding_rows(
    monkeypatch, tokens_lens, tp_rank, padding_rows
):
    """With Attention TP 2 and 3 tokens in the stage, rank 0 sends tokens 0-1
    with their routes, and rank 1 sends token 2 plus one padding row whose
    top-k IDs are all ``UNROUTED_EXPERT_ID``. A microbatch's metadata holds
    only its own stage's length; a whole batch's holds every stage's.

    Why: the native MoE block pads the tokens to a multiple of the TP size and
    gives each rank a slice, and the TP gather drops the padding rows. A
    routed padding row would make the FFN run experts at every MoE layer for a
    row that is thrown away.
    """
    events: list[tuple] = []
    afd_metadata = install_proxy_context(monkeypatch, events, stage_idx=STAGE_IDX)
    afd_metadata.tokens_lens = tokens_lens
    monkeypatch.setattr(remote, "get_tensor_model_parallel_rank", lambda: tp_rank)
    config = attention_moe_vllm_config(
        connector=DEEPEP_M2N_CONNECTOR, compute_gate_on_attention=True
    )
    config.parallel_config.tensor_parallel_size = 2
    runner = _build_runner(monkeypatch, config, gate=_gate())
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    runner.routed_experts.forward_modular(
        x=_hidden_states(2),
        topk_weights=torch.full((2, TOP_K), 0.5),
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    (send,) = [event for event in events if event[0] == "send"]
    sent_ids = send[3]["topk_ids"]
    unrouted = (sent_ids == remote.UNROUTED_EXPERT_ID).all(dim=1)
    assert unrouted.nonzero().flatten().tolist() == padding_rows
    assert torch.equal(sent_ids[~unrouted], topk_ids[~unrouted])


class _MicrobatchTurns:
    """Two threads take turns as vLLM's microbatch threads do: one runs at a
    time and hands over at AFD's DBO yield."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._turn = 0
        self._done = [False, False]

    def wait_turn(self, ubatch_id: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self._turn == ubatch_id, MICROBATCH_HANDOFF_TIMEOUT_S
            )

    def yield_to_peer(self, ubatch_id: int) -> None:
        peer = 1 - ubatch_id
        with self._condition:
            if self._done[peer]:
                return
            self._turn = peer
            self._condition.notify_all()
            assert self._condition.wait_for(
                lambda: self._turn == ubatch_id, MICROBATCH_HANDOFF_TIMEOUT_S
            )

    def finish(self, ubatch_id: int) -> None:
        with self._condition:
            self._done[ubatch_id] = True
            self._turn = 1 - ubatch_id
            self._condition.notify_all()


@pytest.mark.parametrize(
    "enable_dbo", [True, False], ids=["enable-dbo", "ubatch-size-2-without-dbo"]
)
def test_routed_experts_keep_shared_output_per_microbatch(monkeypatch, enable_dbo):
    """Two microbatches that interleave at the DBO yield each read their own
    shared experts' output, with ``--enable-dbo`` or with ``--ubatch-size 2``
    alone.

    Why: the routed experts store the shared output and yield before vLLM's
    runner reads it, so each microbatch needs its own output slot; one slot
    fails vLLM's empty-slot check when the second microbatch arrives.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events, stage_idx=0)
    config = attention_moe_vllm_config(
        connector=DEEPEP_M2N_CONNECTOR, compute_gate_on_attention=True
    )
    config.parallel_config.enable_dbo = enable_dbo
    config.parallel_config.ubatch_size = 0 if enable_dbo else 2
    runner = _build_runner(
        monkeypatch, config, gate=_gate(), shared_experts=_ScaledSharedExperts()
    )
    turns = _MicrobatchTurns()
    ubatch_of: dict[int, int] = {}

    def dbo_yield(tensor, *, role):
        turns.yield_to_peer(ubatch_of[threading.get_ident()])
        return tensor

    monkeypatch.setattr(remote, "maybe_apply_dbo_yield", dbo_yield)
    results: dict[int, tuple] = {}
    errors: dict[int, BaseException] = {}

    def run_microbatch(ubatch_id: int) -> None:
        thread_id = threading.get_ident()
        ubatch_of[thread_id] = ubatch_id
        # As vLLM's UBatchContext does, so dbo_current_ubatch_id() works.
        ubatching._THREAD_ID_TO_CONTEXT[thread_id] = ubatch_id
        turns.wait_turn(ubatch_id)
        try:
            hidden_states = _hidden_states(2) + 100 * ubatch_id
            runner.routed_experts.forward_modular(
                x=hidden_states,
                topk_weights=torch.full((2, TOP_K), 0.5),
                topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
                shared_experts=runner.shared_experts,
                shared_experts_input=hidden_states,
            )
            # vLLM's runner reads the shared output after forward_modular.
            results[ubatch_id] = (hidden_states, runner.shared_experts.output)
        except BaseException as error:  # noqa: BLE001 - reported below
            errors[ubatch_id] = error
        finally:
            turns.finish(ubatch_id)

    threads = [threading.Thread(target=run_microbatch, args=(i,)) for i in (0, 1)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(MICROBATCH_HANDOFF_TIMEOUT_S)
    finally:
        for thread in threads:
            ubatching._THREAD_ID_TO_CONTEXT.pop(thread.ident, None)

    assert not errors, errors
    assert [event[0] for event in events] == ["send", "send", "recv", "recv"]
    assert sorted(results) == [0, 1]
    for hidden_states, shared_output in results.values():
        assert torch.equal(shared_output, hidden_states * SHARED_EXPERTS_SCALE)


@pytest.mark.parametrize(
    ("local_input", "message"),
    [
        ("input_ids", "input_ids transport"),
        ("shared_experts_input", "no local shared experts"),
    ],
)
def test_runner_rejects_local_inputs_before_sending(monkeypatch, local_input, message):
    """Inputs for local routing or shared experts fail before any transfer.

    Why: the FFN protocol carries no token ids for this boundary, and the
    Attention role builds no shared experts to feed.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    runner = _build_runner(monkeypatch, attention_moe_vllm_config())
    hidden_states = _hidden_states(2)
    local_inputs = {
        "input_ids": torch.tensor([1, 2]),
        "shared_experts_input": hidden_states,
    }

    with pytest.raises(NotImplementedError, match=message):
        runner(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            **{local_input: local_inputs[local_input]},
        )

    assert events == []
