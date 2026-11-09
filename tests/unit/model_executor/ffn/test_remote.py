# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-role MoE runners built through vLLM's native FusedMoEFactory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer  # noqa: E402
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)

from afd_plugin.model_executor.ffn import remote  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
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
# FusedMoEFactory arguments the AFD factory fixes for a weightless runner.
RESERVED_FACTORY_ARGUMENTS = (
    "quant_config",
    "shared_experts",
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
        return remote.build_attention_moe_runner(
            config,
            gate=gate,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            prefix=PREFIX,
            **kwargs,
        )


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

    expected_type = (
        remote.AFDExternalRoutingMoERunner
        if compute_gate_on_attention
        else remote.AFDRemoteMoERunner
    )
    assert type(runner) is expected_type
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
        ("cuda", "DeepEPM2NAFDConnector"),
        ("npu", "CAMP2pAFDConnector"),
        ("cpu", "P2pNcclAFDConnector"),
    ],
)
def test_factory_rejects_unsupported_placement_before_registration(
    monkeypatch, device_type, connector
):
    config = attention_moe_vllm_config(connector=connector)

    with pytest.raises(ValueError, match="unsupported Attention remote MoE"):
        _build_runner(monkeypatch, config, device_type=device_type)

    assert config.compilation_config.static_forward_context == {}


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


def test_runner_descriptor_ignores_attention_parallel_layout(monkeypatch):
    """The runner describes unsharded experts with the model's dimensions.

    Why: Attention TP and DP must not shard or map experts that live on the
    FFN role.
    """
    attention_tp_size = 4
    monkeypatch.setattr(
        fused_moe_layer,
        "get_tensor_model_parallel_world_size",
        lambda: attention_tp_size,
    )

    runner = _build_runner(monkeypatch, attention_moe_vllm_config())

    parallel = runner.moe_config.moe_parallel_config
    assert (parallel.tp_size, parallel.dp_size, parallel.pcp_size) == (1, 1, 1)
    assert (parallel.ep_size, parallel.use_ep) == (1, False)
    assert runner.moe_config.hidden_dim == HIDDEN_SIZE
    assert runner.moe_config.intermediate_size_per_partition == INTERMEDIATE_SIZE
    assert runner.expert_map is None


@pytest.mark.parametrize("num_tokens", [1, 7])
def test_runner_without_gate_sends_hidden_states_only(monkeypatch, num_tokens):
    """With the gate on FFN, the runner sends only the hidden states, tagged
    with its layer and stage, and returns the FFN output.

    Why: the FFN role runs its native internal-router MoE on the rows it
    receives; the send, DBO yield and receive order is the stage contract.
    """
    events: list[tuple] = []
    install_proxy_context(remote, monkeypatch, events, stage_idx=STAGE_IDX)
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
    """With the gate on Attention, the runner applies its gate and sends the
    logits with the hidden states, ignoring its router_logits argument.

    Why: native DeepseekV2MoE passes the hidden states as router_logits and
    lets the runner's gate route, as native MoERunner does; the FFN role
    needs the real logits for its external-router experts.
    """
    events: list[tuple] = []
    install_proxy_context(remote, monkeypatch, events, stage_idx=STAGE_IDX)
    gate = _gate()
    config = attention_moe_vllm_config(compute_gate_on_attention=True)
    runner = _build_runner(monkeypatch, config, gate=gate)
    hidden_states = _hidden_states(3)

    output = runner(hidden_states=hidden_states, router_logits=hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    _, sent_states, context, send_kwargs = events[0]
    assert sent_states is hidden_states
    assert context.states is None
    assert set(send_kwargs) == {"router_logits"}
    assert torch.equal(send_kwargs["router_logits"], hidden_states @ gate.weight.t())
    assert torch.equal(output, hidden_states * FFN_OUTPUT_SCALE)


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
    install_proxy_context(remote, monkeypatch, events)
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
