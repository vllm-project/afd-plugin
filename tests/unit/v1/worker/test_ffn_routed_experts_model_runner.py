# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for ``GPUFFNRoutedExpertsModelRunner`` with a fake connector."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.v1.worker import (  # noqa: E402
    ffn_routed_experts_model_runner as routed_module,
)
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner  # noqa: E402
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (  # noqa: E402
    GPUFFNRoutedExpertsModelRunner,
    local_experts_from_vllm,
    select_routed_moe_backend,
)
from tests.unit.fakes.connector import (  # noqa: E402
    FakeRoutedConnector,
)

MOE_LAYERS = (1, 3)
NUM_LOCAL_EXPERTS = 2
HIDDEN = 256
INTERMEDIATE = 128
FP8_BLOCK = 128


def _runner(connector=None):
    runner = object.__new__(GPUFFNRoutedExpertsModelRunner)
    runner.connector = connector or FakeRoutedConnector()
    runner.moe_layer_indices = MOE_LAYERS
    runner.local_experts = {
        layer: SimpleNamespace(weights=f"w{layer}") for layer in MOE_LAYERS
    }
    runner.use_cuda_graph = False
    runner._cuda_graphs = {}
    runner.prof = None
    return runner


def _dp(*tokens):
    return SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(tokens))


def test_routed_runner_reuses_the_connector_driven_ffn_runner():
    """The routed runner is the FFN runner with a different forward.

    Why: graph capture, replay, the loop contract and shutdown are shared.
    """
    assert issubclass(GPUFFNRoutedExpertsModelRunner, GPUFFNModelRunner)


def _recv_and_run(stage, layer):
    return [("recv", stage), ("run", f"rows{stage}", f"w{layer}")]


def _send(stage):
    return [("send", f"out(rows{stage})", stage)]


def test_forward_follows_attention_dbo_order_then_joins_lanes():
    """For layers (1, 3) and stages (0, 1) the FFN issues Attention's DBO
    micro-batch order: both stages dispatch layer 1, then each stage sends
    its layer-1 result just before dispatching layer 3, and the last results
    go back at the end. Each recv runs that layer's experts; the lanes rejoin
    once at the end.

    Why: Attention's micro-batch threads issue D0 D1 C0 D0' C1 D1' and
    dispatch/combine can block the host, so sending stage 0 right after its
    experts (D0 C0 D1) hung every rank in the graph+DBO warmup. A CUDA-graph
    capture also fails if forked lane streams are not joined.
    """
    runner = _runner()

    output = runner._ffn_forward(dp_metadata_list={1: _dp(3), 0: _dp(5)})

    first, last = MOE_LAYERS
    expected = [
        *_recv_and_run(0, first),
        *_recv_and_run(1, first),
        *_send(0),
        *_recv_and_run(0, last),
        *_send(1),
        *_recv_and_run(1, last),
        *_send(0),
        *_send(1),
        ("sync",),
    ]
    assert runner.connector.events == expected
    assert output == "out(rows1)"
    assert len(runner.connector.control_updates) == 1


def test_forward_with_one_stage_sends_each_layer_before_the_next_recv():
    """With one stage the order is recv, run, send per MoE layer.

    Why: without DBO, Attention combines a layer before it dispatches the next.
    """
    runner = _runner()

    runner._ffn_forward(dp_metadata_list={0: _dp(5)})

    first, last = MOE_LAYERS
    assert runner.connector.events == [
        *_recv_and_run(0, first),
        *_send(0),
        *_recv_and_run(0, last),
        *_send(0),
        ("sync",),
    ]


def test_forward_skips_control_update_inside_capture():
    runner = _runner()

    runner._ffn_forward(dp_metadata_list={0: _dp(1)}, update_connector_state=False)

    assert runner.connector.control_updates == []
    assert runner.connector.events[-1] == ("sync",)


def test_eager_fallback_keeps_attention_replay_flag():
    """A step Attention replays but the FFN has no graph for runs eagerly
    with ``is_graph_replaying`` still set in the control payload.

    Why: the connector then plans the step like Attention (no host sync).
    """
    runner = _runner()
    runner.use_cuda_graph = True

    runner.execute_model(
        dp_metadata_list={0: _dp(4)},
        is_graph_replaying=True,
    )

    (payload,) = runner.connector.control_updates
    assert payload.is_graph_replaying is True
    assert payload.is_graph_capturing is False


def test_load_model_takes_each_moe_layers_local_experts(monkeypatch):
    """After the native load, the runner asks the model for its MoE layers and
    converts each layer's vLLM experts once.

    Why: weights are read after vLLM's post-load processing, not per step.
    """
    routed_experts = {layer: object() for layer in MOE_LAYERS}
    model = SimpleNamespace(
        get_experts_layer_indices=lambda: list(MOE_LAYERS),
        get_routed_experts=routed_experts.__getitem__,
    )

    def native_load(self, *, load_dummy_weights=False, **kwargs):
        self.model = model

    monkeypatch.setattr(GPUFFNModelRunner, "load_model", native_load)
    monkeypatch.setattr(
        routed_module,
        "local_experts_from_vllm",
        lambda experts: ("converted", experts),
    )
    runner = _runner()

    runner.load_model()

    assert runner.moe_layer_indices == MOE_LAYERS
    assert runner.local_experts == {
        layer: ("converted", routed_experts[layer]) for layer in MOE_LAYERS
    }


def test_initialize_passes_moe_layers_and_experts_to_the_connector():
    runner = _runner()

    runner.initialize_afd_connector()

    assert runner.connector.init_kwargs == {
        "moe_layer_indices": MOE_LAYERS,
        "local_experts": runner.local_experts,
    }


def test_constructor_rejects_connectors_without_routed_rows(monkeypatch):
    def base_init(self, vllm_config, device):
        self.connector = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(GPUFFNModelRunner, "__init__", base_init)

    with pytest.raises(TypeError, match="is not an AFDRoutedExpertsConnector"):
        GPUFFNRoutedExpertsModelRunner(SimpleNamespace(), torch.device("cpu"))


def test_ffn_runner_adopts_bootstrapped_connector_and_closes_placeholder():
    """The worker's bootstrapped connector replaces the one built at
    construction, which is closed.

    Why: bootstrap must precede vLLM's distributed init, construction cannot.
    """
    events: list[str] = []
    runner = _runner(SimpleNamespace(close=lambda: events.append("close")))
    bootstrapped = FakeRoutedConnector()

    runner.adopt_bootstrapped_connector(bootstrapped)

    assert runner.connector is bootstrapped
    assert events == ["close"]


def _routed_experts(*, fp8: bool, expert_ids=(4, 5), backend: str = "TRITON"):
    num_experts = len(expert_ids)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    selected = SimpleNamespace(name=backend)
    experts = SimpleNamespace(
        quant_method=(
            SimpleNamespace(fp8_backend=selected)
            if fp8
            else SimpleNamespace(unquantized_backend=selected)
        ),
        w13_weight=torch.nn.Parameter(
            torch.zeros(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=dtype),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(num_experts, HIDDEN, INTERMEDIATE, dtype=dtype),
            requires_grad=False,
        ),
        expert_map_manager=SimpleNamespace(
            get_local_expert_ids=lambda: list(expert_ids),
        ),
    )
    if fp8:
        experts.w13_weight_scale_inv = torch.nn.Parameter(
            torch.ones(num_experts, 2 * INTERMEDIATE // FP8_BLOCK, HIDDEN // FP8_BLOCK),
            requires_grad=False,
        )
        experts.w2_weight_scale_inv = torch.nn.Parameter(
            torch.ones(num_experts, HIDDEN // FP8_BLOCK, INTERMEDIATE // FP8_BLOCK),
            requires_grad=False,
        )
    return experts


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_local_experts_are_views_of_vllm_parameters(fp8):
    """The executor's weights share storage with vLLM's parameters, take the
    block scales only for FP8, and carry vLLM's local expert ids.

    Why: no second copy of the experts, and the ids feed the placement check.
    """
    experts = _routed_experts(fp8=fp8)

    local = local_experts_from_vllm(experts)

    assert local.expert_ids == (4, 5)
    assert local.weights.w13.data_ptr() == experts.w13_weight.data_ptr()
    assert local.weights.w2.data_ptr() == experts.w2_weight.data_ptr()
    assert local.weights.is_fp8 is fp8
    if fp8:
        assert (
            local.weights.w13_scale.data_ptr()
            == experts.w13_weight_scale_inv.data_ptr()
        )
    else:
        assert local.weights.w13_scale is None


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "backend", ["FLASHINFER_CUTLASS", "FLASHINFER_TRTLLM", "MARLIN"]
)
def test_local_experts_reject_backends_that_reorder_weights(fp8, backend):
    """Experts processed for a backend that reorders them in place are refused.

    Why: FlashInfer swaps w13 to [up; gate] (TRT-LLM and Marlin also shuffle)
    without changing dtype or shape; the executor would read [gate; up] and
    every MoE layer would compute silu(up) * gate, with no error.
    """
    with pytest.raises(RuntimeError, match="moe-backend triton"):
        local_experts_from_vllm(_routed_experts(fp8=fp8, backend=backend))


def _kernel_config(moe_backend: str) -> SimpleNamespace:
    return SimpleNamespace(kernel_config=SimpleNamespace(moe_backend=moe_backend))


def test_routed_ffn_pins_auto_moe_backend_to_triton():
    """``auto`` becomes ``triton``; an explicit ``triton`` stays.

    Why: auto picks FlashInfer CUTLASS on Hopper with EP > 1 for block FP8,
    which reorders the experts; the FFN never runs vLLM's MoE forward, so the
    backend only decides the weight layout.
    """
    for configured in ("auto", "triton"):
        vllm_config = _kernel_config(configured)
        select_routed_moe_backend(vllm_config)
        assert vllm_config.kernel_config.moe_backend == "triton"


@pytest.mark.parametrize("configured", ["flashinfer_cutlass", "deep_gemm", "marlin"])
def test_routed_ffn_rejects_other_explicit_moe_backends(configured):
    """An explicit backend other than triton fails before the weights load.

    Why: the failure must be loud at startup, not wrong tokens at serving.
    """
    with pytest.raises(RuntimeError, match="moe-backend triton"):
        select_routed_moe_backend(_kernel_config(configured))
