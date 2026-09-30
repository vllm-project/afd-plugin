# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real vLLM construction and transport contracts for parameter-free MoE."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.layers import fused_moe  # noqa: E402
from vllm.model_executor.layers.fused_moe import layer as moe_layer  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (  # noqa: E402
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (  # noqa: E402
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (  # noqa: E402
    MoERunner,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E402
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor import remote_moe  # noqa: E402
from afd_plugin.model_executor.remote_moe import (  # noqa: E402
    AFDRemoteMoEMethod,
    AFDRemoteMoERunner,
    AFDRemoteRoutedExperts,
)

pytestmark = pytest.mark.vllm_runtime
PREFIX = "model.layers.3.mlp.experts"


def _make_runner(
    config,
    *,
    device_type="cuda",
    connector="P2pNcclAFDConnector",
    compute_gate_on_attention=False,
    **kwargs,
):
    config.additional_config["afd"] = {
        "role": "attention",
        "connector": connector,
        "compute_gate_on_attention": compute_gate_on_attention,
    }
    with pytest.MonkeyPatch.context() as patch, set_current_vllm_config(config):
        patch.setattr(
            remote_moe, "current_platform", SimpleNamespace(device_type=device_type)
        )
        return remote_moe.build_attention_moe_runner(
            config,
            num_experts=4,
            top_k=2,
            hidden_size=7,
            intermediate_size=11,
            params_dtype=torch.bfloat16,
            prefix=PREFIX,
            **kwargs,
        )


def _unexpected_local_compute(*args, **kwargs):
    pytest.fail("Remote MoE reached local expert allocation or computation")


@pytest.fixture
def remote_runner():
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    return config, _make_runner(config)


def test_real_factory_preserves_inherited_constructors_and_registration(monkeypatch):
    monkeypatch.setattr(
        UnquantizedFusedMoEMethod, "__init__", _unexpected_local_compute
    )
    monkeypatch.setattr(
        FusedMoEMethodBase, "maybe_make_prepare_finalize", _unexpected_local_compute
    )
    configs = [VllmConfig(device_config=DeviceConfig("cpu")) for _ in range(2)]
    runners = [_make_runner(config) for config in configs]

    assert AFDRemoteMoERunner.__init__ is MoERunner.__init__
    assert AFDRemoteRoutedExperts.__init__ is RoutedExperts.__init__
    assert AFDRemoteMoEMethod.__init__ is FusedMoEMethodBase.__init__
    assert (
        AFDRemoteMoEMethod.process_weights_after_loading
        is QuantizeMethodBase.process_weights_after_loading
    )
    for actual, inherited in (
        (AFDRemoteMoERunner.forward, MoERunner.forward),
        (
            AFDRemoteMoERunner.maybe_init_modular_kernel,
            MoERunner.maybe_init_modular_kernel,
        ),
        (AFDRemoteRoutedExperts._get_quant_method, RoutedExperts._get_quant_method),
        (AFDRemoteMoEMethod.create_weights, FusedMoEMethodBase.create_weights),
        (
            AFDRemoteMoEMethod.get_fused_moe_quant_config,
            FusedMoEMethodBase.get_fused_moe_quant_config,
        ),
        (
            AFDRemoteMoEMethod.maybe_roundup_sizes,
            FusedMoEMethodBase.maybe_roundup_sizes,
        ),
    ):
        assert inspect.signature(actual, eval_str=True) == inspect.signature(
            inherited, eval_str=True, locals={"RoutedExperts": RoutedExperts}
        )
    for config, runner in zip(configs, runners, strict=True):
        assert type(runner) is AFDRemoteMoERunner
        assert type(runner.routed_experts) is AFDRemoteRoutedExperts
        assert type(runner.routed_experts.quant_method) is AFDRemoteMoEMethod
        assert runner.gate is None
        assert runner.shared_experts is None
        assert runner.shared_expert_gate is None
        assert runner.is_internal_router
        assert runner.layer_id == 3
        assert config.compilation_config.static_forward_context == {PREFIX: runner}
        assert config.compilation_config.static_all_moe_layers == [PREFIX]
        with pytest.raises(ValueError, match="Duplicate layer name"):
            _make_runner(config)
    assert runners[0] is not runners[1]


def test_real_post_load_is_parameter_free_and_keeps_quant_method(remote_runner):
    config, runner = remote_runner
    method = runner.routed_experts.quant_method
    assert list(runner.parameters()) == []
    assert list(runner.routed_experts.get_expert_weights()) == []
    assert "quant_method" not in runner.__dict__
    assert list(runner.load_weights([])) == []
    with set_current_vllm_config(config):
        process_weights_after_loading(
            runner, SimpleNamespace(quantization=None), torch.device("cpu")
        )
        runner.maybe_init_modular_kernel()
    assert runner.routed_experts.quant_method is method
    assert method.moe_quant_config is None
    assert method.moe_kernel is None
    assert list(runner.parameters()) == []
    # No-weight construction must not turn unfiltered checkpoints into success.
    with pytest.raises(AttributeError, match="w13_weight"):
        list(runner.load_weights([("0.gate_proj.weight", torch.ones(11, 7))]))


@pytest.mark.parametrize("compute_gate", [False, True])
def test_factory_selects_gpu_runner_without_local_weights(compute_gate):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    runner = _make_runner(config, compute_gate_on_attention=compute_gate)
    expected = (
        remote_moe.AFDExternalRoutingMoERunner if compute_gate else AFDRemoteMoERunner
    )
    assert type(runner) is expected
    assert runner.is_internal_router is not compute_gate
    assert runner.gate is runner.shared_experts is runner.shared_expert_gate is None
    assert list(runner.parameters()) == []
    assert config.compilation_config.static_forward_context == {PREFIX: runner}


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("compute_gate", [False, True])
def test_factory_rejects_nonfinite_routed_scaling_factor(
    monkeypatch, factor, compute_gate
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(ValueError, match="routed_scaling_factor must be finite"):
        _make_runner(
            config,
            compute_gate_on_attention=compute_gate,
            routed_scaling_factor=factor,
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.parametrize(
    ("scale_kwargs", "expected"),
    [
        ({}, 1.0),
        ({"routed_scaling_factor": 0.0}, 0.0),
        ({"routed_scaling_factor": -2.5}, -2.5),
        ({"routed_scaling_factor": 2.5}, 2.5),
    ],
    ids=["native-default", "zero", "negative", "positive"],
)
def test_factory_preserves_finite_routed_scaling_factor(scale_kwargs, expected):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    runner = _make_runner(config, **scale_kwargs)
    assert runner.routed_scaling_factor == expected


@pytest.mark.parametrize(
    "reserved_name",
    [
        "gate",
        "quant_config",
        "shared_experts",
        "shared_expert_gate",
        "routed_input_transform",
        "routed_output_transform",
        "n_shared_experts",
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
    ],
)
def test_factory_rejects_reserved_native_arguments(monkeypatch, reserved_name):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(
        ValueError, match=f"factory reserves these arguments: {reserved_name}"
    ):
        _make_runner(config, **{reserved_name: None})
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.parametrize(
    "setting",
    ["enable_eplb", "num_redundant_experts", "enable_return_routed_experts"],
)
def test_factory_rejects_local_eplb_and_capture_before_native_construction(
    monkeypatch, setting
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    if setting == "enable_eplb":
        config.parallel_config.enable_eplb = True
        message = "enable_eplb"
    elif setting == "num_redundant_experts":
        config.parallel_config.eplb_config.num_redundant_experts = 1
        message = "redundant experts"
    else:
        config.model_config = SimpleNamespace(enable_return_routed_experts=True)
        message = "routed_experts capture"
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(RuntimeError, match=message):
        _make_runner(config)
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


def test_unit_descriptor_does_not_inherit_attention_tp(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.parallel_config.tensor_parallel_size = 4
    monkeypatch.setattr(moe_layer, "get_tensor_model_parallel_world_size", lambda: 4)
    runner = _make_runner(config)
    parallel = runner.moe_config.moe_parallel_config
    assert (
        parallel.tp_size,
        parallel.dp_size,
        parallel.pcp_size,
        parallel.ep_size,
    ) == (
        1,
        1,
        1,
        1,
    )
    assert config.parallel_config.tensor_parallel_size == 4
    assert runner.moe_config.hidden_dim == 7
    assert runner.moe_config.hidden_dim_unpadded == 7
    assert runner.moe_config.intermediate_size_per_partition == 11
    assert runner.routed_experts.expert_map is None


def _context(connector, stage_idx):
    metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0, 2],
        requests_start_loc=[0, 1],
        stage_idx=stage_idx,
        connector=connector,
        tokens_lens=[2, 1],
        num_stages=2,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        additional_kwargs={"afd_metadata": metadata},
    )
    context.ubatch_idx = stage_idx
    return context


@pytest.mark.parametrize("profile", [False, True])
def test_forward_uses_live_context_and_bypasses_native_math(
    monkeypatch, remote_runner, profile
):
    _, runner = remote_runner
    events: list[tuple] = []
    output = torch.zeros(2, 7, dtype=torch.bfloat16)
    hidden = torch.arange(14, dtype=torch.bfloat16).view(2, 7)
    original = hidden.clone()
    refs = [hidden.clone(), hidden[:1].clone()]
    contexts: list[ForwardContext] = []

    class Connector:
        def send_attn_output(self, hidden_states, context, **kwargs):
            live = get_forward_context()
            events.append(("send", live, hidden_states, context, kwargs))
            live.additional_kwargs["pending_transfer"] = context

        def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
            live = get_forward_context()
            assert live is contexts[ubatch_idx]
            assert live.additional_kwargs["pending_transfer"] is events[-2][3]
            assert ref_tensor is refs[ubatch_idx]
            events.append(("recv", live, ref_tensor, ubatch_idx))
            return output

    def yield_stage(hidden_states, *, role):
        live = get_forward_context()
        events.append(("yield", live, role))
        return refs[live.ubatch_idx]

    for name in (
        "_forward_entry",
        "_forward_impl",
        "apply_routed_input_transform",
        "_maybe_pad_hidden_states",
        "_maybe_apply_routed_scale_to_output",
        "apply_routed_output_transform",
        "_maybe_reduce_final_output",
    ):
        monkeypatch.setattr(runner, name, _unexpected_local_compute)
    monkeypatch.setattr(runner.router, "select_experts", _unexpected_local_compute)
    monkeypatch.setattr(remote_moe, "maybe_apply_dbo_yield", yield_stage)
    connector = Connector()
    contexts.extend(_context(connector, stage) for stage in range(2))
    state_keys = set(runner.__dict__)
    for stage, context in enumerate(contexts):
        context.in_profile_run = profile
        metadata = context.additional_kwargs["afd_metadata"]
        metadata.stage_idx = 9
        inputs = hidden if stage == 0 else hidden[:1]
        with override_forward_context(context):
            assert runner(inputs, torch.empty(0)) is output
        sent = events[stage * 3]
        assert sent[2] is inputs
        assert sent[4] == {}
        assert sent[3].metadata.layer_idx == 3
        assert sent[3].metadata.stage_idx == stage
        assert sent[3].metadata.seq_lens == [inputs.shape[0]]
        assert metadata.stage_idx == stage
    assert [event[0] for event in events] == ["send", "yield", "recv"] * 2
    assert events[1][2] == events[4][2] == "attention"
    assert set(runner.__dict__) == state_keys
    assert torch.equal(hidden, original)


def test_rejects_ids_and_missing_metadata_before_sending(remote_runner):
    _, runner = remote_runner
    connector = SimpleNamespace(send_attn_output=_unexpected_local_compute)
    context = _context(connector, 0)
    hidden = torch.ones(1, 7)
    with override_forward_context(context):
        with pytest.raises(NotImplementedError, match="input_ids"):
            runner(hidden, hidden, input_ids=torch.tensor([1]))
        context.additional_kwargs.clear()
        with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
            runner(hidden, hidden)


@pytest.mark.parametrize("failure", ["send", "yield", "recv"])
def test_exchange_propagates_failures_without_later_operations(
    monkeypatch, remote_runner, failure
):
    _, runner = remote_runner
    events = []

    def operation(name):
        def run(*args, **kwargs):
            events.append(name)
            if name == failure:
                raise RuntimeError("transport failed")
            return args[0] if args else kwargs["ref_tensor"]

        return run

    connector = SimpleNamespace(
        send_attn_output=operation("send"), recv_ffn_output=operation("recv")
    )
    monkeypatch.setattr(remote_moe, "maybe_apply_dbo_yield", operation("yield"))
    hidden = torch.ones(1, 7)
    with (
        override_forward_context(_context(connector, 0)),
        pytest.raises(RuntimeError, match="transport failed"),
    ):
        runner(hidden, hidden)
    order = ["send", "yield", "recv"]
    assert events == order[: order.index(failure) + 1]


def test_external_router_sends_logits_without_local_moe_math(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    runner = _make_runner(config, compute_gate_on_attention=True)
    assert runner.forward.__func__ is AFDRemoteMoERunner.forward
    assert not runner.is_internal_router
    assert list(runner.parameters()) == []
    monkeypatch.setattr(runner, "_forward_entry", _unexpected_local_compute)
    monkeypatch.setattr(runner.router, "select_experts", _unexpected_local_compute)
    hidden = torch.ones(2, 7, dtype=torch.bfloat16)
    logits = torch.arange(8).view(2, 4).float()
    output = hidden + 1
    sent = []

    def send(hidden_states, context, **kwargs):
        sent.append((hidden_states, kwargs))

    connector = SimpleNamespace(
        send_attn_output=send,
        recv_ffn_output=lambda **kwargs: output,
    )
    monkeypatch.setattr(
        remote_moe,
        "maybe_apply_dbo_yield",
        lambda hidden_states, **kwargs: hidden_states,
    )
    with override_forward_context(_context(connector, 0)):
        assert runner(hidden, logits) is output
        with pytest.raises(NotImplementedError, match="input_ids"):
            runner(hidden, logits, input_ids=torch.tensor([1, 2]))
    assert len(sent) == 1
    assert sent[0][0] is hidden
    assert set(sent[0][1]) == {"router_logits"}
    assert sent[0][1]["router_logits"] is logits


@pytest.mark.parametrize(
    ("device", "connector", "compute_gate"),
    [
        ("cuda", "CAMAsyncAFDConnector", True),
        ("npu", "CAMP2pAFDConnector", False),
        ("npu", "CAMAsyncAFDConnector", True),
        ("cpu", "P2pNcclAFDConnector", False),
    ],
)
def test_factory_rejects_unsupported_paths_before_native_construction(
    monkeypatch, device, connector, compute_gate
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(ValueError, match="unsupported Attention remote MoE"):
        _make_runner(
            config,
            device_type=device,
            connector=connector,
            compute_gate_on_attention=compute_gate,
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.gpu
def test_native_kernel_loopback_equivalence(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    program = f"""
import runpy
from pathlib import Path
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
namespace['_run_native_kernel_loopback'](Path({str(tmp_path)!r}))
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=240)


def _run_native_kernel_loopback(model_dir):
    # Isolate backend registration and process groups from ordinary unit tests.
    from transformers import DeepseekV2Config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models import deepseek_v2 as native
    from vllm.utils.network_utils import get_open_port
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.gpu_worker import Worker

    from afd_plugin.model_executor.models import deepseek_v2 as adapter

    hf_config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        max_position_embeddings=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        routed_scaling_factor=2.5,
    )
    hf_config.save_pretrained(model_dir)
    config = EngineArgs(
        model=str(model_dir),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.05,
    ).create_engine_config()
    worker = Worker(
        vllm_config=config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        is_driver_worker=True,
    )
    try:
        with set_current_vllm_config(config):
            worker.init_device()
        native.FusedMoE = fused_moe.FusedMoE
        with (
            set_current_vllm_config(config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            reference = native.DeepseekV2MoE(
                config=hf_config,
                parallel_config=config.parallel_config,
                prefix="model.layers.0.mlp",
                apply_routed_scale_to_output=True,
            )
            config.additional_config["afd"] = {
                "role": "attention",
                "connector": "P2pNcclAFDConnector",
                "compute_gate_on_attention": False,
            }
            with torch.no_grad():
                for parameter in reference.parameters():
                    fixed = torch.arange(parameter.numel(), device="cuda")
                    parameter.copy_((fixed.remainder(31) - 15).view_as(parameter) / 128)
            process_weights_after_loading(
                reference, config.model_config, torch.device("cuda")
            )
            reference.experts.maybe_init_modular_kernel()
            assert not isinstance(reference.experts, AFDRemoteMoERunner)
            assert list(reference.experts.parameters())
            assert reference.shared_experts is not None
            assert reference.routed_scaling_factor == 2.5

        class LoopbackConnector:
            def send_attn_output(self, hidden_states, context, **kwargs):
                assert set(kwargs) == (
                    {"router_logits"} if compute_gate_on_attention else set()
                )
                self.received_input = hidden_states
                with set_forward_context(
                    None, config, num_tokens=hidden_states.shape[0]
                ):
                    self.output = (
                        reference.experts(hidden_states, kwargs["router_logits"])
                        if compute_gate_on_attention
                        else reference(hidden_states)
                    )

            def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
                assert ref_tensor is self.received_input
                return self.output

        class PreviousRemoteExperts(nn.Module):
            """Frozen pre-runner boundary for the before/after kernel comparison."""

            def __init__(self):
                super().__init__()
                self.layer_idx = 1
                self.is_internal_router = True

            def forward(self, hidden_states, router_logits, input_ids=None):
                if input_ids is not None:
                    raise NotImplementedError(
                        "experts-boundary input_ids transport is not implemented"
                    )
                send_kwargs = (
                    {} if self.is_internal_router else {"router_logits": router_logits}
                )
                return remote_moe.remote_ffn_forward(
                    hidden_states, layer_idx=self.layer_idx, **send_kwargs
                )

        connector = LoopbackConnector()
        context = _context(connector, 0)
        previous_boundary = native.DeepseekV2MoE.__new__(native.DeepseekV2MoE)
        nn.Module.__init__(previous_boundary)
        previous_boundary.is_sequence_parallel = False
        previous_boundary.gate = None
        previous_boundary.experts = PreviousRemoteExperts()
        with (
            set_current_vllm_config(config),
            torch.inference_mode(),
            override_forward_context(context),
        ):
            for compute_gate_on_attention in (False, True):
                config.additional_config["afd"]["compute_gate_on_attention"] = (
                    compute_gate_on_attention
                )
                layer_idx = 1 + int(compute_gate_on_attention)
                with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
                    remote = adapter.AFDDeepseekV2RemoteExpertsMoE(
                        config=hf_config,
                        vllm_config=config,
                        prefix=f"model.layers.{layer_idx}.mlp",
                    )
                if compute_gate_on_attention:
                    remote.gate.load_state_dict(reference.gate.state_dict())
                    reference.experts.gate = None
                    previous_boundary.gate = remote.gate
                    previous_boundary.experts.is_internal_router = False
                    previous_boundary.experts.layer_idx = layer_idx
                    assert (
                        type(remote.experts) is remote_moe.AFDExternalRoutingMoERunner
                    )
                    assert list(remote.experts.parameters()) == []
                else:
                    assert list(remote.parameters()) == []
                for token_count in (1, 7):
                    hidden = torch.arange(
                        token_count * hf_config.hidden_size, device="cuda"
                    ).reshape(token_count, hf_config.hidden_size)
                    hidden = ((hidden.remainder(43) - 21) / 32).to(torch.bfloat16)
                    expected = previous_boundary(hidden.clone()).clone()
                    repeated = previous_boundary(hidden.clone())
                    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
                    actual = remote(hidden.clone())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
