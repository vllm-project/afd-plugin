# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real vLLM construction and transport contracts for parameter-free MoE."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import Tensor, nn  # noqa: E402
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
    AFDRemoteMoERunnerBase,
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
    routed_scaling_factor=2.5,
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
            routed_scaling_factor=routed_scaling_factor,
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

    assert AFDRemoteMoERunnerBase.__init__ is MoERunner.__init__
    assert AFDRemoteMoERunner.__init__ is MoERunner.__init__
    assert inspect.isabstract(AFDRemoteMoERunnerBase)
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


@pytest.mark.parametrize(
    ("device_type", "connector", "compute_gate", "runner_name"),
    [
        ("cuda", "P2pNcclAFDConnector", False, "AFDRemoteMoERunner"),
        ("cuda", "P2pNcclAFDConnector", True, "AFDExternalRoutingMoERunner"),
        ("npu", "CAMP2pAFDConnector", False, "AFDRemoteMoERunner"),
        ("npu", "CAMAsyncAFDConnector", True, "AFDCAMAsyncMoERunner"),
    ],
)
def test_factory_builds_selected_backend_with_runner_kwargs(
    monkeypatch, device_type, connector, compute_gate, runner_name
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.additional_config["mix_placement"] = False
    if device_type == "npu":
        from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

        monkeypatch.setattr(npu_remote_moe, "validate_remote_moe_config", lambda: None)
    imported = []
    import_module = remote_moe.importlib.import_module

    def track_import(name):
        imported.append(name)
        return import_module(name)

    monkeypatch.setattr(remote_moe.importlib, "import_module", track_import)
    gate = nn.Linear(7, 4, bias=False) if compute_gate else None
    shared = (
        nn.Linear(7, 7, bias=False) if connector == "CAMAsyncAFDConnector" else None
    )
    runner = _make_runner(
        config,
        device_type=device_type,
        connector=connector,
        compute_gate_on_attention=compute_gate,
        gate=gate,
        attention_shared_experts=shared,
        shared_output_divisor_fp16=2.5,
        num_shared_experts=3,
    )

    if device_type == "cuda":
        assert "afd_plugin.model_executor.remote_moe" in imported
        assert not any(
            name.startswith(("afd_plugin.model_executor.npu", "vllm_ascend"))
            for name in imported
        )
    assert type(runner).__name__ == runner_name
    assert type(runner.routed_experts) is AFDRemoteRoutedExperts
    assert list(runner.routed_experts.get_expert_weights()) == []
    assert runner.shared_experts is None
    assert config.compilation_config.static_forward_context == {PREFIX: runner}
    if runner_name == "AFDCAMAsyncMoERunner":
        assert runner.gate is gate
        assert runner.num_shared_experts == 3
        assert runner.shared_output_divisor_fp16 == 2.5
        assert "attention_shared_experts" not in runner._modules
        assert not isinstance(runner, AFDRemoteMoERunner)
        assert isinstance(runner, AFDRemoteMoERunnerBase)
        assert list(runner.parameters()) == list(gate.parameters())
    else:
        assert runner.gate is None
        assert list(runner.parameters()) == []


@pytest.mark.parametrize(
    ("device_type", "connector", "compute_gate"),
    [
        ("cuda", "P2pNcclAFDConnector", False),
        ("cuda", "P2pNcclAFDConnector", True),
        ("npu", "CAMP2pAFDConnector", False),
    ],
)
def test_factory_rejects_shared_experts_on_synchronous_paths(
    monkeypatch, device_type, connector, compute_gate
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(ValueError, match="attention_shared_experts requires CAMAsync"):
        _make_runner(
            config,
            device_type=device_type,
            connector=connector,
            compute_gate_on_attention=compute_gate,
            attention_shared_experts=nn.Identity(),
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


def test_factory_rejects_cam_without_gate_before_native_construction(monkeypatch):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(ValueError, match="requires an Attention gate"):
        _make_runner(
            config,
            device_type="npu",
            connector="CAMAsyncAFDConnector",
            compute_gate_on_attention=True,
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.parametrize(
    "divisor", [0.0, -1.0, float("inf"), -float("inf"), float("nan")]
)
def test_factory_rejects_invalid_local_shared_divisor(monkeypatch, divisor):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(
        ValueError, match="shared_output_divisor_fp16 must be finite and positive"
    ):
        _make_runner(
            config,
            device_type="npu",
            connector="CAMAsyncAFDConnector",
            compute_gate_on_attention=True,
            gate=nn.Identity(),
            attention_shared_experts=nn.Identity(),
            shared_output_divisor_fp16=divisor,
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    ("device_type", "connector", "compute_gate", "has_local_shared"),
    [
        ("cuda", "P2pNcclAFDConnector", False, False),
        ("cuda", "P2pNcclAFDConnector", True, False),
        ("npu", "CAMP2pAFDConnector", False, False),
        ("npu", "CAMAsyncAFDConnector", True, False),
        ("npu", "CAMAsyncAFDConnector", True, True),
    ],
)
def test_factory_rejects_nonfinite_routed_scaling_factor(
    monkeypatch, factor, device_type, connector, compute_gate, has_local_shared
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(ValueError, match="routed_scaling_factor must be finite"):
        _make_runner(
            config,
            device_type=device_type,
            connector=connector,
            compute_gate_on_attention=compute_gate,
            gate=nn.Identity() if compute_gate else None,
            attention_shared_experts=nn.Identity() if has_local_shared else None,
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
def test_factory_preserves_finite_routed_scaling_factor_and_shared_divisor(
    monkeypatch, scale_kwargs, expected
):
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.additional_config["afd"] = {
        "role": "attention",
        "connector": "CAMAsyncAFDConnector",
        "compute_gate_on_attention": True,
    }
    monkeypatch.setattr(
        remote_moe, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(npu_remote_moe, "validate_remote_moe_config", lambda: None)
    shared = nn.Identity()
    with set_current_vllm_config(config):
        runner = remote_moe.build_attention_moe_runner(
            config,
            gate=nn.Identity(),
            attention_shared_experts=shared,
            shared_output_divisor_fp16=3.0,
            num_experts=4,
            top_k=2,
            hidden_size=7,
            intermediate_size=11,
            prefix=PREFIX,
            **scale_kwargs,
        )
    assert runner.routed_scaling_factor == expected
    assert runner.shared_output_divisor_fp16 == 3.0
    assert runner._attention_shared_experts() is shared


@pytest.mark.parametrize("shared_count", [None, 0, 2])
@pytest.mark.parametrize("has_local_shared", [False, True])
def test_factory_rejects_mixed_placement_before_native_construction(
    monkeypatch, shared_count, has_local_shared
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.additional_config["mix_placement"] = True
    monkeypatch.setattr(remote_moe.fused_moe, "FusedMoE", _unexpected_local_compute)
    with pytest.raises(RuntimeError, match="routed-only expert IDs.*mix_placement"):
        _make_runner(
            config,
            device_type="npu",
            connector="CAMAsyncAFDConnector",
            compute_gate_on_attention=True,
            gate=nn.Identity(),
            num_shared_experts=shared_count,
            attention_shared_experts=nn.Identity() if has_local_shared else None,
        )
    assert not config.compilation_config.static_forward_context
    assert not config.compilation_config.static_all_moe_layers


@pytest.mark.parametrize(
    "reserved_name",
    [
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
    ("device_type", "connector", "setting"),
    [
        ("cuda", "P2pNcclAFDConnector", "enable_eplb"),
        ("cuda", "P2pNcclAFDConnector", "num_redundant_experts"),
        ("cuda", "P2pNcclAFDConnector", "enable_return_routed_experts"),
        ("npu", "CAMP2pAFDConnector", "dynamic_eplb"),
        ("npu", "CAMP2pAFDConnector", "expert_map_path"),
        ("npu", "CAMP2pAFDConnector", "ascend_num_redundant_experts"),
    ],
)
def test_factory_rejects_local_eplb_and_capture_before_native_construction(
    monkeypatch, device_type, connector, setting
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.additional_config["afd"] = {
        "role": "attention",
        "connector": connector,
        "compute_gate_on_attention": False,
    }
    monkeypatch.setattr(
        remote_moe, "current_platform", SimpleNamespace(device_type=device_type)
    )
    message = setting
    if setting == "enable_eplb":
        config.parallel_config.enable_eplb = True
        message = "enable_eplb"
    elif setting == "num_redundant_experts":
        config.parallel_config.eplb_config.num_redundant_experts = 1
        message = "redundant experts"
    elif setting == "enable_return_routed_experts":
        config.model_config = SimpleNamespace(enable_return_routed_experts=True)
        message = "routed_experts capture"
    else:
        ascend_config = SimpleNamespace(
            eplb_config=SimpleNamespace(
                dynamic_eplb=False,
                expert_map_path=None,
                num_redundant_experts=0,
            )
        )
        if setting == "dynamic_eplb":
            ascend_config.eplb_config.dynamic_eplb = True
        elif setting == "expert_map_path":
            ascend_config.eplb_config.expert_map_path = "expert-map.json"
        else:
            ascend_config.eplb_config.num_redundant_experts = 1
            message = "Ascend redundant experts"
        ascend_config_module = ModuleType("vllm_ascend.ascend_config")
        monkeypatch.setattr(
            ascend_config_module,
            "get_ascend_config",
            lambda: ascend_config,
            raising=False,
        )
        monkeypatch.setitem(
            sys.modules, "vllm_ascend.ascend_config", ascend_config_module
        )

    monkeypatch.setattr(
        remote_moe.fused_moe,
        "FusedMoE",
        lambda **_kwargs: pytest.fail("native MoE factory must not be called"),
    )
    context_before = dict(config.compilation_config.static_forward_context)
    with pytest.raises(RuntimeError, match=message):
        remote_moe.build_attention_moe_runner(
            config,
            num_experts=4,
            top_k=2,
            hidden_size=7,
            intermediate_size=11,
            prefix=PREFIX,
        )
    assert config.compilation_config.static_forward_context == context_before


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
            live.additional_kwargs["camp2p_transfer"] = context

        def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
            live = get_forward_context()
            assert live is contexts[ubatch_idx]
            assert live.additional_kwargs["camp2p_transfer"] is events[-2][3]
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


def test_runner_selection_is_immutable_and_rejects_unsupported_paths():
    with pytest.raises(TypeError):
        remote_moe._ATTENTION_MOE_RUNNERS[("cuda", "P2pNcclAFDConnector", False)] = (
            "wrong.module",
            "WrongRunner",
        )
    for name in ("create", "register_runner", "_registry", "get_factory_kwargs"):
        assert name not in vars(AFDRemoteMoERunner)
        assert name not in vars(AFDRemoteMoERunnerBase)
    for device, connector, on_attention in (
        ("cuda", "CAMAsyncAFDConnector", True),
        ("npu", "CAMP2pAFDConnector", True),
        ("npu", "CAMAsyncAFDConnector", False),
        ("cpu", "P2pNcclAFDConnector", False),
    ):
        with pytest.raises(ValueError, match="unsupported Attention remote MoE"):
            _make_runner(
                VllmConfig(device_config=DeviceConfig("cpu")),
                device_type=device,
                connector=connector,
                compute_gate_on_attention=on_attention,
            )


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("shared_count", [None, 2])
def test_cam_runner_uses_native_selector_once(monkeypatch, balanced, shared_count):
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    hidden = torch.ones(2, 7, dtype=torch.bfloat16)
    logits = torch.arange(8).reshape(2, 4).float()
    correction_bias = torch.zeros(4)
    selected = []
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            assert hidden_states is hidden
            events.append("gate")
            return logits, None

    def select_experts(**kwargs):
        events.append("select")
        selected.append(kwargs)
        weights = torch.full((2, 2), 0.25, dtype=torch.float16)
        weights *= kwargs["routed_scaling_factor"]
        ids = torch.full((2, 2), 3, dtype=torch.int32)
        return weights, ids

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        SimpleNamespace(select_experts=select_experts),
    )
    monkeypatch.setattr(
        npu_remote_moe, "force_balanced_topk_ids_enabled", lambda: balanced
    )
    gate = Gate()
    gate.weight = torch.nn.Parameter(torch.ones(4, 7))
    config.additional_config["mix_placement"] = False
    monkeypatch.setattr(npu_remote_moe, "validate_remote_moe_config", lambda: None)
    runner = _make_runner(
        config,
        device_type="npu",
        connector="CAMAsyncAFDConnector",
        compute_gate_on_attention=True,
        gate=gate,
        num_shared_experts=shared_count,
        use_grouped_topk=True,
        num_expert_group=2,
        topk_group=1,
        renormalize=False,
        scoring_func="sigmoid",
        e_score_correction_bias=correction_bias,
    )
    assert runner.gate is gate
    assert runner.is_internal_router
    assert list(runner.named_parameters()) == [("gate.weight", gate.weight)]
    assert runner.routed_experts.routed_scaling_factor == 1.0
    assert runner.routed_scaling_factor == 2.5
    monkeypatch.setattr(runner, "_forward_entry", _unexpected_local_compute)
    monkeypatch.setattr(runner.router, "select_experts", _unexpected_local_compute)
    monkeypatch.setattr(
        runner, "_maybe_apply_routed_scale_to_output", _unexpected_local_compute
    )

    weights, ids, computed_logits = runner._route_native(hidden)
    assert events == ["gate", "select"]
    assert selected[0] == {
        "hidden_states": hidden,
        "router_logits": logits,
        "top_k": 2,
        "use_grouped_topk": True,
        "renormalize": False,
        "topk_group": 1,
        "num_expert_group": 2,
        "custom_routing_function": None,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "e_score_correction_bias": correction_bias,
        "mix_placement": False,
        "num_logical_experts": 4,
        "num_shared_experts": shared_count or 0,
        "num_experts": 4,
    }
    assert weights.dtype is torch.float32
    torch.testing.assert_close(
        weights,
        torch.full_like(weights, 0.25),
        rtol=0,
        atol=0,
    )
    expected_ids = (
        torch.arange(ids.numel(), dtype=torch.int32).reshape(ids.shape).remainder(4)
        if balanced
        else torch.full_like(ids, 3)
    )
    assert torch.equal(ids, expected_ids)
    assert computed_logits is logits


@pytest.fixture
def attention_gate_runner(monkeypatch):
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    monkeypatch.setattr(npu_remote_moe, "validate_remote_moe_config", lambda: None)
    runner = _make_runner(
        VllmConfig(device_config=DeviceConfig("cpu")),
        device_type="npu",
        connector="CAMAsyncAFDConnector",
        compute_gate_on_attention=True,
        gate=nn.Linear(7, 4, bias=False),
    )
    for name in (
        "_route_native",
        "_forward_entry",
        "_forward_impl",
        "_maybe_apply_shared_experts",
        "_maybe_apply_routed_scale_to_output",
        "_maybe_reduce_final_output",
    ):
        monkeypatch.setattr(runner, name, _unexpected_local_compute)
    monkeypatch.setattr(runner.gate, "forward", _unexpected_local_compute)
    monkeypatch.setattr(runner.router, "select_experts", _unexpected_local_compute)
    monkeypatch.setattr(remote_moe, "maybe_apply_dbo_yield", _unexpected_local_compute)
    return runner


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "use_sequence_parallel"),
    [(1, 0, False), (2, 0, False), (2, 1, False), (2, 1, True)],
    ids=["tp1", "tp2-rank0", "tp2-rank1", "flashcomm1"],
)
def test_cam_dispatch_combine_uses_live_connector_and_explicit_stage(
    monkeypatch, attention_gate_runner, tp_size, tp_rank, use_sequence_parallel
):
    from afd_plugin.model_executor.models.npu import async_cam_layout

    runner = attention_gate_runner
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank),
    )
    state_before = dict(runner.__dict__)
    requires_gather = tp_size > 1 and not use_sequence_parallel
    for stage_idx, rows in enumerate((5, 3)):
        hidden = torch.arange(rows * 7, dtype=torch.bfloat16).reshape(rows, 7)
        weights = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
        ids = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
        logits = (
            torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
            if stage_idx == 0
            else None
        )
        sent: list[tuple] = []

        def send(hidden_states, transfer_context, *, sent=sent, **kwargs):
            sent.append((hidden_states, transfer_context.metadata, kwargs))

        send_connector = SimpleNamespace(
            send_attn_output=send, recv_ffn_output=_unexpected_local_compute
        )
        send_context = _context(send_connector, 8)
        send_metadata = send_context.additional_kwargs["afd_metadata"]
        send_metadata.stage_idx = 9
        with override_forward_context(send_context):
            dispatch_ref, layout = runner._dispatch_cam(
                hidden,
                weights,
                ids,
                logits,
                stage_idx=stage_idx,
                use_sequence_parallel=use_sequence_parallel,
            )
        assert len(sent) == 1
        sent_hidden, metadata, routing = sent[0]
        assert dispatch_ref is sent_hidden
        assert metadata.layer_idx == runner.layer_id == 3
        assert metadata.stage_idx == stage_idx
        assert metadata.seq_lens == [dispatch_ref.shape[0]]
        assert send_metadata.stage_idx == 9
        assert send_context.ubatch_idx == 8
        assert set(routing) == {"topk_weights", "topk_ids", "router_logits"}
        assert layout.parent_tokens == rows
        assert layout.requires_tp_all_gather is requires_gather
        assert layout.use_sequence_parallel is use_sequence_parallel
        assert (layout.tp_size, layout.tp_rank) == (tp_size, tp_rank)
        for actual, original in zip(
            (
                dispatch_ref,
                routing["topk_weights"],
                routing["topk_ids"],
                routing["router_logits"],
            ),
            (hidden, weights, ids, logits),
            strict=True,
        ):
            if original is None:
                assert actual is None
            elif requires_gather:
                padding = original.new_zeros((1, original.shape[1]))
                expected = torch.cat((original, padding))[layout.local_token_slice]
                assert actual.dtype == original.dtype
                assert torch.equal(actual, expected)
            else:
                assert actual is original
        assert layout.padded_tokens == rows + int(requires_gather)
        gathers: list[Tensor] = []
        if requires_gather:
            global_output = (
                torch.arange(layout.padded_tokens * 7, dtype=hidden.dtype).reshape(
                    layout.padded_tokens, 7
                )
                + 100
            )
            local_output = global_output[layout.local_token_slice]

            def all_gather(
                tensor,
                token_dim,
                *,
                local_output=local_output,
                gathers=gathers,
                global_output=global_output,
            ):
                assert tensor is local_output
                assert token_dim == 0
                gathers.append(tensor)
                return global_output

            monkeypatch.setattr(
                async_cam_layout, "tensor_model_parallel_all_gather", all_gather
            )
        else:
            local_output = hidden + 100
            monkeypatch.setattr(
                async_cam_layout,
                "tensor_model_parallel_all_gather",
                _unexpected_local_compute,
            )
        received: list[Tensor] = []

        def receive(
            *,
            ref_tensor,
            ubatch_idx,
            dispatch_ref=dispatch_ref,
            stage_idx=stage_idx,
            received=received,
            local_output=local_output,
        ):
            assert ref_tensor is dispatch_ref
            assert ubatch_idx == stage_idx
            received.append(ref_tensor)
            return local_output

        recv_connector = SimpleNamespace(
            send_attn_output=_unexpected_local_compute, recv_ffn_output=receive
        )
        recv_context = _context(recv_connector, 8)
        recv_metadata = recv_context.additional_kwargs["afd_metadata"]
        recv_metadata.stage_idx = 9
        with override_forward_context(recv_context):
            output = runner._combine_cam(dispatch_ref, layout, stage_idx=stage_idx)
        assert len(received) == 1
        assert len(gathers) == int(requires_gather)
        if requires_gather:
            assert torch.equal(output, global_output[:rows])
        else:
            assert output is local_output
        assert recv_metadata.stage_idx == 9
        assert recv_context.ubatch_idx == 8
        assert runner.__dict__.keys() == state_before.keys()
        assert all(
            runner.__dict__[name] is value for name, value in state_before.items()
        )


@pytest.mark.parametrize("failure", ["send", "recv"])
def test_cam_dispatch_combine_propagates_original_error(
    monkeypatch, attention_gate_runner, failure
):
    from afd_plugin.model_executor.models.npu import async_cam_layout

    runner = attention_gate_runner
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    hidden = torch.ones(2, 7, dtype=torch.bfloat16)
    weights = torch.ones(2, 2, dtype=torch.float32)
    ids = torch.ones(2, 2, dtype=torch.int32)
    error = RuntimeError("CAM transport failed")
    events = []

    def send(*args, **kwargs):
        events.append("send")
        if failure == "send":
            raise error

    def receive(**kwargs):
        events.append("recv")
        raise error

    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    state_before = dict(runner.__dict__)
    with (
        override_forward_context(_context(connector, 9)),
        pytest.raises(RuntimeError) as caught,
    ):
        dispatch_ref, layout = runner._dispatch_cam(
            hidden,
            weights,
            ids,
            None,
            stage_idx=1,
            use_sequence_parallel=False,
        )
        monkeypatch.setattr(
            async_cam_layout,
            "tensor_model_parallel_all_gather",
            _unexpected_local_compute,
        )
        runner._combine_cam(dispatch_ref, layout, stage_idx=1)
    assert caught.value is error
    assert events == (["send"] if failure == "send" else ["send", "recv"])
    assert runner.__dict__.keys() == state_before.keys()
    assert all(runner.__dict__[name] is value for name, value in state_before.items())


@pytest.mark.npu
@pytest.mark.parametrize("import_order", ["before", "after"])
@pytest.mark.parametrize("connector", ["CAMP2pAFDConnector", "CAMAsyncAFDConnector"])
def test_real_ascend_factory_and_post_load_in_isolated_process(import_order, connector):
    pytest.importorskip("vllm_ascend")
    runner_name = (
        "AFDCAMAsyncMoERunner"
        if connector == "CAMAsyncAFDConnector"
        else "AFDRemoteMoERunner"
    )
    program = f"""
import runpy
import vllm_ascend.ops
if {import_order!r} == 'before':
    import afd_plugin.model_executor.models.deepseek_v2
from vllm_ascend.platform import NPUPlatform
NPUPlatform.pre_register_and_update()
from vllm_ascend.patch.platform import patch_fused_moe
if {import_order!r} == 'after':
    import afd_plugin.model_executor.models.deepseek_v2
from vllm_ascend.ascend_config import init_ascend_config
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
config = namespace['VllmConfig'](device_config=namespace['DeviceConfig']('cpu'))
init_ascend_config(config)
assert namespace['fused_moe'].FusedMoE is patch_fused_moe._ascend_FusedMoE
runner = namespace['_make_runner'](
    config, device_type='npu', connector={connector!r},
    compute_gate_on_attention={connector == "CAMAsyncAFDConnector"!r},
    gate=(namespace['nn'].Identity()
          if {connector == "CAMAsyncAFDConnector"!r} else None),
)
assert type(runner).__name__ == {runner_name!r}
assert type(runner.routed_experts) is namespace['AFDRemoteRoutedExperts']
check = namespace['test_real_post_load_is_parameter_free_and_keeps_quant_method']
check((config, runner))
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=120)


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("npu", marks=pytest.mark.npu),
    ],
)
def test_native_kernel_loopback_equivalence(tmp_path, device_type):
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA hardware required")
        preimport = ""
    else:
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU hardware required")
        preimport = "import vllm_ascend.ops\n"
    program = (
        preimport
        + f"""
import runpy
from pathlib import Path
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
namespace['_run_native_kernel_loopback'](
    Path({str(tmp_path)!r}), {device_type!r},
    include_external_routing={device_type == "cuda"!r},
)
"""
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=240)


def _run_native_kernel_loopback(
    model_dir, device_type, *, include_external_routing=False
):
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

    from afd_plugin.model_executor.models import deepseek_v2 as adapter

    if device_type == "npu":
        from vllm_ascend.worker.worker import NPUWorker as Worker

        from afd_plugin.compat.npu.forward_context import ascend_forward_context
    else:
        from vllm.v1.worker.gpu_worker import Worker

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
            torch.device(device_type),
        ):
            reference = native.DeepseekV2MoE(
                config=hf_config,
                parallel_config=config.parallel_config,
                prefix="model.layers.0.mlp",
                apply_routed_scale_to_output=True,
            )
            config.additional_config["afd"] = {
                "role": "attention",
                "connector": (
                    "CAMP2pAFDConnector"
                    if device_type == "npu"
                    else "P2pNcclAFDConnector"
                ),
                "compute_gate_on_attention": False,
            }
            with torch.no_grad():
                for parameter in reference.parameters():
                    fixed = torch.arange(parameter.numel(), device=device_type)
                    parameter.copy_((fixed.remainder(31) - 15).view_as(parameter) / 128)
            process_weights_after_loading(
                reference, config.model_config, torch.device(device_type)
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
                if device_type == "npu":
                    forward_scope = ascend_forward_context(
                        vllm_config=config,
                        afd_metadata=metadata,
                        model_instance=reference,
                        num_tokens=hidden_states.shape[0],
                    )
                else:
                    forward_scope = set_forward_context(
                        None, config, num_tokens=hidden_states.shape[0]
                    )
                with forward_scope:
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
        metadata = context.additional_kwargs["afd_metadata"]
        if device_type == "npu":
            previous_boundary = adapter.RemoteFFNProxy(layer_idx=1)
        else:
            previous_boundary = native.DeepseekV2MoE.__new__(native.DeepseekV2MoE)
            torch.nn.Module.__init__(previous_boundary)
            previous_boundary.is_sequence_parallel = False
            previous_boundary.gate = None
            previous_boundary.experts = PreviousRemoteExperts()
        with (
            set_current_vllm_config(config),
            torch.inference_mode(),
            override_forward_context(context),
        ):
            routing_modes = (False, True) if include_external_routing else (False,)
            for compute_gate_on_attention in routing_modes:
                config.additional_config["afd"]["compute_gate_on_attention"] = (
                    compute_gate_on_attention
                )
                layer_idx = 1 + int(compute_gate_on_attention)
                with set_default_torch_dtype(torch.bfloat16), torch.device(device_type):
                    remote = adapter.AFDDeepseekV2RemoteExpertsMoE(
                        config=hf_config,
                        vllm_config=config,
                        layer_idx=layer_idx,
                        prefix=f"model.layers.{layer_idx}.mlp",
                    )
                if compute_gate_on_attention:
                    assert device_type == "cuda"
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
                        token_count * hf_config.hidden_size, device=device_type
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
