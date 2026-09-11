from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from vllm.config import CompilationMode  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402

CONSTRUCTOR_DIGESTS = {
    "AFDDeepseekV2Model": (
        "b2e17233e01c98dce0d6640ad97059ee37f70a07488f9c37c2c1885780a43cd7"
    ),
    "AFDDeepseekV2DecoderLayer": (
        "f1d0b52f12063de217103f8ff1a716e62b743e1653c4dde5ef608fc39a49a780"
    ),
}


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, calls: dict[str, list[str]], *args, prefix="", **kwargs):
        super().__init__()
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


def _stage_type(kind: str):
    return type(f"Fake{kind.title()}", (_FakeStage,), {"kind": kind})


@pytest.fixture
def construction_env(monkeypatch):
    calls = {
        "attention": [],
        "dense": [],
        "gate": [],
        "moe": [],
        "norm": [],
        "stage": [],
    }

    def bind(stage_type):
        return lambda *args, **kwargs: stage_type(calls, *args, **kwargs)

    attention_type = _stage_type("attention")
    dense_type = _stage_type("dense")
    moe_type = _stage_type("moe")
    gate_type = _stage_type("gate")
    norm_type = _stage_type("norm")

    monkeypatch.setattr(adapter.native, "DeepseekAttention", bind(attention_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2Attention", bind(attention_type))
    monkeypatch.setattr(
        adapter.native,
        "DeepseekV2MLAAttention",
        bind(attention_type),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", bind(dense_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", bind(moe_type))
    monkeypatch.setattr(adapter, "ReplicatedLinear", bind(gate_type))
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(norm_type))
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )
    return calls


def _vllm_config(*, layer_count: int = 2):
    config = SimpleNamespace(
        first_k_dense_replace=1,
        hidden_act="silu",
        hidden_size=8,
        intermediate_size=16,
        model_type="deepseek",
        moe_intermediate_size=8,
        moe_layer_freq=1,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts_per_tok=2,
        num_hidden_layers=layer_count,
        q_lora_rank=None,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-6,
        routed_scaling_factor=1.0,
        topk_method="noaux_tc",
        v_head_dim=0,
        vocab_size=32,
    )
    return SimpleNamespace(
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_config=config, use_mla=False),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )


def _make_layer(
    monkeypatch,
    *,
    role: str,
    layer_idx: int,
    attention_gate: bool = False,
    vllm_config=None,
):
    afd_config = AFDConfig(
        role=role,
        compute_gate_on_attention=attention_gate,
    )
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    if vllm_config is None:
        vllm_config = _vllm_config()
    return adapter.AFDDeepseekV2DecoderLayer(
        vllm_config,
        f"model.layers.{layer_idx}",
    )


def _parameter_names(module: nn.Module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def _constructor_source(class_name: str) -> str:
    source = Path(adapter.__file__).read_text()
    source_lines = source.splitlines()
    module = ast.parse(source)
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    constructor = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return "\n".join(
        source_lines[constructor.lineno - 1 : constructor.end_lineno],
    )


def _masked_patch_sha256(source: str) -> str:
    masked_lines = []
    in_patch = False
    for line in source.splitlines():
        if "# ### PATCH START" in line:
            assert not in_patch
            indentation = line[: len(line) - len(line.lstrip())]
            masked_lines.append(f"{indentation}# <AFD PATCH>")
            in_patch = True
        elif "# ### PATCH END" in line:
            assert in_patch
            in_patch = False
        elif not in_patch:
            masked_lines.append(line.rstrip())
    assert not in_patch
    masked_source = "\n".join(masked_lines).strip() + "\n"
    return hashlib.sha256(masked_source.encode()).hexdigest()


def test_pinned_constructor_signatures_match_native_vllm():
    assert inspect.signature(adapter.AFDDeepseekV2Model.__init__) == inspect.signature(
        adapter.native.DeepseekV2Model.__init__,
    )
    assert inspect.signature(
        adapter.AFDDeepseekV2DecoderLayer.__init__,
    ) == inspect.signature(adapter.native.DeepseekV2DecoderLayer.__init__)


@pytest.mark.parametrize(
    ("class_name", "expected_digest"),
    CONSTRUCTOR_DIGESTS.items(),
)
def test_non_patch_constructor_source_matches_pinned_vllm(
    class_name: str,
    expected_digest: str,
) -> None:
    assert _masked_patch_sha256(_constructor_source(class_name)) == expected_digest


def test_standard_attention_constructs_no_ffn_parameters(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(monkeypatch, role="attention", layer_idx=0)
    moe = _make_layer(monkeypatch, role="attention", layer_idx=1)

    assert construction_env["attention"] == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    assert construction_env["dense"] == []
    assert construction_env["moe"] == []
    assert isinstance(dense.mlp, adapter.RemoteFFNProxy)
    assert isinstance(moe.mlp, adapter.RemoteFFNProxy)
    assert not any(name.startswith("mlp.") for name in _parameter_names(dense))
    assert not any(name.startswith("mlp.") for name in _parameter_names(moe))


def test_attention_gate_keeps_dense_local_and_gate_at_mlp_path(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        attention_gate=True,
    )
    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=True,
    )

    assert construction_env["dense"] == ["model.layers.0.mlp"]
    assert construction_env["moe"] == []
    assert construction_env["gate"] == ["model.layers.1.mlp.gate"]
    assert isinstance(moe.mlp, adapter.GateOnlyRemoteMoE)
    assert "mlp.gate.weight" in _parameter_names(moe)
    assert "mlp.gate.e_score_correction_bias" in _parameter_names(moe)
    assert not any("experts" in name for name in _parameter_names(moe))
    assert not isinstance(dense.mlp, adapter.RemoteFFNProxy)


def test_cuda_attention_gate_uses_v026_native_gate_contract(
    monkeypatch,
    construction_env,
):
    gate_calls = []

    class _FakeGate(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            gate_calls.append((input_size, output_size, kwargs))
            self.weight = nn.Parameter(torch.empty(output_size, input_size))

    vllm_config = _vllm_config()
    vllm_config.model_config.hf_config.moe_router_dtype = "float32"
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(adapter.native, "GateLinear", _FakeGate)
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        adapter.native,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1),
            rank_in_group=0,
        ),
    )

    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=True,
        vllm_config=vllm_config,
    )

    assert isinstance(moe.mlp, adapter.AFDDeepseekV2RemoteExpertsMoE)
    assert isinstance(moe.mlp.experts, adapter.AFDAttentionFusedMoE)
    assert gate_calls == [
        (
            8,
            4,
            {
                "out_dtype": torch.float32,
                "prefix": "model.layers.1.mlp.gate",
            },
        ),
    ]
    assert list(moe.mlp.experts.parameters()) == []
    assert list(moe.mlp.experts.buffers()) == []


def test_cuda_remote_experts_reject_eplb_on_attention(
    monkeypatch,
    construction_env,
):
    vllm_config = _vllm_config()
    vllm_config.parallel_config.enable_eplb = True
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )

    with pytest.raises(RuntimeError, match="do not support EPLB"):
        _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=1,
            attention_gate=True,
            vllm_config=vllm_config,
        )


def test_cuda_ffn_gate_uses_parameter_free_internal_router_shell(
    monkeypatch,
    construction_env,
):
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(
        adapter.native,
        "GateLinear",
        lambda *_args, **_kwargs: pytest.fail(
            "FFN-side gate must not construct an Attention gate",
        ),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        adapter.native,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1),
            rank_in_group=0,
        ),
    )

    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=False,
    )

    assert isinstance(moe.mlp, adapter.AFDDeepseekV2RemoteExpertsMoE)
    assert isinstance(moe.mlp.experts, adapter.AFDAttentionFusedMoE)
    assert moe.mlp.gate is None
    assert moe.mlp.experts.is_internal_router
    assert "forward" not in type(moe.mlp).__dict__
    assert list(moe.mlp.experts.parameters()) == []
    assert list(moe.mlp.experts.buffers()) == []
    assert not any(name.startswith("mlp.") for name in _parameter_names(moe))


def test_ffn_constructs_no_real_attention(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(monkeypatch, role="ffn", layer_idx=0)
    moe = _make_layer(monkeypatch, role="ffn", layer_idx=1)

    assert construction_env["attention"] == []
    assert construction_env["dense"] == ["model.layers.0.mlp"]
    assert construction_env["moe"] == ["model.layers.1.mlp"]
    assert isinstance(dense.self_attn, adapter.native.PPMissingLayer)
    assert isinstance(moe.self_attn, adapter.native.PPMissingLayer)
    assert not any(name.startswith("self_attn.") for name in _parameter_names(dense))
    assert not any(name.startswith("self_attn.") for name in _parameter_names(moe))


def test_attention_shell_computes_gate_before_transport(
    monkeypatch,
    construction_env,
):
    """CUDA compute_gate_on_attention=True ships real router logits.

    vLLM 0.28.0's native MoE forward passes router_logits=hidden_states and
    expects the experts runner to hold the gate. The Attention-side shell
    owns the gate, so its forward must compute the logits before transport;
    shipping hidden states breaks the P2P width contract (hidden_size vs
    n_routed_experts) on the FFN side.
    """
    construction_env(monkeypatch)
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1),
            rank_in_group=0,
        ),
    )
    monkeypatch.setattr(adapter.native, "_get_moe_router_dtype", lambda _c: None)

    gate_calls = []

    class _FakeGate(nn.Module):
        def __init__(self, hidden_size, n_experts, out_dtype=None, prefix=""):
            super().__init__()
            self.n_experts = n_experts

        def forward(self, hidden_states):
            gate_calls.append(tuple(hidden_states.shape))
            return torch.zeros((hidden_states.shape[0], self.n_experts)), None

    monkeypatch.setattr(adapter.native, "GateLinear", _FakeGate)

    sent = []

    class _RecordingProxy(adapter.AFDAttentionFusedMoE):
        def _send_and_receive(self, hidden_states, **send_kwargs):
            sent.append(send_kwargs)
            return hidden_states

    monkeypatch.setattr(adapter, "AFDAttentionFusedMoE", _RecordingProxy)

    vllm_config = _vllm_config()
    shell = adapter.AFDDeepseekV2RemoteExpertsMoE(
        config=vllm_config.model_config.hf_config,
        parallel_config=vllm_config.parallel_config,
        layer_idx=1,
        prefix="model.layers.1.mlp",
        compute_gate_on_attention=True,
    )
    hidden_states = torch.zeros((2, 8))
    output = shell.forward(hidden_states)

    assert gate_calls == [(2, 8)]
    assert len(sent) == 1
    assert sent[0]["router_logits"].shape == (
        2,
        vllm_config.model_config.hf_config.n_routed_experts,
    )
    assert torch.equal(output, hidden_states)


def test_attention_shell_without_gate_keeps_native_delegation(
    monkeypatch,
    construction_env,
):
    """With the gate on the FFN side, the shell ships hidden states only."""
    construction_env(monkeypatch)
    monkeypatch.setattr(
        adapter.native,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1),
            rank_in_group=0,
        ),
    )
    monkeypatch.setattr(adapter.native, "_get_moe_router_dtype", lambda _c: None)

    sent = []

    class _RecordingProxy(adapter.AFDAttentionFusedMoE):
        def _send_and_receive(self, hidden_states, **send_kwargs):
            sent.append(send_kwargs)
            return hidden_states

    monkeypatch.setattr(adapter, "AFDAttentionFusedMoE", _RecordingProxy)

    vllm_config = _vllm_config()
    shell = adapter.AFDDeepseekV2RemoteExpertsMoE(
        config=vllm_config.model_config.hf_config,
        parallel_config=vllm_config.parallel_config,
        layer_idx=1,
        prefix="model.layers.1.mlp",
        compute_gate_on_attention=False,
    )
    assert shell.gate is None
    assert shell.experts.is_internal_router is True

    hidden_states = torch.zeros((2, 8))
    output = shell.forward(hidden_states)

    assert sent == [{}]
    assert torch.equal(output, hidden_states)


def test_npu_ffn_refreshes_native_fused_moe_factory(
    monkeypatch,
    construction_env,
):
    stale_factory = object()
    ascend_factory = object()
    factories_seen_by_native_moe = []

    class _FakeMoE(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            factories_seen_by_native_moe.append(adapter.native.FusedMoEFactory)

    monkeypatch.setattr(adapter.native, "FusedMoEFactory", stale_factory)
    monkeypatch.setattr(adapter.fused_moe, "FusedMoEFactory", ascend_factory)
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)

    _make_layer(monkeypatch, role="ffn", layer_idx=1)

    assert factories_seen_by_native_moe == [ascend_factory]


@pytest.mark.parametrize(
    ("aiter_enabled", "apply_routed_scale_to_output"),
    [(False, True), (True, False)],
)
def test_ffn_moe_preserves_v026_native_routed_scale_placement(
    monkeypatch,
    construction_env,
    aiter_enabled,
    apply_routed_scale_to_output,
):
    moe_calls = []

    class _FakeMoE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            moe_calls.append(kwargs)
            self.experts = SimpleNamespace(gate=object())

    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)
    monkeypatch.setattr(
        adapter.native.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: aiter_enabled,
    )

    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=1,
        attention_gate=True,
    )

    assert moe_calls[0]["apply_routed_scale_to_output"] is apply_routed_scale_to_output
    assert moe.mlp.experts.gate is None


def test_ffn_side_gate_keeps_native_internal_router(
    monkeypatch,
    construction_env,
):
    native_gate = object()

    class _FakeMoE(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.experts = SimpleNamespace(gate=native_gate)

    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)
    monkeypatch.setattr(
        adapter.native.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: False,
    )

    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=1,
        attention_gate=False,
    )

    assert moe.mlp.experts.gate is native_gate


def test_attention_gate_ffn_dense_uses_non_executing_placeholder(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=0,
        attention_gate=True,
    )
    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=1,
        attention_gate=True,
    )

    assert construction_env["attention"] == []
    assert construction_env["dense"] == []
    assert construction_env["moe"] == ["model.layers.1.mlp"]
    assert isinstance(dense.mlp, adapter.native.PPMissingLayer)
    assert isinstance(moe.self_attn, adapter.native.PPMissingLayer)


def _patch_model_constructor_dependencies(
    monkeypatch,
    construction_env,
):
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        adapter.native,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: _FakeStage(construction_env, *args, **kwargs),
    )

    def make_layers(count, factory, *, prefix):
        layers = nn.ModuleList(
            factory(f"{prefix}.{layer_idx}") for layer_idx in range(count)
        )
        return 0, count, layers

    monkeypatch.setattr(adapter.native, "make_layers", make_layers)
    monkeypatch.setattr(
        adapter.native,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapter.native.DeepseekV2Model,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("native constructor was called"),
    )


def test_model_constructor_uses_role_aware_layers(
    monkeypatch,
    construction_env,
):
    vllm_config = _vllm_config()
    afd_config = AFDConfig(role="attention")
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    _patch_model_constructor_dependencies(monkeypatch, construction_env)

    model = adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")

    assert isinstance(model, adapter.native.DeepseekV2Model)
    assert all(
        isinstance(layer, adapter.AFDDeepseekV2DecoderLayer) for layer in model.layers
    )
    assert construction_env["dense"] == []
    assert construction_env["moe"] == []
    assert model.hidden_size == vllm_config.model_config.hf_config.hidden_size
    assert model.use_mha
    assert model.num_redundant_experts == 0
    assert (
        sum(
            base.__name__ == "TorchCompileWithNoGuardsWrapper"
            for base in type(model).__mro__
        )
        == 1
    )


@pytest.mark.parametrize(
    ("role", "expected_allocations"),
    [("attention", 1), ("ffn", 0)],
)
def test_v32_indexer_buffer_is_allocated_only_on_attention(
    monkeypatch,
    construction_env,
    role,
    expected_allocations,
):
    vllm_config = _vllm_config()
    vllm_config.model_config.hf_config.index_topk = 2048
    afd_config = AFDConfig(role=role)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    _patch_model_constructor_dependencies(monkeypatch, construction_env)

    original_empty = torch.empty
    indexer_allocations = []

    def track_empty(*size, **kwargs):
        if size[:2] == (8, 2048):
            indexer_allocations.append((size, kwargs.copy()))
            kwargs = {key: value for key, value in kwargs.items() if key != "device"}
        return original_empty(*size, **kwargs)

    monkeypatch.setattr(adapter.torch, "empty", track_empty)

    model = adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")

    assert model.is_v32
    assert len(indexer_allocations) == expected_allocations


def test_afd_alias_without_activation_fails_before_construction(monkeypatch):
    vllm_config = SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")
    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV2DecoderLayer(vllm_config, "model.layers.0")

    assert issubclass(
        adapter.AFDDeepseekV2ForCausalLM,
        adapter.native.DeepseekV2ForCausalLM,
    )
    assert adapter.AFDDeepseekV2ForCausalLM.model_cls is adapter.AFDDeepseekV2Model


def test_model_constructor_rejects_sequence_parallel_moe_before_allocation(
    monkeypatch,
    construction_env,
):
    vllm_config = _vllm_config()
    vllm_config.parallel_config.use_sequence_parallel_moe = True
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="ffn"),
    )

    with pytest.raises(RuntimeError, match="sequence-parallel MoE"):
        adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")

    assert all(not calls for calls in construction_env.values())
