# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip("torch")
    nn = torch.nn
pytest.importorskip("vllm")
from torch import nn  # noqa: E402
from vllm.config import CompilationMode, set_current_vllm_config  # noqa: E402
from vllm.distributed import parallel_state  # noqa: E402
from vllm.model_executor.layers import fused_moe  # noqa: E402

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.ffn import remote  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.model_layers import stage_type as _stage_type  # noqa: E402
from tests.unit.fakes.vllm_config import attention_moe_vllm_config  # noqa: E402

DEEPEP_M2N_ATTENTION = AFDConfig(
    role="attention",
    connector=DEEPEP_M2N_CONNECTOR,
    compute_gate_on_attention=True,
)
NATIVE_MOE_BLOCK = adapter.native.Qwen3MoeSparseMoeBlock
NATIVE_MLP = adapter.native.Qwen3MoeMLP


class _FakeMissingLayer(nn.Module):
    pass


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list[str]] = {
        "attention": [],
        "dense": [],
        "gate": [],
        "moe": [],
        "norm": [],
    }

    def bind(stage_type):
        return lambda *args, **kwargs: stage_type(calls, *args, **kwargs)

    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeAttention",
        bind(_stage_type("attention")),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeMLP",
        bind(_stage_type("dense")),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeSparseMoeBlock",
        bind(_stage_type("moe")),
    )
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(_stage_type("norm")))
    monkeypatch.setattr(
        adapter.native,
        "GateLinear",
        bind(_stage_type("gate")),
    )
    monkeypatch.setattr(adapter.native, "PPMissingLayer", _FakeMissingLayer)
    return calls


def _model_config(
    *,
    layer_count: int = 48,
    mlp_only_layers: list[int] | None = None,
    decoder_sparse_step: int = 1,
):
    config = SimpleNamespace(
        attention_bias=False,
        decoder_sparse_step=decoder_sparse_step,
        head_dim=8,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=4096,
        mlp_only_layers=mlp_only_layers or [],
        moe_intermediate_size=16,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=layer_count,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        tie_word_embeddings=False,
        vocab_size=128,
    )
    return SimpleNamespace(
        cache_config=None,
        lora_config=None,
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        speculative_config=None,
    )


def _platform(*, is_cuda: bool, device_type: str = "cuda") -> SimpleNamespace:
    return SimpleNamespace(device_type=device_type, is_cuda=lambda: is_cuda)


def _make_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    layer_idx: int,
    vllm_config=None,
    afd_config: AFDConfig | None = None,
):
    if afd_config is None:
        afd_config = AFDConfig(role=role)
    # The model and the native-factory runner read the same placement.
    for module in (adapter, remote):
        monkeypatch.setattr(
            module,
            "parse_afd_config",
            lambda *_args, **_kwargs: afd_config,
        )
    if vllm_config is None:
        vllm_config = _model_config()
    return adapter.AFDQwen3MoeDecoderLayer(
        vllm_config,
        f"model.layers.{layer_idx}",
    )


def _single_ep_rank() -> SimpleNamespace:
    return SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1))


@pytest.fixture
def attention_moe_env(monkeypatch: pytest.MonkeyPatch):
    """A real vLLM config and CUDA placement for the Attention MoE runners."""
    config = attention_moe_vllm_config(
        hf_text_config=_model_config().model_config.hf_text_config,
    )
    monkeypatch.setattr(remote, "current_platform", _platform(is_cuda=True))
    monkeypatch.setattr(adapter.native, "get_ep_group", _single_ep_rank)
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_world_size", lambda: 1
    )
    with set_current_vllm_config(config):
        yield config


@pytest.fixture
def cuda_construction_env(
    monkeypatch: pytest.MonkeyPatch, construction_env, attention_moe_env
):
    """Stand-in native stages, except vLLM's MoE block with real runners."""
    monkeypatch.setattr(adapter.native, "Qwen3MoeSparseMoeBlock", NATIVE_MOE_BLOCK)
    return attention_moe_env


def test_attention_role_constructs_only_attention_and_norms(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """With the gate on the FFN, the Attention MoE block owns no parameters.

    Why: the FFN role owns the gate and every expert.
    """
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        vllm_config=cuda_construction_env,
    )

    assert isinstance(layer.mlp, NATIVE_MOE_BLOCK)
    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    assert list(layer.mlp.buffers()) == []


def test_deepep_m2n_attention_owns_only_the_native_gate(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """Under DeepEP M2N, Attention owns only the native gate, at mlp.gate.weight.

    Why: Attention routes with the checkpoint's gate; FFN owns every expert.
    """
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=5,
        vllm_config=cuda_construction_env,
        afd_config=DEEPEP_M2N_ATTENTION,
    )

    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "mlp.gate.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    assert list(layer.mlp.experts.buffers()) == []


@pytest.mark.parametrize(
    ("afd_config", "tp_size", "sequence_parallel"),
    [
        (DEEPEP_M2N_ATTENTION, 1, False),
        (DEEPEP_M2N_ATTENTION, 2, True),
        (AFDConfig(role="attention"), 2, False),
    ],
    ids=["m2n-tp1", "m2n-tp2", "p2p-tp2"],
)
def test_attention_tp_ranks_send_their_own_slice_only_with_routes(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
    afd_config: AFDConfig,
    tp_size: int,
    sequence_parallel: bool,
) -> None:
    """Only with routes and Attention TP > 1 is the MoE block sequence-parallel.

    Why: TP peers hold the same tokens, so each must send a disjoint slice.
    """
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        vllm_config=cuda_construction_env,
        afd_config=afd_config,
    )

    assert layer.mlp.is_sequence_parallel is sequence_parallel


def test_attention_moe_with_ffn_gate_sends_hidden_states_only(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """With the gate on the FFN, the MoE forward sends only the hidden states.

    Why: the FFN role runs its own native gate and experts on them.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events, stage_idx=0)
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=3,
        vllm_config=cuda_construction_env,
    )
    hidden_states = torch.ones((2, 16))

    output = layer.mlp(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert events[0][2].metadata.layer_idx == 3
    assert torch.equal(events[0][1], hidden_states)
    assert events[0][3] == {}
    assert torch.equal(output, hidden_states * 0.25)


@pytest.mark.parametrize(
    ("tp_size", "outcome"),
    [
        (1, nullcontext()),
        (16, pytest.raises(ValueError, match="greater than the number of experts")),
    ],
    ids=["built", "native-error"],
)
def test_attention_moe_restores_vllm_experts_factory(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
    tp_size: int,
    outcome,
) -> None:
    """vLLM's experts factory is restored after the Attention MoE block,
    also when its constructor raised.

    Why: models built later in the process need vLLM's experts.
    """
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    with outcome:
        _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=0,
            vllm_config=cuda_construction_env,
        )

    assert adapter.native.FusedMoEFactory is fused_moe.FusedMoEFactory


def test_p2p_attention_tp2_builds_no_shared_expert(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """With P2P at Attention TP 2, Attention builds no shared expert, even one
    that TP 2 cannot split, and the model config keeps its size.

    Why: the FFN (here TP 1) runs it; vLLM's TP linear layers reject the size.
    """
    text_config = cuda_construction_env.model_config.hf_text_config
    text_config.shared_expert_intermediate_size = 3
    monkeypatch.setattr(adapter.native, "Qwen3MoeMLP", NATIVE_MLP)
    # Attention rank 0 of TP 2, as vLLM's linear layers see it.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=2, rank_in_group=0)
    )
    monkeypatch.setattr(
        adapter.native, "get_tensor_model_parallel_world_size", lambda: 2
    )

    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        vllm_config=cuda_construction_env,
    )

    assert layer.mlp.shared_expert is None
    assert layer.mlp.shared_expert_gate is None
    assert text_config.shared_expert_intermediate_size == 3


def _start_attention_build(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config,
    layer_idx: int,
    built: list[nn.Module | BaseException],
) -> threading.Thread:
    """Build an Attention decoder layer on a new thread; record it or its error."""

    def build() -> None:
        try:
            built.append(
                _make_layer(
                    monkeypatch,
                    role="attention",
                    layer_idx=layer_idx,
                    vllm_config=vllm_config,
                )
            )
        except BaseException as error:
            built.append(error)

    # Daemon: a thread stuck on a broken lock must not hang the test session.
    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    return thread


def _swap_is_owned() -> bool:
    """Whether another thread owns the experts-factory swap; never keeps it."""
    if adapter._FACTORY_SWAP_LOCK.acquire(blocking=False):
        adapter._FACTORY_SWAP_LOCK.release()
        return False
    return True


def test_attention_moe_swap_is_scoped_to_its_thread(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """While one thread builds an Attention MoE block, it still owns the swap,
    and another thread calling the factory gets vLLM's experts.

    Why: the swap rebinds a vLLM module attribute that every thread sees.
    """

    def vllm_factory(**kwargs):
        return "vllm experts"

    monkeypatch.setattr(fused_moe, "FusedMoEFactory", vllm_factory)
    monkeypatch.setattr(adapter.native, "FusedMoEFactory", vllm_factory)
    in_swap, release = threading.Event(), threading.Event()

    def paused_build_expert(vllm_config, **kwargs):
        in_swap.set()
        assert release.wait(timeout=30)
        return nn.Identity()

    monkeypatch.setattr(adapter, "build_expert", paused_build_expert)
    built: list[nn.Module | BaseException] = []
    builder = _start_attention_build(monkeypatch, cuda_construction_env, 0, built)
    try:
        assert in_swap.wait(timeout=30)
        assert adapter.native.FusedMoEFactory(num_experts=8) == "vllm experts"
        assert _swap_is_owned()
    finally:
        release.set()
        builder.join(timeout=30)

    [layer] = built
    assert isinstance(layer, nn.Module), layer
    assert isinstance(layer.mlp.experts, nn.Identity)
    assert adapter.native.FusedMoEFactory is vllm_factory


def test_attention_moe_swap_is_owned_before_vllm_factory_is_read(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """A constructor paused between reading vLLM's factory and installing its
    own already owns the swap; a second constructor waits, then swaps too.

    Why: two constructors that both read vLLM's factory would install over
    each other, and one would build local experts.
    """

    def vllm_factory(**kwargs):
        return "vllm experts"

    monkeypatch.setattr(adapter.native, "FusedMoEFactory", vllm_factory)
    read_factory, release = threading.Event(), threading.Event()

    class PausingFusedMoE:
        """Pauses the first identity check, after qwen3_moe's factory is read."""

        def __getattr__(self, name: str):
            assert name == "FusedMoEFactory", name
            if not read_factory.is_set():
                read_factory.set()
                assert release.wait(timeout=30)
            return vllm_factory

    monkeypatch.setattr(adapter, "fused_moe", PausingFusedMoE())
    monkeypatch.setattr(
        adapter, "build_expert", lambda vllm_config, **kwargs: nn.Identity()
    )
    built: list[nn.Module | BaseException] = []
    threads = [_start_attention_build(monkeypatch, cuda_construction_env, 0, built)]
    try:
        assert read_factory.wait(timeout=30)
        assert _swap_is_owned()
        threads.append(
            _start_attention_build(monkeypatch, cuda_construction_env, 1, built)
        )
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=30)

    assert len(built) == 2, built
    for layer in built:
        assert isinstance(layer, nn.Module), layer
        assert isinstance(layer.mlp.experts, nn.Identity)
    assert adapter.native.FusedMoEFactory is vllm_factory


def test_attention_moe_refuses_a_replaced_experts_factory(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """Attention refuses to swap qwen3_moe.FusedMoEFactory once it is not vLLM's.

    Why: a nested swap, or someone else's, would stack with this one.
    """
    monkeypatch.setattr(adapter.native, "FusedMoEFactory", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="already replaced"):
        _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=0,
            vllm_config=cuda_construction_env,
        )


def test_ffn_role_constructs_only_native_moe(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=0)

    assert set(dict(layer.named_parameters())) == {"mlp.weight"}


def test_model_injects_role_aware_decoder_through_native_hook(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:

    def fake_native_init(
        self,
        *,
        vllm_config,
        prefix="",
        decoder_layer_type=adapter.native.Qwen3MoeDecoderLayer,
    ) -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.layers = nn.ModuleList(
            [decoder_layer_type(vllm_config=vllm_config, prefix="model.layers.0")]
        )

    vllm_config = _model_config()
    vllm_config.compilation_config = SimpleNamespace(mode=CompilationMode.NONE)
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=True),
    )
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="ffn"),
    )
    monkeypatch.setattr(
        adapter.native.Qwen3MoeModel,
        "__init__",
        fake_native_init,
    )

    model = adapter.AFDQwen3MoeModel(vllm_config=vllm_config)

    assert set(dict(model.named_parameters())) == {"layers.0.mlp.weight"}
    assert model.afd_role == "ffn"


def test_routed_experts_shell_builds_only_native_moe_blocks(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    vllm_config = _model_config(layer_count=2)
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(
            role="ffn",
            connector=DEEPEP_M2N_CONNECTOR,
            compute_gate_on_attention=True,
        ),
    )

    model = adapter.AFDQwen3MoeRoutedExperts(
        vllm_config=vllm_config,
        prefix="draft",
    )

    assert construction_env["moe"] == [
        "draft.model.layers.0.mlp",
        "draft.model.layers.1.mlp",
    ]
    assert set(dict(model.named_parameters())) == {
        "model.layers.0.mlp.weight",
        "model.layers.1.mlp.weight",
    }
    assert model.get_experts_layer_indices() == (0, 1)
    assert model.is_unused_checkpoint_weight("lm_head.weight")
    assert not model.is_unused_checkpoint_weight("model.layers.0.mlp.gate.weight")


def test_attention_causal_lm_reports_its_remote_moe_blocks_as_native(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    """The Attention causal LM reports its MoE blocks through native metadata.

    Why: the native constructor raises if it finds no native MoE block.
    """
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        vllm_config=cuda_construction_env,
    )

    class FakeAFDModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList((layer,))
            self.embed_tokens = nn.Embedding(128, 16)
            self.make_empty_intermediate_tensors = lambda *args, **kwargs: None

    class FakeLMHead(nn.Module):
        def __init__(self, vocab_size, hidden_size, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    monkeypatch.setattr(
        adapter,
        "AFDQwen3MoeModel",
        lambda **_kwargs: FakeAFDModel(),
    )
    monkeypatch.setattr(adapter.native, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(adapter.native, "LogitsProcessor", lambda *_args: object())

    model = adapter.AFDQwen3MoeForCausalLM(vllm_config=cuda_construction_env)

    assert model.moe_layers == [layer.mlp.experts]
    assert model.num_moe_layers == 1
    assert model.num_logical_experts == 8
    assert model.num_local_physical_experts == 8


@pytest.mark.parametrize(
    ("is_cuda", "afd_config", "parallel_overrides", "message"),
    [
        (False, AFDConfig(), {}, "CUDA only"),
        (
            True,
            AFDConfig(compute_gate_on_attention=True),
            {},
            "compute_gate_on_attention=false",
        ),
        (
            True,
            AFDConfig(),
            {"use_sequence_parallel_moe": True},
            "sequence-parallel MoE",
        ),
        (True, AFDConfig(), {"enable_eplb": True}, "EPLB"),
        (
            True,
            AFDConfig(),
            {"pipeline_parallel_size": 2},
            "pipeline parallelism",
        ),
    ],
)
def test_unsupported_modes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    afd_config: AFDConfig,
    parallel_overrides: dict[str, int | bool],
    message: str,
) -> None:
    vllm_config = _model_config()
    for name, value in parallel_overrides.items():
        setattr(vllm_config.parallel_config, name, value)
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=is_cuda),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, afd_config)


def test_gate_on_attention_rejects_dense_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate-on-Attention requires every layer to be MoE.

    Why: the routed FFN runner serves MoE layers only, so a dense layer's MLP
    would run on neither role.
    """
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))

    with pytest.raises(RuntimeError, match="every layer to be MoE"):
        adapter._validate_supported_config(
            _model_config(layer_count=4, mlp_only_layers=[3]),
            AFDConfig(
                connector=DEEPEP_M2N_CONNECTOR,
                compute_gate_on_attention=True,
            ),
        )


def test_deepep_m2n_rejects_shared_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepEP M2N fails closed on a Qwen3 checkpoint with shared experts.

    Why: neither the Attention proxy nor the routed FFN runner computes a
    shared expert, so its output would be dropped without an error.
    """
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))
    vllm_config = _model_config()
    vllm_config.model_config.hf_text_config.shared_expert_intermediate_size = 64

    with pytest.raises(RuntimeError, match="does not support shared experts"):
        adapter._validate_supported_config(
            vllm_config,
            AFDConfig(
                connector=DEEPEP_M2N_CONNECTOR,
                compute_gate_on_attention=True,
            ),
        )


def test_rocm_platform_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=False, device_type="cuda"),
    )

    with pytest.raises(RuntimeError, match="CUDA only"):
        adapter._validate_supported_config(_model_config(), AFDConfig())


@pytest.mark.parametrize(
    ("config_name", "message"),
    [
        ("speculative_config", "speculative decoding"),
        ("lora_config", "LoRA"),
    ],
)
def test_unsupported_model_features_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    message: str,
) -> None:
    vllm_config = _model_config()
    setattr(vllm_config, config_name, SimpleNamespace())
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=True),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, AFDConfig())


def test_ffn_compute_delegates_to_native_mlp(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=0)
    expected = torch.full((2, 4), 7.0)
    layer.mlp.forward = lambda hidden_states: expected

    output = layer.compute_ffn_output(torch.zeros(2, 4))

    assert output is expected


def test_attention_role_rejects_local_ffn_compute(
    monkeypatch: pytest.MonkeyPatch,
    cuda_construction_env,
) -> None:
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        vllm_config=cuda_construction_env,
    )

    with pytest.raises(RuntimeError, match="requires the AFD FFN role"):
        layer.compute_ffn_output(torch.zeros(2, 4))
