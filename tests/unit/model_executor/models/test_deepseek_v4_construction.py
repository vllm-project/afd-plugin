# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")
nn = torch.nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper  # noqa: E402
from vllm.config import CompilationMode, CUDAGraphMode  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, calls, *args, prefix="", **kwargs):
        super().__init__()
        del args, kwargs
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


def _stage_type(kind: str):
    return type(f"Fake{kind.title()}", (_FakeStage,), {"kind": kind})


@pytest.fixture
def construction_env(monkeypatch):
    calls = {"attention": [], "moe": [], "norm": [], "embedding": []}

    attention_type = _stage_type("attention")
    moe_type = _stage_type("moe")
    norm_type = _stage_type("norm")
    embedding_type = _stage_type("embedding")

    monkeypatch.setattr(
        adapter.native,
        "DeepseekV4Attention",
        lambda *args, **kwargs: attention_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "DeepseekV4MoE",
        lambda *args, **kwargs: moe_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "RMSNorm",
        lambda *args, **kwargs: norm_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: embedding_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return calls


@pytest.fixture
def mtp_construction_env(monkeypatch, construction_env):
    calls = construction_env
    calls.update(linear=[], shared_head=[], mtp_embedding=[])
    linear_type = _stage_type("linear")
    shared_head_type = _stage_type("shared_head")
    mtp_embedding_type = _stage_type("mtp_embedding")

    monkeypatch.setattr(
        adapter.native_mtp,
        "ReplicatedLinear",
        lambda *args, **kwargs: linear_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "RMSNorm",
        lambda *args, **kwargs: _stage_type("norm")(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "SharedHead",
        lambda *args, **kwargs: shared_head_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: mtp_embedding_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "LogitsProcessor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    return calls


def _vllm_config(
    *,
    role: str,
    layer_count: int = 43,
    mtp: bool = False,
    dspark: bool = False,
):
    assert not (mtp and dspark)
    config = SimpleNamespace(
        hc_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hidden_size=8,
        index_topk=4,
        n_group=1,
        num_attention_heads=8,
        num_hidden_layers=layer_count,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
        rope_parameters={"original_max_position_embeddings": 64},
        routed_scaling_factor=1.5,
        vocab_size=32,
    )
    if dspark:
        config.dspark_block_size = 2
        config.dspark_target_layer_ids = [0, 2]
    return SimpleNamespace(
        additional_config={"afd": {"role": role}},
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        speculative_config=(
            SimpleNamespace(
                method="mtp",
                enforce_eager=True,
                num_speculative_tokens=1,
                draft_model_config=SimpleNamespace(hf_config=config),
            )
            if mtp or dspark
            else None
        ),
    )


def _make_layer(monkeypatch, *, role: str, layer_idx: int):
    return adapter.AFDDeepseekV4DecoderLayer(
        _vllm_config(role=role),
        f"model.layers.{layer_idx}",
    )


def _parameter_names(module: nn.Module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def test_pinned_constructor_signatures_match_native_model():
    def without_annotations(function):
        signature = inspect.signature(function)
        parameters = [
            parameter.replace(annotation=inspect.Parameter.empty)
            for parameter in signature.parameters.values()
        ]
        return signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )

    afd_layer_signature = without_annotations(
        adapter.AFDDeepseekV4DecoderLayer.__init__
    )
    native_layer_signature = without_annotations(
        adapter.native.DeepseekV2DecoderLayer.__init__
    )
    if "attn_cls" not in native_layer_signature.parameters:
        native_layer_signature = native_layer_signature.replace(
            parameters=[
                *native_layer_signature.parameters.values(),
                afd_layer_signature.parameters["attn_cls"],
            ]
        )
    assert afd_layer_signature == native_layer_signature
    assert without_annotations(
        adapter.AFDDeepseekV4Model.__init__
    ) == without_annotations(adapter.native.DeepseekV4Model.__init__)
    assert without_annotations(
        adapter.AFDDeepSeekMultiTokenPredictorLayer.__init__
    ) == without_annotations(
        adapter.native_mtp.DeepSeekMultiTokenPredictorLayer.__init__
    )
    assert without_annotations(
        adapter.AFDDeepSeekMultiTokenPredictor.__init__
    ) == without_annotations(adapter.native_mtp.DeepSeekMultiTokenPredictor.__init__)
    assert without_annotations(
        adapter.AFDDeepSeekV4MTP.__init__
    ) == without_annotations(adapter.native_mtp.DeepSeekV4MTP.__init__)
    assert without_annotations(adapter.AFDDeepSeekV4MTP.forward) == without_annotations(
        adapter.native_mtp.DeepSeekV4MTP.forward
    )


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 42])
def test_attention_layer_owns_attention_hc_and_norm_only(
    monkeypatch,
    construction_env,
    layer_idx,
):
    layer = _make_layer(monkeypatch, role="attention", layer_idx=layer_idx)
    names = _parameter_names(layer)

    assert construction_env["attention"] == [f"model.layers.{layer_idx}.self_attn"]
    assert construction_env["moe"] == []
    assert isinstance(layer.mlp, adapter.AFDDeepseekV4RemoteMoEProxy)
    assert "self_attn.weight" in names
    assert "input_layernorm.weight" in names
    assert "post_attention_layernorm.weight" in names
    assert "hc_attn_fn" in names
    assert "hc_ffn_fn" in names
    assert not any(name.startswith("mlp.") for name in names)


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 42])
def test_ffn_layer_owns_moe_only(
    monkeypatch,
    construction_env,
    layer_idx,
):
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=layer_idx)
    names = _parameter_names(layer)

    assert construction_env["attention"] == []
    assert construction_env["moe"] == [f"model.layers.{layer_idx}.mlp"]
    assert isinstance(layer.self_attn, adapter.native.PPMissingLayer)
    assert names == {"mlp.weight"}


def _patch_make_layers(monkeypatch):
    def make_layers(count, factory, *, prefix):
        layers = nn.ModuleList(
            factory(prefix=f"{prefix}.{idx}") for idx in range(count)
        )
        return 0, count, layers

    monkeypatch.setattr(adapter.native, "make_layers", make_layers)


@pytest.mark.parametrize(
    ("role", "expected_embedding", "expected_topk_buffer"),
    [("attention", True, True), ("ffn", False, False)],
)
def test_model_constructor_enforces_role_ownership(
    monkeypatch,
    construction_env,
    role,
    expected_embedding,
    expected_topk_buffer,
):
    _patch_make_layers(monkeypatch)
    model = adapter.AFDDeepseekV4Model(
        vllm_config=_vllm_config(role=role, layer_count=2),
        prefix="model",
    )
    names = _parameter_names(model)

    assert ("embed_tokens.weight" in names) is expected_embedding
    assert (model.topk_indices_buffer is not None) is expected_topk_buffer
    assert model.aux_hidden_state_layers == ()
    assert ("hc_head_fn" in names) is (role == "attention")
    if role == "attention":
        assert not any(".mlp.weight" in name for name in names)
    else:
        assert set(names) == {"layers.0.mlp.weight", "layers.1.mlp.weight"}


def test_attention_target_allocates_mtp_hidden_buffer_only_when_enabled(
    monkeypatch,
    construction_env,
):
    _patch_make_layers(monkeypatch)
    model = adapter.AFDDeepseekV4Model(
        vllm_config=_vllm_config(role="attention", layer_count=1, mtp=True),
        prefix="model",
    )

    assert model._mtp_hidden_buffer.shape == (8, 32)
    assert model._mtp_hidden_buffer.dtype == torch.bfloat16


def test_attention_target_allocates_dspark_hidden_buffer_without_mtp(
    monkeypatch,
    construction_env,
):
    _patch_make_layers(monkeypatch)
    model = adapter.AFDDeepseekV4Model(
        vllm_config=_vllm_config(
            role="attention",
            layer_count=3,
            dspark=True,
        ),
        prefix="model",
    )

    assert model.dspark_enabled is True
    assert model.mtp_enabled is False
    assert not hasattr(model, "_mtp_hidden_buffer")
    assert model._dspark_hidden_buffer.shape == (8, 16)
    assert model._dspark_hidden_buffer.dtype == torch.bfloat16


@pytest.mark.parametrize("speculative_mode", [None, "mtp", "dspark"])
def test_attention_layer_major_u2_runs_layer_then_stage(
    monkeypatch,
    speculative_mode,
):
    events = []
    active_context = [None]

    @contextmanager
    def use_forward_context(forward_context):
        previous = active_context[0]
        active_context[0] = forward_context
        try:
            yield
        finally:
            active_context[0] = previous

    class FakeConnector:
        def require_attention_pipeline_idle(self):
            events.append(("idle",))

        def wait_for_attention_stage_receive(self, *, stage_idx, tensor):
            events.append(("wait", stage_idx, tuple(tensor.shape)))

        def reset_attention_pipeline_state(self):
            events.append(("reset",))

    class FakeEmbedding(nn.Module):
        def forward(self, input_ids):
            events.append(("embed", active_context[0].ubatch_idx))
            return input_ids.float().unsqueeze(-1)

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

        def forward_attention_to_remote_ffn(
            self,
            _positions,
            hidden_states,
            _residual,
            _scaling,
        ):
            events.append(("layer", self.layer_idx, active_context[0].ubatch_idx))
            continuation = (hidden_states, hidden_states, hidden_states)
            return hidden_states + self.layer_idx + 1, continuation

        def complete_remote_ffn(self, ffn_output, _continuation):
            events.append(("complete", self.layer_idx, active_context[0].ubatch_idx))
            return ffn_output

    connector = FakeConnector()
    model = object.__new__(adapter.AFDDeepseekV4Model)
    nn.Module.__init__(model)
    model.afd_role = "attention"
    model.mtp_enabled = speculative_mode == "mtp"
    if model.mtp_enabled:
        model._mtp_hidden_buffer = torch.empty((3, 1), dtype=torch.float32)
    model.dspark_enabled = speculative_mode == "dspark"
    model._dspark_target_layer_ids = [0, 1] if model.dspark_enabled else []
    model._dspark_target_layer_id_set = frozenset(model._dspark_target_layer_ids)
    if model.dspark_enabled:
        model._dspark_hidden_buffer = torch.empty((3, 2), dtype=torch.float32)
    model.hc_mult = 1
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
    model.embed_tokens = FakeEmbedding()
    model.aux_hidden_state_layers = ()
    model.hc_head = lambda hidden_states, *_args: hidden_states.squeeze(1)
    model.hc_head_fn = None
    model.hc_head_scale = None
    model.hc_head_base = None
    model.norm = nn.Identity()

    def stage(input_ids, stage_idx):
        forward_context = SimpleNamespace(
            additional_kwargs={
                "afd_metadata": SimpleNamespace(connector=connector),
            },
            ubatch_idx=stage_idx,
        )
        return SimpleNamespace(
            context=SimpleNamespace(forward_context=forward_context),
            input_ids=input_ids,
            inputs_embeds=None,
            intermediate_tensors=None,
            positions=torch.arange(input_ids.shape[0]),
        )

    monkeypatch.setattr(adapter, "override_forward_context", use_forward_context)
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    metadata = [
        stage(torch.tensor([1, 2]), 0),
        stage(torch.tensor([10]), 1),
    ]

    outputs = model.forward_ubatches_layer_major(metadata)

    assert [output.tolist() for output in outputs] == [
        [[4.0], [5.0]],
        [[13.0]],
    ]
    assert events == [
        ("idle",),
        ("embed", 0),
        ("embed", 1),
        ("layer", 0, 0),
        ("layer", 0, 1),
        ("wait", 0, (2, 1, 1)),
        ("complete", 0, 0),
        ("layer", 1, 0),
        ("wait", 1, (1, 1, 1)),
        ("complete", 0, 1),
        ("layer", 1, 1),
        ("wait", 0, (2, 1, 1)),
        ("complete", 1, 0),
        ("wait", 1, (1, 1, 1)),
        ("complete", 1, 1),
    ]
    assert all(
        metadata[stage_idx].context.forward_context.afd_layer_major_u2
        for stage_idx in range(2)
    )
    if model.mtp_enabled:
        assert model._mtp_hidden_buffer.tolist() == [[4.0], [5.0], [13.0]]
    if model.dspark_enabled:
        assert model._dspark_hidden_buffer.tolist() == [
            [2.0, 4.0],
            [3.0, 5.0],
            [11.0, 13.0],
        ]


@pytest.mark.parametrize("hybrid_dag", [True, False])
def test_attention_graph_u2_builds_stage_local_receive_dependencies(
    monkeypatch,
    hybrid_dag,
):
    events = []
    active_context = [None]
    pending_recv_layers = {}

    @contextmanager
    def use_forward_context(forward_context):
        previous = active_context[0]
        active_context[0] = forward_context
        try:
            yield
        finally:
            active_context[0] = previous

    class FakeConnector:
        def require_attention_pipeline_idle(self):
            events.append(("idle",))

        def wait_for_attention_stage_receive(self, *, stage_idx, tensor):
            assert tensor is not None
            events.append(("wait_recv", pending_recv_layers.pop(stage_idx), stage_idx))

        def wait_for_attention_graph_receive(self, *, layer_idx, stage_idx, tensor):
            assert tensor is not None
            events.append(("graph_wait_recv", layer_idx, stage_idx))

        def reset_attention_pipeline_state(self):
            events.append(("reset",))

        def attention_graph_compute_pipeline_active(self):
            return True

        def attention_graph_hybrid_dag_active(self):
            return hybrid_dag

        @contextmanager
        def attention_graph_compute(
            self,
            *,
            layer_idx,
            stage_idx,
            tensors,
            wait_for_receive_layer_idx,
        ):
            assert tensors
            events.append(
                (
                    "fork",
                    layer_idx,
                    stage_idx,
                    wait_for_receive_layer_idx,
                )
            )
            yield
            events.append(("compute_done", layer_idx, stage_idx))

        def wait_for_attention_graph_compute(
            self,
            *,
            layer_idx,
            stage_idx,
            tensors,
        ):
            assert tensors
            events.append(("join", layer_idx, stage_idx))

        def join_attention_graph_compute(
            self,
            *,
            layer_idx,
            stage_idx,
            tensors,
        ):
            assert tensors
            events.append(("final_join", layer_idx, stage_idx))

    class FakeEmbedding(nn.Module):
        def forward(self, input_ids):
            return input_ids.float().unsqueeze(-1)

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx

        def forward_attention_to_remote_ffn_input(
            self,
            _positions,
            hidden_states,
            _residual,
            _scaling,
        ):
            stage_idx = active_context[0].ubatch_idx
            events.append(("compute", self.layer_idx, stage_idx))
            continuation = (hidden_states, hidden_states, hidden_states)
            return hidden_states + self.layer_idx + 1, continuation

        def dispatch_remote_ffn(self, hidden_states):
            stage_idx = active_context[0].ubatch_idx
            events.append(("send", self.layer_idx, stage_idx))
            return SimpleNamespace(output=hidden_states + 10)

        def receive_remote_ffn(self, transfer):
            stage_idx = active_context[0].ubatch_idx
            events.append(("recv", self.layer_idx, stage_idx))
            pending_recv_layers[stage_idx] = self.layer_idx
            return transfer.output

        def complete_remote_ffn(self, ffn_output, _continuation):
            stage_idx = active_context[0].ubatch_idx
            events.append(("complete", self.layer_idx, stage_idx))
            return ffn_output

    connector = FakeConnector()
    model = object.__new__(adapter.AFDDeepseekV4Model)
    nn.Module.__init__(model)
    model.afd_role = "attention"
    model.config = SimpleNamespace(num_hidden_layers=2)
    model.mtp_enabled = False
    model.dspark_enabled = True
    model._dspark_target_layer_ids = [0, 1]
    model._dspark_target_layer_id_set = frozenset(model._dspark_target_layer_ids)
    model._dspark_hidden_buffer = torch.empty((3, 2), dtype=torch.float32)
    model.hc_mult = 1
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
    model.embed_tokens = FakeEmbedding()
    model.aux_hidden_state_layers = ()
    model.hc_head = lambda hidden_states, *_args: hidden_states.squeeze(1)
    model.hc_head_fn = None
    model.hc_head_scale = None
    model.hc_head_base = None
    model.norm = nn.Identity()

    def stage(input_ids, stage_idx):
        forward_context = SimpleNamespace(
            additional_kwargs={
                "afd_metadata": SimpleNamespace(connector=connector),
            },
            ubatch_idx=stage_idx,
        )
        return SimpleNamespace(
            context=SimpleNamespace(forward_context=forward_context),
            input_ids=input_ids,
            inputs_embeds=None,
            intermediate_tensors=None,
            positions=torch.arange(input_ids.shape[0]),
        )

    monkeypatch.setattr(adapter, "override_forward_context", use_forward_context)
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    outputs = model.forward_ubatches_layer_major(
        [stage(torch.tensor([1, 2]), 0), stage(torch.tensor([10]), 1)]
    )

    assert [output.tolist() for output in outputs] == [
        [[24.0], [25.0]],
        [[33.0]],
    ]
    assert model._dspark_hidden_buffer.tolist() == [
        [12.0, 24.0],
        [13.0, 25.0],
        [21.0, 33.0],
    ]
    schedule = [event for event in events if event[0] in {"compute", "send", "recv"}]
    assert schedule == [
        ("compute", 0, 0),
        ("compute", 0, 1),
        ("send", 0, 0),
        ("recv", 0, 0),
        ("send", 0, 1),
        ("recv", 0, 1),
        ("compute", 1, 0),
        ("compute", 1, 1),
        ("send", 1, 0),
        ("recv", 1, 0),
        ("send", 1, 1),
        ("recv", 1, 1),
    ]
    for layer_idx in range(2):
        assert events.index(("send", layer_idx, 0)) < events.index(
            ("recv", layer_idx, 0)
        )
        assert events.index(("recv", layer_idx, 0)) < events.index(
            ("send", layer_idx, 1)
        )
    for layer_idx in range(2):
        for stage_idx in range(2):
            assert events.index(("join", layer_idx, stage_idx)) < events.index(
                ("send", layer_idx, stage_idx)
            )
            assert events.index(("recv", layer_idx, stage_idx)) < events.index(
                ("wait_recv", layer_idx, stage_idx),
            )
    expected_dependencies = (
        [
            ("fork", 0, 0, None),
            ("fork", 0, 1, None),
            ("fork", 1, 0, 0),
            ("fork", 1, 1, 0),
            ("fork", 2, 0, 1),
            ("fork", 2, 1, 1),
        ]
        if hybrid_dag
        else [
            ("fork", 0, 0, None),
            ("fork", 0, 1, None),
            ("fork", 1, 0, None),
            ("fork", 1, 1, None),
            ("fork", 2, 0, None),
            ("fork", 2, 1, None),
        ]
    )
    assert [event for event in events if event[0] == "fork"] == expected_dependencies
    assert ("final_join", 2, 0) in events
    assert ("final_join", 2, 1) in events


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_mtp_constructor_enforces_role_ownership(
    mtp_construction_env,
    role,
):
    model = adapter.AFDDeepSeekV4MTP(
        vllm_config=_vllm_config(role=role, layer_count=1, mtp=True),
    )
    names = _parameter_names(model)

    assert model.model.start_layer == 0
    assert model.model.end_layer == 1
    if role == "attention":
        assert mtp_construction_env["attention"] == ["mtp.0.self_attn"]
        assert mtp_construction_env["moe"] == []
        assert "model.layers.0.e_proj.weight" in names
        assert "model.layers.0.h_proj.weight" in names
        assert "model.layers.0.shared_head.weight" in names
        assert "model.embed_tokens.weight" in names
        assert not any("mtp_block.mlp" in name for name in names)
    else:
        assert mtp_construction_env["attention"] == []
        assert mtp_construction_env["moe"] == ["mtp.0.mlp"]
        assert names == {"model.layers.0.mtp_block.mlp.weight"}


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_mtp_eager_steps_survive_compiled_target_configuration(
    monkeypatch, mtp_construction_env, num_speculative_tokens, enforce_eager
):
    config = _vllm_config(role="attention", layer_count=1, mtp=True)
    config.speculative_config.enforce_eager = enforce_eager
    config.speculative_config.num_speculative_tokens = num_speculative_tokens
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    compiled_graphs = []

    def backend(graph, _example_inputs, **_kwargs):
        compiled_graphs.append(graph)
        return graph.forward

    def init_compiled_model(model, **_kwargs):
        # Exercise the real model decorator and its compile decision, using
        # vLLM's guard policy with a CPU backend instead of Ascend kernels.
        compiled = torch.compile(
            model.forward,
            backend=backend,
            fullgraph=True,
            dynamic=False,
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )
        model.aot_compiled_fn = lambda _model, *args, **kwargs: compiled(
            *args, **kwargs
        )

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "__init__", init_compiled_model
    )

    class StepOutput(nn.Module):
        def forward(
            self, input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        ):
            return torch.full_like(hidden_states, spec_step_idx)

    model = adapter.AFDDeepSeekV4MTP(vllm_config=config)
    model.model = StepOutput()
    torch._dynamo.reset_code(model.forward.__func__.__code__)
    inputs = torch.ones((4, 2))
    observed = [
        int(model(inputs, inputs, inputs)[0, 0])
        for _ in range(2 * num_speculative_tokens)
    ]

    if enforce_eager and num_speculative_tokens > 1:
        assert observed == list(range(num_speculative_tokens)) * 2
        assert compiled_graphs == []
    else:
        # Graph draft and single-token eager draft keep compilation. Their
        # wire step is zero, so neither relies on a changing Python counter.
        assert len(compiled_graphs) == 1
        assert observed == [0] * (2 * num_speculative_tokens)
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_afd_activation_is_required_before_construction():
    config = _vllm_config(role="attention")
    config.additional_config = {}

    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV4DecoderLayer(config, "model.layers.0")
    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV4Model(vllm_config=config, prefix="model")
