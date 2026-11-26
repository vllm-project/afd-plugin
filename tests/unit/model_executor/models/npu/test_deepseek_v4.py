# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role ownership and CPU construction; no Ascend kernels are executed."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest


@pytest.fixture
def torch() -> ModuleType:
    return pytest.importorskip("torch")


class _OtherImpl:
    """A different implementation has no DSA reservation protocol."""


def _compile_constructors(*names: str):
    # Execute the real constructors with CPU collaborators instead of importing
    # the NPU package and replacing its complete runtime module graph.
    path = (
        Path(__file__).resolve().parents[5]
        / "afd_plugin/model_executor/models/npu/deepseek_v4.py"
    )
    constructors = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            node.decorator_list = []
            node.body = [
                member
                for member in node.body
                if isinstance(member, ast.Assign)
                or isinstance(member, ast.FunctionDef)
                and member.name == "__init__"
            ]
            constructors.append(node)
    assert {node.name for node in constructors} == set(names)
    source = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *constructors,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(source), str(path), "exec")


@pytest.fixture
def construct_model(torch):
    layers = importlib.import_module("tests.unit.fakes.model_layers")
    nn = torch.nn
    constructors = _compile_constructors(
        "AFDDeepseekV4DecoderLayer",
        "AFDDeepseekV4Model",
        "AFDDeepseekV4ForCausalLM",
    )

    def construct(
        *,
        role="attention",
        rope_parameters=None,
        spec="none",
        last_rank=True,
        implementation=layers.DSAImpl,
    ):
        config = SimpleNamespace(
            hidden_size=4,
            vocab_size=16,
            index_topk=2,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            rope_parameters={} if rope_parameters is None else rope_parameters,
            max_position_embeddings=128,
        )
        vllm_config = SimpleNamespace(
            afd=SimpleNamespace(
                role=role,
                connector="CAMP2pAFDConnector",
                compute_gate_on_attention=False,
            ),
            model_config=SimpleNamespace(hf_config=config, dtype=torch.float16),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            parallel_config=SimpleNamespace(
                use_sequence_parallel_moe=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
            cache_config=None,
            quant_config=None,
            compilation_config=None,
            speculative_config=None
            if spec == "none"
            else SimpleNamespace(
                use_eagle=lambda: spec == "eagle",
                uses_draft_model=lambda: spec == "draft",
            ),
        )

        def make_layers(count, factory, *, prefix):
            return (
                0,
                count,
                nn.ModuleList(factory(f"{prefix}.{i}") for i in range(count)),
            )

        # Ascend 0.34 chooses DSA context parallelism from the topology. Its
        # native Attention constructor rejects legacy DSA-specific keywords.
        def make_attention(
            vllm_config,
            config,
            max_position_embeddings=0,
            cache_config=None,
            quant_config=None,
            prefix="",
            topk_indices_buffer=None,
            reduce_results=True,
        ):
            return layers.DSAAttention(
                implementation=implementation,
                max_position_embeddings=max_position_embeddings,
            )

        native = SimpleNamespace(
            DeepseekV4DecoderLayer=layers.DeepseekV4Decoder,
            DeepseekV4Model=nn.Module,
            AscendDeepseekV4ForCausalLM=layers.DeepseekV4CausalLM,
            current_platform=SimpleNamespace(device_type="cpu"),
            get_pp_group=lambda: SimpleNamespace(
                is_first_rank=True, is_last_rank=last_rank
            ),
            DeepseekV4Attention=make_attention,
            DeepseekV4MoE=layers.DeepseekV4MoE,
            PPMissingLayer=nn.Identity,
            RMSNorm=lambda *args, **kwargs: nn.Identity(),
            VocabParallelEmbedding=lambda *args, **kwargs: nn.Identity(),
            make_layers=make_layers,
            make_pp_empty_intermediate_tensors=lambda _model, factory: factory,
            OProjBufferPool=lambda: SimpleNamespace(output=None),
            AscendDSAImpl=layers.DSAImpl,
        )
        namespace = {
            "torch": torch,
            "cast": cast,
            "nn": nn,
            "native": native,
            "AFD_ASYNC_CONNECTOR": "CAMAsyncAFDConnector",
            "parse_afd_config": lambda config, **kwargs: config.afd,
            "_refresh_ascend_fused_moe": lambda: None,
            "AFDDeepseekV4RemoteMoE": lambda **kwargs: nn.Identity(),
        }
        exec(constructors, namespace)
        return namespace["AFDDeepseekV4ForCausalLM"](vllm_config=vllm_config)

    return construct


@pytest.fixture
def construct_gate(torch):
    nn = torch.nn

    def make_linear(input_size, output_size, *, bias, **_):
        return nn.Linear(input_size, output_size, bias=bias)

    namespace = {
        "nn": nn,
        "torch": torch,
        "RemoteFFNProxy": nn.Identity,
        "ReplicatedLinear": make_linear,
    }
    exec(_compile_constructors("AFDDeepseekV4AttentionGateRemoteMoE"), namespace)
    gate = namespace["AFDDeepseekV4AttentionGateRemoteMoE"]
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )

    def construct(routing_config, layer_idx, is_draft_layer):
        config = SimpleNamespace(
            hidden_size=2,
            vocab_size=4,
            num_experts_per_tok=2,
            n_routed_experts=3,
            n_shared_experts=None,
            norm_topk_prob=True,
            **routing_config,
        )
        return gate(
            config=config,
            layer_idx=layer_idx,
            prefix=f"model.layers.{layer_idx}.mlp",
            vllm_config=vllm_config,
            is_draft_layer=is_draft_layer,
        )

    return construct


@pytest.mark.parametrize(
    "routing_config, layer_idx, is_draft_layer, uses_hash",
    [
        ({"mlp_layer_types": ["moe", "hash_moe", "moe", "hash_moe"]}, 0, False, False),
        ({"mlp_layer_types": ["moe", "hash_moe", "moe", "hash_moe"]}, 3, False, True),
        ({"mlp_layer_types": ["hash_moe"], "num_hash_layers": 2}, 1, False, False),
        ({"num_hash_layers": 2}, 1, False, True),
        ({"mlp_layer_types": ["hash_moe"]}, 0, True, False),
        ({}, 0, False, False),
    ],
)
def test_gate_allocates_the_checkpoint_parameters_for_its_moe_type(
    construct_gate, torch, routing_config, layer_idx, is_draft_layer, uses_hash
):
    state = construct_gate(routing_config, layer_idx, is_draft_layer).state_dict()
    if uses_hash:
        assert state["gate.tid2eid"].dtype == torch.int32
        assert state["gate.tid2eid"].tolist() == [[0, 1], [1, 2], [2, 0], [0, 1]]
        assert "gate.e_score_correction_bias" not in state
    else:
        assert "gate.tid2eid" not in state
        correction_bias = state["gate.e_score_correction_bias"]
        assert correction_bias.dtype == torch.float32
        assert correction_bias.shape == (3,)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_model_constructs_only_role_owned_stages(construct_model, role):
    model = construct_model(role=role)
    parameters = set(dict(model.named_parameters()))

    assert ("model.layers.0.self_attn.weight" in parameters) == (role == "attention")
    assert ("model.layers.0.mlp.weight" in parameters) == (role == "ffn")


def test_attention_layers_share_model_local_reserved_output_storage(construct_model):
    model = construct_model()
    first = model.model.layers[0].self_attn.project(1)
    second = model.model.layers[1].self_attn.project(2)
    other_model = construct_model()
    other = other_model.model.layers[0].self_attn.project(3)

    assert first.data_ptr() == second.data_ptr()
    assert first.tolist() == [2]
    assert other.data_ptr() != first.data_ptr()


def test_other_attention_implementation_needs_no_dsa_reservation(
    construct_model, torch
):
    model = construct_model(implementation=_OtherImpl)

    assert model.model.layers[0].self_attn(torch.tensor([127, 128])).tolist() == [
        True,
        False,
    ]


@pytest.mark.parametrize(
    "rope_parameters, max_position",
    [
        ({"original_max_position_embeddings": 32}, 32),
        (
            {
                "original_max_position_embeddings": 32,
                "compress": {"original_max_position_embeddings": 64},
            },
            64,
        ),
        ({"original_max_position_embeddings": 32, "compress": {}}, 128),
        ({}, 128),
    ],
)
def test_decoder_preserves_native_rope_position_range(
    construct_model, torch, rope_parameters, max_position
):
    model = construct_model(rope_parameters=rope_parameters)
    positions = torch.tensor([max_position - 1, max_position])

    assert model.model.layers[0].self_attn(positions).tolist() == [True, False]


@pytest.mark.parametrize(
    "spec, last_rank, needs_hidden_states",
    [
        ("none", True, False),
        ("ngram", True, False),
        ("eagle", True, True),
        ("draft", True, True),
    ],
)
def test_mtp_target_hidden_states_exist_before_execution(
    construct_model, torch, spec, last_rank, needs_hidden_states
):
    model = construct_model(spec=spec, last_rank=last_rank)
    hidden_states = model.get_mtp_target_hidden_states()

    if needs_hidden_states:
        assert hidden_states.shape == (8, 8)
        assert hidden_states.dtype == torch.float16
        assert hidden_states.device.type == "cpu"
        assert model.get_mtp_target_hidden_states() is hidden_states
    else:
        assert hidden_states is None


ATTENTION_ROLES = frozenset(("attention",))
FFN_ROLES = frozenset(("ffn",))
BOTH_ROLES = frozenset(("attention", "ffn"))


@pytest.fixture(scope="module")
def checkpoint_weight_roles() -> Callable[..., frozenset[str]]:
    # Execute the pure helper against the shared role constants without
    # importing torch or the native model class graph.
    path = (
        Path(__file__).resolve().parents[5]
        / "afd_plugin/model_executor/models/npu/deepseek_v4.py"
    )
    helpers: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_weight_roles"
    ]
    from afd_plugin.model_executor.models._weight_utils import (
        ATTENTION_ROLE,
        BOTH_ROLES,
        FFN_ROLE,
        layer_weight_path,
    )

    namespace: dict[str, object] = {
        "ATTENTION_ROLE": ATTENTION_ROLE,
        "BOTH_ROLES": BOTH_ROLES,
        "FFN_ROLE": FFN_ROLE,
        "layer_weight_path": layer_weight_path,
    }
    module = ast.Module(body=helpers, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return cast(Callable[..., frozenset[str]], namespace["_checkpoint_weight_roles"])


@pytest.mark.parametrize(
    "name,ffn_gate_roles,attention_gate_roles",
    [
        ("model.layers.0.mlp.gate.tid2eid", FFN_ROLES, BOTH_ROLES),
        ("model.layers.7.ffn.gate.tid2eid", FFN_ROLES, BOTH_ROLES),
        ("model.layers.0.mlp.gate.weight", FFN_ROLES, BOTH_ROLES),
        ("model.layers.3.ffn.gate.weight", FFN_ROLES, BOTH_ROLES),
        ("model.layers.3.mlp.gate.e_score_correction_bias", FFN_ROLES, BOTH_ROLES),
        ("model.layers.0.attn.q_a_proj.weight", ATTENTION_ROLES, ATTENTION_ROLES),
        ("model.layers.0.self_attn.o_proj.weight", ATTENTION_ROLES, ATTENTION_ROLES),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", FFN_ROLES, FFN_ROLES),
        ("model.layers.0.ffn.shared_experts.gate_proj.weight", FFN_ROLES, FFN_ROLES),
        ("model.embed_tokens.weight", BOTH_ROLES, BOTH_ROLES),
        ("model.layers.0.hc_attn_fn", BOTH_ROLES, BOTH_ROLES),
        ("lm_head.weight", BOTH_ROLES, BOTH_ROLES),
    ],
)
def test_checkpoint_weights_follow_role_ownership(
    checkpoint_weight_roles, name, ffn_gate_roles, attention_gate_roles
):
    # Hash tables follow the router: sending one to an Attention role without
    # a gate raises KeyError in the native loader, so gate-on-FFN stays FFN-only.
    assert checkpoint_weight_roles(name, attn_owns_gate=False) == ffn_gate_roles
    assert checkpoint_weight_roles(name, attn_owns_gate=True) == attention_gate_roles


@pytest.mark.parametrize("stage", ["ffn", "mlp"])
def test_async_shared_weights_follow_attention_for_checkpoint_and_runtime_names(
    checkpoint_weight_roles, stage
):
    name = f"model.layers.0.{stage}.shared_experts.gate_proj.weight"
    assert (
        checkpoint_weight_roles(name, attention_shared_experts=True) == ATTENTION_ROLES
    )
    assert checkpoint_weight_roles(name) == FFN_ROLES
