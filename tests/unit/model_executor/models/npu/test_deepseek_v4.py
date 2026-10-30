# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU construction contracts; these fixtures do not execute Ascend kernels."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


class _NativeDecoder(nn.Module):
    pass


class _NativeCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = self.model_cls(vllm_config=vllm_config, prefix=prefix)

    def get_mtp_target_hidden_states(self):
        # The native public getter exposes the model-owned residual buffer.
        return self.model._mtp_hidden_buffer


class _DSAImpl:
    _oproj_buffer_pool: SimpleNamespace

    def reserve_oproj_buffers(self, device, dtype):
        pool = self._oproj_buffer_pool
        if pool.output is None:
            pool.output = torch.empty(1, device=device, dtype=dtype)
        self.output = pool.output

    def project(self, value):
        self.output.fill_(value)
        return self.output


class _OtherImpl:
    """A different implementation has no DSA reservation protocol."""


class _Attention(nn.Module):
    def __init__(self, *, implementation, max_position_embeddings, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.max_position_embeddings = max_position_embeddings
        self.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=implementation()))

    def forward(self, positions):
        return positions < self.max_position_embeddings

    def project(self, value):
        return self.dsa_attn.dsa_attn.impl.project(value)


class _MoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))


@pytest.fixture
def construct_model():
    # Execute the real constructors with CPU collaborators instead of importing
    # the NPU package and replacing its complete runtime module graph.
    path = (
        Path(__file__).resolve().parents[5]
        / "afd_plugin/model_executor/models/npu/deepseek_v4.py"
    )
    names = {
        "AFDDeepseekV4DecoderLayer",
        "AFDDeepseekV4Model",
        "AFDDeepseekV4ForCausalLM",
    }
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
    assert {node.name for node in constructors} == names
    source = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *constructors,
        ],
        type_ignores=[],
    )

    def construct(
        *,
        role="attention",
        rope_parameters=None,
        spec="none",
        last_rank=True,
        implementation=_DSAImpl,
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

        native = SimpleNamespace(
            DeepseekV4DecoderLayer=_NativeDecoder,
            DeepseekV4Model=nn.Module,
            AscendDeepseekV4ForCausalLM=_NativeCausalLM,
            current_platform=SimpleNamespace(device_type="cpu"),
            get_pp_group=lambda: SimpleNamespace(
                is_first_rank=True, is_last_rank=last_rank
            ),
            enable_dsa_cp=lambda: False,
            DeepseekV4Attention=lambda **kwargs: _Attention(
                implementation=implementation, **kwargs
            ),
            DeepseekV4MoE=_MoE,
            PPMissingLayer=nn.Identity,
            RMSNorm=lambda *args, **kwargs: nn.Identity(),
            VocabParallelEmbedding=lambda *args, **kwargs: nn.Identity(),
            make_layers=make_layers,
            make_pp_empty_intermediate_tensors=lambda _model, factory: factory,
            OProjBufferPool=lambda: SimpleNamespace(output=None),
            AscendDSAImpl=_DSAImpl,
        )
        namespace = {
            "torch": torch,
            "cast": cast,
            "nn": nn,
            "native": native,
            "_ATTENTION_ROLE": "attention",
            "_FFN_ROLE": "ffn",
            "AFD_ASYNC_CONNECTOR": "CAMAsyncAFDConnector",
            "parse_afd_config": lambda config, **kwargs: config.afd,
            "_refresh_ascend_fused_moe": lambda: None,
            "AFDDeepseekV4RemoteMoE": lambda **kwargs: nn.Identity(),
        }
        exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
        return namespace["AFDDeepseekV4ForCausalLM"](vllm_config=vllm_config)

    return construct


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


def test_other_attention_implementation_needs_no_dsa_reservation(construct_model):
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
    construct_model, rope_parameters, max_position
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
        ("eagle", False, False),
        ("draft", False, False),
    ],
)
def test_mtp_target_hidden_states_exist_before_execution(
    construct_model, spec, last_rank, needs_hidden_states
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
