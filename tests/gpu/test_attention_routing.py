# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare AFD's Attention-side CUDA MoE output with native vLLM's, bit for
bit."""

import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402
from transformers import DeepseekV2Config, Qwen3MoeConfig  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
    set_forward_context,
)
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)
from vllm.model_executor.models import deepseek_v2 as native  # noqa: E402
from vllm.model_executor.models import qwen3_moe as qwen3_native  # noqa: E402
from vllm.utils.network_utils import get_open_port  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402
from vllm.v1.worker.gpu_worker import (  # noqa: E402
    init_worker_distributed_environment,
)
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    reset_workspace_manager,
)

from afd_plugin.config import DEEPEP_M2N_CONNECTOR  # noqa: E402
from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as qwen3_adapter  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402

pytestmark = pytest.mark.gpu

# The child process starts vLLM and compiles the Triton MoE kernels.
PARITY_TIMEOUT_S = 600
PARITY_TOKEN_COUNTS = (1, 7)
PARITY_HIDDEN_SIZE = 128
PARITY_MODEL_LEN = 128
PARITY_RANK = 0
# Deterministic values ``(i % period - period // 2) / scale``, exact in BF16.
WEIGHT_PATTERN = (31, 128)
INPUT_PATTERN = (43, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native MoE needs CUDA")
@pytest.mark.parametrize(
    "routing",
    [
        pytest.param(
            {
                "topk_method": "greedy",
                "scoring_func": "softmax",
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": False,
            },
            id="greedy-softmax",
        ),
        pytest.param(
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": True,
            },
            id="noaux-tc-sigmoid",
        ),
    ],
)
def test_attention_moe_runner_matches_native_moe_bit_for_bit(tmp_path, routing):
    """The split Attention MoE returns the native MoE's output bit for bit,
    for both gate placements and for DeepEP M2N routes.

    Why: splitting the MoE between the roles must not change any output.
    """
    _run_parity_child(tmp_path, "_run_attention_moe_parity", routing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native MoE needs CUDA")
@pytest.mark.parametrize("renormalize", [True, False])
def test_qwen3_attention_moe_matches_native_moe_bit_for_bit(tmp_path, renormalize):
    """Qwen3's split MoE matches the native block bit for bit, for P2P and
    DeepEP M2N.

    Why: splitting the MoE between the roles must not change any output.
    """
    _run_parity_child(tmp_path, "_run_qwen3_attention_moe_parity", renormalize)


def _run_parity_child(tmp_path: Path, function: str, argument: object) -> None:
    """Run one parity check in a child process with its own vLLM rank."""
    program = (
        "import runpy\n"
        f"module = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        f"module[{function!r}]({str(tmp_path)!r}, {argument!r})\n"
    )
    # Keep vLLM's process groups and MoE workspace out of this test process.
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        cwd=REPO_ROOT,
        timeout=PARITY_TIMEOUT_S,
    )


class _LoopbackFFN:
    """Runs the FFN role's native MoE in-process on what Attention sends."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        internal_router_moe: nn.Module,
        external_router_moe: nn.Module,
    ) -> None:
        self.vllm_config = vllm_config
        self.internal_router_moe = internal_router_moe
        self.external_router_moe = external_router_moe
        self.expected_payload: set[str] = set()
        self.output: torch.Tensor | None = None

    def send_attn_output(self, hidden_states, context, **payload) -> None:
        assert set(payload) == self.expected_payload
        num_tokens = hidden_states.shape[0]
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            if "topk_ids" in payload:
                # As the DeepEP M2N FFN, run only the routed experts on the
                # routes Attention selected.
                routed_experts = self.external_router_moe.experts.routed_experts
                output = routed_experts.forward_modular(
                    x=hidden_states,
                    topk_weights=payload["topk_weights"],
                    topk_ids=payload["topk_ids"],
                )
                assert isinstance(output, torch.Tensor)
                self.output = output.clone()
            elif payload:
                self.output = self.external_router_moe.experts(
                    hidden_states=hidden_states,
                    router_logits=payload["router_logits"],
                ).clone()
            else:
                self.output = self.internal_router_moe(hidden_states).clone()

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx) -> torch.Tensor:
        assert self.output is not None
        return self.output


def _pattern(count: int, pattern: tuple[int, int]) -> torch.Tensor:
    period, scale = pattern
    values = torch.arange(count, device="cuda").remainder(period) - period // 2
    return values / scale


def _built_moe(
    module_factory: Callable[[], nn.Module],
    vllm_config: VllmConfig,
) -> nn.Module:
    """Construct a MoE as the model loader does, with deterministic weights."""
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        module = module_factory()
    with torch.no_grad():
        for parameter in module.parameters():
            values = _pattern(parameter.numel(), WEIGHT_PATTERN)
            parameter.copy_(values.view_as(parameter))
    process_weights_after_loading(
        module, vllm_config.model_config, torch.device("cuda")
    )
    return module


def _check_attention_moe_parity(
    vllm_config: VllmConfig,
    hf_config: DeepseekV2Config,
) -> None:
    def native_moe(layer_idx: int) -> nn.Module:
        return _built_moe(
            partial(
                native.DeepseekV2MoE,
                config=hf_config,
                parallel_config=vllm_config.parallel_config,
                prefix=f"model.layers.{layer_idx}.mlp",
                # As the FFN role builds it on CUDA.
                apply_routed_scale_to_output=True,
            ),
            vllm_config,
        )

    internal_router_moe = native_moe(0)
    external_router_moe = native_moe(1)
    # As the FFN role builds it when Attention computes the gate.
    external_router_moe.experts.gate = None
    loopback = _LoopbackFFN(vllm_config, internal_router_moe, external_router_moe)
    attention_context = _attention_context(vllm_config, loopback)
    for layer_idx, connector, gate_on_attention, payload in (
        (2, "P2pNcclAFDConnector", False, set()),
        (3, "P2pNcclAFDConnector", True, {"router_logits"}),
        (4, DEEPEP_M2N_CONNECTOR, True, {"topk_ids", "topk_weights"}),
    ):
        vllm_config.additional_config["afd"] = {
            "role": "attention",
            "connector": connector,
            "compute_gate_on_attention": gate_on_attention,
        }
        # Equal shapes get equal values, so the gate and the shared experts
        # match the native ones.
        attention_moe = _built_moe(
            partial(
                adapter.AFDDeepseekV2RemoteExpertsMoE,
                config=hf_config,
                vllm_config=vllm_config,
                prefix=f"model.layers.{layer_idx}.mlp",
                compute_gate_on_attention=gate_on_attention,
                sends_routes=connector == DEEPEP_M2N_CONNECTOR,
            ),
            vllm_config,
        )
        loopback.expected_payload = payload
        _assert_matches_native(
            attention_moe, internal_router_moe, attention_context, vllm_config
        )


def _check_qwen3_attention_moe_parity(vllm_config: VllmConfig) -> None:
    native_moe = _built_moe(
        partial(
            qwen3_native.Qwen3MoeSparseMoeBlock,
            vllm_config=vllm_config,
            prefix="model.layers.0.mlp",
        ),
        vllm_config,
    )
    # FFN runs the whole block, or under DeepEP M2N only its routed experts.
    loopback = _LoopbackFFN(vllm_config, native_moe, native_moe)
    attention_context = _attention_context(vllm_config, loopback)
    for layer_idx, connector, gate_on_attention, payload in (
        (1, "P2pNcclAFDConnector", False, set()),
        (2, DEEPEP_M2N_CONNECTOR, True, {"topk_ids", "topk_weights"}),
    ):
        vllm_config.additional_config["afd"] = {
            "role": "attention",
            "connector": connector,
            "compute_gate_on_attention": gate_on_attention,
        }
        # Equal shapes get equal values, so the gate matches the native one.
        attention_layer = _built_moe(
            partial(
                qwen3_adapter.AFDQwen3MoeDecoderLayer,
                vllm_config=vllm_config,
                prefix=f"model.layers.{layer_idx}",
            ),
            vllm_config,
        )
        loopback.expected_payload = payload
        _assert_matches_native(
            attention_layer.mlp, native_moe, attention_context, vllm_config
        )


def _attention_context(
    vllm_config: VllmConfig, connector: _LoopbackFFN
) -> ForwardContext:
    """Forward context of an Attention rank whose connector is ``connector``."""
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={
            "afd_metadata": AFDForwardContextMetadata(
                tokens_start_loc=[0],
                requests_start_loc=[0],
                stage_idx=0,
                connector=connector,
                tokens_lens=[0],
                num_stages=1,
            ),
        },
    )


def _assert_matches_native(
    attention_moe: nn.Module,
    native_moe: nn.Module,
    attention_context: ForwardContext,
    vllm_config: VllmConfig,
) -> None:
    for num_tokens in PARITY_TOKEN_COUNTS:
        hidden_states = _pattern(num_tokens * PARITY_HIDDEN_SIZE, INPUT_PATTERN)
        hidden_states = hidden_states.view(num_tokens, -1).to(torch.bfloat16)
        with torch.inference_mode():
            with override_forward_context(attention_context):
                actual = attention_moe(hidden_states.clone()).clone()
            with set_forward_context(None, vllm_config, num_tokens=num_tokens):
                native_output = native_moe(hidden_states.clone()).clone()
        torch.testing.assert_close(actual, native_output, rtol=0, atol=0)


def _run_attention_moe_parity(
    model_dir: str,
    routing: dict[str, str | int | bool],
) -> None:
    """Child-process body of the DeepSeek V2 parity test."""
    hf_config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        vocab_size=128,
        hidden_size=PARITY_HIDDEN_SIZE,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        max_position_embeddings=PARITY_MODEL_LEN,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        routed_scaling_factor=2.5,
        **routing,
    )
    _run_in_vllm_rank(
        model_dir,
        hf_config,
        lambda vllm_config: _check_attention_moe_parity(vllm_config, hf_config),
    )


def _run_qwen3_attention_moe_parity(model_dir: str, renormalize: bool) -> None:
    """Child-process body of the Qwen3 MoE parity test."""
    hf_config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=128,
        hidden_size=PARITY_HIDDEN_SIZE,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=PARITY_MODEL_LEN,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=renormalize,
    )
    _run_in_vllm_rank(model_dir, hf_config, _check_qwen3_attention_moe_parity)


def _run_in_vllm_rank(
    model_dir: str,
    hf_config: DeepseekV2Config | Qwen3MoeConfig,
    check: Callable[[VllmConfig], None],
) -> None:
    """Run ``check`` on one CUDA rank of vLLM built from ``hf_config``."""
    hf_config.save_pretrained(model_dir)
    vllm_config = EngineArgs(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=PARITY_MODEL_LEN,
        max_num_batched_tokens=PARITY_MODEL_LEN,
    ).create_engine_config()
    device = torch.device("cuda", PARITY_RANK)
    torch.cuda.set_device(device)
    try:
        with set_current_vllm_config(vllm_config):
            init_worker_distributed_environment(
                vllm_config,
                rank=PARITY_RANK,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=PARITY_RANK,
            )
            init_workspace_manager(device)
            check(vllm_config)
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
