# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare AFD Attention-side CUDA routing and MoE output with native and
mathematical oracles."""

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
from transformers import DeepseekV2Config  # noqa: E402
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
from vllm.model_executor.layers.fused_moe.router import fused_topk_router  # noqa: E402
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    process_weights_after_loading,
)
from vllm.model_executor.models import deepseek_v2 as native  # noqa: E402
from vllm.utils.network_utils import get_open_port  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402
from vllm.v1.worker.gpu_worker import (  # noqa: E402
    init_worker_distributed_environment,
)
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    reset_workspace_manager,
)

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.model_layers import GateLinear  # noqa: E402

pytestmark = pytest.mark.gpu

ROUTING_HIDDEN_SIZE = 16
ROUTING_NUM_EXPERTS = 8
ROUTING_TOP_K = 2

# The parity child process imports vLLM, initializes its distributed
# environment and compiles the Triton MoE kernels on first use.
PARITY_TIMEOUT_S = 600
PARITY_TOKEN_COUNTS = (1, 7)
PARITY_HIDDEN_SIZE = 128
PARITY_MODEL_LEN = 128
# The single rank of the child process's vLLM world.
PARITY_RANK = 0
# Deterministic values ``(i % period - period // 2) / scale``, exact in BF16.
WEIGHT_PATTERN = (31, 128)
INPUT_PATTERN = (43, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused top-k needs CUDA")
@pytest.mark.parametrize("renormalize", [True, False])
def test_routing_proxy_matches_native_router(monkeypatch, renormalize):
    """On GPU the proxy's routes equal vLLM's native fused top-k router.

    Why: the FFN applies the weights the Attention side computes; any drift
    from native routing changes the model's output.
    """
    torch.manual_seed(0)
    events: list[tuple] = []
    install_proxy_context(adapter, monkeypatch, events, stage_idx=0)
    gate = (
        GateLinear(ROUTING_HIDDEN_SIZE, ROUTING_NUM_EXPERTS, out_dtype=torch.float32)
        .cuda()
        .to(torch.bfloat16)
    )
    proxy = adapter.RemoteFFNProxy(layer_idx=3)
    proxy.gate = gate
    proxy.top_k = ROUTING_TOP_K
    proxy.renormalize = renormalize
    hidden_states = torch.randn(
        33, ROUTING_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )

    proxy(hidden_states)

    sent = events[0][3]
    router_logits, _ = gate(hidden_states)
    native_weights, native_ids = fused_topk_router.FusedTopKRouter(
        top_k=ROUTING_TOP_K,
        global_num_experts=ROUTING_NUM_EXPERTS,
        renormalize=renormalize,
    )._compute_routing(hidden_states, router_logits, indices_type=None)
    assert torch.equal(sent["topk_ids"], native_ids)
    assert torch.equal(sent["topk_weights"], native_weights)

    reference = torch.softmax(router_logits.float(), dim=-1)
    reference_weights, reference_ids = reference.topk(ROUTING_TOP_K, dim=-1)
    if renormalize:
        reference_weights = reference_weights / reference_weights.sum(-1, keepdim=True)
    assert torch.equal(
        sent["topk_ids"].long().sort(-1).values, reference_ids.sort(-1).values
    )
    torch.testing.assert_close(
        sent["topk_weights"].sort(-1).values,
        reference_weights.sort(-1).values,
        rtol=1e-3,
        atol=1e-3,
    )


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
    """On CUDA, the Attention MoE built by vLLM's FusedMoEFactory sends the
    payload the gate placement requires and returns, bit for bit, the output
    of the unsplit native MoE, for both gate placements and for 1 and 7
    tokens.

    A loopback connector runs the FFN role's native MoE in-process on what
    Attention sends: the complete MoE on the hidden states when FFN owns the
    gate, or its external-router experts on the sent router logits when
    Attention does.

    Why: splitting the MoE between the roles must not change any output.
    Both sides run the same kernels on the same inputs, so the comparison
    uses rtol=0 and atol=0.
    """
    program = (
        "import runpy\n"
        f"module = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        f"module['_run_attention_moe_parity']({str(tmp_path)!r}, {routing!r})\n"
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
            if payload:
                # FFN compute_experts_output: native runner, external routing.
                self.output = self.external_router_moe.experts(
                    hidden_states=hidden_states,
                    router_logits=payload["router_logits"],
                ).clone()
            else:
                # FFN compute_ffn_output: the complete native MoE.
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
    attention_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={
            "afd_metadata": AFDForwardContextMetadata(
                tokens_start_loc=[0],
                requests_start_loc=[0],
                stage_idx=0,
                connector=loopback,
                tokens_lens=[0],
                num_stages=1,
            ),
        },
    )
    for layer_idx, gate_on_attention in ((2, False), (3, True)):
        vllm_config.additional_config["afd"] = {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "compute_gate_on_attention": gate_on_attention,
        }
        # Equal shapes get equal values, so the gate matches the native one.
        attention_moe = _built_moe(
            partial(
                adapter.AFDDeepseekV2RemoteExpertsMoE,
                config=hf_config,
                vllm_config=vllm_config,
                layer_idx=layer_idx,
                prefix=f"model.layers.{layer_idx}.mlp",
                compute_gate_on_attention=gate_on_attention,
            ),
            vllm_config,
        )
        loopback.expected_payload = {"router_logits"} if gate_on_attention else set()
        for num_tokens in PARITY_TOKEN_COUNTS:
            hidden_states = _pattern(num_tokens * PARITY_HIDDEN_SIZE, INPUT_PATTERN)
            hidden_states = hidden_states.view(num_tokens, -1).to(torch.bfloat16)
            with torch.inference_mode():
                with override_forward_context(attention_context):
                    actual = attention_moe(hidden_states.clone()).clone()
                with set_forward_context(None, vllm_config, num_tokens=num_tokens):
                    native_output = internal_router_moe(hidden_states.clone()).clone()
            torch.testing.assert_close(actual, native_output, rtol=0, atol=0)


def _run_attention_moe_parity(
    model_dir: str,
    routing: dict[str, str | int | bool],
) -> None:
    """Child-process body of the parity test: one CUDA rank of vLLM."""
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
            _check_attention_moe_parity(vllm_config, hf_config)
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
