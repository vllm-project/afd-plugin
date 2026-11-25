# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare ``AFDFFNExpertsModel`` expert views with role-aware FFN CausalLM
models, bit for bit, when loaded through vLLM's ``DefaultModelLoader``."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import DeepseekV2Config, Qwen3MoeConfig  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.model_loader.default_loader import (  # noqa: E402
    DefaultModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (  # noqa: E402
    filter_safetensors_files_by_weight_name,
)
from vllm.utils.network_utils import get_open_port  # noqa: E402
from vllm.v1.worker.gpu_worker import (  # noqa: E402
    init_worker_distributed_environment,
)
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    reset_workspace_manager,
)

from afd_plugin import register_afd  # noqa: E402
from afd_plugin.config import DEEPEP_M2N_CONNECTOR  # noqa: E402
from afd_plugin.model_executor.ffn.experts_shell import (  # noqa: E402
    AFDFFNExpertsModel,
)
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
)
from afd_plugin.model_executor.models.model_utils import (  # noqa: E402
    get_afd_model_config,
)
from afd_plugin.model_executor.models.qwen3_moe import (  # noqa: E402
    AFDQwen3MoeForCausalLM,
)
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (  # noqa: E402
    local_experts_from_vllm,
    select_routed_moe_backend,
)
from tests.conftest import REPO_ROOT  # noqa: E402

pytestmark = pytest.mark.gpu

# The child process initializes vLLM distributed state and loads both models.
PARITY_TIMEOUT_S: Final[int] = 600
PARITY_RANK: Final[int] = 0
PARITY_VOCAB_SIZE: Final[int] = 128
PARITY_HIDDEN_SIZE: Final[int] = 128
PARITY_INTERMEDIATE_SIZE: Final[int] = 256
PARITY_MOE_INTERMEDIATE_SIZE: Final[int] = 128
PARITY_MODEL_LEN: Final[int] = 128
PARITY_NUM_HEADS: Final[int] = 2
PARITY_NUM_EXPERTS: Final[int] = 4
PARITY_TOP_K: Final[int] = 2

QWEN3_NUM_LAYERS: Final[int] = 2
DEEPSEEK_NUM_LAYERS: Final[int] = 3
DEEPSEEK_FIRST_K_DENSE: Final[int] = 1
DEEPSEEK_NUM_SHARED_EXPERTS: Final[int] = 1
DEEPSEEK_KV_LORA_RANK: Final[int] = 64
DEEPSEEK_QK_NOPE_HEAD_DIM: Final[int] = 32
DEEPSEEK_QK_ROPE_HEAD_DIM: Final[int] = 32
DEEPSEEK_V_HEAD_DIM: Final[int] = 32
DEEPSEEK_ROUTED_SCALING_FACTOR: Final[float] = 2.5

# Deterministic BF16/FP32 weight pattern with per-tensor seed offsets so no
# two layers, experts, or projections share identical values.
WEIGHT_PERIOD: Final[int] = 61
WEIGHT_SCALE: Final[float] = 128.0
WEIGHT_BASE_SCALE: Final[float] = 64.0
WEIGHT_SEED_STRIDE: Final[int] = 13

SHARD_NON_ROUTED: Final[str] = "model-00001-of-00002.safetensors"
SHARD_ROUTED: Final[str] = "model-00002-of-00002.safetensors"
WEIGHTS_INDEX_NAME: Final[str] = "model.safetensors.index.json"

FORBIDDEN_SHELL_TOKENS: Final[tuple[str, ...]] = (
    "embed_tokens",
    "lm_head",
    "norm",
    "shared_experts",
    "self_attn",
)

RoutingParams = dict[str, str | int | bool]
ParityCase = tuple[str, RoutingParams]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="routed-experts shell parity needs CUDA",
)
@pytest.mark.parametrize(
    ("family", "routing"),
    [
        pytest.param("qwen3_moe", {}, id="qwen3-moe"),
        pytest.param(
            "deepseek_v2",
            {
                "topk_method": "greedy",
                "scoring_func": "softmax",
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": False,
            },
            id="deepseek-v2-greedy-softmax",
        ),
        pytest.param(
            "deepseek_v2",
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": True,
            },
            id="deepseek-v2-noaux-tc-sigmoid",
        ),
    ],
)
def test_routed_experts_shell_matches_role_aware_ffn_bitwise(
    tmp_path: Path,
    family: str,
    routing: RoutingParams,
) -> None:
    """``AFDFFNExpertsModel`` subclasses produce bitwise-identical
    ``FFNLocalExperts`` views to their role-aware FFN CausalLM models when both
    load through vLLM's ``DefaultModelLoader`` from the same sharded
    safetensors checkpoint.
    """
    _run_parity_child(tmp_path, "_run_experts_shell_parity", (family, routing))


def _run_parity_child(
    tmp_path: Path,
    function: str,
    argument: ParityCase,
) -> None:
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


def _pattern_tensor(
    shape: tuple[int, ...],
    seed: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    numel = math.prod(shape)
    values = (
        (torch.arange(numel, dtype=torch.float32) + seed * WEIGHT_SEED_STRIDE)
        .remainder(WEIGHT_PERIOD)
        .sub(WEIGHT_PERIOD // 2)
        .div(WEIGHT_SCALE)
        .add((seed + 1) / WEIGHT_BASE_SCALE)
    )
    return values.view(shape).to(dtype)


def _build_qwen3_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=PARITY_VOCAB_SIZE,
        hidden_size=PARITY_HIDDEN_SIZE,
        intermediate_size=PARITY_INTERMEDIATE_SIZE,
        moe_intermediate_size=PARITY_MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=QWEN3_NUM_LAYERS,
        num_attention_heads=PARITY_NUM_HEADS,
        num_key_value_heads=PARITY_NUM_HEADS,
        max_position_embeddings=PARITY_MODEL_LEN,
        num_experts=PARITY_NUM_EXPERTS,
        num_experts_per_tok=PARITY_TOP_K,
        norm_topk_prob=True,
        tie_word_embeddings=False,
    )


def _build_deepseek_v2_config(routing: RoutingParams) -> DeepseekV2Config:
    return DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        vocab_size=PARITY_VOCAB_SIZE,
        hidden_size=PARITY_HIDDEN_SIZE,
        intermediate_size=PARITY_INTERMEDIATE_SIZE,
        moe_intermediate_size=PARITY_MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=DEEPSEEK_NUM_LAYERS,
        num_attention_heads=PARITY_NUM_HEADS,
        num_key_value_heads=PARITY_NUM_HEADS,
        q_lora_rank=None,
        kv_lora_rank=DEEPSEEK_KV_LORA_RANK,
        qk_nope_head_dim=DEEPSEEK_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DEEPSEEK_QK_ROPE_HEAD_DIM,
        v_head_dim=DEEPSEEK_V_HEAD_DIM,
        max_position_embeddings=PARITY_MODEL_LEN,
        n_routed_experts=PARITY_NUM_EXPERTS,
        n_shared_experts=DEEPSEEK_NUM_SHARED_EXPERTS,
        num_experts_per_tok=PARITY_TOP_K,
        first_k_dense_replace=DEEPSEEK_FIRST_K_DENSE,
        routed_scaling_factor=DEEPSEEK_ROUTED_SCALING_FACTOR,
        tie_word_embeddings=False,
        **routing,
    )


def _write_synthetic_checkpoint(
    checkpoint_dir: Path,
    family: str,
    hf_config: Qwen3MoeConfig | DeepseekV2Config,
    *,
    moe_layers: tuple[int, ...],
    dense_layers: tuple[int, ...],
    has_correction_bias: bool,
) -> tuple[Path, Path, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Write a two-shard safetensors checkpoint plus its index file.

    Shard 1 holds all non-routed weights needed by the full role-aware FFN
    model (embeddings, norms, lm_head, attention, dense MLP, and shared
    experts). Shard 2 holds only the MoE layers' routed experts and gates.
    """
    hf_config.save_pretrained(str(checkpoint_dir))
    non_routed: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _pattern_tensor(
            (PARITY_VOCAB_SIZE, PARITY_HIDDEN_SIZE), seed=1
        ),
        "model.norm.weight": _pattern_tensor((PARITY_HIDDEN_SIZE,), seed=2),
        "lm_head.weight": _pattern_tensor(
            (PARITY_VOCAB_SIZE, PARITY_HIDDEN_SIZE), seed=3
        ),
    }

    seed = 10
    num_layers = int(hf_config.num_hidden_layers)
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        non_routed[f"{prefix}.input_layernorm.weight"] = _pattern_tensor(
            (PARITY_HIDDEN_SIZE,), seed=seed
        )
        non_routed[f"{prefix}.post_attention_layernorm.weight"] = _pattern_tensor(
            (PARITY_HIDDEN_SIZE,), seed=seed + 1
        )
        seed += 2
        if family == "qwen3_moe":
            head_dim = PARITY_HIDDEN_SIZE // PARITY_NUM_HEADS
            for proj_name, shape in (
                ("q_proj", (PARITY_HIDDEN_SIZE, PARITY_HIDDEN_SIZE)),
                ("k_proj", (PARITY_HIDDEN_SIZE, PARITY_HIDDEN_SIZE)),
                ("v_proj", (PARITY_HIDDEN_SIZE, PARITY_HIDDEN_SIZE)),
                ("o_proj", (PARITY_HIDDEN_SIZE, PARITY_HIDDEN_SIZE)),
                ("q_norm", (head_dim,)),
                ("k_norm", (head_dim,)),
            ):
                non_routed[f"{prefix}.self_attn.{proj_name}.weight"] = _pattern_tensor(
                    shape, seed=seed
                )
                seed += 1
        else:
            qk_head_dim = DEEPSEEK_QK_NOPE_HEAD_DIM + DEEPSEEK_QK_ROPE_HEAD_DIM
            kv_proj_out = PARITY_NUM_HEADS * (
                DEEPSEEK_QK_NOPE_HEAD_DIM + DEEPSEEK_V_HEAD_DIM
            )
            for proj_name, shape in (
                ("q_proj", (PARITY_NUM_HEADS * qk_head_dim, PARITY_HIDDEN_SIZE)),
                (
                    "kv_a_proj_with_mqa",
                    (
                        DEEPSEEK_KV_LORA_RANK + DEEPSEEK_QK_ROPE_HEAD_DIM,
                        PARITY_HIDDEN_SIZE,
                    ),
                ),
                ("kv_a_layernorm", (DEEPSEEK_KV_LORA_RANK,)),
                ("kv_b_proj", (kv_proj_out, DEEPSEEK_KV_LORA_RANK)),
                (
                    "o_proj",
                    (PARITY_HIDDEN_SIZE, PARITY_NUM_HEADS * DEEPSEEK_V_HEAD_DIM),
                ),
            ):
                non_routed[f"{prefix}.self_attn.{proj_name}.weight"] = _pattern_tensor(
                    shape, seed=seed
                )
                seed += 1

    for layer_idx in dense_layers:
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        for proj_name, shape in (
            ("gate_proj", (PARITY_INTERMEDIATE_SIZE, PARITY_HIDDEN_SIZE)),
            ("up_proj", (PARITY_INTERMEDIATE_SIZE, PARITY_HIDDEN_SIZE)),
            ("down_proj", (PARITY_HIDDEN_SIZE, PARITY_INTERMEDIATE_SIZE)),
        ):
            non_routed[f"{mlp_prefix}.{proj_name}.weight"] = _pattern_tensor(
                shape, seed=seed
            )
            seed += 1

    if family == "deepseek_v2":
        shared_intermediate = PARITY_MOE_INTERMEDIATE_SIZE * DEEPSEEK_NUM_SHARED_EXPERTS
        for layer_idx in moe_layers:
            shared_prefix = f"model.layers.{layer_idx}.mlp.shared_experts"
            for proj_name, shape in (
                ("gate_proj", (shared_intermediate, PARITY_HIDDEN_SIZE)),
                ("up_proj", (shared_intermediate, PARITY_HIDDEN_SIZE)),
                ("down_proj", (PARITY_HIDDEN_SIZE, shared_intermediate)),
            ):
                non_routed[f"{shared_prefix}.{proj_name}.weight"] = _pattern_tensor(
                    shape, seed=seed
                )
                seed += 1

    routed: dict[str, torch.Tensor] = {}
    expected_w13: dict[int, torch.Tensor] = {}
    expected_w2: dict[int, torch.Tensor] = {}
    for layer_idx in moe_layers:
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        routed[f"{mlp_prefix}.gate.weight"] = _pattern_tensor(
            (PARITY_NUM_EXPERTS, PARITY_HIDDEN_SIZE), seed=seed
        )
        seed += 1
        if has_correction_bias:
            routed[f"{mlp_prefix}.gate.e_score_correction_bias"] = _pattern_tensor(
                (PARITY_NUM_EXPERTS,), seed=seed, dtype=torch.float32
            )
            seed += 1

        layer_w13: list[torch.Tensor] = []
        layer_w2: list[torch.Tensor] = []
        for expert_idx in range(PARITY_NUM_EXPERTS):
            expert_prefix = f"{mlp_prefix}.experts.{expert_idx}"
            gate_proj = _pattern_tensor(
                (PARITY_MOE_INTERMEDIATE_SIZE, PARITY_HIDDEN_SIZE), seed=seed
            )
            up_proj = _pattern_tensor(
                (PARITY_MOE_INTERMEDIATE_SIZE, PARITY_HIDDEN_SIZE), seed=seed + 1
            )
            down_proj = _pattern_tensor(
                (PARITY_HIDDEN_SIZE, PARITY_MOE_INTERMEDIATE_SIZE), seed=seed + 2
            )
            seed += 3
            routed[f"{expert_prefix}.gate_proj.weight"] = gate_proj
            routed[f"{expert_prefix}.up_proj.weight"] = up_proj
            routed[f"{expert_prefix}.down_proj.weight"] = down_proj
            layer_w13.append(torch.cat([gate_proj, up_proj], dim=0))
            layer_w2.append(down_proj)
        expected_w13[layer_idx] = torch.stack(layer_w13, dim=0)
        expected_w2[layer_idx] = torch.stack(layer_w2, dim=0)

    assert non_routed.keys().isdisjoint(routed.keys())
    shard1_path = checkpoint_dir / SHARD_NON_ROUTED
    shard2_path = checkpoint_dir / SHARD_ROUTED
    save_file(non_routed, str(shard1_path))
    save_file(routed, str(shard2_path))

    weight_map = {
        **{name: SHARD_NON_ROUTED for name in non_routed},
        **{name: SHARD_ROUTED for name in routed},
    }
    total_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*non_routed.values(), *routed.values())
    )
    (checkpoint_dir / WEIGHTS_INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        ),
        encoding="utf-8",
    )
    return shard1_path, shard2_path, expected_w13, expected_w2


def _build_ffn_vllm_config(model_dir: str) -> VllmConfig:
    return EngineArgs(
        model=model_dir,
        enable_expert_parallel=True,
        dtype="bfloat16",
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=PARITY_MODEL_LEN,
        max_num_batched_tokens=PARITY_MODEL_LEN,
        additional_config={
            "afd": {
                "role": "ffn",
                "connector": DEEPEP_M2N_CONNECTOR,
                "compute_gate_on_attention": True,
                "num_attention_ranks": 1,
                "num_ffn_ranks": 1,
            }
        },
    ).create_engine_config()


def _load_ffn_model(
    model_dir: str,
    *,
    routed_experts_connector: bool,
) -> nn.Module:
    load_vllm_config = _build_ffn_vllm_config(model_dir)
    select_routed_moe_backend(load_vllm_config)
    afd_model_config = get_afd_model_config(
        load_vllm_config.model_config,
        device_type="cuda",
        routed_experts_connector=routed_experts_connector,
    )
    load_vllm_config.model_config = afd_model_config
    loader = DefaultModelLoader(load_vllm_config.load_config)
    with set_current_vllm_config(load_vllm_config):
        return loader.load_model(
            vllm_config=load_vllm_config,
            model_config=afd_model_config,
        )


def _assert_shell_parity(
    *,
    shell: nn.Module,
    full_model: nn.Module,
    expected_shell_cls: str,
    expected_full_cls: str,
    expected_moe_layers: tuple[int, ...],
    dense_layers: tuple[int, ...],
    expected_w13: dict[int, torch.Tensor],
    expected_w2: dict[int, torch.Tensor],
    shard_paths: list[str],
    routed_shard_path: str,
    has_correction_bias: bool,
) -> None:
    assert type(shell).__name__ == expected_shell_cls
    assert type(full_model).__name__ == expected_full_cls
    assert isinstance(shell, AFDFFNExpertsModel)
    assert isinstance(full_model, (AFDQwen3MoeForCausalLM, AFDDeepseekV2ForCausalLM))

    assert shell.get_experts_layer_indices() == expected_moe_layers
    assert shell.get_experts_layer_indices() == full_model.get_experts_layer_indices()

    for layer_idx in shell.get_experts_layer_indices():
        shell_routed = shell.get_routed_experts(layer_idx)
        full_routed = full_model.get_routed_experts(layer_idx)
        shell_params = {
            (name, tuple(param.shape), param.dtype)
            for name, param in shell_routed.named_parameters()
        }
        full_params = {
            (name, tuple(param.shape), param.dtype)
            for name, param in full_routed.named_parameters()
        }
        assert shell_params == full_params

        shell_local = local_experts_from_vllm(shell_routed)
        full_local = local_experts_from_vllm(full_routed)
        assert shell_local.expert_ids == full_local.expert_ids
        assert torch.equal(shell_local.weights.w13, full_local.weights.w13)
        assert torch.equal(shell_local.weights.w2, full_local.weights.w2)
        assert shell_local.weights.w13_scale is None
        assert full_local.weights.w13_scale is None
        assert shell_local.weights.w2_scale is None
        assert full_local.weights.w2_scale is None

        # Verify against the written checkpoint directly so a silent no-op load
        # or [gate; up] swap on both models cannot pass parity.
        assert torch.equal(shell_local.weights.w13.cpu(), expected_w13[layer_idx])
        assert torch.equal(shell_local.weights.w2.cpu(), expected_w2[layer_idx])

    shell_param_names = {name for name, _ in shell.named_parameters()}
    full_param_names = {name for name, _ in full_model.named_parameters()}
    assert any("embed_tokens" in name for name in full_param_names)
    assert any("lm_head" in name for name in full_param_names)

    forbidden_tokens = (
        *FORBIDDEN_SHELL_TOKENS,
        *(f"layers.{layer_idx}." for layer_idx in dense_layers),
    )
    assert not any(
        token in name for name in shell_param_names for token in forbidden_tokens
    )
    for layer_idx in expected_moe_layers:
        bias_name = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        assert (bias_name in shell_param_names) is has_correction_bias

    kept_shards = filter_safetensors_files_by_weight_name(
        shard_paths,
        shell.is_unused_checkpoint_weight,
    )
    assert kept_shards == [routed_shard_path]


def _run_experts_shell_parity(model_dir: str, case: ParityCase) -> None:
    """Child-process body verifying shell vs role-aware FFN expert parity."""
    family, routing = case
    if family == "qwen3_moe":
        hf_config: Qwen3MoeConfig | DeepseekV2Config = _build_qwen3_config()
        expected_full_cls = "AFDQwen3MoeForCausalLM"
        expected_shell_cls = "AFDQwen3MoeRoutedExperts"
        moe_layers = tuple(range(QWEN3_NUM_LAYERS))
        dense_layers: tuple[int, ...] = ()
        has_correction_bias = False
    elif family == "deepseek_v2":
        hf_config = _build_deepseek_v2_config(routing)
        expected_full_cls = "AFDDeepseekV2ForCausalLM"
        expected_shell_cls = "AFDDeepseekV2RoutedExperts"
        moe_layers = tuple(range(DEEPSEEK_FIRST_K_DENSE, DEEPSEEK_NUM_LAYERS))
        dense_layers = tuple(range(DEEPSEEK_FIRST_K_DENSE))
        has_correction_bias = routing.get("topk_method") == "noaux_tc"
    else:
        raise ValueError(f"Unsupported parity test family: {family!r}")

    checkpoint_dir = Path(model_dir)
    shard1_path, shard2_path, expected_w13, expected_w2 = _write_synthetic_checkpoint(
        checkpoint_dir,
        family,
        hf_config,
        moe_layers=moe_layers,
        dense_layers=dense_layers,
        has_correction_bias=has_correction_bias,
    )

    register_afd()

    base_vllm_config = _build_ffn_vllm_config(model_dir)
    select_routed_moe_backend(base_vllm_config)

    device = torch.device("cuda", PARITY_RANK)
    torch.cuda.set_device(device)
    try:
        with set_current_vllm_config(base_vllm_config):
            init_worker_distributed_environment(
                base_vllm_config,
                rank=PARITY_RANK,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=PARITY_RANK,
            )
            init_workspace_manager(device)

        full_model = _load_ffn_model(model_dir, routed_experts_connector=False)
        shell = _load_ffn_model(model_dir, routed_experts_connector=True)
        _assert_shell_parity(
            shell=shell,
            full_model=full_model,
            expected_shell_cls=expected_shell_cls,
            expected_full_cls=expected_full_cls,
            expected_moe_layers=moe_layers,
            dense_layers=dense_layers,
            expected_w13=expected_w13,
            expected_w2=expected_w2,
            shard_paths=[str(shard1_path), str(shard2_path)],
            routed_shard_path=str(shard2_path),
            has_correction_bias=has_correction_bias,
        )
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
