# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-checkpoint MoE reference probe for the CUDA AFD FFN role.

Run on one otherwise idle CUDA GPU. For each selected MoE layer this builds the
checkpoint truncated after that layer and computes the layer's MoE output for
fixed inputs along four paths:

- native: the native vLLM DeepSeek layer MLP, whose runner owns the gate;
- afd: the FFN role's ``compute_ffn_output`` with the gate on FFN;
- afd-gate-on-attention: the Attention role's gate logits passed to the FFN
  role's ``compute_experts_output``;
- fp32: an independent reference computed from the raw checkpoint tensors.

    python -m tests.e2e.operators.moe_reference_cuda \
        --model /path/to/DeepSeek-V2-Lite --output /logs/dsv2-moe-reference-cuda

Both AFD paths must equal native bit for bit. Every path must match the fp32
reference within a fixed relative L2 bound on tokens whose top-k selection is
not a near tie. Outputs: report.json plus input, logits and outputs per case.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import PretrainedConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models import deepseek_v2 as native
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from afd_plugin.model_executor.models.deepseek_v2 import AFDDeepseekV2ForCausalLM
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index

TOKEN_COUNTS = (1, 7, 64, 512)
SEED = 42
# Relative L2 bound of a bfloat16 path against the fp32 reference.
REFERENCE_REL_L2 = 0.02
# A token is a near tie when its k-th and (k+1)-th routing probabilities differ
# by less than this fraction of the k-th; bfloat16 logits may flip such picks.
NEAR_TIE_RELATIVE_GAP = 0.02


def checkpoint_tensors(
    model_path: Path, keep: Callable[[str], bool]
) -> Iterator[tuple[str, torch.Tensor]]:
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as file:
            # safe_open exposes keys(), but is not an iterable mapping.
            for name in file.keys():  # noqa: SIM118
                if keep(name):
                    yield name, file.get_tensor(name)


def within_layers(name: str, last_layer: int) -> bool:
    """Whether a tensor is outside the decoder stack or at a layer <= last_layer."""
    parts = name.split(".")
    return "layers" not in parts or int(parts[parts.index("layers") + 1]) <= last_layer


def layer_tensors(model_path: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."
    return {
        name.removeprefix(prefix): tensor
        for name, tensor in checkpoint_tensors(
            model_path,
            lambda name: (
                name.startswith(prefix)
                and (".mlp." in name or "post_attention_layernorm" in name)
            ),
        )
    }


def fp32_reference(
    tensors: dict[str, torch.Tensor], hf: PretrainedConfig, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE output from raw checkpoint tensors, independent of vLLM kernels."""
    x = hidden.float()
    logits = x @ tensors["mlp.gate.weight"].cuda().float().T
    if hf.scoring_func != "softmax":
        raise ValueError(
            f"fp32 reference supports softmax scoring, not {hf.scoring_func}"
        )
    probs = logits.softmax(dim=-1)
    if hf.topk_method == "group_limited_greedy":
        grouped = probs.view(probs.shape[0], hf.n_group, -1)
        groups = grouped.amax(dim=-1).topk(hf.topk_group, dim=-1).indices
        keep = torch.zeros_like(grouped[..., 0], dtype=torch.bool)
        keep.scatter_(1, groups, True)
        scores = grouped.masked_fill(~keep[..., None], 0.0).flatten(1)
    elif hf.topk_method == "greedy":
        scores = probs
    else:
        raise ValueError(f"fp32 reference does not implement {hf.topk_method}")
    top = scores.topk(hf.num_experts_per_tok + 1, dim=-1).values
    near_tie = (top[:, -2] - top[:, -1]) < NEAR_TIE_RELATIVE_GAP * top[:, -2]
    weights, ids = scores.topk(hf.num_experts_per_tok, dim=-1)
    if hf.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    def mlp(inputs: torch.Tensor, prefix: str) -> torch.Tensor:
        gate = inputs @ tensors[f"{prefix}.gate_proj.weight"].cuda().float().T
        up = inputs @ tensors[f"{prefix}.up_proj.weight"].cuda().float().T
        activated = torch.nn.functional.silu(gate) * up
        return activated @ tensors[f"{prefix}.down_proj.weight"].cuda().float().T

    routed = torch.zeros_like(x)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        output = mlp(x[rows], f"mlp.experts.{expert}")
        routed.index_add_(0, rows, output * weights[rows, slots, None])
    shared = mlp(x, "mlp.shared_experts")
    return logits, routed * hf.routed_scaling_factor + shared, near_tie


def build_config(
    model_path: Path, last_layer: int, afd: Mapping[str, object] | None
) -> VllmConfig:
    """Engine config truncated after last_layer; afd selects the AFD role."""
    topology = {
        "connector": "P2pNcclAFDConnector",
        "num_attention_ranks": 1,
        "num_ffn_ranks": 1,
    }
    return EngineArgs(
        model=str(model_path),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.5,
        hf_overrides={"num_hidden_layers": last_layer + 1},
        additional_config={} if afd is None else {"afd": {**topology, **afd}},
    ).create_engine_config()


def load_model(
    model_cls: type[torch.nn.Module],
    config: VllmConfig,
    model_path: Path,
    last_layer: int,
):
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(config.model_config.dtype),
        torch.device("cuda"),
    ):
        model = model_cls(vllm_config=config)
        loaded = model.load_weights(
            checkpoint_tensors(model_path, lambda name: within_layers(name, last_layer))
        )
        assert loaded, "checkpoint filter did not load any parameters"
        process_weights_after_loading(model, config.model_config, torch.device("cuda"))
    return model.eval()


@contextmanager
def moe_forward_context(config: VllmConfig, layer_idx: int) -> Iterator[None]:
    """The FFN runner's forward context for one MoE layer."""
    with (
        torch.inference_mode(),
        set_current_vllm_config(config),
        set_forward_context(None, config),
    ):
        _set_moe_layer_index(get_forward_context(), layer_idx)
        yield


def path_output(
    name: str,
    model,
    layer_idx: int,
    hidden: torch.Tensor,
    gate_logits: torch.Tensor | None,
) -> torch.Tensor:
    if name == "native":
        return model.model.layers[layer_idx].mlp(hidden)
    if name == "afd":
        return model.compute_ffn_output(hidden, layer_idx)
    if name == "attention-gate":
        return model.model.layers[layer_idx].mlp.gate(hidden)[0]
    return model.compute_experts_output(hidden, layer_idx, gate_logits)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor, rows) -> float:
    error = (actual.float() - expected.float())[rows]
    return (error.norm() / expected.float()[rows].norm().clamp_min(1e-12)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29681)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    probe_config = build_config(args.model, 0, None)
    hf = probe_config.model_config.hf_config
    layers = args.layers or [hf.first_k_dense_replace]
    with set_current_vllm_config(probe_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{args.port}",
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
    init_workspace_manager(torch.device("cuda"))

    reports = []
    try:
        for layer_idx in layers:
            tensors = layer_tensors(args.model, layer_idx)
            norm_weight = tensors["post_attention_layernorm.weight"].cuda()
            inputs = {}
            for count in TOKEN_COUNTS:
                generator = torch.Generator(device="cuda").manual_seed(SEED + count)
                raw = torch.randn(
                    count, hf.hidden_size, device="cuda", generator=generator
                )
                variance = raw.pow(2).mean(-1, keepdim=True)
                normed = raw * torch.rsqrt(variance + hf.rms_norm_eps)
                inputs[count] = (normed * norm_weight.float()).to(torch.bfloat16)

            outputs: dict[int, dict[str, torch.Tensor]] = {
                count: {} for count in TOKEN_COUNTS
            }
            paths = (
                ("native", native.DeepseekV2ForCausalLM, None),
                ("afd", AFDDeepseekV2ForCausalLM, {"role": "ffn"}),
                (
                    "attention-gate",
                    AFDDeepseekV2ForCausalLM,
                    {"role": "attention", "compute_gate_on_attention": True},
                ),
                (
                    "afd-gate-on-attention",
                    AFDDeepseekV2ForCausalLM,
                    {"role": "ffn", "compute_gate_on_attention": True},
                ),
            )
            for name, model_cls, afd in paths:
                config = build_config(args.model, layer_idx, afd)
                model = load_model(model_cls, config, args.model, layer_idx)
                for count, hidden in inputs.items():
                    with moe_forward_context(config, layer_idx):
                        result = path_output(
                            name,
                            model,
                            layer_idx,
                            hidden,
                            outputs[count].get("attention-gate"),
                        )
                    outputs[count][name] = result.clone()
                del model
                config.compilation_config.static_forward_context.clear()
                torch.cuda.empty_cache()

            for count, hidden in inputs.items():
                ref_logits, ref_out, near_tie = fp32_reference(tensors, hf, hidden)
                case = outputs[count]
                rows = ~near_tie
                report = {
                    "layer": layer_idx,
                    "tokens": count,
                    "near_tie_tokens": int(near_tie.sum().item()),
                    "attention_gate_logits_dtype": str(case["attention-gate"].dtype),
                    "attention_gate_logits_rel_l2": rel_l2(
                        case["attention-gate"], ref_logits, slice(None)
                    ),
                }
                for name in ("afd", "afd-gate-on-attention"):
                    report[f"{name}_equals_native"] = torch.equal(
                        case[name], case["native"]
                    )
                    report[f"{name}_max_abs_vs_native"] = (
                        (case[name].float() - case["native"].float()).abs().max().item()
                    )
                for name in ("native", "afd", "afd-gate-on-attention"):
                    report[f"{name}_rel_l2_vs_fp32"] = rel_l2(case[name], ref_out, rows)
                    report[f"{name}_rel_l2_vs_fp32_all_tokens"] = rel_l2(
                        case[name], ref_out, slice(None)
                    )
                reports.append(report)
                torch.save(
                    {
                        "input": hidden.cpu(),
                        "fp32_logits": ref_logits.cpu(),
                        "fp32_output": ref_out.cpu(),
                        "near_tie": near_tie.cpu(),
                        **{name: tensor.cpu() for name, tensor in case.items()},
                    },
                    args.output / f"layer-{layer_idx}-tokens-{count}.pt",
                )
                print(json.dumps(report), flush=True)
            (args.output / "report.json").write_text(json.dumps(reports, indent=2))
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()

    for report in reports:
        for name in ("afd", "afd-gate-on-attention"):
            assert report[f"{name}_equals_native"], report
        for name in ("native", "afd", "afd-gate-on-attention"):
            assert report[f"{name}_rel_l2_vs_fp32"] <= REFERENCE_REL_L2, report


if __name__ == "__main__":
    main()
