# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-checkpoint, fixed-input native-versus-AFD MoE numerical probe.

Run on one otherwise idle 910C after the communication matrix. This loads
only the requested MoE layers, not the attention stack or KV cache. It uses
the pinned native checkpoint loader and post-load quantization transforms.
DSV4's default layers cover Hash and ordinary routing; DSV2 uses its first MoE.

    python -m tests.e2e.operators.moe_checkpoint_reference --model /model \
        --family dsv4 --output /logs/dsv4-moe-reference

Outputs retain input/token IDs, routing, shared/routed/final tensors and error
metrics. This isolates MoE semantics; it does not replace target-topology E2E.
"""

from __future__ import annotations

import argparse
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

TOKENS = 7
SEED = 42
REFERENCE_TOLERANCES = {"float16": (0.001, 0.01), "bfloat16": (0.005, 0.03)}


def checkpoint_weights(model_path: Path, layer_idx: int):
    """Read only this real checkpoint's MoE tensors, including all scales."""
    prefixes = (
        f"model.layers.{layer_idx}.mlp.",
        f"model.layers.{layer_idx}.ffn.",
        f"layers.{layer_idx}.mlp.",
        f"layers.{layer_idx}.ffn.",
    )
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as file:
            # safe_open exposes keys(), but is not an iterable mapping.
            for name in file.keys():  # noqa: SIM118
                if name.startswith(prefixes):
                    yield name, file.get_tensor(name)


def shared_checkpoint_reference(model_path, layer_idx, hidden, swiglu_limit):
    """Independent CPU MLP from raw checkpoint weights, before loader transforms."""
    parameters = {}
    for name, tensor in checkpoint_weights(model_path, layer_idx):
        if ".shared_experts." not in name:
            continue
        name = name.split(".shared_experts.", 1)[1]
        name = (
            name.replace("w1.", "gate_proj.")
            .replace("w2.", "down_proj.")
            .replace("w3.", "up_proj.")
        )
        if name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale"
        parameters[name] = tensor

    def linear(x, projection):
        weight = parameters[f"{projection}.weight"]
        if weight.dtype == torch.int8:
            weight_scale = parameters[f"{projection}.weight_scale"]
            weight_offset = parameters[f"{projection}.weight_offset"]
            assert torch.count_nonzero(weight_offset) == 0, (
                "native W8A8 shared MLP requires symmetric weights"
            )
            scale = x.float().abs().amax(dim=-1, keepdim=True) / 127
            quantized = (x.float() / scale).round().clamp(-127, 127).to(torch.int32)
            accumulator = quantized @ weight.to(torch.int32).T
            output = accumulator.float() * scale * weight_scale.flatten().float()
        else:
            assert weight.is_floating_point(), (
                "shared CPU oracle supports floating or W8A8 weights"
            )
            output = x.float() @ weight.float().T
        bias = parameters.get(f"{projection}.bias")
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype)

    x = hidden.detach().cpu()
    gate, up = linear(x, "gate_proj"), linear(x, "up_proj")
    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    return linear(activated, "down_proj")


def model_holder(native, moe, config, layer_idx, family):
    """Expose the layer under the native loader's canonical parameter names."""
    holder_type = (
        native.AscendDeepseekV4ForCausalLM
        if family == "dsv4"
        else native.DeepseekV2Model
    )
    holder = holder_type.__new__(holder_type)
    torch.nn.Module.__init__(holder)
    holder.config = config
    holder.num_redundant_experts = 0
    layer = torch.nn.Module()
    layer.mlp = moe
    layers = torch.nn.ModuleDict({str(layer_idx): layer})
    if family == "dsv4":
        holder.model = torch.nn.Module()
        holder.model.layers = layers
    else:
        holder.layers = layers
        holder.use_mha = False
    return holder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--family", choices=("dsv2", "dsv4"), required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument(
        "--dtype", choices=tuple(REFERENCE_TOLERANCES), default="bfloat16"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29671)
    args = parser.parse_args()

    # Worker setup owns all pinned Ascend operator/quantization registration.
    # Imports remain deferred until CLI setup, just like the worker process.
    from vllm.config import CompilationConfig, set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers import fused_moe
    from vllm.model_executor.model_loader.utils import (
        configure_quant_config,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_ascend.worker.worker import NPUWorker

    from afd_plugin.compat.npu.forward_context import ascend_forward_context
    from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
        compute_attention_gate_moe_ffn,
    )

    config = EngineArgs(
        model=str(args.model),
        dtype=args.dtype,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        quantization="ascend"
        if (args.model / "quant_model_description.json").is_file()
        else None,
        gpu_memory_utilization=0.8,
    ).create_engine_config()
    worker = NPUWorker(
        vllm_config=config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{args.port}",
        is_driver_worker=True,
    )
    with set_current_vllm_config(config):
        worker.init_device()
    hf = config.model_config.hf_config
    dtype = config.model_config.dtype
    layers = args.layers or (
        [0, hf.num_hash_layers] if args.family == "dsv4" else [hf.first_k_dense_replace]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        for layer_idx in layers:
            prefix = f"model.layers.{layer_idx}.mlp"
            if args.family == "dsv4":
                from vllm_ascend.models import deepseek_v4 as native

                from afd_plugin.model_executor.models.npu import (
                    deepseek_v4_attention_gate,
                )
                from afd_plugin.model_executor.models.npu.deepseek_v4 import (
                    AFDDeepseekV4AttentionGateRemoteMoE,
                    _iter_role_weights,
                )

                moe_type = native.DeepseekV4MoE
                extra = {"is_draft_layer": False}
            else:
                from vllm.model_executor.models import deepseek_v2 as native

                from afd_plugin.model_executor.models.deepseek_v2 import (
                    AFDDeepseekV2RemoteExpertsMoE,
                    _iter_role_weights,
                )

                moe_type = native.DeepseekV2MoE
                extra = {"apply_routed_scale_to_output": True}
            native.FusedMoE = fused_moe.FusedMoE
            if config.quant_config is not None:
                model_class = (
                    native.AscendDeepseekV4ForCausalLM
                    if args.family == "dsv4"
                    else native.DeepseekV2ForCausalLM
                )
                configure_quant_config(config.quant_config, model_class)
            config.additional_config["afd"] = {
                "role": "attention",
                "connector": "CAMAsyncAFDConnector",
                "async": True,
                "compute_gate_on_attention": True,
            }
            with (
                set_current_vllm_config(config),
                set_default_torch_dtype(dtype),
                torch.device("npu"),
            ):
                reference_moe = moe_type(
                    config=hf,
                    parallel_config=config.parallel_config,
                    quant_config=config.quant_config,
                    prefix=prefix,
                    **extra,
                )
                attention_config = config
                if args.family == "dsv4":
                    attention = AFDDeepseekV4AttentionGateRemoteMoE(
                        config=hf,
                        layer_idx=layer_idx,
                        prefix=prefix,
                        vllm_config=config,
                    )
                else:
                    attention_config = copy(config)
                    attention_config.compilation_config = CompilationConfig()
                    with set_current_vllm_config(attention_config):
                        attention = AFDDeepseekV2RemoteExpertsMoE(
                            config=hf,
                            layer_idx=layer_idx,
                            prefix=prefix,
                            vllm_config=attention_config,
                        )
                native_holder = model_holder(
                    native, reference_moe, hf, layer_idx, args.family
                )
                attention_holder = model_holder(
                    native, attention, hf, layer_idx, args.family
                )
                for holder, shared_only in (
                    (native_holder, False),
                    (attention_holder, True),
                ):
                    weights = checkpoint_weights(args.model, layer_idx)
                    if shared_only:
                        if args.family == "dsv4":
                            weights = _iter_role_weights(
                                weights,
                                role="attention",
                                attn_owns_gate=True,
                                attention_shared_experts=True,
                            )
                        else:
                            weights = _iter_role_weights(
                                weights,
                                role="attention",
                                config=hf,
                                compute_gate_on_attention=True,
                                attention_shared_experts=True,
                            )
                    if args.family == "dsv2":
                        weights = (
                            (name.removeprefix("model."), tensor)
                            for name, tensor in weights
                        )
                    owner_config = attention_config if shared_only else config
                    with set_current_vllm_config(owner_config):
                        loaded = holder.load_weights(weights)
                        assert loaded, (
                            "checkpoint filter did not load any MoE parameters"
                        )
                        process_weights_after_loading(
                            holder, owner_config.model_config, torch.device("npu")
                        )

            torch.manual_seed(SEED)
            hidden = (
                torch.randn(TOKENS, hf.hidden_size, dtype=dtype, device="npu") * 0.1
            )
            token_ids = torch.arange(TOKENS, dtype=torch.int64, device="npu") + 17
            from vllm_ascend.ops.fused_moe.experts_selector import select_experts

            afd_metadata = SimpleNamespace(
                connector=SimpleNamespace(select_experts=select_experts)
            )
            with (
                torch.inference_mode(),
                set_current_vllm_config(config),
                ascend_forward_context(
                    vllm_config=config,
                    afd_metadata=afd_metadata,
                    model_instance=native_holder,
                    num_tokens=TOKENS,
                    input_ids=token_ids,
                ),
            ):
                native_final = reference_moe(hidden)
                native_shared = reference_moe.shared_experts(hidden)
                factor = reference_moe.routed_scaling_factor
                if dtype == torch.float16 and args.family == "dsv2":
                    native_shared = native_shared / factor
                # Native MoE includes its own routing, dispatch, expert MLP
                # and combination. Subtract independently computed shared
                # output in FP32 to expose its routed contribution.
                native_routed = native_final.float() - native_shared.float()
                with (
                    set_current_vllm_config(attention_config),
                    ascend_forward_context(
                        vllm_config=attention_config,
                        afd_metadata=afd_metadata,
                        model_instance=attention_holder,
                        num_tokens=TOKENS,
                        input_ids=token_ids,
                    ),
                ):
                    if args.family == "dsv4":
                        weights, ids = (
                            deepseek_v4_attention_gate.compute_attention_gate_topk(
                                attention, hidden
                            )
                        )
                    else:
                        weights, ids, _ = attention.experts.compute_gate_topk(hidden)
                    afd_shared = attention.shared_experts(hidden)
                routing_diagnostic = {}
                routing_tensors = {}
                if args.family == "dsv4":
                    native_logits = torch.nn.functional.linear(
                        hidden.float(), reference_moe.gate.weight_fp32
                    )
                    afd_logits = torch.nn.functional.linear(
                        hidden.float(), attention.gate.weight_fp32
                    )
                    native_weights, native_ids = select_experts(
                        hidden_states=hidden,
                        router_logits=native_logits,
                        top_k=attention.top_k,
                        use_grouped_topk=True,
                        renormalize=attention.renormalize,
                        topk_group=attention.topk_group,
                        num_expert_group=attention.num_expert_group,
                        scoring_func=attention.scoring_func,
                        routed_scaling_factor=attention.routed_scaling_factor,
                        e_score_correction_bias=reference_moe.gate.e_score_correction_bias,
                        tid2eid=reference_moe.gate.tid2eid,
                    )
                    fp32_weights, fp32_ids = (
                        deepseek_v4_attention_gate._compute_sqrtsoftplus_topk(
                            attention, native_logits
                        )
                    )
                    routing_diagnostic = {
                        "native_logit_dtype": str(native_logits.dtype),
                        "afd_logit_dtype": str(afd_logits.dtype),
                        "logit_max_abs": (native_logits - afd_logits.float())
                        .abs()
                        .max()
                        .item(),
                        "selected_id_mismatches": (native_ids != ids).sum().item(),
                        "fp32_selected_id_mismatches": (native_ids != fp32_ids)
                        .sum()
                        .item(),
                        "fp32_weight_max_abs": (native_weights - fp32_weights)
                        .abs()
                        .max()
                        .item(),
                    }
                    routing_tensors = {
                        "native_logits": native_logits,
                        "afd_logits": afd_logits,
                        "native_ids": native_ids,
                        "native_weights": native_weights,
                        "fp32_ids": fp32_ids,
                        "fp32_weights": fp32_weights,
                    }
                order = ids.flatten().argsort(stable=True)
                expanded = hidden.repeat_interleave(hf.num_experts_per_tok, dim=0)[
                    order
                ]
                counts = (
                    ids.flatten()
                    .to(torch.int64)
                    .bincount(minlength=hf.n_routed_experts)
                )
                dynamic_scales = None
                # Quantized FFN checkpoints consume the same per-token int8
                # representation used by source CAM; native quant kernels
                # remain the independent full-MoE reference above.
                from vllm_ascend.quantization.quant_type import QuantType

                if reference_moe.experts.quant_type != QuantType.NONE:
                    import torch_npu

                    expanded, dynamic_scales = torch_npu.npu_dynamic_quant(expanded)
                layer = SimpleNamespace(mlp=reference_moe)
                ffn = compute_attention_gate_moe_ffn(
                    layer,
                    hidden_states=expanded,
                    group_list=counts,
                    dynamic_scales=dynamic_scales,
                    expand_x_shared=hidden,
                    dynamic_scales_shared=None,
                    topk_scales=None,
                    group_list_type=1,
                    routed_scale_applied_in_topk=args.family == "dsv4",
                )
                routed = ffn.routed_output[order.argsort()].reshape(
                    TOKENS, hf.num_experts_per_tok, hf.hidden_size
                )
                afd_routed = (routed.float() * weights[:, :, None]).sum(dim=1).to(dtype)
                if dtype == torch.float16 and args.family == "dsv2":
                    afd_shared = afd_shared / factor
                afd_final = afd_routed + afd_shared

            cpu_shared = shared_checkpoint_reference(
                args.model,
                layer_idx,
                hidden,
                getattr(hf, "swiglu_limit", None) if args.family == "dsv4" else None,
            )
            if dtype == torch.float16 and args.family == "dsv2":
                cpu_shared = cpu_shared / factor
            tensors = {
                **routing_tensors,
                "native_shared_cpu": cpu_shared,
                "afd_shared_cpu": afd_shared,
                "input": hidden,
                "input_ids": token_ids,
                "ids": ids,
                "weights": weights,
                "native_shared": native_shared,
                "afd_shared": afd_shared,
                "native_routed": native_routed,
                "afd_routed": afd_routed,
                "native_final": native_final,
                "afd_final": afd_final,
            }
            tensors = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
            torch.save(tensors, args.output / f"layer-{layer_idx}.pt")
            atol, rtol = REFERENCE_TOLERANCES[args.dtype]
            report = {
                "layer": layer_idx,
                "hash": args.family == "dsv4" and layer_idx < hf.num_hash_layers,
                "routing_diagnostic": routing_diagnostic,
            }
            for part in ("shared", "shared_cpu", "routed", "final"):
                actual, expected = tensors[f"afd_{part}"], tensors[f"native_{part}"]
                error = actual.float() - expected.float()
                relative_l2 = error.norm() / expected.float().norm().clamp_min(1e-12)
                report[part] = {
                    "max_abs_error": error.abs().max().item(),
                    "relative_l2": relative_l2.item(),
                }
            reports.append(report)
            (args.output / "report.json").write_text(json.dumps(reports, indent=2))
            for part in ("shared", "shared_cpu", "routed", "final"):
                assert report[part]["relative_l2"] <= rtol, report
                torch.testing.assert_close(
                    tensors[f"afd_{part}"].float(),
                    tensors[f"native_{part}"].float(),
                    atol=atol,
                    rtol=rtol,
                )
            print(json.dumps(report), flush=True)
            del reference_moe, attention, native_holder, attention_holder, tensors
            config.compilation_config.static_forward_context.clear()
            config.compilation_config.static_all_moe_layers.clear()
            torch.npu.empty_cache()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
