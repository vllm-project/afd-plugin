# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-checkpoint check of the FastAFD M2N MegaMoE kernels on SM100.

Run with one process per GPU; the last ``--num-ffn-ranks`` ranks are FFN
(expert) ranks and the rest are Attention ranks, matching the connector's
Attention-first union layout:

    torchrun --nproc-per-node 2 -m tests.e2e.operators.fastafd_m2n_reference \
        --model /path/to/Qwen3.6-35B-A3B-FP8 --output /logs/fastafd-m2n

``--model`` takes a Qwen3.5/3.6 MoE or Qwen3 MoE FP8 checkpoint.

FFN ranks requantize their checkpoint experts with the plugin helpers and run
one persistent expert kernel per step over all selected layers and lanes.
Attention ranks route fixed inputs and run the Attention kernels per layer and
lane, either fused (phase 0) or as the serving path does it: a dispatch (phase
1) for every lane of the layer, then the matching combines (phase 2). Routing
is either the checkpoint gate (softmax top-k, renormalized) or skewed, with
every token on the same top-k experts. Each output is compared with two fp32
references:

- matched: the kernel's numerics (per-32 UE8M0 FP8 activations, requantized
  weights, FP8 intermediate after SwiGLU and the top-k weight);
- model: block-FP8 checkpoint weights with unquantized activations.

The matched bound checks the kernel; the model figure reports the error the
M2N quantization adds relative to the checkpoint math.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open

from afd_plugin.compat.gpu.fastafd import (
    FASTAFD_CHECKPOINT_BLOCK,
    FASTAFD_SCALE_GROUP,
    ceil_to_ue8m0,
    fastafd_ops,
    make_weight_descriptors,
    requant_fp8_block_to_per32,
    transform_weights_for_m2n,
)
from afd_plugin.config import afd_config_from_mapping
from afd_plugin.connectors import AFDDPMetadata
from afd_plugin.connectors.gpu.fastafd import (
    FastAFDAFDConnector,
    M2NLaneBuffer,
    M2NShape,
)
from afd_plugin.v1.worker.fastafd_ffn_model_runner import FastAFDExpertExecutor

TOKEN_COUNTS = (1, 7, 64)
# Lane-0 token counts of the --production steps (lane 1 runs half as many).
# The empty steps sit between non-empty ones so the next step reuses buffers
# that an empty step went through.
PRODUCTION_COUNTS = (7, 0, 7, 64, 0, 1)
PHASES = {"fused": (0,), "split": (1, 2)}
SEED = 1234
# Relative L2 bound against the matched reference: both sides quantize and
# round at the same points, so the gap is accumulation order, fast-math SiLU
# and FP8 values that land on a rounding boundary.
MATCHED_REL_L2 = 0.01
FP8_MAX = 448.0
# The dispatch kernel floors each 32-element input amax at 1e-4; the expert
# kernel uses the raw amax of the intermediate.
INPUT_AMAX_FLOOR = 1e-4
INTERMEDIATE_AMAX_FLOOR = 1e-30
ATTENTION_SMS = 24


def text_config(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    return config.get("text_config", config)


class Checkpoint:
    def __init__(self, model: Path) -> None:
        self.model = model
        self.index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        # Qwen3.5/3.6 nest the text model under language_model; Qwen3 MoE does not.
        nested = any(name.startswith("model.language_model.") for name in self.index)
        self.layers = "model.language_model.layers" if nested else "model.layers"

    def mlp_prefix(self, layer: int) -> str:
        return f"{self.layers}.{layer}.mlp"

    def tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self.index[name], []).append(name)
        out = {}
        for file_name, file_names in by_file.items():
            with safe_open(str(self.model / file_name), framework="pt") as file:
                for name in file_names:
                    out[name] = file.get_tensor(name)
        return out


def expert_weights(
    checkpoint: Checkpoint, layer: int, experts: range
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacked FP8 ``gate_up [E, 2I, H]``, ``down [E, H, I]`` and block scales."""
    prefix = checkpoint.mlp_prefix(layer)
    names = [
        f"{prefix}.experts.{expert}.{proj}.{kind}"
        for expert in experts
        for proj in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    ]
    tensors = checkpoint.tensors(names)

    def stack(proj: str, kind: str) -> torch.Tensor:
        return torch.stack(
            [tensors[f"{prefix}.experts.{e}.{proj}.{kind}"] for e in experts]
        ).cuda()

    gate_up = torch.cat(
        [stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1
    )
    gate_up_scales = torch.cat(
        [stack("gate_proj", "weight_scale_inv"), stack("up_proj", "weight_scale_inv")],
        dim=1,
    )
    return (
        gate_up,
        gate_up_scales,
        stack("down_proj", "weight"),
        stack("down_proj", "weight_scale_inv"),
    )


def block_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = torch.repeat_interleave(scales, FASTAFD_CHECKPOINT_BLOCK, dim=-2)
    expanded = torch.repeat_interleave(expanded, FASTAFD_CHECKPOINT_BLOCK, dim=-1)
    return weight.float() * expanded[..., : weight.shape[-2], : weight.shape[-1]]


def requant_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 values of the per-32 UE8M0 weights the kernel consumes."""
    rounded = ceil_to_ue8m0(scales.float())
    ratio = torch.where(
        rounded > 0, scales.float() / rounded, torch.zeros_like(rounded)
    )
    requantized = (block_dequant(weight, ratio)).to(torch.float8_e4m3fn)
    return block_dequant(requantized, rounded)


def fp8_per32(x: torch.Tensor, amax_floor: float) -> torch.Tensor:
    """fp32 values after per-32 UE8M0 FP8 quantization, as the kernels do it."""
    if not x.numel():
        return x.float()
    groups = x.float().reshape(*x.shape[:-1], -1, FASTAFD_SCALE_GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(amax_floor)
    scales = ceil_to_ue8m0(amax / FP8_MAX)
    quantized = (groups / scales).to(torch.float8_e4m3fn).float() * scales
    return quantized.reshape(x.shape)


def moe_reference(
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    matched: bool,
) -> torch.Tensor:
    """MoE output; ``matched`` follows the kernels' quantization and rounding.

    The matched path quantizes inputs per 32 with an amax floor of 1e-4,
    rounds gate and up to bfloat16 before SwiGLU, applies the top-k weight
    before quantizing the intermediate, and rounds every expert output to
    bfloat16 before the fp32 top-k sum.
    """
    x = fp8_per32(hidden, INPUT_AMAX_FLOOR) if matched else hidden.float()
    intermediate = gate_up.shape[1] // 2
    out = torch.zeros_like(x)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        projected = x[rows] @ gate_up[expert].T
        if matched:
            projected = projected.to(torch.bfloat16).float()
        activated = torch.nn.functional.silu(projected[:, :intermediate])
        activated = activated * projected[:, intermediate:]
        activated = activated * weights[rows, slots, None]
        if matched:
            activated = fp8_per32(activated, INTERMEDIATE_AMAX_FLOOR)
        expert_out = activated @ down[expert].T
        if matched:
            expert_out = expert_out.to(torch.bfloat16).float()
        out.index_add_(0, rows, expert_out)
    return out


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = actual.float() - expected.float()
    return (error.norm() / expected.float().norm().clamp_min(1e-12)).item()


def make_input(rank: int, count: int, lane: int, hidden_size: int) -> torch.Tensor:
    """RMS-normalized random rows, fixed per rank, token count and lane."""
    generator = torch.Generator(device="cuda").manual_seed(
        SEED + 101 * rank + 17 * count + lane
    )
    raw = torch.randn(count, hidden_size, device="cuda", generator=generator)
    norm = raw.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return (raw / norm).to(torch.bfloat16)


def route(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    top_k: int,
    *,
    routing: str,
    norm_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 top-k ids and fp32 weights: checkpoint gate, or all on experts 0..k-1."""
    if routing == "gate":
        probs = torch.nn.functional.linear(hidden, gate).float().softmax(dim=-1)
        weights, ids = probs.topk(top_k, dim=-1)
        if norm_topk:
            weights = weights / weights.sum(dim=-1, keepdim=True)
    else:
        count = hidden.shape[0]
        ids = torch.arange(top_k, device="cuda").expand(count, -1).clone()
        weights = torch.full((count, top_k), 1.0 / top_k, device="cuda")
    return ids.to(torch.int32).contiguous(), weights.contiguous()


def compare(
    output: torch.Tensor,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    reference_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    **case: object,
) -> dict:
    """One report row: finiteness and the error against both references."""
    report = {**case, "finite": bool(torch.isfinite(output).all().item())}
    for kind in ("matched", "model"):
        gate_up, down = reference_weights[kind]
        reference = moe_reference(
            hidden, ids.long(), weights, gate_up, down, matched=kind == "matched"
        )
        report[f"{kind}_rel_l2"] = rel_l2(output, reference)
        report[f"{kind}_max_abs"] = (
            (output.float() - reference).abs().max().item() if output.numel() else 0.0
        )
    print(json.dumps(report), flush=True)
    return report


def load_attention_side(
    checkpoint: Checkpoint, layers: list[int], num_experts: int
) -> tuple[dict, dict]:
    """Checkpoint gates and both reference weight sets for the selected layers."""
    gates = {}
    reference_weights = {}
    for layer in layers:
        name = f"{checkpoint.mlp_prefix(layer)}.gate.weight"
        # The Qwen3-235B-A22B FP8 checkpoint stores the gate in float32; route in
        # bf16 like the model does.
        gates[layer] = checkpoint.tensors([name])[name].to("cuda", torch.bfloat16)
        gate_up, gate_up_scales, down, down_scales = expert_weights(
            checkpoint, layer, range(num_experts)
        )
        reference_weights[layer] = {
            "matched": (
                requant_dequant(gate_up, gate_up_scales),
                requant_dequant(down, down_scales),
            ),
            "model": (
                block_dequant(gate_up, gate_up_scales),
                block_dequant(down, down_scales),
            ),
        }
    return gates, reference_weights


def lane_tokens(count: int, lane: int) -> int:
    return count if lane == 0 else count // 2


def production_connector(
    args: argparse.Namespace, hf: dict, rank: int, num_attention: int
) -> tuple[SimpleNamespace, FastAFDAFDConnector]:
    """This rank's FastAFDAFDConnector, initialized from the checkpoint config."""
    if args.layers != list(range(len(args.layers))):
        raise SystemExit("--production serves the first n layers: pass --layers 0..n-1")
    role = "attention" if rank < num_attention else "ffn"
    config = json.loads((args.model / "config.json").read_text())
    vllm_config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            model=str(args.model),
            revision=None,
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                quantization_config=config.get("quantization_config")
            ),
            hf_text_config=SimpleNamespace(
                **{**hf, "num_hidden_layers": len(args.layers)}
            ),
        ),
        load_config=SimpleNamespace(download_dir=None),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=args.max_tokens),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            enable_dbo=max(args.lanes) > 1,
            use_ubatching=False,
            tensor_parallel_size=1,
        ),
    )
    port = [0]
    if rank == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port[0] = sock.getsockname()[1]
    dist.broadcast_object_list(port, src=0)
    # Under torchrun every rank would join the agent's store as a client; the
    # connector's own rendezvous needs union rank 0 to host its store.
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    afd_config = afd_config_from_mapping(
        {
            "connector": "FastAFDAFDConnector",
            "role": role,
            "host": "127.0.0.1",
            "port": port[0],
            "num_attention_ranks": num_attention,
            "num_ffn_ranks": args.num_ffn_ranks,
            "compute_gate_on_attention": True,
        },
    )
    connector = FastAFDAFDConnector(
        rank,
        int(os.environ.get("LOCAL_RANK", rank)),
        vllm_config,  # type: ignore[arg-type]
        afd_config,
        rank if role == "attention" else rank - num_attention,
    )
    connector.init_afd_connector()
    return vllm_config, connector


def run_production(args: argparse.Namespace, hf: dict, rank: int, world: int) -> None:
    """Split dispatch/combine through FastAFDAFDConnector and FastAFDExpertExecutor."""
    num_attention = world - args.num_ffn_ranks
    vllm_config, connector = production_connector(args, hf, rank, num_attention)
    counts = (*PRODUCTION_COUNTS, args.max_tokens)
    steps = [
        (lanes, count, routing)
        for lanes in args.lanes
        for count in counts
        for routing in args.routings
    ]
    if rank >= num_attention:
        executor = FastAFDExpertExecutor(vllm_config, connector)  # type: ignore[arg-type]
        dist.barrier()
        for lanes, count, _routing in steps:
            executor.submit(
                {
                    lane: AFDDPMetadata(
                        num_tokens_across_dp_cpu=[lane_tokens(count, lane)]
                        * num_attention
                    )
                    for lane in range(lanes)
                }
            )
            torch.cuda.synchronize()
        assert executor.drain(0.0)
        connector.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    gates, reference_weights = load_attention_side(
        Checkpoint(args.model), args.layers, connector.shape.num_experts
    )
    norm_topk = bool(hf.get("norm_topk_prob", True))
    dist.barrier()
    reports = []
    for lanes, count, routing in steps:
        inputs = [
            make_input(
                rank, lane_tokens(count, lane), lane, connector.shape.hidden_size
            )
            for lane in range(lanes)
        ]
        outputs = {}
        routes = {}
        for layer in args.layers:
            for lane, hidden in enumerate(inputs):
                routes[(layer, lane)] = route(
                    hidden,
                    gates[layer],
                    connector.shape.top_k,
                    routing=routing,
                    norm_topk=norm_topk,
                )
                connector.dispatch(hidden, *routes[(layer, lane)], lane)
            for lane, hidden in enumerate(inputs):
                outputs[(layer, lane)] = connector.combine(
                    hidden, *routes[(layer, lane)], lane
                )
        torch.cuda.synchronize()
        for (layer, lane), output in outputs.items():
            ids, weights = routes[(layer, lane)]
            reports.append(
                compare(
                    output,
                    inputs[lane],
                    ids,
                    weights,
                    reference_weights[layer],
                    path="production",
                    rank=rank,
                    phase="split",
                    routing=routing,
                    lanes=lanes,
                    tokens=lane_tokens(count, lane),
                    layer=layer,
                    lane=lane,
                )
            )
    connector.close()
    finish(args, rank, reports)


def finish(args: argparse.Namespace, rank: int, reports: list[dict]) -> None:
    """Write this rank's report, shut down, and apply the bounds."""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"report-rank{rank}.json").write_text(json.dumps(reports, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    for report in reports:
        assert report["finite"], report
        assert report["matched_rel_l2"] <= MATCHED_REL_L2, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-ffn-ranks", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="buffer capacity per Attention rank; also run as a token count",
    )
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument(
        "--routings", nargs="+", choices=("gate", "skewed"), default=["gate", "skewed"]
    )
    parser.add_argument(
        "--empty",
        action="store_true",
        help="also run a final step with no tokens",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="drive split dispatch/combine through FastAFDAFDConnector and "
        "FastAFDExpertExecutor instead of the raw kernels",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    hf = text_config(args.model)
    if args.production:
        run_production(args, hf, rank, world)
        return
    num_attention = world - args.num_ffn_ranks
    shape = M2NShape(
        num_attention_ranks=num_attention,
        num_ffn_ranks=args.num_ffn_ranks,
        num_layers=hf["num_hidden_layers"],
        num_experts=hf["num_experts"],
        top_k=hf["num_experts_per_tok"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["moe_intermediate_size"],
        max_tokens_per_rank=args.max_tokens,
    )
    group = dist.group.WORLD
    # A lane buffer must stay on one Attention phase mode: fused and split
    # launches keep different deferred-barrier state, so each mode gets its own.
    lane_buffers_by_phase = {
        phase: [M2NLaneBuffer(shape, group) for _ in range(max(args.lanes))]
        for phase in args.phases
    }
    ops = fastafd_ops()
    checkpoint = Checkpoint(args.model)
    counts = (*TOKEN_COUNTS, args.max_tokens)
    steps = [
        (lanes, count, phase, routing)
        for lanes in args.lanes
        for count in counts
        for phase in args.phases
        for routing in args.routings
    ]
    if args.empty:
        steps.append((1, 0, args.phases[-1], args.routings[0]))
    layers = args.layers

    if rank >= num_attention:
        ffn_rank = rank - num_attention
        local = shape.num_experts_per_ffn_rank
        experts = range(ffn_rank * local, (ffn_rank + 1) * local)
        kept, blobs, l1_ptrs = [], [], []
        for layer in layers:
            gate_up, gate_up_scales, down, down_scales = expert_weights(
                checkpoint, layer, experts
            )
            l1, l2 = transform_weights_for_m2n(
                requant_fp8_block_to_per32(gate_up, gate_up_scales),
                requant_fp8_block_to_per32(down, down_scales),
            )
            kept.append((l1, l2))
            blobs.append(
                make_weight_descriptors(
                    l1,
                    l2,
                    num_experts_per_rank=local,
                    hidden_size=shape.hidden_size,
                    intermediate_size=shape.intermediate_size,
                ).cuda()
            )
            l1_ptrs.append(l1[0].data_ptr())
        descriptors = torch.stack(blobs).contiguous()
        pointers = torch.tensor(l1_ptrs, dtype=torch.int64, device="cuda")
        dist.barrier(group)
        for lanes, count, phase, _routing in steps:
            buffers = lane_buffers_by_phase[phase]
            lane_buffers = [
                buffers[lane] if lane < lanes else None for lane in range(4)
            ]
            ptrs = [
                (lane_buffer or buffers[0]).peer_ptrs for lane_buffer in lane_buffers
            ]
            ops.mega_moe_m2n_eg(
                descriptors,
                pointers,
                len(layers),
                local,
                shape.hidden_size,
                shape.intermediate_size,
                kept[0][0][0].numel(),
                True,
                None,
                buffers[0].buffer,
                *(
                    lane_buffer.buffer if lane_buffer else None
                    for lane_buffer in lane_buffers[1:]
                ),
                *ptrs,
                rank,
                num_attention,
                args.num_ffn_ranks,
                buffers[0].max_tokens_per_rank,
                shape.num_experts,
                shape.top_k,
                max(1, count),
                None,
                True,
                0,
                0,
                None,
            )
            torch.cuda.synchronize()
        dist.barrier(group)
        dist.destroy_process_group()
        return

    gates, reference_weights = load_attention_side(
        checkpoint, layers, shape.num_experts
    )
    norm_topk = bool(hf.get("norm_topk_prob", True))
    dist.barrier(group)
    reports = []
    for lanes, count, phase, routing in steps:
        buffers = lane_buffers_by_phase[phase]
        inputs = [
            make_input(rank, count, lane, shape.hidden_size) for lane in range(lanes)
        ]
        outputs = {}
        routes = {}
        for layer in layers:
            for lane, hidden in enumerate(inputs):
                routes[(layer, lane)] = route(
                    hidden,
                    gates[layer],
                    shape.top_k,
                    routing=routing,
                    norm_topk=norm_topk,
                )
                outputs[(layer, lane)] = torch.empty_like(hidden)
            # Split mode dispatches every lane before the first combine, as two
            # interleaved microbatches do; each lane gets exactly one combine.
            for kernel_phase in PHASES[phase]:
                for lane, hidden in enumerate(inputs):
                    ids, weights = routes[(layer, lane)]
                    ops.mega_moe_m2n_ag(
                        outputs[(layer, lane)],
                        hidden,
                        ids,
                        weights,
                        None,
                        norm_topk,
                        buffers[lane].buffer,
                        buffers[lane].peer_ptrs,
                        rank,
                        num_attention,
                        args.num_ffn_ranks,
                        buffers[lane].max_tokens_per_rank,
                        shape.num_experts,
                        shape.top_k,
                        shape.intermediate_size,
                        ATTENTION_SMS,
                        kernel_phase,
                        False,
                        False,
                        shape.top_k,
                        shape.num_experts,
                        0.0,
                        None,
                    )
        torch.cuda.synchronize()
        for (layer, lane), output in outputs.items():
            ids, weights = routes[(layer, lane)]
            reports.append(
                compare(
                    output,
                    inputs[lane],
                    ids,
                    weights,
                    reference_weights[layer],
                    path="kernel",
                    rank=rank,
                    phase=phase,
                    routing=routing,
                    lanes=lanes,
                    tokens=count,
                    layer=layer,
                    lane=lane,
                )
            )
    finish(args, rank, reports)


if __name__ == "__main__":
    main()
