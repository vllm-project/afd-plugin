# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Full BF16 V2-Lite inference oracle: native TP2 versus independent FFN FaaS.

Run each inference in its own process/allocation subset, then compare saved
teacher-forced prompt log probabilities and greedy generated token IDs. Fixed
local inputs need no downloaded evaluation dataset. This is a correctness probe,
not a throughput benchmark by default. The separate benchmark launcher opts
into process-local warmup/repeats and records generate-only timings separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPTS = (
    "The capital of France is",
    "Question: What is 7 plus 5? Answer:",
    "Write a short sentence about the ocean.",
    "def add(a, b):\n    return",
)
# Fixed before execution. BF16 expert acceptance uses NRMSE <= .02; the final
# log-softmax adds a .15 absolute log-probability bound for near-zero values.
LOGPROB_ATOL = 0.15
LOGPROB_NRMSE = 0.02


def inference_afd_config(mode: str, port: int, *, role: str = "attention") -> dict:
    """Share the reviewed 2A2F configuration with the old CAM F launcher."""
    if mode == "baseline":
        return {}
    extra = {"attn_ranks_per_dp": 2, "hccl_buffer_size": 64}
    if mode == "legacy-attention":
        extra["dynamicQuant"] = 0
        extra["ffn_two_stream"] = False
    else:
        extra.update(capacity=8, slots=2, timeout_s=30, startup_timeout_s=180)
    return {
        "afd": {
            "connector": (
                "CAMAsyncAFDConnector"
                if mode == "legacy-attention"
                else "FaaSAsyncAFDConnector"
            ),
            "role": role,
            "async": True,
            "compute_gate_on_attention": True,
            "num_attention_ranks": 2,
            "num_ffn_ranks": 2,
            "host": "127.0.0.1",
            "port": port,
            "connector_extra_config": extra,
        }
    }


def inference_records(outputs) -> list[dict]:
    records = []
    for output in outputs:
        prompt_values = []
        for position, values in enumerate(output.prompt_logprobs):
            if values is not None:
                prompt_values.append(
                    {
                        "position": position,
                        "token": output.prompt_token_ids[position],
                        "logprob": values[output.prompt_token_ids[position]].logprob,
                    }
                )
        completion = output.outputs[0]
        records.append(
            {
                "prompt": output.prompt,
                "prompt_token_ids": output.prompt_token_ids,
                "prompt_logprobs": prompt_values,
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "logprobs": [
                    values[token].logprob
                    for values, token in zip(
                        completion.logprobs, completion.token_ids, strict=True
                    )
                ],
            }
        )
    return records


def infer(args: argparse.Namespace) -> None:
    import torch
    from vllm import LLM, SamplingParams

    started = time.monotonic()
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        seed=0,
        distributed_executor_backend="mp",
        additional_config=inference_afd_config(args.mode, args.port),
        worker_extension_cls=(
            "tests.e2e.models.deepseek_v2_lite.ffn_faas_boundaries.FFNBoundaryWorker"
            if args.boundary_directory is not None
            else ""
        ),
    )
    load_seconds = time.monotonic() - started
    if args.boundary_directory is not None:
        target_ids = (
            None
            if args.boundary_prompt_index is None
            else llm.get_tokenizer().encode(PROMPTS[args.boundary_prompt_index])
        )
        llm.collective_rpc(
            "arm_ffn_boundaries", args=(str(args.boundary_directory), 1, target_ids)
        )
    parameters = SamplingParams(
        temperature=0.0, max_tokens=16, prompt_logprobs=1, logprobs=1
    )
    if args.inference_marker is not None:
        args.inference_marker.write_text(
            json.dumps({"monotonic": time.monotonic(), "phase": "inference-started"})
            + "\n"
        )
    samples = []
    try:
        for iteration in range(args.warmups + args.repeats):
            started = time.monotonic()
            outputs = llm.generate(list(PROMPTS), parameters, use_tqdm=False)
            generate_seconds = time.monotonic() - started
            if args.boundary_directory is not None:
                llm.collective_rpc("flush_ffn_boundaries")
            records = inference_records(outputs)
            inference_seconds = time.monotonic() - started
            output_tokens = sum(len(record["token_ids"]) for record in records)
            sample = {
                "iteration": iteration,
                "phase": "warmup" if iteration < args.warmups else "measured",
                "generate_seconds": generate_seconds,
                "inference_seconds": inference_seconds,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / generate_seconds,
                "records": records,
            }
            samples.append(sample)
            if args.samples_output is not None:
                with args.samples_output.open("a") as sample_file:
                    sample_file.write(json.dumps(sample) + "\n")
        report = {
            "mode": args.mode,
            "torch": str(torch.__version__),
            "load_seconds": load_seconds,
            "inference_seconds": samples[-1]["inference_seconds"],
            "records": samples[-1]["records"],
            "warmups": args.warmups,
            "repeats": args.repeats,
            "samples": samples,
            "timing": "generate_seconds excludes record conversion and file I/O",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=120)


def compare(args: argparse.Namespace) -> None:
    import math

    baseline, candidate = (
        json.loads(p.read_text()) for p in (args.baseline, args.candidate)
    )
    count = 0
    max_error = 0.0
    error_square = 0.0
    reference_square = 0.0
    for reference, actual in zip(
        baseline["records"], candidate["records"], strict=True
    ):
        assert reference["prompt_token_ids"] == actual["prompt_token_ids"]
        assert reference["token_ids"] == actual["token_ids"], (
            reference["text"],
            actual["text"],
        )
        pairs = list(
            zip(reference["prompt_logprobs"], actual["prompt_logprobs"], strict=True)
        )
        values = []
        for left, right in pairs:
            assert (left["position"], left["token"]) == (
                right["position"],
                right["token"],
            )
            values.append((left["logprob"], right["logprob"]))
        values.extend(zip(reference["logprobs"], actual["logprobs"], strict=True))
        for left, right in values:
            assert math.isfinite(left) and math.isfinite(right)
            difference = abs(left - right)
            assert difference <= LOGPROB_ATOL, difference
            max_error = max(max_error, difference)
            error_square += difference**2
            reference_square += left**2
            count += 1
    nrmse = math.sqrt(error_square / max(reference_square, 1e-24))
    assert count > 0 and nrmse <= LOGPROB_NRMSE, (count, nrmse)
    report = {
        "passed": True,
        "prompts": len(PROMPTS),
        "compared_logprobs": count,
        "greedy_token_ids_equal": True,
        "max_logprob_error": max_error,
        "logprob_nrmse": nrmse,
        "logprob_atol": LOGPROB_ATOL,
        "logprob_nrmse_limit": LOGPROB_NRMSE,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["baseline", "attention", "legacy-attention", "compare"]
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=29660)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--inference-marker", type=Path)
    parser.add_argument("--boundary-directory", type=Path)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples-output", type=Path)
    parser.add_argument(
        "--boundary-prompt-index", type=int, choices=range(len(PROMPTS)), default=None
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats positive")
    if args.boundary_directory is not None and (args.warmups, args.repeats) != (0, 1):
        parser.error("boundary diagnostics require zero warmups and one inference")
    if args.samples_output is not None:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        args.samples_output.touch(exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "compare":
        compare(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
