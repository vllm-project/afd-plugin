# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small, frozen V4 requests for the native reference and independent FaaS.

Freeze once with the actual V4 tokenizer, run the same file against each
mode, then compare saved token IDs/log probabilities using explicit bounds.
The mode labels the run; actual kernel/source identity belongs in deployment
evidence. The plan subcommand emits launch commands without executing them.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from shlex import join as shell_join

import httpx

from afd_plugin.config import afd_config_from_mapping
from afd_plugin.ffn_faas.settings import FaaSOptions

PROMPTS = (
    "The capital of France is",
    "Question: What is 7 plus 5? Answer:",
    "Write a short sentence about the ocean.",
    "def add(a, b):\n    return",
)
MAX_OUTPUT_TOKENS = 16
CONCURRENCY = 2
MODEL_METADATA = (
    "config.json",
    "quant_model_weights.safetensors.index.json",
    "quant_model_description.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKEN_ID_PREFIX = "token_id:"
REFERENCE_MODE = "native-reference"
FAAS_MODE = "faas-native"
PROBE_MODULE = "tests.e2e.models.deepseek_v4_flash.ffn_faas_inference"


def sampling_parameters() -> dict:
    return {
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "logprobs": 1,
        "echo": True,
        "return_tokens_as_token_ids": True,
    }


def freeze(args: argparse.Namespace) -> None:
    # Tokenizer loading is isolated to this subcommand; HTTP execution and
    # result comparison require neither tensors nor the inference runtime.
    from transformers import AutoTokenizer

    root = args.model.resolve(strict=True)
    metadata = {name: (root / name).read_bytes() for name in MODEL_METADATA}
    config = json.loads(metadata["config.json"])
    if config["architectures"] != ["DeepseekV4ForCausalLM"]:
        raise ValueError("the frozen workload requires the actual V4 checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    workload = {
        "schema_version": 1,
        "architecture": "DeepseekV4ForCausalLM",
        "model_metadata_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in metadata.items()
        },
        "requests": [
            {
                "index": index,
                "prompt": prompt,
                "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=True),
            }
            for index, prompt in enumerate(PROMPTS)
        ],
        "sampling": sampling_parameters(),
        "concurrency": CONCURRENCY,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(workload, ensure_ascii=False, indent=2) + "\n")


def read_workload(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    workload = json.loads(raw)
    if (
        workload["schema_version"] != 1
        or workload["architecture"] != "DeepseekV4ForCausalLM"
        or len(workload["requests"]) != len(PROMPTS)
        or workload["concurrency"] != CONCURRENCY
        or workload["sampling"] != sampling_parameters()
    ):
        raise ValueError("expected the frozen four-request V4 workload")
    for index, item in enumerate(workload["requests"]):
        if (
            item["index"] != index
            or not item["prompt_token_ids"]
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in item["prompt_token_ids"]
            )
        ):
            raise ValueError("frozen request indices/tokens are invalid")
    return workload, hashlib.sha256(raw).hexdigest()


def completion_record(response: dict, request: dict, max_tokens: int) -> dict:
    choices = response["choices"]
    if len(choices) != 1 or choices[0]["finish_reason"] not in ("stop", "length"):
        raise ValueError("expected one completed greedy response")
    logprobs = choices[0]["logprobs"]
    ids = []
    for value in logprobs["tokens"]:
        if not isinstance(value, str) or not value.startswith(TOKEN_ID_PREFIX):
            raise ValueError("the API did not return requested numeric token IDs")
        ids.append(int(value.removeprefix(TOKEN_ID_PREFIX)))
    prompt_ids = request["prompt_token_ids"]
    prompt_length = len(prompt_ids)
    generated_ids = ids[prompt_length:]
    values = logprobs["token_logprobs"]
    if (
        ids[:prompt_length] != prompt_ids
        or not 0 < len(generated_ids) <= max_tokens
        or len(values) != len(ids)
        or any(value is None or not math.isfinite(value) for value in values[1:])
    ):
        raise ValueError("response IDs/log probabilities do not match frozen input")
    usage = response["usage"]
    if usage["prompt_tokens"] != prompt_length or usage["completion_tokens"] != len(
        generated_ids
    ):
        raise ValueError("response usage disagrees with returned token IDs")
    return {
        "index": request["index"],
        "prompt_token_ids": prompt_ids,
        "token_ids": generated_ids,
        "all_token_ids": ids,
        "token_logprobs": values,
        "finish_reason": choices[0]["finish_reason"],
    }


async def request_round(
    client: httpx.AsyncClient, args: argparse.Namespace, workload: dict, sample: dict
) -> None:
    semaphore = asyncio.Semaphore(workload["concurrency"])
    items = [
        {"index": item["index"], "error": "unfinished"} for item in workload["requests"]
    ]
    sample["requests"] = items

    async def request(item: dict, frozen: dict) -> None:
        async with semaphore:
            item["started_at"] = time.perf_counter()
            try:
                response = await client.post(
                    args.url,
                    json={
                        "model": args.served_model,
                        "prompt": frozen["prompt_token_ids"],
                        **workload["sampling"],
                    },
                )
                item["finished_at"] = time.perf_counter()
                item["latency_seconds"] = item["finished_at"] - item["started_at"]
                item["status_code"] = response.status_code
                item["response_body"] = response.text
                response.raise_for_status()
                item["record"] = completion_record(
                    response.json(), frozen, workload["sampling"]["max_tokens"]
                )
                del item["error"]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                item["error"] = f"{type(error).__name__}: {error}"

    await asyncio.gather(
        *(
            request(item, frozen)
            for item, frozen in zip(items, workload["requests"], strict=True)
        )
    )
    errors = [item for item in items if "error" in item]
    if errors:
        raise RuntimeError(f"V4 request round failed: {errors}")
    sample["end_to_end_seconds"] = max(item["finished_at"] for item in items) - min(
        item["started_at"] for item in items
    )
    sample["output_tokens"] = sum(len(item["record"]["token_ids"]) for item in items)


def run(args: argparse.Namespace) -> None:
    workload, digest = read_workload(args.workload)
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode,
        "mode_label_is_execution_proof": False,
        "url": args.url,
        "served_model": args.served_model,
        "workload_sha256": digest,
        "workload": workload,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timing": (
            "first send to last HTTP response, including client scheduling; "
            "excludes service startup and final file I/O"
        ),
        "samples": [],
        "passed": False,
    }

    async def execute() -> None:
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
            for iteration in range(args.warmups + args.repeats):
                sample = {
                    "iteration": iteration,
                    "phase": "warmup" if iteration < args.warmups else "measured",
                }
                report["samples"].append(sample)
                await request_round(client, args, workload, sample)
                with (args.output_dir / "samples.jsonl").open("a") as output:
                    output.write(json.dumps(sample) + "\n")

    try:
        asyncio.run(execute())
        report["passed"] = True
    finally:
        (args.output_dir / "inference.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )


def compare(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.logprob_atol)
        or args.logprob_atol < 0
        or not math.isfinite(args.logprob_nrmse)
        or args.logprob_nrmse < 0
    ):
        raise ValueError("comparison bounds must be finite and nonnegative")
    reference, candidate = (
        json.loads(path.read_text()) for path in (args.reference, args.candidate)
    )
    if (
        not reference["passed"]
        or not candidate["passed"]
        or reference["workload_sha256"] != candidate["workload_sha256"]
        or reference["workload"] != candidate["workload"]
        or reference["probe_sha256"] != candidate["probe_sha256"]
        or reference["mode"] != REFERENCE_MODE
        or candidate["mode"] != FAAS_MODE
    ):
        raise ValueError(
            "comparison requires successful runs with identical frozen inputs/probe"
        )
    measured = [
        [sample for sample in report["samples"] if sample["phase"] == "measured"]
        for report in (reference, candidate)
    ]
    if not all(measured):
        raise ValueError("both runs require measured samples")
    oracle = [item["record"] for item in reference["samples"][0]["requests"]]
    quality: list[list[dict]] = [[], []]
    for side, saved in enumerate((reference, candidate)):
        for sample in saved["samples"]:
            maximum = error_square = reference_square = 0.0
            compared = 0
            for expected, item in zip(oracle, sample["requests"], strict=True):
                actual = item["record"]
                if (
                    expected["index"] != actual["index"]
                    or expected["all_token_ids"] != actual["all_token_ids"]
                    or expected["finish_reason"] != actual["finish_reason"]
                ):
                    raise ValueError("greedy token IDs or request association disagree")
                for left, right in zip(
                    expected["token_logprobs"][1:],
                    actual["token_logprobs"][1:],
                    strict=True,
                ):
                    if not math.isfinite(left) or not math.isfinite(right):
                        raise ValueError("nonfinite saved log probability")
                    difference = abs(left - right)
                    maximum = max(maximum, difference)
                    error_square += difference**2
                    reference_square += left**2
                    compared += 1
            nrmse = math.sqrt(error_square / max(reference_square, 1e-24))
            quality[side].append(
                {
                    "iteration": sample["iteration"],
                    "phase": sample["phase"],
                    "compared_logprobs": compared,
                    "max_logprob_error": maximum,
                    "logprob_nrmse": nrmse,
                    "passed": (
                        compared > 0
                        and maximum <= args.logprob_atol
                        and nrmse <= args.logprob_nrmse
                    ),
                }
            )
    passed = all(sample["passed"] for group in quality for sample in group)
    timings = [
        [sample["end_to_end_seconds"] for sample in samples] for samples in measured
    ]
    if any(
        not math.isfinite(value) or value <= 0 for group in timings for value in group
    ):
        raise ValueError("end-to-end times must be finite and positive")
    means = [statistics.mean(values) for values in timings]
    report = {
        "passed": passed,
        "workload_sha256": reference["workload_sha256"],
        "greedy_token_ids_equal": True,
        "compared_logprobs": sum(item["compared_logprobs"] for item in quality[1]),
        "max_logprob_error": max(item["max_logprob_error"] for item in quality[1]),
        "logprob_nrmse": max(item["logprob_nrmse"] for item in quality[1]),
        "reference_repeat_checks": quality[0],
        "candidate_checks": quality[1],
        "limits": {
            "logprob_atol": args.logprob_atol,
            "logprob_nrmse": args.logprob_nrmse,
        },
        "reference_mode": reference["mode"],
        "candidate_mode": candidate["mode"],
        "reference_seconds": timings[0],
        "candidate_seconds": timings[1],
        "reference_mean_seconds": means[0],
        "candidate_mean_seconds": means[1],
        "reference_median_seconds": statistics.median(timings[0]),
        "candidate_median_seconds": statistics.median(timings[1]),
        "reference_stdev_seconds": (
            statistics.stdev(timings[0]) if len(timings[0]) > 1 else None
        ),
        "candidate_stdev_seconds": (
            statistics.stdev(timings[1]) if len(timings[1]) > 1 else None
        ),
        "reference_sample_count": len(timings[0]),
        "candidate_sample_count": len(timings[1]),
        "timing_is_performance_comparison": False,
        "evidence_limit": (
            "numerical check only; the native reference and FaaS use different "
            "device budgets; verify actual source/precision in deployment evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError("V4 log probability comparison exceeded the supplied bounds")


def plan(args: argparse.Namespace) -> None:
    """Emit this one V4 validation cell; process/device ownership stays external."""
    extra = FaaSOptions.from_mapping(
        {
            "attn_ranks_per_dp": 4,
            "capacity": 8,
            "slots": 2,
            "hccl_buffer_size": 64,
            "timeout_s": 120,
            "startup_timeout_s": 1200,
        }
    )
    afd = {
        "connector": "FaaSAsyncAFDConnector",
        "role": "attention",
        "async": True,
        "compute_gate_on_attention": True,
        "num_attention_ranks": 12,
        "num_ffn_ranks": 4,
        "host": args.afd_host,
        "port": args.afd_port,
        "connector_extra_config": extra.to_mapping(),
    }
    afd_config_from_mapping(afd)
    common_ascend = {
        "enable_prefill_mc2": False,
        "enable_force_eplb": False,
        "enable_cpu_binding": False,
        "ascend_fusion_config": {"fusion_ops_gmmswigluquant": True},
    }
    common_engine = [
        args.vllm_bin,
        "serve",
        str(args.model),
        "--served-model-name",
        args.served_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.api_port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--attention-config",
        json.dumps({"indexer_kv_dtype": "int8"}),
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--enforce-eager",
        "--shutdown-timeout",
        "120",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "32",
        "--max-num-seqs",
        "2",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.7",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
        "--api-server-count",
        "1",
        "--data-parallel-address",
        args.afd_host,
        "--data-parallel-rpc-port",
        str(args.dp_rpc_port),
    ]
    common_env = {
        "OMP_NUM_THREADS": "2",
        "HCCL_BUFFSIZE": str(extra.hccl_buffer_size),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "120",
        "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
        "AFD_FORCE_BALANCED_TOPK_IDS": "0",
        "AFD_CAM_OP_IO_LOG": "0",
        "AFD_ASYNC_MOE_LAYOUT_LOG": "0",
    }
    native = common_engine + [
        "--data-parallel-size",
        "1",
        "--data-parallel-size-local",
        "1",
        "--additional-config",
        json.dumps(common_ascend),
    ]
    attention = common_engine + [
        "--data-parallel-size",
        "3",
        "--data-parallel-size-local",
        "3",
        "--additional-config",
        json.dumps({**common_ascend, "afd": afd}),
    ]
    ffn = [
        args.python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "afd_plugin.ffn_faas",
        "serve",
        str(args.model),
        "--attention-ranks",
        "12",
        "--attention-tp",
        "4",
        "--ffn-ranks",
        "4",
        "--devices",
        "0,1,2,3",
        "--host",
        args.afd_host,
        "--port",
        str(args.afd_port),
        "--capacity",
        str(extra.capacity),
        "--slots",
        str(extra.slots),
        "--window-mb",
        str(extra.hccl_buffer_size),
        "--timeout",
        str(extra.timeout_s),
        "--startup-timeout",
        str(extra.startup_timeout_s),
        "--status-port",
        str(args.status_port),
        "--status-file",
        str(args.run_dir / "ffn-status.json"),
    ]

    def launch(argv: list[str], mask: str, plugins: str, topology: str) -> dict:
        environment = {
            **common_env,
            "ASCEND_RT_VISIBLE_DEVICES": mask,
            "VLLM_PLUGINS": plugins,
        }
        return {
            "argv": argv,
            "environment_updates": environment,
            "topology": topology,
            "shell": shell_join(
                [
                    "env",
                    *(f"{key}={value}" for key, value in environment.items()),
                    *argv,
                ]
            ),
        }

    module = [args.python_bin, "-m", PROBE_MODULE]
    frozen = args.run_dir / "frozen.json"
    # The same allowlist filters platform and general plugin entry points.
    # Ascend model registration is separate from the platform named "ascend".
    native_plugins = "ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
    report = {
        "scope": "V4 framework numerical validation; plan only, nothing launched",
        "execution_order": [
            "finish the currently owned F device task and verify quiescence",
            "freeze inputs; launch native reference and wait for API readiness",
            "save reference requests; shut down and quiesce its owned processes",
            "launch FaaS F and A before waiting for their paired ready state",
            "run the same frozen requests; compare; drain A then F and clean up",
        ],
        "inherited_environment": (
            "preserve CANN PYTHONPATH/LD_LIBRARY_PATH/ASCEND_CUSTOM_OPP_PATH and "
            "the verified NIC/HCCL settings; env updates do not clear them"
        ),
        "native_reference": launch(
            native, "0,1,2,3", native_plugins, "4 DIE TP4/EP4/DP1"
        ),
        "faas_attention": launch(
            attention,
            "0,1,2,3,4,5,6,7,8,9,10,11",
            f"{native_plugins},afd",
            "12 DIE DP3/TP4",
        ),
        "faas_ffn": launch(ffn, "12,13,14,15", native_plugins, "4 DIE EP4/TP1"),
        "freeze": module
        + ["freeze", "--model", str(args.model), "--output", str(frozen)],
        "requests": {
            mode: module
            + [
                "run",
                "--workload",
                str(frozen),
                "--url",
                f"http://{args.afd_host}:{args.api_port}/v1/completions",
                "--served-model",
                args.served_model,
                "--mode",
                mode,
                "--output-dir",
                str(args.run_dir / mode),
            ]
            for mode in (REFERENCE_MODE, FAAS_MODE)
        },
        "compare": module
        + [
            "compare",
            "--reference",
            str(args.run_dir / REFERENCE_MODE / "inference.json"),
            "--candidate",
            str(args.run_dir / FAAS_MODE / "inference.json"),
            "--logprob-atol",
            "0.01",
            "--logprob-nrmse",
            "0.001",
            "--output",
            str(args.run_dir / "numerical-comparison.json"),
        ],
        "numerical_limits_source": (
            "preselected engineering check by coordinator before candidate results"
        ),
        "performance_claim": (
            "none: native 4 DIE versus FaaS 16 DIE are unequal device budgets"
        ),
        "preflight": (
            "verify actual imports, native/control binaries and checkpoint; record "
            "effective SP and prefix caching; the mode label alone is not proof"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--model", required=True, type=Path)
    frozen.add_argument("--output", required=True, type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("--workload", required=True, type=Path)
    runner.add_argument("--url", required=True, help="full /v1/completions URL")
    runner.add_argument("--served-model", required=True)
    runner.add_argument("--mode", choices=(REFERENCE_MODE, FAAS_MODE), required=True)
    runner.add_argument("--output-dir", required=True, type=Path)
    runner.add_argument("--warmups", type=int, default=1)
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--timeout", type=float, default=300)
    comparator = commands.add_parser("compare")
    comparator.add_argument("--reference", required=True, type=Path)
    comparator.add_argument("--candidate", required=True, type=Path)
    comparator.add_argument("--logprob-atol", required=True, type=float)
    comparator.add_argument("--logprob-nrmse", required=True, type=float)
    comparator.add_argument("--output", required=True, type=Path)
    planner = commands.add_parser("plan")
    planner.add_argument("--model", required=True, type=Path)
    planner.add_argument("--served-model", required=True)
    planner.add_argument("--afd-host", required=True)
    planner.add_argument("--api-port", type=int, default=19280)
    planner.add_argument("--afd-port", type=int, default=29660)
    planner.add_argument("--dp-rpc-port", type=int, default=29661)
    planner.add_argument("--status-port", type=int, default=29760)
    planner.add_argument("--vllm-bin", default="vllm")
    planner.add_argument("--python-bin", default="python3")
    planner.add_argument("--run-dir", required=True, type=Path)
    planner.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "compare": compare, "plan": plan}[args.command](args)


if __name__ == "__main__":
    main()
