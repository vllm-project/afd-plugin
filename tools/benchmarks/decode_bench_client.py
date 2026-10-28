# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Decode-only benchmark over the decode-bench prompt pool, run with vllm bench.

Drives an Attention instance whose ``AFDDecodeBenchConnector`` has
``num_prompts`` > 0 (see ``tools/benchmarks/decode_bench.py``):

1. Warmup: one ``vllm bench serve`` run per Attention DP rank, each pinned to
   its rank with the ``X-data-parallel-rank`` header, sends ``--pool-prompts``
   requests, so every rank prefills and pins its whole prompt pool.
2. Check: the server's prefix-cache counters must show each warmup request
   looked up ``--prompt-len`` prompt tokens, i.e. the server's pool prompts are
   as long as the measured prompts below.
3. Measured: one ``vllm bench serve`` run with ``--random-input-len
   --prompt-len``. Every argument after ``--`` goes to it unchanged (request
   rate, concurrency, number of requests, output length, ...). The server remaps
   each request onto a pool prompt, so the prompt text is only a placeholder.
4. Check: over the measured run the server must have prefilled exactly one
   token per finished request (the last prompt token), with no failed requests
   and no preemptions; otherwise this script exits non-zero.

vllm bench writes its usual result JSON for each run into ``--result-dir``;
this script adds ``decode_bench_report.json`` with the checks.

Usage:
    python tools/benchmarks/decode_bench_client.py --pool-prompts 1000 \
        --prompt-len 1025 -- --num-prompts 4000 --random-output-len 256 \
        --max-concurrency 64 --request-rate 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_PARALLEL_RANK_HEADER = "X-data-parallel-rank"
MODELS_PATH = "/v1/models"
METRICS_PATH = "/metrics"
PREFIX_CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries"
PREFIX_CACHE_HITS_METRIC = "vllm:prefix_cache_hits"
REQUEST_SUCCESS_METRIC = "vllm:request_success"
PREEMPTIONS_METRIC = "vllm:num_preemptions"
HTTP_OK = 200
READY_POLL_INTERVAL_S = 5.0
# The server forces warmup requests to one output token as well.
WARMUP_OUTPUT_TOKENS = 1
# A decode-only request prefills only its last prompt token.
DECODE_ONLY_PREFILLED_TOKENS = 1.0
PREFILLED_TOKENS_TOLERANCE = 0.01
PROMPT_LEN_TOLERANCE = 0.5
# vllm bench flags this script sets for every run; overriding them would break
# the prompt-length contract or lose the result file.
CLIENT_OWNED_BENCH_FLAGS = frozenset(
    {
        "--host",
        "--port",
        "--base-url",
        "--model",
        "--tokenizer",
        "--dataset-name",
        "--random-input-len",
        "--random-range-ratio",
        "--random-prefix-len",
        "--save-result",
        "--result-dir",
        "--result-filename",
    }
)
# The warmup also owns how many requests each rank gets and where they go.
WARMUP_OWNED_BENCH_FLAGS = CLIENT_OWNED_BENCH_FLAGS | {
    "--num-prompts",
    "--random-output-len",
    "--header",
    "--max-concurrency",
}
WARMUP_RESULT_FILENAME = "warmup-rank{rank}.json"
WARMUP_LOG_FILENAME = "warmup-rank{rank}.log"
MEASURED_RESULT_FILENAME = "measured.json"
REPORT_FILENAME = "decode_bench_report.json"
MEASURED_SUMMARY_KEYS = (
    "completed",
    "failed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_input_tokens",
    "total_output_tokens",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


@dataclass(frozen=True)
class ServerCounters:
    """Server-wide totals of the counters the checks need, summed over engines."""

    prefix_cache_queries: float = 0.0
    prefix_cache_hits: float = 0.0
    finished_requests: float = 0.0
    preemptions: float = 0.0

    def __sub__(self, other: ServerCounters) -> ServerCounters:
        return ServerCounters(
            prefix_cache_queries=self.prefix_cache_queries - other.prefix_cache_queries,
            prefix_cache_hits=self.prefix_cache_hits - other.prefix_cache_hits,
            finished_requests=self.finished_requests - other.finished_requests,
            preemptions=self.preemptions - other.preemptions,
        )

    @property
    def prefilled_tokens(self) -> float:
        return self.prefix_cache_queries - self.prefix_cache_hits


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split ``argv`` at ``--``: client flags before, measured vllm bench after."""
    argv = list(argv)
    split = argv.index("--") if "--" in argv else len(argv)
    client_argv, bench_argv = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Arguments after `--` are passed to the measured `vllm bench serve`.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18305)
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name; defaults to the first one in /v1/models.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer vllm bench uses to build prompts; defaults to --model.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=2,
        help="Attention data-parallel ranks; each gets its own warmup run.",
    )
    parser.add_argument(
        "--pool-prompts",
        type=int,
        required=True,
        help="Pool prompts per Attention DP rank; must match the server's "
        "kv_connector_extra_config num_prompts.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=1025,
        help="Prompt length of the warmup and measured runs; must equal the "
        "server's pool prompt length (checked after the warmup).",
    )
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=64,
        help="In-flight requests of each rank's warmup run.",
    )
    parser.add_argument(
        "--warmup-bench-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra vllm bench arguments for the warmup runs; must come last "
        "before `--`.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Measure only, against a server whose pool is already warm.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("decode_bench_results"),
    )
    parser.add_argument("--ready-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(client_argv)
    reject_owned_flags(args.warmup_bench_args, WARMUP_OWNED_BENCH_FLAGS, "warmup")
    reject_owned_flags(bench_argv, CLIENT_OWNED_BENCH_FLAGS, "measured")
    return args, bench_argv


def reject_owned_flags(
    bench_args: Sequence[str], owned_flags: frozenset[str], phase: str
) -> None:
    for bench_arg in bench_args:
        flag = bench_arg.split("=", 1)[0]
        if flag in owned_flags:
            raise SystemExit(
                f"{flag} is set by decode_bench_client.py for the {phase} vllm "
                f"bench run; drop it (owned: {', '.join(sorted(owned_flags))})"
            )


def wait_for_model(base_url: str, timeout_s: float) -> str:
    """Poll until the server answers /v1/models; return the first model id."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}{MODELS_PATH}") as response:
                if response.status == HTTP_OK:
                    return json.loads(response.read())["data"][0]["id"]
        except (urllib.error.URLError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"{base_url} not ready after {timeout_s:.0f}s")
        time.sleep(READY_POLL_INTERVAL_S)


def read_counters(base_url: str) -> ServerCounters:
    with urllib.request.urlopen(f"{base_url}{METRICS_PATH}") as response:
        text = response.read().decode()
    totals = {
        PREFIX_CACHE_QUERIES_METRIC: 0.0,
        PREFIX_CACHE_HITS_METRIC: 0.0,
        REQUEST_SUCCESS_METRIC: 0.0,
        PREEMPTIONS_METRIC: 0.0,
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        metric = sample.split("{", 1)[0].removesuffix("_total")
        if metric in totals:
            totals[metric] += float(value)
    return ServerCounters(
        prefix_cache_queries=totals[PREFIX_CACHE_QUERIES_METRIC],
        prefix_cache_hits=totals[PREFIX_CACHE_HITS_METRIC],
        finished_requests=totals[REQUEST_SUCCESS_METRIC],
        preemptions=totals[PREEMPTIONS_METRIC],
    )


def bench_command(
    args: argparse.Namespace,
    model: str,
    result_filename: str,
    bench_args: Sequence[str],
) -> list[str]:
    """A ``vllm bench serve`` command over --prompt-len random prompts."""
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        model,
        "--tokenizer",
        args.tokenizer or model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.prompt_len),
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        result_filename,
        *bench_args,
    ]


def read_bench_result(args: argparse.Namespace, filename: str) -> dict[str, object]:
    return json.loads((args.result_dir / filename).read_text())


def run_warmup(args: argparse.Namespace, model: str) -> list[dict[str, object]]:
    """Run one pinned vllm bench per rank in parallel; return their results."""
    processes = []
    for rank in range(args.dp_size):
        command = bench_command(
            args,
            model,
            WARMUP_RESULT_FILENAME.format(rank=rank),
            [
                "--num-prompts",
                str(args.pool_prompts),
                "--random-output-len",
                str(WARMUP_OUTPUT_TOKENS),
                "--max-concurrency",
                str(args.warmup_concurrency),
                "--header",
                f"{DATA_PARALLEL_RANK_HEADER}={rank}",
                *args.warmup_bench_args,
            ],
        )
        log_path = args.result_dir / WARMUP_LOG_FILENAME.format(rank=rank)
        with log_path.open("w") as log:
            processes.append(
                (rank, log_path, subprocess.Popen(command, stdout=log, stderr=log))
            )
    results = []
    for rank, log_path, process in processes:
        if process.wait() != 0:
            raise RuntimeError(
                f"warmup vllm bench for rank {rank} exited {process.returncode}; "
                f"see {log_path}"
            )
        results.append(
            read_bench_result(args, WARMUP_RESULT_FILENAME.format(rank=rank))
        )
    return results


def check_warmup(
    args: argparse.Namespace,
    results: list[dict[str, object]],
    delta: ServerCounters,
) -> list[Check]:
    num_failed = sum(int(str(result["failed"])) for result in results)
    checks = [
        Check(
            "warmup requests succeeded",
            num_failed == 0,
            f"{num_failed} failed; a failed warmup request leaves its pool entry "
            "for a later request to prefill",
        )
    ]
    min_requests = args.dp_size * args.pool_prompts
    checks.append(
        Check(
            "every rank received a full pool of warmup requests",
            delta.finished_requests >= min_requests,
            f"{delta.finished_requests:.0f} finished, need >= {min_requests}",
        )
    )
    tokens_per_request = (
        delta.prefix_cache_queries / delta.finished_requests
        if delta.finished_requests
        else 0.0
    )
    prompt_len_matches = (
        abs(tokens_per_request - args.prompt_len) <= PROMPT_LEN_TOLERANCE
    )
    checks.append(
        Check(
            "server pool prompts match --prompt-len",
            prompt_len_matches,
            f"{tokens_per_request:.2f} prompt tokens per warmup request, "
            f"--prompt-len {args.prompt_len}"
            + (
                ""
                if prompt_len_matches
                else "; set the server's prompt_len to one token past a block "
                "boundary and pass the same value here"
            ),
        )
    )
    return checks


def check_measured(result: dict[str, object], delta: ServerCounters) -> list[Check]:
    prefilled_per_request = (
        delta.prefilled_tokens / delta.finished_requests
        if delta.finished_requests
        else 0.0
    )
    return [
        Check(
            "measured requests succeeded",
            int(str(result["failed"])) == 0,
            f"{result['failed']} failed of {result['completed']} completed",
        ),
        Check(
            "decode only: one prefilled token per request",
            abs(prefilled_per_request - DECODE_ONLY_PREFILLED_TOKENS)
            <= PREFILLED_TOKENS_TOLERANCE,
            f"{prefilled_per_request:.4f} prefilled tokens per finished request "
            f"({delta.prefilled_tokens:.0f} tokens, "
            f"{delta.finished_requests:.0f} requests)",
        ),
        Check(
            "no preemptions",
            delta.preemptions == 0,
            f"{delta.preemptions:.0f} preemptions; a preempted request recomputes "
            "its context",
        ),
    ]


def print_checks(phase: str, checks: list[Check]) -> None:
    for check in checks:
        verdict = "PASS" if check.passed else "FAIL"
        print(f"[{phase}] {verdict} {check.name}: {check.detail}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args, measured_bench_args = parse_args(sys.argv[1:] if argv is None else argv)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    served_model = wait_for_model(base_url, args.ready_timeout_s)
    model = args.model or served_model
    report: dict[str, object] = {
        "config": {key: str(value) for key, value in vars(args).items()}
        | {"model": model, "measured_bench_args": measured_bench_args},
    }
    checks: list[Check] = []

    if not args.skip_warmup:
        print(
            f"Warmup: vllm bench x {args.dp_size} ranks, {args.pool_prompts} "
            f"prompts of {args.prompt_len} tokens each",
            flush=True,
        )
        before = read_counters(base_url)
        warmup_results = run_warmup(args, model)
        warmup_delta = read_counters(base_url) - before
        warmup_checks = check_warmup(args, warmup_results, warmup_delta)
        print_checks("warmup", warmup_checks)
        checks += warmup_checks
        report["warmup"] = {
            "server_counters": asdict(warmup_delta),
            "checks": [asdict(check) for check in warmup_checks],
        }
        if not all(check.passed for check in warmup_checks):
            write_report(args, report, checks)
            return 1

    print(f"Measured: vllm bench serve {' '.join(measured_bench_args)}", flush=True)
    before = read_counters(base_url)
    completed = subprocess.run(
        bench_command(args, model, MEASURED_RESULT_FILENAME, measured_bench_args)
    )
    if completed.returncode != 0:
        print(f"measured vllm bench exited {completed.returncode}", flush=True)
        return completed.returncode
    measured_delta = read_counters(base_url) - before
    measured_result = read_bench_result(args, MEASURED_RESULT_FILENAME)
    measured_checks = check_measured(measured_result, measured_delta)
    print_checks("measured", measured_checks)
    checks += measured_checks
    report["measured"] = {
        "vllm_bench_result": str(args.result_dir / MEASURED_RESULT_FILENAME),
        "summary": {key: measured_result.get(key) for key in MEASURED_SUMMARY_KEYS},
        "server_counters": asdict(measured_delta),
        "checks": [asdict(check) for check in measured_checks],
    }
    write_report(args, report, checks)
    return 0 if all(check.passed for check in checks) else 1


def write_report(
    args: argparse.Namespace, report: dict[str, object], checks: list[Check]
) -> None:
    report["passed"] = all(check.passed for check in checks)
    path = args.result_dir / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2))
    print(f"{'PASS' if report['passed'] else 'FAIL'}: report saved to {path}")


if __name__ == "__main__":
    sys.exit(main())
