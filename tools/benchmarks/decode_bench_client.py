# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-phase load generator for the decode benchmark's prompt-pool mode.

Run it against an Attention instance whose ``AFDDecodeBenchConnector`` has
``num_prompts`` > 0 (see ``tools/benchmarks/decode_bench.py``):

1. Warmup: sends ``--num-prompts`` requests to every Attention DP rank, pinned
   with the ``X-data-parallel-rank`` header, so each rank prefills and pins its
   whole prompt pool. Their metrics are discarded.
2. Measured: sends ``--num-requests`` requests through vLLM's load balancer.
   The server remaps each onto a pool prompt, so the prompt sent here is only a
   placeholder and every request decodes over real, prefix-cached KV.

The server's prefix-cache counters before and after the measured phase give the
prompt tokens prefilled per measured request; a clean decode-only run prefills
exactly one.

Usage:
    python tools/benchmarks/decode_bench_client.py --num-prompts 1000 \
        --num-requests 4000 --output-len 256 --max-concurrency 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np

DATA_PARALLEL_RANK_HEADER = "X-data-parallel-rank"
COMPLETIONS_PATH = "/v1/completions"
MODELS_PATH = "/v1/models"
METRICS_PATH = "/metrics"
PREFIX_CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries"
PREFIX_CACHE_HITS_METRIC = "vllm:prefix_cache_hits"
SSE_DATA_PREFIX = b"data: "
SSE_DONE = b"[DONE]"
HTTP_OK = 200
# Placeholder prompt token ids stay below this, which every vocabulary covers.
PLACEHOLDER_TOKEN_ID_LIMIT = 1000
# The server forces warmup requests to one output token as well.
WARMUP_OUTPUT_TOKENS = 1
READY_POLL_INTERVAL_S = 5.0
PERCENTILES = (50, 90, 99)
MS_PER_S = 1000.0


@dataclass
class RequestResult:
    """Timing of one streamed completion."""

    ok: bool
    ttft_s: float = 0.0
    latency_s: float = 0.0
    output_tokens: int = 0
    itl_s: list[float] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class PlannedRequest:
    payload: dict[str, object]
    headers: dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18305)
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name; defaults to the first one in /v1/models.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=2,
        help="Attention data-parallel ranks; each is warmed separately.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Pool prompts per Attention DP rank; must match the server's "
        "kv_connector_extra_config num_prompts.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Measured requests.",
    )
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=math.inf,
        help="Measured requests per second (Poisson); inf sends them as fast "
        "as --max-concurrency allows.",
    )
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=64,
        help="In-flight warmup requests across all ranks.",
    )
    parser.add_argument(
        "--placeholder-prompt-len",
        type=int,
        default=8,
        help="Tokens in each request's placeholder prompt; the server replaces "
        "them with a pool prompt.",
    )
    parser.add_argument(
        "--pin-measured-ranks",
        action="store_true",
        help="Spread measured requests round-robin over DP ranks with the "
        f"{DATA_PARALLEL_RANK_HEADER} header instead of vLLM's load balancer.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Measure only, against a server whose pool is already warm.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--result-file", type=Path, default=None)
    return parser.parse_args(argv)


async def wait_for_model(
    session: aiohttp.ClientSession,
    base_url: str,
    timeout_s: float,
) -> str:
    """Poll until the server answers /v1/models; return the first model id."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            async with session.get(f"{base_url}{MODELS_PATH}") as response:
                if response.status == HTTP_OK:
                    models = await response.json()
                    return models["data"][0]["id"]
        except aiohttp.ClientError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"{base_url} not ready after {timeout_s:.0f}s")
        await asyncio.sleep(READY_POLL_INTERVAL_S)


async def read_prefix_cache_counters(
    session: aiohttp.ClientSession,
    base_url: str,
) -> tuple[float, float]:
    """Return the (queries, hits) prefix-cache token counters, summed."""
    async with session.get(f"{base_url}{METRICS_PATH}") as response:
        text = await response.text()
    totals = {PREFIX_CACHE_QUERIES_METRIC: 0.0, PREFIX_CACHE_HITS_METRIC: 0.0}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        metric = sample.split("{", 1)[0].removesuffix("_total")
        if metric in totals:
            totals[metric] += float(value)
    return totals[PREFIX_CACHE_QUERIES_METRIC], totals[PREFIX_CACHE_HITS_METRIC]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request: PlannedRequest,
) -> RequestResult:
    start = time.perf_counter()
    first_chunk_time: float | None = None
    last_chunk_time = start
    itl_s: list[float] = []
    output_tokens = 0
    try:
        async with session.post(
            url, json=request.payload, headers=request.headers
        ) as response:
            if response.status != HTTP_OK:
                body = await response.text()
                return RequestResult(ok=False, error=f"HTTP {response.status}: {body}")
            async for raw_line in response.content:
                line = raw_line.strip()
                if not line.startswith(SSE_DATA_PREFIX):
                    continue
                data = line.removeprefix(SSE_DATA_PREFIX)
                if data == SSE_DONE:
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    output_tokens = usage["completion_tokens"]
                if not chunk.get("choices"):
                    continue
                now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = now
                else:
                    itl_s.append(now - last_chunk_time)
                last_chunk_time = now
    except aiohttp.ClientError as error:
        return RequestResult(ok=False, error=repr(error))
    if first_chunk_time is None:
        return RequestResult(ok=False, error="stream ended without output")
    return RequestResult(
        ok=True,
        ttft_s=first_chunk_time - start,
        latency_s=last_chunk_time - start,
        output_tokens=output_tokens,
        itl_s=itl_s,
    )


async def run_phase(
    session: aiohttp.ClientSession,
    url: str,
    requests: list[PlannedRequest],
    max_concurrency: int,
    request_rate: float,
    rng: random.Random,
) -> tuple[list[RequestResult], float]:
    """Send ``requests``; return their results and the phase duration."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited(request: PlannedRequest) -> RequestResult:
        async with semaphore:
            return await send_request(session, url, request)

    start = time.perf_counter()
    tasks = []
    for request in requests:
        tasks.append(asyncio.create_task(limited(request)))
        if not math.isinf(request_rate):
            await asyncio.sleep(rng.expovariate(request_rate))
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - start


def build_payload(
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        # Per-chunk usage gives the exact output token count even when one
        # chunk carries several tokens.
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }


def placeholder_prompt(rng: random.Random, length: int) -> list[int]:
    return [rng.randrange(PLACEHOLDER_TOKEN_ID_LIMIT) for _ in range(length)]


def plan_warmup(
    args: argparse.Namespace, model: str, rng: random.Random
) -> list[PlannedRequest]:
    # Interleave ranks so every rank warms its pool concurrently.
    return [
        PlannedRequest(
            payload=build_payload(
                model,
                placeholder_prompt(rng, args.placeholder_prompt_len),
                WARMUP_OUTPUT_TOKENS,
            ),
            headers={DATA_PARALLEL_RANK_HEADER: str(rank)},
        )
        for _ in range(args.num_prompts)
        for rank in range(args.dp_size)
    ]


def plan_measured(
    args: argparse.Namespace, model: str, rng: random.Random
) -> list[PlannedRequest]:
    return [
        PlannedRequest(
            payload=build_payload(
                model,
                placeholder_prompt(rng, args.placeholder_prompt_len),
                args.output_len,
            ),
            headers=(
                {DATA_PARALLEL_RANK_HEADER: str(index % args.dp_size)}
                if args.pin_measured_ranks
                else {}
            ),
        )
        for index in range(args.num_requests)
    ]


def distribution_ms(values_s: list[float]) -> dict[str, float]:
    if not values_s:
        return {}
    values_ms = np.asarray(values_s) * MS_PER_S
    summary = {"mean": float(values_ms.mean())}
    for percentile in PERCENTILES:
        summary[f"p{percentile}"] = float(np.percentile(values_ms, percentile))
    return summary


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, object]:
    completed = [result for result in results if result.ok]
    output_tokens = sum(result.output_tokens for result in completed)
    tpot_s = [
        (result.latency_s - result.ttft_s) / (result.output_tokens - 1)
        for result in completed
        if result.output_tokens > 1
    ]
    return {
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "duration_s": duration_s,
        "request_throughput": len(completed) / duration_s,
        "output_token_throughput": output_tokens / duration_s,
        "total_output_tokens": output_tokens,
        "ttft_ms": distribution_ms([result.ttft_s for result in completed]),
        "tpot_ms": distribution_ms(tpot_s),
        "itl_ms": distribution_ms(
            [itl for result in completed for itl in result.itl_s]
        ),
        "e2el_ms": distribution_ms([result.latency_s for result in completed]),
    }


def first_error(results: list[RequestResult]) -> str:
    return next((result.error for result in results if not result.ok), "")


async def run(args: argparse.Namespace) -> dict[str, object]:
    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}{COMPLETIONS_PATH}"
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        served_model = await wait_for_model(session, base_url, args.ready_timeout_s)
        model = args.model or served_model
        report: dict[str, object] = {"config": vars(args) | {"model": model}}

        if not args.skip_warmup:
            warmup = plan_warmup(args, model, rng)
            print(
                f"Warmup: {args.num_prompts} prompts x {args.dp_size} ranks",
                flush=True,
            )
            warmup_results, warmup_duration_s = await run_phase(
                session, url, warmup, args.warmup_concurrency, math.inf, rng
            )
            num_failed = sum(not result.ok for result in warmup_results)
            if num_failed:
                raise RuntimeError(
                    f"{num_failed} warmup requests failed; first error: "
                    f"{first_error(warmup_results)}"
                )
            report["warmup"] = {
                "requests": len(warmup_results),
                "duration_s": warmup_duration_s,
            }
            print(f"Warmup done in {warmup_duration_s:.1f}s", flush=True)

        queries_before, hits_before = await read_prefix_cache_counters(
            session, base_url
        )
        print(f"Measured: {args.num_requests} requests", flush=True)
        measured_results, measured_duration_s = await run_phase(
            session,
            url,
            plan_measured(args, model, rng),
            args.max_concurrency,
            args.request_rate,
            rng,
        )
        queries_after, hits_after = await read_prefix_cache_counters(session, base_url)

        measured = summarize(measured_results, measured_duration_s)
        if measured["failed"]:
            measured["first_error"] = first_error(measured_results)
        num_completed = sum(result.ok for result in measured_results)
        prompt_tokens = queries_after - queries_before
        hit_tokens = hits_after - hits_before
        measured["prefix_cache"] = {
            "queried_tokens": prompt_tokens,
            "hit_tokens": hit_tokens,
            "hit_rate": hit_tokens / prompt_tokens if prompt_tokens else 0.0,
            # 1.0 when every measured request prefilled only its last token.
            "prefilled_tokens_per_request": (
                (prompt_tokens - hit_tokens) / num_completed if num_completed else 0.0
            ),
        }
        report["measured"] = measured
        return report


def print_summary(measured: dict[str, object]) -> None:
    prefix_cache = measured["prefix_cache"]
    assert isinstance(prefix_cache, dict)
    print(
        f"completed={measured['completed']} failed={measured['failed']} "
        f"duration={measured['duration_s']:.1f}s"
    )
    print(
        f"request throughput={measured['request_throughput']:.2f} req/s "
        f"output throughput={measured['output_token_throughput']:.1f} tok/s"
    )
    for metric in ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms"):
        values = measured[metric]
        assert isinstance(values, dict)
        formatted = " ".join(f"{key}={value:.2f}" for key, value in values.items())
        print(f"{metric}: {formatted}")
    print(
        f"prefix cache hit rate={prefix_cache['hit_rate']:.4f} "
        f"prefilled tokens/request={prefix_cache['prefilled_tokens_per_request']:.2f} "
        "(1.00 = decode only)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(run(args))
    measured = report["measured"]
    assert isinstance(measured, dict)
    print_summary(measured)
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(report, indent=2, default=str))
        print(f"Saved {args.result_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
