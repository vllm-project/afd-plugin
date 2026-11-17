#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Measure prefill TTFT by request length with identical seeded arrival plans.

One generated token isolates prefill. X-data-parallel-rank selects an actual
vLLM DP replica, including for synchronous connectors. No server probe is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp

SHORT_TOKENS = 256
LONG_TOKENS = 16384
BALANCED_TOKENS = round((4 * SHORT_TOKENS + LONG_TOKENS) / 5)
VOCAB_MIN = 1000
VOCAB_MAX = 30000
REQUEST_TIMEOUT_S = 120
TTFT_SLO_S = 5.0
SHORT_TTFT_SLO_S = 1.0
SHORT_ARRIVAL_DELAY_S = 0.05


@dataclass(frozen=True)
class Arrival:
    arrival_s: float
    input_tokens: int
    dp_rank: int
    token_seed: int
    cohort: str


def plan(profile: str, rate: float, duration: float, seed: int) -> list[Arrival]:
    arrivals_rng = random.Random(seed)
    lengths_rng = random.Random(seed + 1)
    tokens_rng = random.Random(seed + 2)
    arrivals: list[Arrival] = []
    timestamp = 0.0
    lengths: list[int] = []
    assigned_tokens = [0, 0]
    while (timestamp := timestamp + arrivals_rng.expovariate(rate)) < duration:
        if not lengths:
            lengths = [SHORT_TOKENS] * 4 + [LONG_TOKENS]
            lengths_rng.shuffle(lengths)
        length = BALANCED_TOKENS if profile == "balanced" else lengths.pop()
        cohort = (
            "balanced"
            if profile == "balanced"
            else ("short" if length == SHORT_TOKENS else "long")
        )
        if profile == "skewed":
            rank = 0 if cohort == "short" else 1
        elif profile == "token_balanced":
            # A fixed greedy input-token control, not a feedback scheduler.
            rank = 0 if assigned_tokens[0] <= assigned_tokens[1] else 1
        else:
            rank = len(arrivals) % 2
        assigned_tokens[rank] += length
        arrivals.append(
            Arrival(timestamp, length, rank, tokens_rng.randrange(2**31), cohort)
        )
    return arrivals


def payload(arrival: Arrival, model: str) -> bytes:
    rng = random.Random(arrival.token_seed)
    return json.dumps(
        {
            "model": model,
            "prompt": [
                rng.randrange(VOCAB_MIN, VOCAB_MAX) for _ in range(arrival.input_tokens)
            ],
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()


async def request(
    session: aiohttp.ClientSession,
    url: str,
    arrival: Arrival,
    body: bytes,
    epoch: float,
) -> dict:
    scheduled = epoch + arrival.arrival_s
    await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
    started = time.perf_counter()
    first = None
    usage = None
    error = None
    text_parts = []
    try:
        async with session.post(
            url + "/v1/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-data-parallel-rank": str(arrival.dp_rank),
            },
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:500]}"
                )
            async for line in response.content:
                if not line.startswith(b"data: "):
                    continue
                encoded = line[6:].strip()
                if encoded == b"[DONE]":
                    continue
                event = json.loads(encoded)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("choices") and first is None:
                    first = time.perf_counter()
                for choice in event.get("choices", []):
                    text_parts.append(choice.get("text", ""))
                if event.get("usage"):
                    usage = event["usage"]
        if first is None or usage is None:
            raise RuntimeError("Missing first token or final usage")
        if (
            usage["prompt_tokens"] != arrival.input_tokens
            or usage["completion_tokens"] != 1
        ):
            raise RuntimeError(f"Unexpected token usage: {usage}")
    except (aiohttp.ClientError, TimeoutError, RuntimeError, ValueError) as exc:
        error = str(exc)
    finished = time.perf_counter()
    return dict(
        **asdict(arrival),
        send_delay_s=started - scheduled,
        ttft_s=None if first is None else first - scheduled,
        service_ttft_s=None if first is None else first - started,
        elapsed_s=finished - scheduled,
        finished_at_s=finished - epoch,
        usage=usage,
        output_text="".join(text_parts),
        error=error,
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows: list[dict], duration: float) -> dict:
    elapsed = max(duration, max((r["finished_at_s"] for r in rows), default=duration))
    groups = {}
    for name in ("all", "short", "long", "balanced"):
        selected = [r for r in rows if name == "all" or r["cohort"] == name]
        successful = [r for r in selected if r["error"] is None]
        values = [r["ttft_s"] for r in successful]
        if not selected:
            continue
        groups[name] = dict(
            requests=len(selected),
            successful=len(successful),
            errors=len(selected) - len(successful),
            ttft_mean_s=statistics.mean(values) if values else None,
            ttft_p50_s=percentile(values, 0.5),
            ttft_p95_s=percentile(values, 0.95),
            ttft_p99_s=percentile(values, 0.99),
            max_send_delay_s=max(r["send_delay_s"] for r in selected),
            offered_rps=len(selected) / duration,
            achieved_rps=len(successful) / elapsed,
            input_tokens_per_s=sum(r["input_tokens"] for r in successful) / elapsed,
            goodput_5s_rps=sum(v <= TTFT_SLO_S for v in values) / elapsed,
            arrival_window_goodput_5s_rps=sum(v <= TTFT_SLO_S for v in values)
            / duration,
            arrival_window_goodput_1s_rps=sum(v <= SHORT_TTFT_SLO_S for v in values)
            / duration,
            fraction_under_5s=sum(v <= TTFT_SLO_S for v in values) / len(selected),
            fraction_under_1s=sum(v <= SHORT_TTFT_SLO_S for v in values)
            / len(selected),
        )
    return dict(duration_s=duration, elapsed_s=elapsed, cohorts=groups)


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warm every cohort on both DP replicas, outside measurement.
        for length in (SHORT_TOKENS, BALANCED_TOKENS, LONG_TOKENS):
            arrivals = [
                Arrival(0, length, rank, args.seed + rank, "warmup")
                for rank in range(2)
            ]
            epoch = time.perf_counter()
            rows = await asyncio.gather(
                *[
                    request(session, args.url, a, payload(a, args.model), epoch)
                    for a in arrivals
                ]
            )
            if any(row["error"] for row in rows):
                raise RuntimeError(f"Warmup failed: {rows}")
        if args.mode in ("pairs", "both"):
            for scenario, lengths, short_delay in (
                ("short_alone", [SHORT_TOKENS], 0.0),
                ("short_pair", [SHORT_TOKENS, SHORT_TOKENS], 0.0),
                ("mixed_pair", [SHORT_TOKENS, LONG_TOKENS], 0.0),
                (
                    "long_inflight",
                    [SHORT_TOKENS, LONG_TOKENS],
                    SHORT_ARRIVAL_DELAY_S,
                ),
            ):
                rows = []
                scenario_epoch = time.perf_counter()
                for iteration in range(args.pairs):
                    arrivals = [
                        Arrival(
                            short_delay if rank == 0 else 0,
                            n,
                            (rank + iteration) % 2,
                            args.seed + iteration * 2 + rank,
                            "short" if n == SHORT_TOKENS else "long",
                        )
                        for rank, n in enumerate(lengths)
                    ]
                    bodies = [payload(a, args.model) for a in arrivals]
                    epoch = time.perf_counter()
                    trial_rows = await asyncio.gather(
                        *[
                            request(session, args.url, a, b, epoch)
                            for a, b in zip(arrivals, bodies, strict=True)
                        ]
                    )
                    for row in trial_rows:
                        row["finished_at_s"] += epoch - scenario_epoch
                    rows.extend(trial_rows)
                summary = summarize(rows, time.perf_counter() - scenario_epoch)
                pair_result = dict(
                    scenario=scenario,
                    summary=summary,
                    requests=rows,
                )
                (output / f"{scenario}.json").write_text(
                    json.dumps(pair_result, indent=2)
                )
                print(json.dumps(dict(scenario=scenario, **summary)), flush=True)
            if args.mode == "pairs":
                return
        for repeat in range(args.repeats):
            for profile in args.profiles:
                for rate in args.rates:
                    seed = args.seed + repeat * 100000 + round(rate * 100)
                    arrivals = plan(profile, rate, args.duration, seed)
                    assert arrivals
                    bodies = [payload(a, args.model) for a in arrivals]
                    encoded_plan = json.dumps([asdict(a) for a in arrivals]).encode()
                    epoch = time.perf_counter()
                    rows = await asyncio.gather(
                        *[
                            request(session, args.url, a, b, epoch)
                            for a, b in zip(arrivals, bodies, strict=True)
                        ]
                    )
                    await asyncio.sleep(
                        max(0.0, epoch + args.duration - time.perf_counter())
                    )
                    result = dict(
                        profile=profile,
                        target_rps=rate,
                        repeat=repeat,
                        seed=seed,
                        plan_sha256=hashlib.sha256(encoded_plan).hexdigest(),
                        summary=summarize(rows, args.duration),
                        requests=rows,
                    )
                    (output / f"{profile}-rps{rate:g}-rep{repeat}.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in result.items()
                                if key != "requests"
                            }
                        ),
                        flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18507")
    parser.add_argument("--model", default="prefill-bench")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("pairs", "poisson", "both"), default="poisson"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("balanced", "mixed", "skewed", "token_balanced"),
        default=["balanced", "mixed"],
    )
    parser.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if (
        min(args.rates) <= 0
        or args.duration <= 0
        or args.repeats <= 0
        or args.pairs <= 0
    ):
        parser.error("rates, duration, repeats and pairs must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
