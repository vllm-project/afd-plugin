#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check A5 requests through Proxy or Attention; select smoke, eos, tail, or full."""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

CONCURRENCIES = (1, 8, 32)
BOUNDARY_LENGTHS = (31, 32, 33, 63, 64, 65)
LONG_INPUT_LENGTH = 1024
REQUEST_TIMEOUT = 300
DRAIN_TIMEOUT = 60
REQUEST_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


async def run_checks(args: argparse.Namespace) -> dict:
    tokens = []
    if args.suite == "full":
        token_data = json.loads(Path(args.token_pool).read_text())
        tokens = token_data["tokens"]
        if len(tokens) < LONG_INPUT_LENGTH or any(type(t) is not int for t in tokens):
            raise ValueError("token pool must contain at least 1024 integer token IDs")
    records = []
    eos_seen = False

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=REQUEST_TIMEOUT,
        limits=httpx.Limits(max_connections=40),
        trust_env=False,
    ) as client:

        async def complete(
            name: str,
            prompt: str | list[int],
            budget: int,
            force_length: bool = False,
            *,
            chat: bool = False,
        ) -> dict:
            body = {
                "model": args.model,
                "temperature": 0,
                "max_tokens": budget,
                "stream": False,
            }
            if chat:
                body["messages"] = [{"role": "user", "content": prompt}]
                body["chat_template_kwargs"] = {
                    "thinking": False,
                    "enable_thinking": False,
                }
            else:
                body["prompt"] = prompt
            if force_length:
                body.update(ignore_eos=True, min_tokens=budget)
            endpoint = "/v1/chat/completions" if chat else "/v1/completions"
            Path(args.output, f"{name}.request.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=2) + "\n"
            )
            start = time.monotonic()
            start_timestamp_ns = time.time_ns()
            response = await client.post(
                endpoint,
                json=body,
                headers={"X-Request-Id": f"{args.run_id}-{name}"},
            )
            response.raise_for_status()
            data = response.json()
            Path(args.output, f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
            if data.get("error") or not data.get("choices"):
                raise ValueError(f"{name}: missing choices or API error")
            choice = data["choices"][0]
            finish = choice.get("finish_reason")
            usage = data.get("usage", {})
            count = usage.get("completion_tokens", 0)
            text = (
                choice.get("message", {}).get("content") if chat else choice.get("text")
            )
            if not text or not 0 < count <= budget:
                raise ValueError(f"{name}: empty output or invalid token count")
            if finish not in ("stop", "length"):
                raise ValueError(f"{name}: invalid finish_reason={finish}")
            if force_length and (finish != "length" or count != budget):
                raise ValueError(f"{name}: max_tokens truncation did not execute")
            if isinstance(prompt, list) and usage.get("prompt_tokens") != len(prompt):
                raise ValueError(f"{name}: token-ID prompt length changed in PD")
            return {
                "name": name,
                "endpoint": endpoint,
                "response_id": data.get("id"),
                "start_timestamp_ns": start_timestamp_ns,
                "end_timestamp_ns": time.time_ns(),
                "completion_tokens": count,
                "prompt_tokens": usage.get("prompt_tokens"),
                "finish_reason": finish,
                "elapsed_seconds": time.monotonic() - start,
            }

        # Run natural EOS first so an unsuitable case does not repeat load tests.
        print(
            f"[{args.suite}] checking natural EOS through chat completion", flush=True
        )
        result = await complete("eos", args.eos_prompt, args.eos_max_tokens, chat=True)
        records.append(result)
        eos_seen = result["finish_reason"] == "stop"
        if not eos_seen:
            raise ValueError(
                "EOS was not observed with the chat template; rerun --suite eos "
                "with a frozen --eos-prompt/--eos-max-tokens "
                "before repeating load tests"
            )

        if args.suite == "full":
            for concurrency in CONCURRENCIES:
                for kind, prompt in (
                    ("short", "Reply with a short greeting."),
                    ("long", tokens[:LONG_INPUT_LENGTH]),
                ):
                    print(
                        f"[full] {kind} input, {concurrency} concurrent requests",
                        flush=True,
                    )
                    results = await asyncio.gather(
                        *[
                            complete(f"c{concurrency}-{kind}-{index}", prompt, 64, True)
                            for index in range(concurrency)
                        ]
                    )
                    records.extend(results)
            for length in BOUNDARY_LENGTHS:
                records.append(
                    await complete(f"boundary-{length}", tokens[:length], 32, True)
                )
        elif args.suite == "smoke":
            records.append(
                await complete("smoke-short", "Reply with a short greeting.", 64, True)
            )

        async def stream_check(cancel: bool) -> None:
            name = "cancel" if cancel else "stream"
            body = {
                "model": args.model,
                "prompt": "Count the integers starting at one.",
                "temperature": 0,
                "max_tokens": 512 if cancel else 32,
                "ignore_eos": True,
                "min_tokens": 512 if cancel else 32,
                "stream": True,
            }
            saw_text = False
            saw_done = False
            saw_finish = False
            response_ids = set()
            start_timestamp_ns = time.time_ns()
            lines = []
            async with client.stream(
                "POST",
                "/v1/completions",
                json=body,
                headers={"X-Request-Id": f"{args.run_id}-{name}"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    lines.append(line)
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        raise ValueError(f"{name}: SSE error")
                    if chunk.get("id"):
                        response_ids.add(chunk["id"])
                    for choice in chunk.get("choices", []):
                        saw_text |= bool(choice.get("text"))
                        saw_finish |= choice.get("finish_reason") in ("stop", "length")
                    if cancel and saw_text:
                        if saw_finish:
                            raise ValueError("cancel: request already finished")
                        break
            Path(args.output, f"{name}.sse").write_text("\n".join(lines) + "\n")
            if not saw_text or (not cancel and not (saw_done and saw_finish)):
                raise ValueError(f"{name}: incomplete SSE response")
            records.append(
                {
                    "name": name,
                    "response_ids": sorted(response_ids),
                    "start_timestamp_ns": start_timestamp_ns,
                    "end_timestamp_ns": time.time_ns(),
                    "cancelled_after_text": cancel,
                    "saw_done": saw_done,
                }
            )

        if args.suite != "eos":
            print(
                f"[{args.suite}] checking SSE, cancellation, and recovery", flush=True
            )
            await stream_check(False)
            await stream_check(True)
            records.append(
                await complete("recovery", "Reply with a short greeting.", 64, True)
            )
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while True:
            response = await client.get(f"{args.attention_url.rstrip('/')}/metrics")
            response.raise_for_status()
            gauges = {}
            for name in REQUEST_GAUGES:
                values = re.findall(
                    rf"^{re.escape(name)}(?:\{{[^\n]*\}})? ([^\s]+)",
                    response.text,
                    re.MULTILINE,
                )
                if not values:
                    raise ValueError(f"missing Attention metric: {name}")
                gauges[name] = [float(value) for value in values]
            snapshot = {"attention_gauges": gauges}
            proxy_idle = True
            if not args.standalone:
                response = await client.get("/healthcheck")
                response.raise_for_status()
                snapshot["proxy"] = response.json()
                proxy_idle = (
                    snapshot["proxy"].get("status") == "ok"
                    and snapshot["proxy"].get("request_num") == 0
                )
            if proxy_idle and all(
                value == 0 for values in gauges.values() for value in values
            ):
                break
            if time.monotonic() >= deadline:
                raise ValueError(
                    f"requests did not drain after cancellation: {snapshot}"
                )
            await asyncio.sleep(1)
    return {
        "run_id": args.run_id,
        "suite": args.suite,
        "request_count": len(records),
        "request_checks_passed": True,
        "full_request_suite_passed": args.suite == "full",
        "f0_passed": None,
        "golden_checked": False,
        "eos_seen": eos_seen,
        "after_recovery": snapshot,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--attention-url", required=True)
    parser.add_argument(
        "--suite",
        choices=("smoke", "eos", "tail", "full"),
        default="full",
        help="smoke: 5 requests; eos: 1; tail: 4 remaining checks; full: 92 (default)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="send to Attention directly and omit proxy healthcheck",
    )
    parser.add_argument("--token-pool", help="required only for --suite full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="dsv4-afd")
    parser.add_argument("--eos-max-tokens", type=int, default=1024)
    parser.add_argument("--eos-prompt", default="Reply with only the word Hello.")
    args = parser.parse_args()
    if args.suite == "full" and not args.token_pool:
        parser.error("--token-pool is required for --suite full")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = asyncio.run(run_checks(args))
    except Exception as exc:
        (output / "request_summary.json").write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "suite": args.suite,
                    "request_checks_passed": False,
                    "full_request_suite_passed": False,
                    "f0_passed": None,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        raise
    (output / "request_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"{args.suite} request checks passed ({result['request_count']} requests); "
        "only full covers the complete request suite. "
        "PD/AFD/DSpark/Graph/U2 evidence and cleanup remain required."
    )


if __name__ == "__main__":
    main()
