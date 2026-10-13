#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check A5 requests through Proxy or Attention, including targeted c32-short."""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

CONCURRENCIES = (1, 8, 32)
DIAGNOSTIC_CONCURRENCY = 32
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
                # Preserve generated IDs without changing detokenization or sampling.
                "return_token_ids": True,
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
            try:
                response = await client.post(
                    endpoint,
                    json=body,
                    headers={"X-Request-Id": f"{args.run_id}-{name}"},
                )
            except httpx.RequestError as exc:
                failure = {
                    "name": name,
                    "endpoint": endpoint,
                    "request_id": f"{args.run_id}-{name}",
                    "exception_type": type(exc).__name__,
                    "error": str(exc) or repr(exc),
                    "elapsed_seconds": time.monotonic() - start,
                    "start_timestamp_ns": start_timestamp_ns,
                    "end_timestamp_ns": time.time_ns(),
                }
                Path(args.output, f"{name}.transport.error.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n"
                )
                raise ValueError(
                    f"{name}: {type(exc).__name__} after "
                    f"{failure['elapsed_seconds']:.2f}s; "
                    "see transport.error.json and Proxy/Attention logs"
                ) from exc
            raw_path = Path(args.output, f"{name}.response.body")
            raw_path.write_bytes(response.content)
            Path(args.output, f"{name}.response.http.json").write_text(
                json.dumps(
                    {
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body_bytes": len(response.content),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{name}: HTTP {response.status_code} returned invalid/empty JSON "
                    f"({len(response.content)} bytes, "
                    "Content-Type="
                    f"{response.headers.get('content-type', '<missing>')}); "
                    f"raw response saved to {raw_path}; inspect Proxy/Attention logs"
                ) from exc
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
                failure = {
                    "name": name,
                    "response_id": data.get("id"),
                    "http_status": response.status_code,
                    "empty_text": not bool(text),
                    "completion_tokens": count,
                    "budget": budget,
                    "finish_reason": finish,
                    "token_ids": choice.get("token_ids"),
                }
                Path(args.output, f"{name}.validation.error.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n"
                )
                raise ValueError(
                    f"{name}: empty output or invalid token count "
                    f"(empty_text={not bool(text)}, completion_tokens={count}, "
                    f"finish_reason={finish}); see validation.error.json"
                )
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

        if args.suite in ("full", "c32-short"):
            concurrencies = (
                (DIAGNOSTIC_CONCURRENCY,)
                if args.suite == "c32-short"
                else CONCURRENCIES
            )
            prompts = [("short", "Reply with a short greeting.")]
            if args.suite == "full":
                prompts.append(("long", tokens[:LONG_INPUT_LENGTH]))
            for concurrency in concurrencies:
                for kind, prompt in prompts:
                    print(
                        f"[{args.suite}] {kind} input, "
                        f"{concurrency} concurrent requests",
                        flush=True,
                    )
                    results = await asyncio.gather(
                        *[
                            complete(f"c{concurrency}-{kind}-{index}", prompt, 64, True)
                            for index in range(concurrency)
                        ],
                        return_exceptions=True,
                    )
                    failures = [
                        {
                            "name": f"c{concurrency}-{kind}-{index}",
                            "exception_type": type(result).__name__,
                            "error": str(result) or repr(result),
                        }
                        for index, result in enumerate(results)
                        if isinstance(result, Exception)
                    ]
                    passed = [
                        result
                        for result in results
                        if not isinstance(result, Exception)
                    ]
                    # Collect the entire in-flight batch before failing. Cancelling
                    # siblings at the first empty output obscures their real results.
                    Path(
                        args.output, f"batch-c{concurrency}-{kind}.summary.json"
                    ).write_text(
                        json.dumps(
                            {
                                "request_count": concurrency,
                                "passed_count": len(passed),
                                "failed_count": len(failures),
                                "request_checks_passed": not failures,
                                "records": passed,
                                "failures": failures,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )
                    records.extend(passed)
                    if failures:
                        raise ValueError(
                            f"c{concurrency}-{kind}: {len(failures)}/{concurrency} "
                            "requests failed; "
                            + "; ".join(failure["error"] for failure in failures)
                        )
        if args.suite == "full":
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

        if args.suite not in ("eos", "c32-short"):
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
        choices=("smoke", "eos", "tail", "full", "c32-short"),
        default="full",
        help=(
            "smoke: 5; eos: 1; tail: 4; full: 92 (default); "
            "c32-short: EOS + 32 short requests for diagnosis only"
        ),
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
                    "exception_type": type(exc).__name__,
                    "error": str(exc) or repr(exc),
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
