# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reproducible prefill-only API smoke for the AsyncCam FFN graph deployment."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 120
CHUNK_SIZE = 8192
CONCURRENT_REQUESTS = 4


def send_completion(
    endpoint: str, model: str, name: str, prompt: str, timeout: int
) -> dict[str, Any]:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    ).encode()
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    result: dict[str, Any] = {"name": name, "started_at": started_at}
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        usage = body.get("usage") or {}
        result.update(
            {
                "ok": bool(body.get("choices")) and usage.get("completion_tokens") == 1,
                "request_id": body.get("id"),
                "usage": usage,
                "completion": [
                    choice.get("text") for choice in body.get("choices", [])
                ],
            }
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        result.update({"ok": False, "error": str(exc)})
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:17100/v1/completions")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Unique numbers avoid tokenizer compression of a repeated word and cross
    # the deployed 8192-token prefill chunk boundary. Verify from API usage.
    long_prompt = "List the next number after: " + " ".join(
        f"{number:05d}" for number in range(8500)
    )
    medium_prompt = "Continue this sequence: " + " ".join(
        f"{number:05d}" for number in range(1000)
    )
    cases = [
        ("short", "The capital of France is"),
        ("medium", medium_prompt),
        ("over_chunk", long_prompt),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w") as output_file:

        def record(result: dict[str, Any]) -> None:
            results.append(result)
            line = json.dumps(result, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            output_file.write(line + "\n")
            output_file.flush()

        for name, prompt in cases:
            result = send_completion(
                args.endpoint, args.model, name, prompt, args.timeout
            )
            record(result)
            if not result["ok"]:
                return 1
            if name == "over_chunk" and (
                (result.get("usage") or {}).get("prompt_tokens", 0) <= CHUNK_SIZE
            ):
                return 1

        start_concurrent = Event()

        def concurrent_case(index: int) -> dict[str, Any]:
            start_concurrent.wait()
            return send_completion(
                args.endpoint,
                args.model,
                f"concurrent_{index}",
                f"Continue: {index}. " + medium_prompt,
                args.timeout,
            )

        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            futures = [
                executor.submit(concurrent_case, index)
                for index in range(CONCURRENT_REQUESTS)
            ]
            start_concurrent.set()
            for future in as_completed(futures):
                record(future.result())

    over_chunk = next(result for result in results if result["name"] == "over_chunk")
    prompt_tokens = (over_chunk.get("usage") or {}).get("prompt_tokens", 0)
    return (
        0
        if all(result["ok"] for result in results) and prompt_tokens > CHUNK_SIZE
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
