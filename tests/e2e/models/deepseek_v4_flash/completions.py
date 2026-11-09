# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cancellable concurrent acceptance with per-request diagnostics."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.case import JsonValue
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_COMPLETION_MAX_TOKENS,
    DSV4_CONCURRENT_REQUESTS,
    DSV4_PROMPT_FIRST_OPERAND,
    DSV4_PROMPT_SECOND_OPERAND,
    DSV4_REQUEST_TIMEOUT_S,
)


def evaluate_completions(
    *,
    url: str,
    model: str,
    output_path: Path,
    check_answer: bool = True,
    answer_repetitions: int = 1,
    first_request_index: int = 0,
    num_requests: int = DSV4_CONCURRENT_REQUESTS,
    max_tokens: int = DSV4_COMPLETION_MAX_TOKENS,
    timeout_s: float = DSV4_REQUEST_TIMEOUT_S,
    first_operand: int = DSV4_PROMPT_FIRST_OPERAND,
    second_operand: int = DSV4_PROMPT_SECOND_OPERAND,
) -> None:
    """Run one batch of ten concurrent requests and check every response.

    ``check_answer`` requires the content to be exactly the expected sum and the
    generation to finish with ``stop``, which is what the asynchronous case
    answers. A profile whose host does not return reliable answers yet passes
    ``check_answer=False``: the ten requests must still be served together and
    each must return a nonempty answer that finished, which is the plumbing this
    smoke case exists to cover. The exact check stays on for every host that has
    it. CUDA replay qualification uses successive rounds of the same prompt
    shape with different operands, so reusing stale Hash-router IDs cannot be
    hidden by sending the same request again. Repeating the requested answer
    exercises live decode after the answer's first token, rather than only
    checking a one-token answer computed by the eager prefill.
    """
    # Case declarations are imported during collection; load the client only
    # when the runner executes this task.
    import httpx

    results: list[dict] = [
        {
            "index": index,
            "prompt": (
                f"Compute {first_operand + index} + "
                f"{second_operand}. "
                + (
                    "Reply with just the number."
                    if answer_repetitions == 1
                    else f"Reply with the result repeated {answer_repetitions} times, "
                    "separated by single spaces and nothing else."
                )
            ),
            "error": "Request cancelled before completion",
        }
        for index in range(first_request_index, first_request_index + num_requests)
    ]

    async def request(client: httpx.AsyncClient, item: dict) -> None:
        item["started_at"] = time.monotonic()
        try:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"thinking": False},
                },
            )
            item["status_code"] = response.status_code
            # Keep even non-JSON error bodies for diagnosing a failed server.
            item["response_body"] = response.text
            response.raise_for_status()
            item["response"] = response.json()
            validate_response(item["response"], check_answer=check_answer)
            expected = first_operand + item["index"] + second_operand
            expected_content = " ".join([str(expected)] * answer_repetitions)
            content = item["response"]["choices"][0]["message"]["content"].strip()
            if check_answer and content != expected_content:
                raise RuntimeError(
                    f"wrong answer: expected {expected_content!r}, got {content!r}"
                )
            del item["error"]
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            item["finished_at"] = time.monotonic()

    async def run_requests() -> None:
        # All requests are scheduled before yielding. Async socket operations
        # cancel immediately, so SIGTERM can reach the runner's service cleanup
        # without waiting for a thread blocked in a 300-second HTTP read.
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            await asyncio.gather(*(request(client, item) for item in results))

    try:
        asyncio.run(run_requests())
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    errors = [
        f"Request {item['index']}: {item['error']}"
        for item in results
        if "error" in item
    ]
    if errors:
        raise RuntimeError("Concurrent completions failed:\n" + "\n".join(errors))
    # Scheduling together is insufficient: retain timings and require every
    # request to start before the first one finishes.
    if max(item["started_at"] for item in results) >= min(
        item["finished_at"] for item in results
    ):
        raise RuntimeError("The completion requests did not overlap")
    for item in results:
        content = item["response"]["choices"][0]["message"]["content"]
        print(f"Request {item['index'] + 1}: {item['prompt']} -> {content}")
    print(f"Concurrent completions: {len(results)}/{num_requests} passed")


@dataclass(frozen=True)
class CompletionTask:
    """The model's concurrent answer/plumbing contract, with recorded workload."""

    check_answer: bool = True
    answer_repetitions: int = 1
    first_request_index: int = 0
    num_requests: int = DSV4_CONCURRENT_REQUESTS
    max_tokens: int = DSV4_COMPLETION_MAX_TOKENS
    timeout_s: float = DSV4_REQUEST_TIMEOUT_S
    first_operand: int = DSV4_PROMPT_FIRST_OPERAND
    second_operand: int = DSV4_PROMPT_SECOND_OPERAND

    def resolve(
        self, environment: Mapping[str, str]
    ) -> tuple[CompletionTask, tuple[str, ...]]:
        return self, ()

    def execute(
        self, *, base_url: str, model_name: str, checkpoint: str, output_path: Path
    ) -> dict[str, JsonValue]:
        evaluate_completions(
            url=f"{base_url}/v1/chat/completions",
            model=model_name,
            output_path=output_path / "completions.json",
            check_answer=self.check_answer,
            answer_repetitions=self.answer_repetitions,
            first_request_index=self.first_request_index,
            num_requests=self.num_requests,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            first_operand=self.first_operand,
            second_operand=self.second_operand,
        )
        return {"request_count": self.num_requests, "answer_checked": self.check_answer}


def validate_response(result: dict, *, check_answer: bool = True) -> None:
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("must return one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise RuntimeError("returned an invalid message")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("returned empty content")
    finish_reason = choice.get("finish_reason")
    if not check_answer:
        # A profile that does not check the answer still requires the request to
        # have finished; anything without a terminal reason did not.
        if not isinstance(finish_reason, str) or not finish_reason:
            raise RuntimeError("did not finish normally")
        return
    if finish_reason != "stop":
        raise RuntimeError("did not finish normally")
