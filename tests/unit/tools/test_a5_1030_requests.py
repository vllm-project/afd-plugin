# SPDX-License-Identifier: Apache-2.0
"""Check the request gate without launching models or using the network."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tools.validation import a5_1030_requests as validation


class CompletionStream(httpx.AsyncByteStream):
    def __init__(self, chunks, on_close):
        self.chunks = chunks
        self.on_close = on_close

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
            await asyncio.sleep(0)

    async def aclose(self):
        self.on_close()


@pytest.fixture
def request_gate(tmp_path, monkeypatch):
    token_pool = tmp_path / "tokens.json"
    token_pool.write_text(json.dumps({"tokens": [10] * 1024}))
    output = tmp_path / "requests"
    output.mkdir()
    args = argparse.Namespace(
        token_pool=str(token_pool),
        output=str(output),
        model="dsv4-afd",
        run_id="test",
        base_url="http://proxy",
        attention_url="http://attention",
        standalone=False,
        suite="full",
        eos_max_tokens=1024,
        eos_prompt="Reply with only the word Hello.",
    )

    def exercise(fault=None):
        active = 0
        peak = 0
        closed_streams = []
        calls = []

        async def handler(request):
            nonlocal active, peak
            calls.append((request.url.host, request.url.path))
            if request.url.path == "/metrics":
                value = 1 if fault == "busy" else 0
                return httpx.Response(
                    200,
                    text="\n".join(
                        f'{name}{{engine="{rank}"}} {value}'
                        for name in validation.REQUEST_GAUGES
                        for rank in range(4)
                    ),
                )
            if request.url.path == "/healthcheck":
                return httpx.Response(200, json={"status": "ok", "request_num": 0})
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            if fault == "http-error":
                return httpx.Response(500, text="backend failed")
            body = json.loads(request.content)
            chat = request.url.path == "/v1/chat/completions"
            if chat:
                assert body["messages"] == [
                    {"role": "user", "content": args.eos_prompt}
                ]
                assert body["chat_template_kwargs"] == {
                    "thinking": False,
                    "enable_thinking": False,
                }
                assert "stop" not in body and not body.get("ignore_eos")
            if body.get("stream"):
                chunks = [
                    b'data: {"choices":[{"text":"hello","finish_reason":null}]}\n\n'
                ]
                if fault != "broken-sse":
                    chunks.extend(
                        [
                            b'data: {"choices":[{"text":"",'
                            b'"finish_reason":"length"}]}\n\n',
                            b"data: [DONE]\n\n",
                        ]
                    )
                name = request.headers["X-Request-Id"]
                return httpx.Response(
                    200,
                    stream=CompletionStream(
                        chunks, lambda: closed_streams.append(name)
                    ),
                )
            if fault == "empty-response":
                return httpx.Response(200, json={"choices": []})
            forced = body.get("ignore_eos", False) or (chat and fault == "eos-length")
            count = body["max_tokens"] if forced else 2
            prompt = body["messages"][0]["content"] if chat else body["prompt"]
            prompt_length = len(prompt) if isinstance(prompt, list) else 8
            if fault == "changed-boundary" and isinstance(prompt, list):
                prompt_length += 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            **(
                                {"message": {"content": "hello"}}
                                if chat
                                else {"text": "hello"}
                            ),
                            "finish_reason": "length" if forced else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_length,
                        "completion_tokens": count,
                    },
                },
            )

        client_class = httpx.AsyncClient
        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            assert kwargs["trust_env"] is False
            return client_class(transport=transport, **kwargs)

        monkeypatch.setattr(
            validation.httpx,
            "AsyncClient",
            client_factory,
        )
        monkeypatch.setattr(validation, "DRAIN_TIMEOUT", 0)
        result = asyncio.run(validation.run_checks(args))
        return result, peak, closed_streams, calls

    return args, exercise


def test_proxy_request_gate_covers_load_boundaries_cancel_and_drain(request_gate):
    args, exercise = request_gate
    result, peak, closed, calls = exercise()
    assert result["request_checks_passed"] is True
    assert result["f0_passed"] is None
    assert result["full_request_suite_passed"] is True
    assert result["request_count"] == 92
    assert peak == 32
    names = {record["name"] for record in result["records"]}
    assert {f"boundary-{length}" for length in (31, 32, 33, 63, 64, 65)} <= names
    assert {"eos", "stream", "cancel", "recovery"} <= names
    assert "test-cancel" in closed
    assert ("proxy", "/healthcheck") in calls
    assert ("attention", "/metrics") in calls
    assert result["after_recovery"]["proxy"]["request_num"] == 0


def test_standalone_gate_uses_attention_metrics_without_proxy(request_gate):
    args, exercise = request_gate
    args.standalone = True
    args.base_url = args.attention_url
    result, _, _, calls = exercise()
    assert result["request_checks_passed"] is True
    assert all(path != "/healthcheck" for _, path in calls)


@pytest.mark.parametrize("suite,count", [("smoke", 5), ("eos", 1), ("tail", 4)])
def test_selected_suite_does_not_repeat_concurrency_and_boundary_load(
    request_gate, suite, count
):
    args, exercise = request_gate
    args.suite = suite
    args.token_pool = None
    result, peak, closed, calls = exercise()
    assert result["request_checks_passed"] is True
    assert result["full_request_suite_passed"] is False
    assert result["request_count"] == count
    assert peak == 1
    names = {record["name"] for record in result["records"]}
    assert not any(
        name.startswith(("c1-", "c8-", "c32-", "boundary-")) for name in names
    )
    assert ("proxy", "/v1/chat/completions") in calls
    assert ("attention", "/metrics") in calls
    if suite == "eos":
        assert not closed
        assert names == {"eos"}
    else:
        assert {"eos", "stream", "cancel", "recovery"} <= names


@pytest.mark.parametrize("suite", ["smoke", "eos", "tail", "full"])
def test_chat_length_limit_is_not_accepted_as_natural_eos(request_gate, suite):
    args, exercise = request_gate
    args.suite = suite
    with pytest.raises(ValueError, match="EOS was not observed with the chat template"):
        exercise("eos-length")
    # EOS runs first; an unsuitable case must not send any concurrency load.
    assert sorted(path.name for path in Path(args.output).iterdir()) == [
        "eos.json",
        "eos.request.json",
    ]


@pytest.mark.parametrize(
    "fault,match",
    [
        ("empty-response", "missing choices"),
        ("broken-sse", "incomplete SSE"),
        ("changed-boundary", "prompt length changed"),
        ("busy", "did not drain"),
    ],
)
def test_invalid_backend_behavior_does_not_pass(request_gate, fault, match):
    _, exercise = request_gate
    with pytest.raises(ValueError, match=match):
        exercise(fault)


def test_backend_http_failure_does_not_pass(request_gate):
    _, exercise = request_gate
    with pytest.raises(httpx.HTTPStatusError):
        exercise("http-error")
