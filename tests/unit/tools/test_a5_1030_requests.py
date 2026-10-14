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
            if fault == "read-timeout":
                raise httpx.ReadTimeout("", request=request)
            if fault == "empty-body":
                return httpx.Response(
                    200, content=b"", headers={"content-type": "application/json"}
                )
            if fault == "html-body":
                return httpx.Response(200, text="<html>gateway error</html>")
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
            assert body["return_token_ids"] is True
            name = request.headers["X-Request-Id"]
            if fault == "timeout-c8" and name in (
                "test-c8-short-2",
                "test-c8-short-4",
                "test-c8-short-6",
            ):
                await asyncio.sleep(0.01)
                raise httpx.ReadTimeout("", request=request)
            empty_text = fault == "empty-c32" and name in (
                "test-c32-short-16",
                "test-c32-short-29",
            )
            if fault == "empty-c32" and name in (
                "test-c32-short-5",
                "test-c32-short-22",
            ):
                # These siblings finish after the empty response. They must
                # retain their results rather than be cancelled by gather.
                await asyncio.sleep(0.01)
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
                                else {"text": "" if empty_text else "hello"}
                            ),
                            "token_ids": [100000 if empty_text else 10] * count,
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


@pytest.mark.parametrize("suite,concurrency", [("c8-short", 8), ("c32-short", 32)])
def test_short_suite_preserves_concurrency_without_repeating_full(
    request_gate, suite, concurrency
):
    args, exercise = request_gate
    args.suite = suite
    args.token_pool = None
    result, peak, closed, calls = exercise()
    assert result["request_checks_passed"] is True
    assert result["full_request_suite_passed"] is False
    assert result["request_count"] == concurrency + 1
    assert result["f0_passed"] is None
    assert peak == concurrency
    assert not closed
    assert {r["name"] for r in result["records"]} == {
        "eos",
        *(f"{suite}-{index}" for index in range(concurrency)),
    }
    batch = json.loads(Path(args.output, f"batch-{suite}.summary.json").read_text())
    assert batch["passed_count"] == concurrency
    assert batch["failed_count"] == 0
    assert ("attention", "/metrics") in calls


def test_empty_text_keeps_token_ids_and_all_concurrent_results(request_gate):
    args, exercise = request_gate
    args.suite = "c32-short"
    args.token_pool = None
    with pytest.raises(ValueError, match="2/32 requests failed"):
        exercise("empty-c32")
    batch = json.loads(Path(args.output, "batch-c32-short.summary.json").read_text())
    assert batch["request_count"] == 32
    assert batch["passed_count"] == 30
    assert batch["failed_count"] == 2
    assert batch["request_checks_passed"] is False
    assert {f["name"] for f in batch["failures"]} == {"c32-short-16", "c32-short-29"}
    for index in range(32):
        assert Path(args.output, f"c32-short-{index}.json").exists()
    for index in (16, 29):
        error = json.loads(
            Path(args.output, f"c32-short-{index}.validation.error.json").read_text()
        )
        assert error["empty_text"] is True
        assert error["http_status"] == 200
        assert error["completion_tokens"] == 64
        assert error["token_ids"] == [100000] * 64
    assert not list(Path(args.output).glob("*.transport.error.json"))


def test_full_prefix_preserves_long_to_short_sequence_and_stops_at_c8(request_gate):
    args, exercise = request_gate
    args.suite = "full-to-c8-short"
    result, peak, closed, calls = exercise()
    assert result["request_checks_passed"] is True
    assert result["full_request_suite_passed"] is False
    assert result["request_count"] == 11
    assert peak == 8
    assert not closed
    assert [r["name"] for r in result["records"]] == [
        "eos",
        "c1-short-0",
        "c1-long-0",
        *(f"c8-short-{index}" for index in range(8)),
    ]
    long_body = json.loads(Path(args.output, "c1-long-0.request.json").read_text())
    assert long_body["prompt"] == [10] * 1024
    short_body = json.loads(Path(args.output, "c1-short-0.request.json").read_text())
    for index in range(8):
        body = json.loads(
            Path(args.output, f"c8-short-{index}.request.json").read_text()
        )
        assert body == short_body
    assert long_body["max_tokens"] == short_body["max_tokens"] == 64
    assert long_body["min_tokens"] == short_body["min_tokens"] == 64
    assert long_body["ignore_eos"] is short_body["ignore_eos"] is True
    assert ("proxy", "/healthcheck") in calls
    assert ("attention", "/metrics") in calls


def test_full_prefix_keeps_completed_siblings_when_c8_requests_time_out(request_gate):
    args, exercise = request_gate
    args.suite = "full-to-c8-short"
    with pytest.raises(ValueError, match="3/8 requests failed"):
        exercise("timeout-c8")
    for name in ("c1-short", "c1-long"):
        batch = json.loads(Path(args.output, f"batch-{name}.summary.json").read_text())
        assert batch["passed_count"] == 1
    batch = json.loads(Path(args.output, "batch-c8-short.summary.json").read_text())
    assert batch["passed_count"] == 5
    assert batch["failed_count"] == 3
    assert batch["request_checks_passed"] is False
    for index in (2, 4, 6):
        error = json.loads(
            Path(args.output, f"c8-short-{index}.transport.error.json").read_text()
        )
        assert error["exception_type"] == "ReadTimeout"
    for index in (0, 1, 3, 5, 7):
        assert Path(args.output, f"c8-short-{index}.json").exists()


def test_full_prefix_requires_token_pool_on_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "a5_1030_requests.py",
            "--suite",
            "full-to-c8-short",
            "--base-url",
            "http://proxy",
            "--attention-url",
            "http://attention",
            "--output",
            str(tmp_path / "requests"),
            "--run-id",
            "test",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        validation.main()
    assert exc.value.code == 2
    assert (
        "--token-pool is required for --suite full-to-c8-short"
        in capsys.readouterr().err
    )
    assert not (tmp_path / "requests").exists()


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


@pytest.mark.parametrize(
    "suite",
    ["smoke", "eos", "tail", "full", "c8-short", "c32-short", "full-to-c8-short"],
)
def test_chat_length_limit_is_not_accepted_as_natural_eos(request_gate, suite):
    args, exercise = request_gate
    args.suite = suite
    with pytest.raises(ValueError, match="EOS was not observed with the chat template"):
        exercise("eos-length")
    # EOS runs first; an unsuitable case must not send any concurrency load.
    assert sorted(path.name for path in Path(args.output).iterdir()) == [
        "eos.json",
        "eos.request.json",
        "eos.response.body",
        "eos.response.http.json",
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
    args, exercise = request_gate
    with pytest.raises(httpx.HTTPStatusError):
        exercise("http-error")
    assert Path(args.output, "eos.response.body").read_text() == "backend failed"
    assert (
        json.loads(Path(args.output, "eos.response.http.json").read_text())[
            "status_code"
        ]
        == 500
    )


@pytest.mark.parametrize("fault", ["empty-body", "html-body"])
def test_invalid_json_records_original_http_response(request_gate, fault):
    args, exercise = request_gate
    with pytest.raises(ValueError, match="HTTP 200 returned invalid/empty JSON"):
        exercise(fault)
    assert Path(args.output, "eos.response.body").exists()
    assert (
        json.loads(Path(args.output, "eos.response.http.json").read_text())[
            "status_code"
        ]
        == 200
    )


def test_timeout_records_request_name_and_exception_type(request_gate):
    args, exercise = request_gate
    with pytest.raises(ValueError, match="eos: ReadTimeout after"):
        exercise("read-timeout")
    failure = json.loads(Path(args.output, "eos.transport.error.json").read_text())
    assert failure["exception_type"] == "ReadTimeout"
    assert failure["name"] == "eos"
    assert failure["error"] == "ReadTimeout('')"
    assert failure["request_id"] == "test-eos"
    assert failure["end_timestamp_ns"] >= failure["start_timestamp_ns"]
