# SPDX-License-Identifier: Apache-2.0
"""A failed decoder must yield an observable error and release PD resources."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from tools import proxy_server as proxy


@pytest.mark.parametrize(
    "stream,backend_status", [(False, 500), (False, 200), (True, 500)]
)
def test_decoder_error_is_not_an_empty_http_200(monkeypatch, stream, backend_status):
    async def exercise():
        releases = []
        info = proxy.InstanceInfo(
            "request-eos", "prefill", 1, "decode", 1, "decode", 8910
        )
        payload = {
            "choices": [{"text": "Hello", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

        async def backend(request):
            if backend_status != 200:
                return httpx.Response(backend_status, json={"error": "KV load failed"})
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(
            base_url="http://decode", transport=httpx.MockTransport(backend)
        ) as client:

            async def get_client(*args):
                return client

            async def schedule(*args, **kwargs):
                releases.append((args, kwargs))

            async def assign(*args, **kwargs):
                return info

            body = {
                "model": "dsv4-afd",
                "prompt": "hello",
                "max_tokens": 32,
                "stream": stream,
            }

            async def request_json():
                return body

            async def request_body():
                return json.dumps(body).encode()

            monkeypatch.setattr(
                proxy,
                "get_runtime",
                lambda: SimpleNamespace(get_client=get_client, schedule=schedule),
            )
            monkeypatch.setattr(
                proxy,
                "get_global_args",
                lambda: SimpleNamespace(max_retries=1, retry_delay=0),
            )
            monkeypatch.setattr(proxy, "assign_instances", assign)
            response = await proxy.handle_completions_impl(
                "/completions", SimpleNamespace(json=request_json, body=request_body)
            )
            if stream:
                response_body = b"".join(
                    [chunk async for chunk in response.body_iterator]
                )
                assert b'"type": "decoder_error"' in response_body
                assert b"request-eos" in response_body
            elif backend_status != 200:
                assert response.status_code == 502
                assert json.loads(response.body)["error"]["type"] == "decoder_error"
            else:
                assert response.status_code == 200
                assert json.loads(response.body) == payload
            # Both successful and failed paths release the request exactly once.
            assert sum(args[0] == "finish_request" for args, _ in releases) == 1
            if backend_status != 200:
                assert releases == [
                    (("finish_request", "prefill", 1, "decode", 1, True), {})
                ]

    asyncio.run(exercise())
