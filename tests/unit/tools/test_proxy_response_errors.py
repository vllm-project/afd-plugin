# SPDX-License-Identifier: Apache-2.0
"""A failed decoder must yield an observable error and release PD resources."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from tools import proxy_server as proxy


def test_backend_requests_and_readiness_bypass_environment_proxy(monkeypatch):
    """An external gateway must not intercept either PD hop or node readiness."""

    async def exercise():
        backend_paths = []
        gateway_requests = []

        async def respond(reader, writer, status, payload, requests):
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                requests.append(headers.split(b"\r\n", 1)[0])
                for line in headers.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        await reader.readexactly(int(line.split(b":", 1)[1]))
                body = json.dumps(payload).encode()
                writer.write(
                    f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async def backend(reader, writer):
            await respond(reader, writer, "200 OK", {"ok": True}, backend_paths)

        async def gateway(reader, writer):
            await respond(
                reader,
                writer,
                "504 Gateway Timeout",
                {"error": "gateway"},
                gateway_requests,
            )

        backend_server = await asyncio.start_server(backend, "127.0.0.1", 0)
        gateway_server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        async with backend_server, gateway_server:
            backend_port = backend_server.sockets[0].getsockname()[1]
            gateway_port = gateway_server.sockets[0].getsockname()[1]
            gateway_url = f"http://127.0.0.1:{gateway_port}"
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                monkeypatch.setenv(name, gateway_url)
                monkeypatch.setenv(name.lower(), gateway_url)
            monkeypatch.setenv("NO_PROXY", "")
            monkeypatch.setenv("no_proxy", "")

            # Demonstrate the same failure with the old HTTPX default.
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{backend_port}/health")
                assert response.status_code == 504

            runtime = proxy.WorkerRuntime(None)
            key = proxy.server_key("127.0.0.1", backend_port)
            try:
                for role in proxy.ServerRole:
                    await runtime._sync_clients(
                        role, {key: ("127.0.0.1", backend_port)}
                    )
                    client = await runtime.get_client(role, key)
                    response = await client.post(
                        "/chat/completions", json={"role": role.value}
                    )
                    assert response.status_code == 200
                assert await proxy.NodeListener.check_instance_status(
                    "127.0.0.1", backend_port
                )
            finally:
                await runtime.close()
            assert len(gateway_requests) == 1
            assert backend_paths == [
                b"POST /v1/chat/completions HTTP/1.1",
                b"POST /v1/chat/completions HTTP/1.1",
                b"GET /v1/models HTTP/1.1",
            ]

    asyncio.run(exercise())


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
