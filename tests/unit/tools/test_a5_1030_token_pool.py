# SPDX-License-Identifier: Apache-2.0
"""Check tokenizer provenance and pool validation without model inference."""

import argparse
import json
from pathlib import Path

import httpx
import pytest

from tools.validation import a5_1030_token_pool as pool


@pytest.fixture
def tokenizer(tmp_path, monkeypatch):
    args = argparse.Namespace(
        attention_url="http://attention/",
        model="dsv4-afd",
        output=str(tmp_path / "diverse-pool"),
    )
    calls = []
    client_class = httpx.Client

    def configure(tokens, status=200):
        def handler(request):
            calls.append(request)
            return httpx.Response(status, json={"tokens": tokens, "count": len(tokens)})

        def client_factory(**kwargs):
            assert kwargs["trust_env"] is False
            return client_class(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(pool.httpx, "Client", client_factory)

    return args, calls, configure


def test_pool_preserves_attention_tokenizer_ids_and_source_text(tokenizer):
    args, calls, configure = tokenizer
    tokens = list(range(2048))
    configure(tokens)
    metadata = pool.create_token_pool(args)
    assert len(calls) == 1
    assert str(calls[0].url) == "http://attention/tokenize"
    body = json.loads(calls[0].content)
    assert body == {
        "model": "dsv4-afd",
        "prompt": pool.DIVERSE_PROMPT,
        "add_special_tokens": False,
    }
    output = Path(args.output)
    assert json.loads((output / "tokenize-request.json").read_text()) == body
    assert json.loads((output / "token-pool.json").read_text())["tokens"] == tokens
    assert metadata["token_count"] == 2048
    assert metadata["long_input_tokens"] == 1024
    assert metadata["long_input_unique_token_count"] == 1024
    assert json.loads((output / "token-pool.metadata.json").read_text()) == metadata
    with pytest.raises(FileExistsError):
        pool.create_token_pool(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "tokens",
    [[10] * 1023, [True] * 1024, ["10"] * 1024, [-1] * 1024],
)
def test_invalid_tokenizer_result_is_saved_without_publishing_pool(tokenizer, tokens):
    args, _, configure = tokenizer
    configure(tokens)
    with pytest.raises(ValueError, match="at least 1024 nonnegative integer IDs"):
        pool.create_token_pool(args)
    output = Path(args.output)
    assert (output / "tokenize.response.body").exists()
    assert not (output / "token-pool.json").exists()


def test_tokenizer_http_error_cannot_publish_pool(tokenizer):
    args, _, configure = tokenizer
    configure([10] * 1024, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        pool.create_token_pool(args)
    output = Path(args.output)
    assert (output / "tokenize.response.body").exists()
    assert not (output / "token-pool.json").exists()
