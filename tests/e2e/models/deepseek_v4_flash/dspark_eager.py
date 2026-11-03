# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check an already running DSpark service; save every raw response.

Use the recipe's worker extensions and --evidence-dir for ownership/dispatch checks.
Omit it to run the same request cases against a native DSpark baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(base: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def metric_totals(metrics: str, name: str) -> dict[str, float]:
    result = {}
    for line in metrics.splitlines():
        if line.startswith(name + "{"):
            match = re.search(r'engine="([^"]+)"', line)
            assert match is not None, line
            result[match[1]] = float(line.rsplit(" ", 1)[1])
    return result


def worker_records(directory: Path) -> dict:
    return {
        path.name: json.loads(path.read_text())
        for path in directory.glob("rank-*.json")
    }


def assert_dispatch(before: dict, after: dict) -> None:
    assert len(after) == 4, after
    for name, state in after.items():
        assert state["draft_wire_layer_ids"] == [43, 44, 45], state
        assert state["owns_routed_experts"], state
        assert not state["owns_attention"] and not state["owns_shared_experts"], state
        if state["ffn_execution"] == "layered":
            assert state["executor_layer_ids"] == list(range(46)), state
        if state["ffn_mode"] == "full":
            assert state["graph_capture_count"] == 1, state
            assert state["graph_replay_count"] > before[name]["graph_replay_count"], (
                state
            )
        required_layers = (
            range(46) if state["ffn_execution"] == "layered" else (43, 44, 45)
        )
        for layer in required_layers:
            key = str(layer)
            old = before[name]["submitted_returns"].get(key, 0)
            assert state["submitted_returns"].get(key, 0) > old, (name, layer, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7100")
    parser.add_argument("--control-url", help="Decode API behind a PD proxy")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--attention-dp", type=int, default=1)
    parser.add_argument("--pd", action="store_true", help="Use single-prompt PD cases")
    args = parser.parse_args()
    control = args.control_url or args.url
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = {}
    if args.evidence_dir is not None:
        before = worker_records(args.evidence_dir)
        assert len(before) == 4, before
        save(output / "ffn-before.json", before)
        attention = post(
            control,
            "/collective_rpc",
            {"method": "afd_dspark_snapshot", "timeout": 30},
        )
        save(output / "attention.json", attention)
        for state in attention["results"]:
            assert not state["owns_routed_experts"], state
            assert state["owns_attention"] and state["owns_shared_experts"], state

    tokens = post(
        control,
        "/tokenize",
        {
            "model": args.model,
            "prompt": "The quick brown fox jumps over the lazy dog. " * 80,
        },
    )["tokens"]
    assert len(tokens) > 513
    cases = [
        ("short", "The capital of France is"),
        ("swa-127", tokens[:127]),
        ("swa-129", tokens[:129]),
        ("chunk-513", tokens[:513]),
        ("batch-two", [tokens[:31], tokens[5:70]]),
    ]
    if args.pd:
        # The native example PD proxy accepts one prompt per handoff. Keep
        # batched-prompt coverage in the colocated test and test concurrent
        # independent PD requests below.
        cases = [("one-token", tokens[:1]), *cases[:-1]]
        cases.insert(3, ("swa-128", tokens[:128]))
    summaries = []

    def complete(name: str, prompt, **overrides):
        request = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
            "seed": 1024,
            "ignore_eos": True,
            "return_token_ids": True,
            **overrides,
        }
        result = post(args.url, "/v1/completions", request)
        save(output / (name + ".json"), result)
        expected = 2 if name == "batch-two" else 1
        assert len(result["choices"]) == expected, result
        if request["ignore_eos"]:
            assert all(
                len(choice["token_ids"]) == 32 for choice in result["choices"]
            ), result
        print(json.dumps({"case": name, "usage": result["usage"]}), flush=True)
        return result

    for name, prompt in cases:
        result = complete(name, prompt)
        summaries.append(
            {
                "name": name,
                "token_ids": [choice["token_ids"] for choice in result["choices"]],
            }
        )
    save(output / "summary.json", summaries)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(complete, "concurrent-short", tokens[:9]),
            pool.submit(complete, "concurrent-long", tokens[:513]),
        ]
        for future in futures:
            future.result()

    short_prompt = next(prompt for name, prompt in cases if name == "short")
    short_tokens = next(
        item["token_ids"][0] for item in summaries if item["name"] == "short"
    )
    early = complete("stop-early", short_prompt, ignore_eos=False, stop=["."])
    assert early["choices"][0]["finish_reason"] == "stop", early
    assert len(early["choices"][0]["token_ids"]) < 32, early

    stream = urllib.request.Request(
        args.url + "/v1/completions",
        data=json.dumps(
            {
                "model": args.model,
                "prompt": tokens[:31],
                "max_tokens": 256,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(stream, timeout=180) as response:
        for line in response:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                (output / "aborted-first-event.txt").write_bytes(line)
                break
        else:
            raise AssertionError("Stream produced no event before cancellation")
    repeat = complete("short-after-stop-and-abort", short_prompt)
    assert repeat["choices"][0]["token_ids"] == short_tokens, repeat

    with urllib.request.urlopen(control + "/metrics", timeout=15) as response:
        metrics = response.read().decode()
    (output / "metrics.txt").write_text(metrics)
    drafts = metric_totals(metrics, "vllm:spec_decode_num_drafts_total")
    accepted = metric_totals(metrics, "vllm:spec_decode_num_accepted_tokens_total")
    assert len(drafts) == args.attention_dp and all(
        value > 0 for value in drafts.values()
    ), drafts
    assert len(accepted) == args.attention_dp and all(
        value > 0 for value in accepted.values()
    ), accepted
    if args.evidence_dir is not None:
        # FFN EngineCore has no HTTP API. Its existing health RPC periodically
        # publishes host-only snapshots; allow one bounded health interval.
        deadline = time.monotonic() + 30
        while True:
            after = worker_records(args.evidence_dir)
            try:
                assert_dispatch(before, after)
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        save(output / "ffn-after.json", after)
    save(
        output / "result.json",
        {"passed": True, "draft_rounds": drafts, "accepted_tokens": accepted},
    )
    print("DSpark eager functional checks passed", flush=True)


if __name__ == "__main__":
    main()
