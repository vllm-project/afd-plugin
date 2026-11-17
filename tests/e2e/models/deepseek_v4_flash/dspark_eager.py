# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check an already running DSpark service; save every raw response.

Use the recipe's worker extensions and --evidence-dir for ownership/dispatch checks.
Add --verify-graph-numerics for complete per-rank graph/eager comparisons.
Omit it to run the same request cases against a native DSpark baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import urllib.request
from pathlib import Path

REQUEST_TIMEOUT_S = 180


def post(base: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
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


def assert_dispatch(
    before: dict,
    after: dict,
    ffn_ranks: int = 4,
    *,
    require_layer_returns=True,
    require_graph=False,
) -> None:
    assert len(after) == ffn_ranks and set(after) == set(before), after
    for name, state in after.items():
        assert state["draft_wire_layer_ids"] == [43, 44, 45], state
        assert state["owns_routed_experts"], state
        assert not state["owns_attention"] and not state["owns_shared_experts"], state
        assert state["ffn_execution"] == "layered", state
        assert state["executor_layer_ids"] == list(range(46)), state
        if require_graph:
            assert state["ffn_mode"] == "full", state
        if state["ffn_mode"] == "full":
            assert state["graph_capture_count"] == 1, state
            assert state["graph_replay_count"] > before[name]["graph_replay_count"], (
                state
            )
        if require_layer_returns:
            for layer in range(46):
                key = str(layer)
                old = before[name]["submitted_returns"].get(key, 0)
                assert state["submitted_returns"].get(key, 0) > old, (
                    name,
                    layer,
                    state,
                )


def counter_delta(before: str, after: str, name: str, ranks: int, label="") -> dict:
    def select(metrics):
        return metric_totals(
            "\n".join(x for x in metrics.splitlines() if label in x), name
        )

    previous, current = select(before), select(after)
    delta = {rank: value - previous.get(rank, 0) for rank, value in current.items()}
    assert len(delta) == ranks and all(value > 0 for value in delta.values()), delta
    return delta


def assert_attention_graph(records: list, ranks: int) -> None:
    assert len(records) == ranks and len({x["rank"] for x in records}) == ranks, records
    for record in records:
        for kind in ("target", "draft"):
            assert any(
                event["tokens"] > 0 and sum(event["real"]) > 0
                for event in record["events"][kind]
            ), (record["rank"], kind, record)


def assert_attention_ownership(records: list, ranks: int) -> None:
    assert len(records) == ranks and len({x["rank"] for x in records}) == ranks, records
    assert_native_source(records)
    for state in records:
        assert not state["owns_routed_experts"], state
        assert state["owns_attention"] and state["owns_shared_experts"], state


def assert_native_source(states: list[dict]) -> None:
    source = Path(__file__).resolve().parents[4] / "afd_plugin"
    for state in states:
        paths = [Path(value) for value in state["native_maps"]]
        required = ("_C_ascend", "libcust_opapi.so")
        for library in required:
            matching = [path for path in paths if library in path.name]
            assert matching and all(path.is_relative_to(source) for path in matching), (
                library,
                state,
            )


def assert_graph_numerics(directory: Path, workers: dict) -> dict:
    proofs = worker_records(directory / "graph-proof")
    assert set(proofs) == set(workers), (
        "graph numerical rank coverage",
        sorted(workers),
        sorted(proofs),
    )
    for name, state in workers.items():
        proof = proofs[name]
        assert proof["rank"] == state["rank"], (name, proof)
        assert state["ffn_mode"] == proof["mode"] == "full", (name, proof)
        assert proof["capture_count"] == 1 and proof["replay_count"] > 0, (name, proof)
        assert proof["submission_unit"] == 1, (name, proof)
        expected = set(state["executor_layer_ids"])
        draft = set(state["draft_wire_layer_ids"])
        assert draft and draft < expected, (name, "target/draft layer table", state)
        records = proof["records"]
        layers = [record["layer"] for record in records]
        assert proof["complete"] is True and set(layers) == expected, (
            name,
            "graph numerical layer coverage",
            sorted(expected - set(layers)),
            layers,
        )
        assert len(layers) == len(expected), (name, "duplicate graph numerical layers")
        for record in records:
            assert record["rows"] > 0, (name, "empty numerical comparison", record)
            if record["bitwise_equal"] is not True or record["max_abs_error"] != 0:
                raise ValueError(f"{name}: graph/eager numerical mismatch: {record}")
    return proofs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7100")
    parser.add_argument("--control-url", help="Decode API behind a PD proxy")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--attention-dp", type=int, default=1)
    parser.add_argument("--ffn-ranks", type=int, default=4)
    parser.add_argument(
        "--verify-graph-numerics",
        action="store_true",
        help="Require complete graph/eager comparisons on every FFN rank",
    )
    parser.add_argument("--pd", action="store_true", help="Use single-prompt PD cases")
    parser.add_argument("--no-speculation", action="store_true")
    parser.add_argument("--strict-deltas", action="store_true")
    parser.add_argument("--observe-graph", action="store_true")
    args = parser.parse_args(argv)
    if args.ffn_ranks < 1:
        parser.error("--ffn-ranks must be positive")
    if args.verify_graph_numerics and args.evidence_dir is None:
        parser.error("--verify-graph-numerics requires --evidence-dir")
    if args.observe_graph and (args.evidence_dir is None or not args.strict_deltas):
        parser.error("--observe-graph requires --evidence-dir and --strict-deltas")
    run(args)


def run(args: argparse.Namespace) -> None:
    control = args.control_url or args.url
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    save(output / "result.json", {"passed": False})
    verify_graph_numerics = args.verify_graph_numerics or (
        args.evidence_dir is not None
        and (args.evidence_dir / "graph-proof" / "enable").is_file()
    )
    before = {}
    if args.evidence_dir is not None:
        before = worker_records(args.evidence_dir)
        if args.strict_deltas and len(before) != args.ffn_ranks:
            deadline = time.monotonic() + 30
            while len(before) != args.ffn_ranks and time.monotonic() < deadline:
                time.sleep(0.5)
                before = worker_records(args.evidence_dir)
        assert len(before) == args.ffn_ranks, before
        save(output / "ffn-before.json", before)
        if not args.strict_deltas:
            attention = post(
                control,
                "/collective_rpc",
                {"method": "afd_dspark_snapshot", "timeout": 30},
            )
            save(output / "attention.json", attention)
            for state in attention["results"]:
                assert not state["owns_routed_experts"], state
                assert state["owns_attention"] and state["owns_shared_experts"], state

    with urllib.request.urlopen(control + "/metrics", timeout=15) as response:
        initial_metrics = response.read().decode()
    (output / "metrics-before.txt").write_text(initial_metrics)
    marker = (
        args.evidence_dir / "requests-active"
        if args.strict_deltas and args.evidence_dir is not None
        else None
    )
    try:
        if marker is not None:
            marker.touch()
        result = exercise(
            args, control, output, before, verify_graph_numerics, initial_metrics
        )
    finally:
        if marker is not None:
            marker.unlink(missing_ok=True)
    if marker is not None:
        attention_ranks = list(
            worker_records(args.evidence_dir / "attention-ranks").values()
        )
        save(output / "attention.json", {"results": attention_ranks})
        assert_attention_ownership(attention_ranks, args.attention_dp)
    if args.observe_graph:
        graph = list(worker_records(args.evidence_dir / "attention-graph").values())
        save(output / "attention-graph.json", {"results": graph})
        assert_attention_graph(graph, args.attention_dp)
    save(output / "result.json", {"passed": True, **result})
    print("DSpark functional checks passed", flush=True)


def exercise(args, control, output, before, verify_graph_numerics, initial_metrics):
    graph_proofs = {}
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
        ("one-token", tokens[:1]),
        ("swa-127", tokens[:127]),
        ("swa-128", tokens[:128]),
        ("swa-129", tokens[:129]),
        ("chunk-513", tokens[:513]),
        ("batch-two", [tokens[:31], tokens[5:70]]),
    ]
    if args.pd:
        # The native example PD proxy accepts one prompt per handoff. Keep
        # batched-prompt coverage in the colocated test and test concurrent
        # independent PD requests below.
        cases = cases[:-1]
    summaries = []

    def completion_request(prompt, **overrides):
        return {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
            "seed": 1024,
            "ignore_eos": True,
            "return_token_ids": True,
            **overrides,
        }

    def record_completion(name, request, result):
        save(output / (name + ".json"), result)
        expected = 2 if name == "batch-two" else 1
        assert len(result["choices"]) == expected, result
        if request["ignore_eos"]:
            assert all(
                len(choice["token_ids"]) == 32 for choice in result["choices"]
            ), result
        print(json.dumps({"case": name, "usage": result["usage"]}), flush=True)
        return result

    def complete(name: str, prompt, **overrides):
        request = completion_request(prompt, **overrides)
        result = post(args.url, "/v1/completions", request)
        return record_completion(name, request, result)

    for name, prompt in cases:
        result = complete(name, prompt)
        summaries.append(
            {
                "name": name,
                "token_ids": [choice["token_ids"] for choice in result["choices"]],
            }
        )
    save(output / "summary.json", summaries)

    async def concurrent_completions():
        # Import only during execution: pytest collection needs no HTTP client.
        import httpx

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:

            async def send(name, prompt):
                request = completion_request(prompt)
                response = await client.post(args.url + "/v1/completions", json=request)
                response.raise_for_status()
                return record_completion(name, request, response.json())

            await asyncio.gather(
                send("concurrent-short", tokens[:9]),
                send("concurrent-long", tokens[:513]),
                send("concurrent-medium", tokens[:128]),
                send("concurrent-other", tokens[1:130]),
            )

    asyncio.run(concurrent_completions())

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
    with urllib.request.urlopen(stream, timeout=REQUEST_TIMEOUT_S) as response:
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
    drafts, accepted, external = {}, {}, {}
    if not args.no_speculation:
        if args.strict_deltas:
            drafts = counter_delta(
                initial_metrics,
                metrics,
                "vllm:spec_decode_num_drafts_total",
                args.attention_dp,
            )
            accepted = counter_delta(
                initial_metrics,
                metrics,
                "vllm:spec_decode_num_accepted_tokens_total",
                args.attention_dp,
            )
        else:
            drafts = metric_totals(metrics, "vllm:spec_decode_num_drafts_total")
            accepted = metric_totals(
                metrics, "vllm:spec_decode_num_accepted_tokens_total"
            )
            assert len(drafts) == args.attention_dp and all(
                value > 0 for value in drafts.values()
            ), drafts
            assert len(accepted) == args.attention_dp and all(
                value > 0 for value in accepted.values()
            ), accepted
    if args.pd and args.strict_deltas:
        external = counter_delta(
            initial_metrics,
            metrics,
            "vllm:prompt_tokens_by_source_total",
            args.attention_dp,
            'source="external_kv_transfer"',
        )
    if args.evidence_dir is not None:
        # FFN EngineCore has no HTTP API. Its existing health RPC periodically
        # publishes host-only snapshots; allow one bounded health interval.
        deadline = time.monotonic() + 30
        while True:
            after = worker_records(args.evidence_dir)
            try:
                assert_dispatch(
                    before,
                    after,
                    args.ffn_ranks,
                    require_graph=args.observe_graph,
                    require_layer_returns=not args.observe_graph
                    or verify_graph_numerics,
                )
                if verify_graph_numerics:
                    graph_proofs = assert_graph_numerics(args.evidence_dir, after)
                if args.strict_deltas:
                    assert_native_source(list(after.values()))
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        save(output / "ffn-after.json", after)
    if verify_graph_numerics:
        save(output / "graph-proof.json", graph_proofs)
    return {
        "draft_rounds": drafts,
        "accepted_tokens": accepted,
        "external_kv_tokens": external,
        "graph_numerics_verified": verify_graph_numerics,
        "ffn_graph_submission": "one-per-fence"
        if verify_graph_numerics
        else "ordinary",
    }


if __name__ == "__main__":
    main()
