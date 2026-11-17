# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-only checker regressions using synthetic HTTP responses and proof files."""

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.e2e.models.deepseek_v4_flash import dspark_eager as checker


def change_proof(directory, fault):
    path = directory / "graph-proof" / "rank-2.json"
    proof = json.loads(path.read_text())
    if fault == "missing-rank":
        path.unlink()
        return
    if fault == "missing-target":
        proof["records"] = proof["records"][1:]
    elif fault == "missing-draft":
        proof["records"] = proof["records"][:-1]
    elif fault == "duplicate-layer":
        proof["records"].append(proof["records"][0])
    elif fault in ("bitwise_equal", "rows", "max_abs_error"):
        proof["records"][0][fault] = {
            "bitwise_equal": False,
            "rows": 0,
            "max_abs_error": 1.0,
        }[fault]
    elif fault == "truthy-equality":
        proof["records"][0]["bitwise_equal"] = "true"
    else:
        proof[fault] = {
            "rank": 99,
            "mode": "eager",
            "complete": False,
            "capture_count": 0,
            "replay_count": 0,
            "submission_unit": 46,
        }[fault]
    checker.save(path, proof)


@pytest.fixture
def run_checker(monkeypatch, tmp_path):
    evidence = tmp_path / "evidence"
    graph = evidence / "graph-proof"
    graph.mkdir(parents=True)
    output = tmp_path / "results"
    output.mkdir()
    # A failed rerun must replace a previous success verdict.
    checker.save(output / "result.json", {"passed": True})

    def run(*, fault=None, ffn_ranks=4, explicit=True, marker=True):
        before, after = {}, {}
        for rank in range(2, 2 + ffn_ranks):
            name = f"rank-{rank}.json"
            state = {
                "rank": rank,
                "draft_wire_layer_ids": [43, 44, 45],
                "owns_routed_experts": True,
                "owns_attention": False,
                "owns_shared_experts": False,
                "ffn_execution": "layered",
                "executor_layer_ids": list(range(46)),
                "ffn_mode": "full",
                "graph_capture_count": 1,
            }
            before[name] = {
                **state,
                "graph_replay_count": 0,
                "submitted_returns": {},
            }
            after[name] = {
                **state,
                "graph_replay_count": 2,
                "submitted_returns": {str(layer): 1 for layer in range(46)},
            }
            checker.save(evidence / name, before[name])
            checker.save(
                graph / name,
                {
                    "rank": rank,
                    "mode": "full",
                    "complete": True,
                    "capture_count": 1,
                    "replay_count": 2,
                    "submission_unit": 1,
                    "records": [
                        {
                            "layer": layer,
                            "rows": 1,
                            "bitwise_equal": True,
                            "max_abs_error": 0.0,
                        }
                        for layer in range(46)
                    ],
                },
            )
        if marker:
            (graph / "enable").touch()
        if fault:
            change_proof(evidence, fault)

        def post(_base, path, payload):
            if path == "/collective_rpc":
                for name, state in after.items():
                    checker.save(evidence / name, state)
                return {
                    "results": [
                        {
                            "owns_routed_experts": False,
                            "owns_attention": True,
                            "owns_shared_experts": True,
                        }
                    ]
                }
            if path == "/tokenize":
                return {"tokens": list(range(600))}
            assert path == "/v1/completions"
            prompt = payload["prompt"]
            count = 2 if isinstance(prompt, list) and isinstance(prompt[0], list) else 1
            stopped = not payload["ignore_eos"]
            return {
                "choices": [
                    {
                        "token_ids": [1] * (1 if stopped else payload["max_tokens"]),
                        "finish_reason": "stop" if stopped else "length",
                    }
                    for _ in range(count)
                ],
                "usage": {},
            }

        def urlopen(request, **_kwargs):
            if isinstance(request, str):
                return io.BytesIO(
                    b'vllm:spec_decode_num_drafts_total{engine="0"} 1\n'
                    b'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 1\n'
                )
            return io.BytesIO(b'data: {"choices": []}\n')

        monkeypatch.setattr(checker, "post", post)
        monkeypatch.setattr(checker.urllib.request, "urlopen", urlopen)
        clock = iter((0, 60))
        monkeypatch.setattr(
            checker, "time", SimpleNamespace(monotonic=lambda: next(clock))
        )
        command = [
            "dspark_eager.py",
            "--output-dir",
            str(output),
            "--evidence-dir",
            str(evidence),
            "--ffn-ranks",
            str(ffn_ranks),
        ]
        if explicit:
            command.append("--verify-graph-numerics")
        monkeypatch.setattr(sys, "argv", command)
        checker.main()
        return output

    return run, output


@pytest.mark.parametrize("ffn_ranks", [4, 8])
def test_checker_requires_complete_numerics_for_every_rank(run_checker, ffn_ranks):
    run, output = run_checker
    run(ffn_ranks=ffn_ranks)
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] and result["graph_numerics_verified"]
    proof = json.loads((output / "graph-proof.json").read_text())
    assert len(proof) == ffn_ranks
    assert all(len(rank["records"]) == 46 for rank in proof.values())


@pytest.mark.parametrize(
    "fault",
    [
        "missing-rank",
        "missing-target",
        "missing-draft",
        "duplicate-layer",
        "complete",
        "bitwise_equal",
        "max_abs_error",
        "truthy-equality",
        "rows",
        "rank",
        "mode",
        "capture_count",
        "replay_count",
        "submission_unit",
    ],
)
def test_checker_rejects_bad_graph_proof(run_checker, fault):
    run, output = run_checker
    with pytest.raises((AssertionError, ValueError)):
        run(fault=fault)
    assert json.loads((output / "result.json").read_text())["passed"] is False
    assert not (output / "graph-proof.json").exists()


def test_enable_marker_enforces_numerical_checks_without_flag(run_checker):
    run, output = run_checker
    with pytest.raises(ValueError, match="graph/eager numerical mismatch"):
        run(fault="bitwise_equal", explicit=False)
    assert json.loads((output / "result.json").read_text())["passed"] is False


def test_ordinary_evidence_does_not_claim_numerical_validation(run_checker):
    run, output = run_checker
    run(fault="missing-rank", explicit=False, marker=False)
    result = json.loads((output / "result.json").read_text())
    assert result["passed"] and result["graph_numerics_verified"] is False
    assert not (output / "graph-proof.json").exists()


def test_numerical_cli_rejects_missing_evidence_directory(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            checker.__file__,
            "--output-dir",
            str(tmp_path),
            "--verify-graph-numerics",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "requires --evidence-dir" in process.stderr
