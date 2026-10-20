# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise shell entrypoints with a local CLI double; no model or device runs."""

import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(
    os.environ.get(
        "AFD_DSV4_SKILL_ROOT",
        str(
            Path(__file__).resolve().parents[2]
            / ".agents/skills/deploy-dsv4-flash-async"
        ),
    )
)
CAPTURE_TIMEOUT_SECONDS = 5
FROZEN_INPUT_TOKENS = 15_803_063
FROZEN_MAX_INPUT_TOKENS = 63_778
FROZEN_REQUESTS = 1536
MOCK_CLI = r"""
import json
import os
import sys
from pathlib import Path

def main():
    args = sys.argv[1:]
    capture = {"args": args, "env": dict(os.environ)}
    Path(os.environ["CAPTURE_DIR"], f"{os.getpid()}.json").write_text(
        json.dumps(capture)
    )
    os.write(int(os.environ["CAPTURE_NOTIFY_FD"]), b"1")

    if args[:2] == ["bench", "serve"]:
        def value(option):
            return args[args.index(option) + 1]

        # Synthetic detailed results exercise the result validator, not a model.
        start = args.index("--metadata") + 1
        end = next(
            (i for i in range(start, len(args)) if args[i].startswith("--")),
            len(args),
        )
        metadata = dict(item.split("=", 1) for item in args[start:end])
        count = 1536
        longest = 63778
        total = 15803063
        length, remainder = divmod(total - longest, count - 1)
        lengths = [longest] + [
            length + (index < remainder) for index in range(count - 1)
        ]
        result = {
            **metadata,
            # vLLM writes the parsed client setting after metadata. In the
            # uncapped case this overwrites the metadata string with JSON null.
            "max_concurrency": int(value("--max-concurrency"))
            if "--max-concurrency" in args
            else None,
            "completed": count,
            "failed": 0,
            "total_input_tokens": total,
            "total_output_tokens": count,
            "input_lens": lengths,
            "output_lens": [1] * count,
            "ttfts": [0.01] * count,
            "errors": [""] * count,
        }
        target = Path(value("--result-dir"), value("--result-filename"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result))


if __name__ == "__main__":
    main()

"""


def option(capture: dict, name: str) -> str:
    args = capture["args"]
    return args[args.index(name) + 1]


class DeploymentScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="afd-dsv4-shell-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.captures = self.root / "captures"
        self.captures.mkdir()
        cli = self.bin / "vllm"
        cli.write_text(f"#!{sys.executable}\n" + MOCK_CLI)
        cli.chmod(0o755)
        curl = self.bin / "curl"
        curl.write_text("#!/usr/bin/env bash\nexit 0\n")
        curl.chmod(0o755)
        dataset = self.root / "synthetic.jsonl"
        dataset.write_text('{"prompt":"test","output_tokens":1}\n')
        self.model = self.root / "checkpoint with spaces"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"num_nextn_predict_layers":1}\n')
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "PYTHON": sys.executable,
            "VLLM_CLI": str(cli),
            "REPO_ROOT": str(self.root / "repo"),
            "PYTHON_VLLM_PATH": str(self.root / "vllm"),
            "PYTHON_VLLM_ASCEND_PATH": str(self.root / "vllm-ascend"),
            "PYTHON_AFD_PLUGIN_PATH": str(self.root / "afd-plugin"),
            "PYTHONPATH": "/cann/python:/cann/tbe",
            "VLLM_PLUGINS": "afd",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "DSV4_MODEL": str(self.model),
            "P_NODE_IP": "192.0.2.1",
            "P_SECONDARY_NODE_IP": "192.0.2.2",
            "D_NODE_IP": "192.0.2.3",
            "NIC_NAME": "testnic",
            "PREFILL_TOPOLOGY": "afd_dp4tp2",
            "PREFILL_NODE_ID": "0",
            "PREFILL_ENABLE_KV_CONNECTOR": "0",
            "PREFILL_ENABLE_DBO": "1",
            "PREFILL_FFN_GRAPH_MODE": "EAGER",
            "AFD_ASYNC_CAM_LAYERED_GMM": "0",
            "MAX_MODEL_LEN": "65536",
            "LOG_DIR": str(self.root / "logs"),
            "PID_DIR": str(self.root / "pids"),
            "CAPTURE_DIR": str(self.captures),
            "DSV4_BENCH_RUN_ID": "runtime030-test",
            "DSV4_BENCH_RESULT_ROOT": str(self.root / "results"),
            "DSV4_BENCH_DATASET_PATH": str(dataset),
            "DSV4_BENCH_DATASET_SHA256": hashlib.sha256(
                dataset.read_bytes()
            ).hexdigest(),
            "DSV4_VLLM_REVISION": "vllm-test",
            "DSV4_ASCEND_REVISION": "ascend-test",
            "DSV4_AFD_REVISION": "afd-test",
        }

    def run_script(
        self, name: str, *args: str, expected_captures: int = 0, **overrides: str
    ) -> tuple[subprocess.CompletedProcess, list[dict]]:
        before = set(self.captures.iterdir())
        reader, writer = os.pipe()
        with (
            os.fdopen(reader, "rb", buffering=0) as notifications,
            os.fdopen(writer, "wb", buffering=0) as notification_writer,
        ):
            result = subprocess.run(
                ["bash", str(SKILL_ROOT / "scripts" / name), *args],
                env={
                    **self.env,
                    **overrides,
                    "CAPTURE_NOTIFY_FD": str(notification_writer.fileno()),
                },
                pass_fds=(notification_writer.fileno(),),
                text=True,
                capture_output=True,
                timeout=15,
            )
            remaining = expected_captures
            while remaining:
                ready, _, _ = select.select(
                    [notifications], [], [], CAPTURE_TIMEOUT_SECONDS
                )
                if not ready:
                    self.fail(f"Missing CLI capture: {result.stdout}\n{result.stderr}")
                remaining -= len(notifications.read(remaining))
        paths = sorted(set(self.captures.iterdir()) - before)
        return result, [json.loads(path.read_text()) for path in paths]

    def test_role_contract_and_layered_isolation(self) -> None:
        result, captures = self.run_script(
            "run_prefill.sh", expected_captures=2, AFD_ASYNC_CAM_LAYERED_GMM="1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        by_role = {
            json.loads(option(capture, "--additional-config"))["afd"]["role"]: capture
            for capture in captures
        }
        attention, ffn = by_role["attention"], by_role["ffn"]
        for capture in captures:
            config = json.loads(option(capture, "--additional-config"))
            self.assertEqual(capture["env"]["VLLM_USE_V2_MODEL_RUNNER"], "1")
            self.assertEqual(capture["env"]["VLLM_PLUGINS"], "ascend,afd")
            self.assertTrue(config["afd"]["async_dp"])
            self.assertFalse(config["enable_force_eplb"])
            self.assertNotIn("enable_force_load_balance", config)
            self.assertNotIn("enable_dsv4_shared_compressor_workspace", config)
            self.assertNotIn("async_moe_split", config["afd"]["connector_extra_config"])
            self.assertEqual(
                option(capture, "--attention_config.indexer_kv_dtype"), "int8"
            )
        self.assertIn("--enable-dbo", attention["args"])
        self.assertNotIn("--enable-dbo", ffn["args"])
        self.assertEqual(attention["env"]["AFD_ASYNC_CAM_LAYERED_GMM"], "0")
        self.assertEqual(ffn["env"]["AFD_ASYNC_CAM_LAYERED_GMM"], "1")
        self.assertEqual(ffn["env"]["VLLM_ASCEND_ENABLE_FLASHCOMM1"], "0")

    def test_attention_dbo_off_and_explicit_threshold(self) -> None:
        result, captures = self.run_script(
            "run_prefill_attention.sh",
            expected_captures=1,
            PREFILL_ENABLE_DBO="0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--enable-dbo", captures[0]["args"])
        result, captures = self.run_script(
            "run_prefill_attention.sh",
            expected_captures=1,
            PREFILL_DBO_PREFILL_TOKEN_THRESHOLD="1024",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(option(captures[0], "--dbo-prefill-token-threshold"), "1024")

    def test_ffn_full_selection_and_missing_prerequisites(self) -> None:
        for settings in (
            {"PREFILL_FFN_GRAPH_MODE": "FULL"},
            {
                "PREFILL_FFN_GRAPH_MODE": "FULL",
                "AFD_ASYNC_CAM_LAYERED_GMM": "1",
                "BATCH_SIZE_FACTOR": "",
            },
        ):
            with self.subTest(settings=settings):
                result, _ = self.run_script(
                    "run_prefill_ffn.sh",
                    PREFILL_FFN_GRAPH_MODE="FULL",
                    AFD_ASYNC_CAM_LAYERED_GMM=settings.get(
                        "AFD_ASYNC_CAM_LAYERED_GMM", "0"
                    ),
                    BATCH_SIZE_FACTOR=settings.get("BATCH_SIZE_FACTOR", ""),
                )
                self.assertNotEqual(result.returncode, 0)
        result, captures = self.run_script(
            "run_prefill_ffn.sh",
            expected_captures=1,
            PREFILL_FFN_GRAPH_MODE="FULL",
            AFD_ASYNC_CAM_LAYERED_GMM="1",
            BATCH_SIZE_FACTOR="0.0625",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--enforce-eager", captures[0]["args"])
        self.assertNotIn("--enable-dbo", captures[0]["args"])
        self.assertEqual(
            json.loads(option(captures[0], "--compilation-config")),
            {"cudagraph_mode": "FULL"},
        )

    def test_cross_node_attention_rank_ranges(self) -> None:
        for topology, global_dp, local_dp, start_rank in (
            ("afd_dp12tp2", 12, 4, 8),
            ("afd_dp10tp2", 10, 4, 6),
            ("afd_dp6tp4", 6, 2, 4),
            ("afd_dp3tp8", 3, 1, 2),
        ):
            with self.subTest(topology=topology):
                result, captures = self.run_script(
                    "run_prefill_attention.sh",
                    expected_captures=1,
                    PREFILL_TOPOLOGY=topology,
                    PREFILL_NODE_ID="1",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                capture = captures[0]
                self.assertEqual(
                    option(capture, "--data-parallel-size"), str(global_dp)
                )
                self.assertEqual(
                    option(capture, "--data-parallel-size-local"), str(local_dp)
                )
                self.assertEqual(
                    option(capture, "--data-parallel-start-rank"), str(start_rank)
                )
                self.assertIn("--headless", capture["args"])

    def test_native_ep_49152_and_plugin_isolation(self) -> None:
        result, captures = self.run_script(
            "run_prefill_full.sh",
            expected_captures=1,
            PREFILL_TOPOLOGY="ep16",
            PREFILL_MAX_NUM_BATCHED_TOKENS="49152",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        capture = captures[0]
        config = json.loads(option(capture, "--additional-config"))
        self.assertEqual(option(capture, "--max-num-batched-tokens"), "49152")
        self.assertNotIn("afd", capture["env"]["VLLM_PLUGINS"].split(","))
        self.assertNotIn(
            self.env["PYTHON_AFD_PLUGIN_PATH"],
            capture["env"]["PYTHONPATH"].split(":"),
        )
        self.assertIn("/cann/tbe", capture["env"]["PYTHONPATH"].split(":"))
        self.assertNotIn("afd", config)
        self.assertNotIn("enable_dsv4_shared_compressor_workspace", config)

    def test_pd_producer_consumer_layouts(self) -> None:
        result, captures = self.run_script(
            "run_prefill_attention.sh",
            expected_captures=1,
            PREFILL_ENABLE_KV_CONNECTOR="1",
            PREFILL_TOPOLOGY="afd_dp6tp4",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        producer = json.loads(option(captures[0], "--kv-transfer-config"))
        self.assertIn(
            "ascend_kv_connector", captures[0]["env"]["VLLM_PLUGINS"].split(",")
        )
        self.assertEqual(producer["kv_role"], "kv_producer")
        self.assertEqual(
            producer["kv_connector_extra_config"]["prefill"],
            {"dp_size": 6, "tp_size": 4},
        )
        result, captures = self.run_script(
            "run_decode.sh",
            expected_captures=1,
            PREFILL_ENABLE_KV_CONNECTOR="1",
            PREFILL_TOPOLOGY="afd_dp6tp4",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        consumer = json.loads(option(captures[0], "--kv-transfer-config"))
        self.assertEqual(consumer["kv_role"], "kv_consumer")
        self.assertEqual(
            consumer["kv_connector_extra_config"],
            producer["kv_connector_extra_config"],
        )
        self.assertNotIn("afd", captures[0]["env"]["VLLM_PLUGINS"].split(","))

    def test_pd_attention_includes_connector_plugin_with_overrides(self) -> None:
        for plugins in (
            "ascend,afd",
            "ascend,afd,custom_plugin",
            "ascend,afd,ascend_kv_connector",
        ):
            with self.subTest(plugins=plugins):
                result, captures = self.run_script(
                    "run_prefill.sh",
                    expected_captures=2,
                    PREFILL_ENABLE_KV_CONNECTOR="1",
                    AFD_VLLM_PLUGINS=plugins,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                by_role = {
                    json.loads(option(capture, "--additional-config"))["afd"][
                        "role"
                    ]: capture
                    for capture in captures
                }
                attention_plugins = by_role["attention"]["env"]["VLLM_PLUGINS"].split(
                    ","
                )
                self.assertEqual(attention_plugins.count("ascend_kv_connector"), 1)
                self.assertTrue(set(plugins.split(",")).issubset(attention_plugins))
                self.assertEqual(by_role["ffn"]["env"]["VLLM_PLUGINS"], plugins)

    def test_decode_selects_checkpoint_speculative_method(self) -> None:
        for metadata, expected in (
            ({"num_nextn_predict_layers": 1}, ("mtp", 1)),
            ({"num_nextn_predict_layers": 1, "dspark_block_size": 5}, ("dspark", 5)),
            ({}, None),
            ({"num_nextn_predict_layers": None}, None),
        ):
            with self.subTest(metadata=metadata):
                (self.model / "config.json").write_text(json.dumps(metadata))
                result, captures = self.run_script("run_decode.sh", expected_captures=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                if expected is None:
                    self.assertNotIn("--speculative-config", captures[0]["args"])
                else:
                    config = json.loads(option(captures[0], "--speculative-config"))
                    self.assertEqual(
                        (config["method"], config["num_speculative_tokens"]), expected
                    )
                    self.assertTrue(config["enforce_eager"])

    def test_decode_speculative_overrides(self) -> None:
        override = {
            "method": "dspark",
            "num_speculative_tokens": 7,
            "enforce_eager": True,
        }
        result, captures = self.run_script(
            "run_decode.sh",
            expected_captures=1,
            DECODE_SPECULATIVE_CONFIG=json.dumps(override),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(option(captures[0], "--speculative-config")), override
        )
        (self.model / "config.json").write_text('{"dspark_block_size":5}\n')
        override = {
            "method": "mtp",
            "num_speculative_tokens": 1,
            "model": "/draft/model",
        }
        result, captures = self.run_script(
            "run_decode.sh",
            expected_captures=1,
            DECODE_SPECULATIVE_CONFIG=json.dumps(override),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(option(captures[0], "--speculative-config")), override
        )
        result, captures = self.run_script(
            "run_decode.sh", expected_captures=1, DECODE_SPECULATIVE_CONFIG="none"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--speculative-config", captures[0]["args"])

    def test_decode_rejects_incompatible_mtp_before_launch(self) -> None:
        (self.model / "config.json").write_text('{"dspark_block_size":5}\n')
        result, captures = self.run_script(
            "run_decode.sh",
            DECODE_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(captures, [])
        self.assertIn("DSpark checkpoint requires method=dspark", result.stderr)

    def test_decode_rejects_invalid_speculative_metadata_before_launch(self) -> None:
        for metadata, override in (
            ({"dspark_block_size": 0}, "auto"),
            ({"dspark_block_size": True}, "auto"),
            ({}, "[]"),
            ({}, "invalid JSON"),
        ):
            with self.subTest(metadata=metadata, override=override):
                (self.model / "config.json").write_text(json.dumps(metadata))
                result, captures = self.run_script(
                    "run_decode.sh", DECODE_SPECULATIVE_CONFIG=override
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(captures, [])

    def test_sweep_accepts_added_topologies(self) -> None:
        for topology in (
            "legacy",
            "afd_dp12tp2",
            "afd_dp10tp2",
            "afd_dp8tp2",
            "afd_dp6tp2",
        ):
            with self.subTest(topology=topology):
                result, _ = self.run_script(
                    "run_bench_sweep.sh",
                    "--topology",
                    topology,
                    "--record-oom",
                    "synthetic test status; no deployment attempted",
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_benchmark_cli_metadata_and_pd_rejection(self) -> None:
        result, captures = self.run_script(
            "run_bench.sh",
            "--max-concurrency",
            "8",
            "--num-warmups",
            "0",
            expected_captures=1,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result_path = Path(option(captures[0], "--result-dir")) / option(
            captures[0], "--result-filename"
        )
        payload = json.loads(result_path.read_text())
        self.assertEqual(payload["max_concurrency"], 8)
        self.assertEqual(payload["num_warmups"], "0")
        self.assertEqual(payload["attention_dbo"], "1")
        self.assertEqual(payload["skill_contract"], "afd-dsv4-v030")
        result, _ = self.run_script("run_bench.sh", PREFILL_ENABLE_KV_CONNECTOR="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prefill-only", result.stderr)

    def test_sweep_reuses_valid_result_and_rejects_stale_results(self) -> None:
        arguments = ("--rates", "2", "--repeats", "1")
        result, captures = self.run_script(
            "run_bench_sweep.sh", *arguments, expected_captures=1
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result_path = (
            self.root
            / "results/runtime030-test/afd_dp4tp2/chunk_8192"
            / "rps_2/repeat_1/result.json"
        )
        original = json.loads(result_path.read_text())
        self.assertIsNone(original["max_concurrency"])
        self.assertEqual(original["total_input_tokens"], FROZEN_INPUT_TOKENS)
        self.assertEqual(max(original["input_lens"]), FROZEN_MAX_INPUT_TOKENS)
        self.assertEqual(original["completed"], FROZEN_REQUESTS)
        count = len(list(self.captures.iterdir()))
        result, _ = self.run_script("run_bench_sweep.sh", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.captures.iterdir())), count)
        for mutation in (
            {"skill_contract": "old-v026"},
            {"attention_dbo": "0"},
            {"failed": 1},
            {"max_concurrency": 8},
            {"max_concurrency": "none"},
            {"max_concurrency": "None"},
        ):
            with self.subTest(mutation=mutation):
                result_path.write_text(json.dumps({**original, **mutation}))
                result, _ = self.run_script("run_bench_sweep.sh", *arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(list(self.captures.iterdir())), count)
        self.assertEqual(len(captures), 1)

    def test_frozen_dataset_and_cache_rejection(self) -> None:
        output = self.root / "frozen.jsonl"
        command = [
            sys.executable,
            str(SKILL_ROOT / "scripts/prepare_bench_dataset.py"),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(
            (SKILL_ROOT / "assets/workloads/manifest.json").read_text()
        )
        self.assertEqual(
            hashlib.sha256(output.read_bytes()).hexdigest(), manifest["sha256"]
        )
        with output.open() as handle:
            requests = [json.loads(line) for line in handle]
        self.assertEqual(len(requests), FROZEN_REQUESTS)
        self.assertTrue(all(request["output_tokens"] == 1 for request in requests))
        modified = output.stat().st_mtime_ns
        subprocess.run(command, check=True, capture_output=True, timeout=15)
        self.assertEqual(output.stat().st_mtime_ns, modified)
        output.write_text("invalid cache\n")
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "invalid cache\n")

    def test_sweep_accepts_native_client_concurrency(self) -> None:
        for cap in ("0", "none", "8"):
            with self.subTest(cap=cap):
                run_id = f"concurrency-{cap}"
                result, captures = self.run_script(
                    "run_bench_sweep.sh",
                    "--rates",
                    "2",
                    "--repeats",
                    "1",
                    expected_captures=1,
                    DSV4_BENCH_MAX_CONCURRENCY=cap,
                    DSV4_BENCH_RUN_ID=run_id,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                result_path = Path(option(captures[0], "--result-dir")) / option(
                    captures[0], "--result-filename"
                )
                payload = json.loads(result_path.read_text())
                expected = 8 if cap == "8" else None
                self.assertEqual(payload["max_concurrency"], expected)


if __name__ == "__main__":
    unittest.main()
